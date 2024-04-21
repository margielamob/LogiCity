import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from logicity.rl_agent.policy.dreamer_helper.utils.module import get_parameters, FreezeParameters
from logicity.rl_agent.policy.dreamer_helper.utils.algorithm import compute_return

from logicity.rl_agent.policy.dreamer_helper.models.actor import DiscreteActionModel
from logicity.rl_agent.policy.dreamer_helper.models.dense import DenseModel
from logicity.rl_agent.policy.dreamer_helper.models.rssm import RSSM
from logicity.rl_agent.policy.dreamer_helper.models.pixel import ObsDecoder, ObsEncoder
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from gymnasium import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DreamerPolicy(BasePolicy):

    def __init__(
        self,
        env,
        config,
        device,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        # init vars
        # MPC policy needs env.get_reward for planning
        self.env = env
        self._model_initialize(config)
        self._optim_initialize(config)

    def _model_initialize(self, config):
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size']
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size*class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size 
    
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type, config.rssm_info).to(self.device)
        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor, config.expl).to(self.device)
        self.RewardDecoder = DenseModel((1,), modelstate_size, config.reward).to(self.device)
        self.ValueModel = DenseModel((1,), modelstate_size, config.critic).to(self.device)
        self.TargetValueModel = DenseModel((1,), modelstate_size, config.critic).to(self.device)
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())
        
        if config.discount['use']:
            self.DiscountModel = DenseModel((1,), modelstate_size, config.discount).to(self.device)
        if config.pixel:
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device)
            self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device)

    def _optim_initialize(self, config):
        model_lr = config.lr['model']
        actor_lr = config.lr['actor']
        value_lr = config.lr['critic']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = self.optimizer_class(get_parameters(self.world_list), model_lr)
        self.actor_optimizer = self.optimizer_class(get_parameters(self.actor_list), actor_lr)
        self.value_optimizer = self.optimizer_class(get_parameters(self.value_list), value_lr)

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' or (self.sample_strategy == 'cem' and obs is None):
            # Random sampling for discrete actions
            return torch.randint(low=0, high=self.ac_dim, size=(num_sequences, horizon), device=self.device)
        elif self.sample_strategy == 'cem':
            # Initialize mean and variance for CEM
            mean = torch.zeros((horizon, self.ac_dim), device=self.device)
            variance = torch.ones((horizon, self.ac_dim), device=self.device) * ((self.high - self.low) / 2)**2

            for i in range(self.cem_iterations):
                if i == 0:
                    # Initial uniform sampling within bounds
                    samples = torch.rand((self.N, horizon, self.ac_dim), device=self.device) * (self.high - self.low) + self.low
                else:
                    # Subsequent samples based on updated mean and variance
                    samples = torch.normal(mean=mean, std=variance.sqrt(), size=(self.N, horizon, self.ac_dim)).to(self.device)
                    samples = torch.clamp(samples, self.low, self.high)  # Ensure samples are within bounds

                # Evaluate and select elites
                rewards = self.evaluate_candidate_sequences(samples, obs)
                _, elite_idxs = torch.topk(rewards, self.cem_num_elites, largest=True, sorted=False)
                elites = samples[elite_idxs]

                # Update mean and variance
                mean = elites.mean(dim=0) * self.cem_alpha + mean * (1 - self.cem_alpha)
                variance = elites.var(dim=0) * self.cem_alpha + variance * (1 - self.cem_alpha)

            # Return the mean as the optimal action sequence
            return mean
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1).squeeze()

        # Sample random actions for all observations in the batch (batch_size x N x horizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=obs.size(0) * self.N, horizon=self.horizon, obs=obs)

        # Reshape to (batch_size, N, horizon) for processing
        candidate_action_sequences = candidate_action_sequences.view(obs.size(0), self.N, self.horizon)

        predicted_rewards_per_ens = []

        for model in self.dyn_models:
            # Expand observations to match the number of sequences (batch_size, N, obs_dim)
            sim_obs = obs.unsqueeze(1).repeat(1, self.N, 1).view(-1, obs.size(1))

            model_rewards = torch.zeros(obs.size(0) * self.N, device=self.device)

            for t in range(self.horizon):
                # Flatten candidate_action_sequences for batch processing in model
                actions = candidate_action_sequences[:, :, t].view(-1)
                rew = self.rew_model(sim_obs, actions, self.data_statistics).view(-1)
                model_rewards += rew
                sim_obs = model.predict(sim_obs, actions, self.data_statistics)
                sim_obs = torch.clamp(sim_obs, 0, 1)
                if deterministic:
                    sim_obs = sim_obs.round()
                else:
                    sim_obs = torch.bernoulli(sim_obs)

            # Reshape rewards back to (batch_size, N) and append
            model_rewards = model_rewards.view(obs.size(0), self.N)
            predicted_rewards_per_ens.append(model_rewards)

        # Calculate mean across ensembles (predicted rewards)
        predicted_rewards = torch.stack(predicted_rewards_per_ens).mean(dim=0)

        # Find best actions for each batch
        best_indices = torch.argmax(predicted_rewards, dim=1)  # Best index for each batch
        actions_to_take = torch.gather(candidate_action_sequences[:, :, 0], 1, best_indices.unsqueeze(1)).squeeze(1)

        return actions_to_take


    def evaluate_candidate_sequences(self, action_sequences, initial_obs):
        """
        Evaluates a batch of action sequences starting from an initial observation.

        Args:
        - action_sequences (numpy.ndarray): A batch of action sequences to evaluate.
        - initial_obs (numpy.ndarray): The initial observation/state from which each sequence begins.

        Returns:
        - numpy.ndarray: A one-dimensional array containing the total rewards for each sequence.
        """
        # a list you can use for storing the predicted reward for each candidate sequence
        predicted_rewards_per_ens = []

        for model in self.dyn_models:
            sim_obs = np.tile(initial_obs, (self.N, 1))
            model_rewards = np.zeros(self.N)

            for t in range(self.horizon):
                rew, _ = self.env.get_reward(sim_obs, action_sequences[:, t, :])
                model_rewards += rew
                sim_obs = model.get_prediction(sim_obs, action_sequences[:, t, :], self.data_statistics)
            predicted_rewards_per_ens.append(model_rewards)

        return np.mean(predicted_rewards_per_ens, axis = 0)
