import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from logicity.rl_agent.alg.infra.utils import *
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from gymnasium import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFModel(nn.Module):

    action_space: spaces.Discrete
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super(FFModel, self).__init__()
        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_dim = int(self.action_space.n)  # number of actions
        delta_network = create_mlp(observation_space.shape[0]+self.action_dim, observation_space.shape[0]*3, self.net_arch, self.activation_fn)
        self.delta_network = nn.Sequential(*delta_network)

    def forward(self, obs, acs, data_statistics):
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        bs = obs.shape[0]
        acs = F.one_hot(acs, num_classes = self.action_dim).float()
        acs = acs.squeeze(1)
        input = torch.cat((obs, acs), dim = 1)
        delta = self.delta_network(input)
        return delta.view(bs, -1, 3)
    
    def predict(self, obs, acs, data_statistics):
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """

        delta = self(obs, acs, data_statistics)
        probabilities = F.softmax(delta, dim=2)
        _, predicted_classes = torch.max(probabilities, dim=2)
        predicted_changes = predicted_classes - 1

        return obs + predicted_changes
    
    def learn(self, obs, acs, next_obs, data_statistics):
        """
        Train the model.
        """
        delta_pred = self(obs, acs, data_statistics)
        delta = next_obs - obs
        target_indices = (delta + 1).long()

        loss = F.cross_entropy(delta_pred.view(-1, 3), target_indices.view(-1), reduction='mean')
        return loss

class RWModel(nn.Module):

    action_space: spaces.Discrete
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super(RWModel, self).__init__()
        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_dim = int(self.action_space.n)  # number of actions
        pred_network = create_mlp(observation_space.shape[0]+self.action_dim, 1, self.net_arch, self.activation_fn)
        self.pred_network = nn.Sequential(*pred_network)
        self.rew_mean = None
        self.rew_std = None

    def forward(self, obs, acs, data_statistics):
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # transform acs to one hot
        acs = F.one_hot(acs, num_classes = self.action_dim).float()
        acs = acs.squeeze(1)
        input = torch.cat((obs, acs), dim = 1)
        unnormed_rew = self.pred_network(input)
        normed_rew = unnormed_rew * data_statistics['rew_std'] + data_statistics['rew_mean']
        return normed_rew
    
    def learn(self, obs, acs, rewards, data_statistics):
        """
        Train the model.
        """
        reward_pred = self(obs, acs, data_statistics)
        loss = F.l1_loss(reward_pred, rewards)
        return loss

class MPCPolicy(BasePolicy):

    def __init__(
        self,
        env,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        dyn_model_kwargs: Dict[str, Any],
        horizon: int,
        n_sequences: int,
        sample_strategy: str,
        cem_iterations: int = 4,
        cem_num_elites: int = 10,
        cem_alpha: float = 1.0,
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
        self.horizon = horizon
        self.N = n_sequences
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = self.ac_space.n

        # dynamics model
        self.dyn_models = []
        self.ensemble_size = dyn_model_kwargs['ensemble_size']
        self.dyn_model_size = dyn_model_kwargs['dyn_model_size']
        self.dyn_model_n_layers = dyn_model_kwargs['dyn_model_n_layers']
        self._build(lr_schedule)
        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        # note: In MBRL, we do not use the feature extractor, the dyn model and reward model have already do this. 
        # features_extractor = self.make_features_extractor()
        self.dyn_models = []
        all_parameters = []  # List to store all model parameters

        for _ in range(self.ensemble_size):
            model = FFModel(
                observation_space=self.observation_space,
                action_space=self.action_space,
                net_arch=[self.dyn_model_size] * self.dyn_model_n_layers,
            )
            model.to(device)
            self.dyn_models.append(model)
            all_parameters += list(model.parameters())  # Collect parameters from each model

        self.rew_model = RWModel(
            observation_space=self.observation_space,
            action_space=self.action_space
        )

        all_parameters += list(self.rew_model.parameters())  # Collect parameters from each model
        # Setup optimizer with all collected parameters
        self.optimizer = self.optimizer_class(
            all_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

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

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> torch.Tensor:
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences = 1, horizon = 1)[0, 0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon, obs=obs)

        if self.sample_strategy == 'random':
            # A list you can use for storing the predicted reward for each candidate sequence
            predicted_rewards_per_ens = []

            for model in self.dyn_models:
                sim_obs = obs.repeat(self.N, 1)
                model_rewards = torch.zeros(self.N, device=self.device)

                for t in range(self.horizon):
                    # Predict rewards using a neural network reward prediction model
                    rew = self.rew_model(sim_obs, candidate_action_sequences[:, t:t+1], self.data_statistics)
                    model_rewards += rew.squeeze(1)
                    sim_obs = model.predict(sim_obs, candidate_action_sequences[:, t:t+1], self.data_statistics)
                    sim_obs = torch.clamp(sim_obs, 0, 1)
                    if deterministic:
                        sim_obs = sim_obs.round()
                    else:
                        sim_obs = torch.bernoulli(sim_obs)

                predicted_rewards_per_ens.append(model_rewards)

            # Calculate mean across ensembles (predicted rewards)
            predicted_rewards = torch.stack(predicted_rewards_per_ens).mean(dim=0)

            # Pick the action sequence and return the 1st element of that sequence
            best_index = torch.argmax(predicted_rewards).item()
            best_action_sequence = candidate_action_sequences[best_index]
            action_to_take = best_action_sequence[0]
        else:
            # If not using random sampling, just take the first action sequence
            action_to_take = candidate_action_sequences[0]

        return action_to_take  # Assuming you need the action in NumPy format outside this function


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
