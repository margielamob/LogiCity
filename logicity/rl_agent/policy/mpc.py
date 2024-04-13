import torch
import torch.nn as nn
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
import torch as th
from gymnasium import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFModel(BasePolicy):

    action_space: spaces.Discrete
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            net_arch = [64, 64]
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)  # number of actions
        delta_network = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.delta_network = nn.Sequential(*delta_network)
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def get_prediction(self, obs, acs, data_statistics):
        if len(obs.shape) == 1 or len(acs.shape) == 1:
            obs = np.squeeze(obs)[None]
            acs = np.squeeze(acs)[None]

        norm_obs = normalize(obs, data_statistics['obs_mean'], data_statistics['obs_std'])
        norm_acs = normalize(acs, data_statistics['acs_mean'], data_statistics['acs_std'])

        norm_input = ptu.from_numpy(np.concatenate((norm_obs, norm_acs), axis = 1))
        norm_delta = ptu.to_numpy(self.delta_network(norm_input))

        delta = unnormalize(norm_delta, data_statistics['delta_mean'], data_statistics['delta_std'])
        return obs + delta

    def update(self, observations, actions, next_observations, data_statistics):

        norm_obs = normalize(np.squeeze(observations), data_statistics['obs_mean'], data_statistics['obs_std'])
        norm_acs = normalize(np.squeeze(actions), data_statistics['acs_mean'], data_statistics['acs_std'])

        pred_delta = self.delta_network(ptu.from_numpy(np.concatenate((norm_obs, norm_acs), axis = 1)))
        true_delta = ptu.from_numpy(normalize(next_observations - observations, data_statistics['delta_mean'], data_statistics['delta_std']))

        loss = nn.functional.mse_loss(true_delta, pred_delta)

        return loss
    
class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 horizon,
                 n_sequences,
                 ensemble_size: int = 5,
                 dyn_model_n_layers: int = 2,
                 dyn_model_size: int = 64,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.horizon = horizon
        self.N = n_sequences
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = self.ac_space.n
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # dynamics model
        self.dyn_models = []
        self.ensemble_size = ensemble_size
        for i in range(self.ensemble_size):
            model = FFModel(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                features_extractor=FlattenExtractor(self.env.observation_space),
                features_dim=self.ob_dim + self.ac_dim,
                net_arch=[dyn_model_size] * dyn_model_n_layers,
                activation_fn=nn.ReLU,
                normalize_images=False
            )
            self.dyn_models.append(model)
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


    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' or (self.sample_strategy == 'cem' and obs is None):
            return np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
        elif self.sample_strategy == 'cem':
            # Initialize mean and variance for CEM
            mean = np.zeros((horizon, self.ac_dim))
            variance = np.ones((horizon, self.ac_dim)) * ((self.high - self.low) / 2)**2
            for i in range(self.cem_iterations):
                if i == 0:
                    # Initial sampling
                    samples = np.random.uniform(low=self.low, high=self.high, size=(self.N, horizon, self.ac_dim))
                else:
                    samples = np.random.normal(loc=mean, scale=np.sqrt(variance), size=(self.N, horizon, self.ac_dim))
                    samples = np.clip(samples, self.low, self.high)  # Ensure samples are within bounds

                # Evaluate and select elites
                rewards = self.evaluate_candidate_sequences(samples, obs)
                elite_idxs = rewards.argsort()[-self.cem_num_elites:]
                elites = samples[elite_idxs]

                # Update mean and variance
                mean = np.mean(elites, axis=0) * self.cem_alpha + mean * (1 - self.cem_alpha)
                variance = np.var(elites, axis=0) * self.cem_alpha + variance * (1 - self.cem_alpha)

            # return the mean as the optimal action sequence, copy 
            return mean  # Return the mean as the optimal action sequence
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences = 1, horizon = 1)[0, 0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon, obs=obs)

        if self.sample_strategy == 'random':
            # a list you can use for storing the predicted reward for each candidate sequence
            predicted_rewards_per_ens = []

            for model in self.dyn_models:
                sim_obs = np.tile(obs, (self.N, 1))
                model_rewards = np.zeros(self.N)

                for t in range(self.horizon):
                    rew, _ = self.env.get_reward(sim_obs, candidate_action_sequences[:, t, :])
                    model_rewards += rew
                    sim_obs = model.get_prediction(sim_obs, candidate_action_sequences[:, t, :], self.data_statistics)
                predicted_rewards_per_ens.append(model_rewards)

            # calculate mean_across_ensembles(predicted rewards).
            # the matrix dimensions should change as follows: [ens,N] --> N
            predicted_rewards = np.mean(predicted_rewards_per_ens, axis = 0) # TODO(Q2)

            # pick the action sequence and return the 1st element of that sequence
            best_index = np.argmax(predicted_rewards) #TODO(Q2)
            best_action_sequence = candidate_action_sequences[best_index] #TODO(Q2)
            action_to_take = best_action_sequence[0] # TODO(Q2)
        else:
            action_to_take =  candidate_action_sequences[0]
        return action_to_take

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
