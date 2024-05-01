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
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from logicity.core.config import *
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
        self.fov = AGENT_FOV * AGENT_FOV
        self.binary_mask = np.zeros((self.fov, observation_space.shape[0]//self.fov))
        # -4 are dx, dy, priority, and path lenth
        self.binary_mask[:, :-4] = 1
        self.activation_fn = activation_fn
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_dim = int(self.action_space.n)  # number of actions
        discrete_delta_network = create_mlp(observation_space.shape[0]+self.action_dim, observation_space.shape[0]*3, self.net_arch, self.activation_fn)
        self.delta_network = nn.Sequential(*discrete_delta_network)

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
        delta = delta.view(bs, -1, 3)
        # check the last dim of obs, if it is 0/1
        if obs[0, -1].round() != obs[0, -1]:
            # the last dim is not binary, we don't need to predict it
            delta[:, -1, :] *= 0
        return delta
    
    def predict(self, obs, acs, data_statistics):
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """

        delta = self(obs, acs, data_statistics)
        predicted_changes = torch.zeros_like(obs)
        probabilities = F.softmax(delta, dim=2)
        _, predicted_classes = torch.max(probabilities, dim=2)

        if obs[0, -1].round() != obs[0, -1]:
            # the last dim is not binary, we don't need to predict it
            predicted_changes[:, :-1] = predicted_classes[:, :-1] - 1
        else:
            predicted_changes = predicted_classes - 1

        return obs + predicted_changes
    
    def learn(self, obs, acs, next_obs, data_statistics):
        """
        Train the model.
        """
        delta_pred = self(obs, acs, data_statistics)
        delta = next_obs - obs
        target_indices = (delta + 1).long()
        if obs[0, -1].round() != obs[0, -1]:
            # the last dim is not binary, we don't need to predict it
            delta_pred = delta_pred[:, :-1, :]
            target_indices = target_indices[:, :-1]

        # Reshape for cross-entropy
        delta_pred = delta_pred.reshape(-1, 3)
        target_indices = target_indices.reshape(-1)

            # Compute class weights
        class_counts = torch.bincount(target_indices, minlength=3)
        total_counts = target_indices.size(0)
        class_weights = total_counts / (class_counts + 1e-6)  # Add a small constant to avoid division by zero

        # Normalize weights so that min weight is 1.0
        class_weights = class_weights / class_weights.min()

        # Apply weights to cross-entropy loss
        loss = F.cross_entropy(delta_pred, target_indices, weight=class_weights, reduction='mean')

        # loss = F.cross_entropy(delta_pred.reshape(-1, 3), target_indices.reshape(-1), reduction='mean')
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
        # Calculate threshold using data statistics
        mean_reward = data_statistics['rew_mean']
        std_reward = data_statistics['rew_std']
        
        # Setting threshold to mean - 2*std could typically cover the lower 5% of data assuming normal distribution
        threshold = mean_reward - 2 * std_reward
        
        # Adjust weight based on standard deviation
        high_penalty_weight = max(10, 5 * std_reward)  # Ensure a minimum weight of 10

        # Create masks for different reward areas based on threshold
        high_penalty_mask = (rewards <= threshold).float()
        low_penalty_mask = (rewards > threshold).float()
        
        # Calculate weights: increased weight for high penalty areas
        weights = high_penalty_weight * high_penalty_mask + low_penalty_mask

        # Calculate weighted L1 loss
        loss = (F.l1_loss(reward_pred, rewards, reduction='none') * weights).mean()
        # loss = F.l1_loss(reward_pred, rewards)
        return loss

class MPCPolicyES(BasePolicy):

    def __init__(
        self,
        env,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        learning_rate: Dict[str, Union[float, Schedule]],
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
        self.dyn_models = ModuleList()
        self.ensemble_size = dyn_model_kwargs['ensemble_size']
        self.dyn_model_size = dyn_model_kwargs['dyn_model_size']
        self.dyn_model_n_layers = dyn_model_kwargs['dyn_model_n_layers']
        self._build(learning_rate)
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

    def _build(self, learning_rate) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
        """
        all_parameters = []

        for _ in range(self.ensemble_size):
            model = FFModel(
                observation_space=self.observation_space,
                action_space=self.action_space,
                net_arch=[self.dyn_model_size] * self.dyn_model_n_layers,
            )
            model.to(device)
            self.dyn_models.append(model)  # Append model to ModuleList
            all_parameters += list(model.parameters())

        self.rew_model = RWModel(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        self.model_optimizer = self.optimizer_class(
            all_parameters,
            lr=learning_rate["model"],
            **self.optimizer_kwargs,
        )
        self.rew_optimizer = self.optimizer_class(
            self.rew_model.parameters(),
            lr=learning_rate["reward"],
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
