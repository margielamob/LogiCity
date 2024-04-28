import numpy as np 
import random 
from collections import namedtuple, deque
from typing import NamedTuple
from gymnasium import spaces
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
import torch as th

from stable_baselines3.common.buffers import BaseBuffer

class TransitionBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs,
        seq_len: int, 
        obs_type=np.float32,
        action_type=np.float32,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
        )
        assert n_envs == 1, "Only one env is supported for now"
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.obs_type = obs_type
        self.action_type = action_type
        self.seq_len = seq_len
        # Dreamer needs one-hot encoded actions
        self.action_dim = action_space.n
        self.pos = 0
        self.full = False
        self.observations = np.empty((self.buffer_size, self.n_envs, *self.obs_shape), dtype=obs_type) 
        self.actions = np.empty((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.buffer_size, self.n_envs), dtype=np.float32) 
        self.dones = np.empty((self.buffer_size, self.n_envs), dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # change action to one hot, use numpy
        reference = np.zeros((self.n_envs, self.action_dim))
        reference[np.arange(self.n_envs), action] = 1

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        self.actions[self.pos] = np.array(reference)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.buffer_size if self.full else self.pos - L)
            idxs = np.arange(idx, idx + L) % self.buffer_size
            valid_idx = not self.pos in idxs[1:] 
        return idxs

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.observations[vec_idxs]
        return observation.reshape(l, n, *self.obs_shape), self.actions[vec_idxs].reshape(l, n, -1), self.rewards[vec_idxs].reshape(l, n), self.dones[vec_idxs].reshape(l, n)

    def sample(self, batch_size: int, env = None):
        n = batch_size
        l = self.seq_len+1
        obs,act,rew,term = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        obs,act,rew,term = self._shift_sequences(obs,act,rew,term)
        # transform action to one hot
        return obs,act,rew,term

    def _get_samples(self, batch_inds: np.ndarray, env = None):
        pass
    
    def _shift_sequences(self, obs, actions, rewards, terminals):
        obs = obs[1:]
        actions = actions[:-1]
        rewards = rewards[:-1]
        terminals = terminals[:-1]
        return obs, actions, rewards, terminals
    