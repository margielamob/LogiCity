import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from torch import optim
from torch import nn

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from logicity.rl_agent.policy import MPCPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update, check_for_correct_spaces
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.vec_env.patch_gym import _convert_space

SelfMBRL = TypeVar("SelfMBRL", bound="MBRL")

class MBRL(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MPCPolicy": MPCPolicy
    }
    policy: MPCPolicy
    def __init__(
        self,
        policy: Union[str, Type[MPCPolicy]],
        env: Union[GymEnv, str],
        mpc_kwargs: Dict[str, Any],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        self.mpc_kwargs = mpc_kwargs
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )

        self.policy = self.policy_class(
            self.env,
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.mpc_kwargs,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Implementation of your training logic
        # For example, use self.buffer to train your model
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = {
            "dyn": [],
            "rew": []
        }
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size * self.policy.ensemble_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            train_losses_dyn = []
            train_losses_rew = []
            ob_no = replay_data.observations
            ac_na = replay_data.actions
            rew = replay_data.rewards
            next_ob_no = replay_data.next_observations

            num_data = ob_no.shape[0]
            num_data_per_ens = int(num_data / self.policy.ensemble_size)

            for i in range(self.policy.ensemble_size):
                # select which datapoints to use for this model of the ensemble
                start_idx = i * num_data_per_ens
                end_idx = (i + 1) * num_data_per_ens if i < self.policy.ensemble_size - 1 else num_data

                observations = ob_no[start_idx:end_idx]
                actions = ac_na[start_idx:end_idx]
                next_observations = next_ob_no[start_idx:end_idx]
                rewards = rew[start_idx:end_idx]

                # use datapoints to update one of the dyn_models
                model = self.policy.dyn_models[i]
                loss_dyn = model.learn(observations, actions, next_observations, self.data_statistics)
                loss_rew = self.policy.rew_model.learn(observations, actions, rewards, self.data_statistics)
                train_losses_dyn.append(loss_dyn)
                train_losses_rew.append(loss_rew)

            # Optimize the policy
            total_loss = sum(train_losses_dyn)/self.policy.ensemble_size + sum(train_losses_rew)/self.policy.ensemble_size
            self.policy.optimizer.zero_grad()
            total_loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses["dyn"].append(sum(train_losses_dyn).detach().cpu().numpy()/self.policy.ensemble_size)
            losses["rew"].append(sum(train_losses_rew).detach().cpu().numpy()/self.policy.ensemble_size)

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss_dyn", np.mean(losses["dyn"]))
        self.logger.record("train/loss_rew", np.mean(losses["rew"]))

    def _on_step(self) -> None:
        """
        Get updated mean/std of the data in our replay buffer
        """
        self._n_calls += 1
        observations = th.tensor(self.replay_buffer.observations, device=self.device)
        actions = th.tensor(self.replay_buffer.actions, device=self.device)
        next_observations = th.tensor(self.replay_buffer.next_observations, device=self.device)
        rewards = th.tensor(self.replay_buffer.rewards, device=self.device)

        deltas = next_observations - observations

        self.data_statistics = {
            # 'obs_mean': observations.mean(dim=0),
            # 'obs_std': observations.std(dim=0),
            # 'acs_mean': actions.mean(dim=0),
            # 'acs_std': actions.std(dim=0),
            # 'delta_mean': deltas.mean(dim=0),
            # 'delta_std': deltas.std(dim=0),
            'rew_mean': rewards.mean(),
            'rew_std': rewards.std(),
        }
        self.policy.data_statistics = self.data_statistics

    def learn(
        self: SelfMBRL,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "MBRL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfMBRL:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # use on-policy data here
        action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state
