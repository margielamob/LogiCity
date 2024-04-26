import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
import pathlib
import io

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
        self.data_statistics = None
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # self._setup_lr_schedule()
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
            self.learning_rate,
            **self.mpc_kwargs,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        if self.data_statistics is not None:
            self.policy.data_statistics = self.data_statistics

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Implementation of your training logic
        # For example, use self.buffer to train your model
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        # self._update_learning_rate(self.policy.optimizer)

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
            loss_dyn_all = sum(train_losses_dyn)/self.policy.ensemble_size
            loss_rew_all = sum(train_losses_rew)/self.policy.ensemble_size

            self.policy.model_optimizer.zero_grad()
            loss_dyn_all.backward()
            self.policy.model_optimizer.step()

            self.policy.rew_optimizer.zero_grad()
            loss_rew_all.backward()
            th.nn.utils.clip_grad_norm_(self.policy.rew_model.parameters(), self.max_grad_norm)
            self.policy.rew_optimizer.step()

            losses["dyn"].append(loss_dyn_all.detach().cpu().numpy())
            losses["rew"].append(loss_rew_all.detach().cpu().numpy())

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


    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfMBRL],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfMBRL:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            data["policy_kwargs"].update(kwargs["policy_kwargs"])

        mpc_kwargs = kwargs["policy_kwargs"]
        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            mpc_kwargs=mpc_kwargs,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model