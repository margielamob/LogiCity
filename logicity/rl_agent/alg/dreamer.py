import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
import pathlib
import io
from copy import deepcopy

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from logicity.rl_agent.policy import DreamerPolicy
from logicity.rl_agent.alg.dreamer_helper.buffer import TransitionBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update, check_for_correct_spaces
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.vec_env.patch_gym import _convert_space

SelfDreamer = TypeVar("SelfDreamer", bound="Dreamer")

class Dreamer(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "DreamerPolicy": DreamerPolicy
    }
    policy: DreamerPolicy
    def __init__(
        self,
        policy: Union[str, Type[DreamerPolicy]],
        env: Union[GymEnv, str],
        config_path: str,
        config_name: str,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        seq_len: int = 50,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class = None,
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
        # dynamic import of config
        config = __import__(config_path, fromlist=[config_name])
        config_class = getattr(config, config_name)
        obs_shape = env.observation_space.shape
        action_size = env.action_space.n
        obs_dtype = bool
        action_dtype = np.float32
        self.config = config_class(
            env='LogiCity',
            obs_shape=obs_shape,
            action_size=action_size,
            obs_dtype = obs_dtype,
            action_dtype = action_dtype,
            seq_len = seq_len,
            batch_size = batch_size
        )
        self.action_size = self.config.action_size
        self.pixel = self.config.pixel
        self.kl_info = self.config.kl
        self.seq_len = self.config.seq_len
        self.batch_size = self.config.batch_size
        self.collect_intervals = self.config.collect_intervals
        self.discount = self.config.discount_
        self.lambda_ = self.config.lambda_
        self.horizon = self.config.horizon
        self.loss_scale = self.config.loss_scale
        self.actor_entropy_scale = self.config.actor_entropy_scale
        self.grad_clip_norm = self.config.grad_clip
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.replay_buffer = TransitionBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            seq_len=self.seq_len,
            obs_type=np.float32,
            action_type=np.float32,
        )

        self.policy = self.policy_class(
            self.env,
            self.config,
            self.device,
            self.observation_space,
            self.action_space,
            self.lr_schedule,
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
            total_loss = sum(train_losses_dyn)/self.policy.ensemble_size + 5 * sum(train_losses_rew)/self.policy.ensemble_size
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

    def _store_transition(
        self,
        replay_buffer: TransitionBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_



    def learn(
        self: SelfDreamer,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "MBRL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDreamer:
        callback.on_training_start(locals(), globals())
        rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
        
        obs = self.env.reset()
        done = False
        prev_rssmstate = self.policy.RSSM._init_rssm_state(1)
        prev_action = th.zeros(1, self.action_size).to(self.device)
        episode_actor_ent = []
        
        while self.num_timesteps < total_timesteps:

            if self.num_timesteps % self.train_freq == 0:
                train_metrics = self.train(train_metrics)

            if self.num_timesteps % self.config.slow_target_update == 0:
                self.update_target()            

            with th.no_grad():
                embed = self.policy.ObsEncoder(th.tensor(obs, dtype=th.float32).unsqueeze(0).to(self.device))  
                _, posterior_rssm_state = self.policy.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                model_state = self.policy.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = self.policy.ActionModel(model_state)
                action = self.policy.ActionModel.add_exploration(action, iter).detach()
                action_ent = th.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = self.env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if done:
                self.replay_buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics['train_rewards'] = score
                train_metrics['action_ent'] =  np.mean(episode_actor_ent)
                wandb.log(train_metrics, step=iter)
                scores.append(score)
                if len(scores)>100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average>best_mean_score:
                        best_mean_score = current_average 
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                
                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

        callback.on_training_end()

        return self

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # use on-policy data here
        action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def collect_rollouts(
            self,
            env,
            callback,
            replay_buffer: TransitionBuffer,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
        ):
            """
            Collect experiences and store them into a ``ReplayBuffer``.

            :param env: The training environment
            :param callback: Callback that will be called at each step
                (and at the beginning and end of the rollout)
            :param train_freq: How much experience to collect
                by doing rollouts of current policy.
                Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
                or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
                with ``<n>`` being an integer greater than 0.
            :param action_noise: Action noise that will be used for exploration
                Required for deterministic policy (e.g. TD3). This can also be used
                in addition to the stochastic policy for SAC.
            :param learning_starts: Number of steps before learning for the warm-up phase.
            :param replay_buffer:
            :param log_interval: Log data every ``log_interval`` episodes
            :return:
            """
            # Switch to eval mode (this affects batch norm / dropout)
            self.policy.set_training_mode(False)

            num_collected_steps, num_collected_episodes = 0, 0
            assert self.n_envs == 1, "Only one env is supported for now"
            if self.use_sde:
                self.actor.reset_noise(env.num_envs)

            callback.on_rollout_start()
            continue_training = True

            while num_collected_steps < learning_starts:
                # Select action randomly or according to policy
                unscaled_action = np.array([self.action_space.sample() for _ in range(self.n_envs)])
                buffer_actions = unscaled_action
                actions = buffer_actions
                # Rescale and perform action
                new_obs, rewards, dones, infos = env.step(actions)

                self.num_timesteps += env.num_envs
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if not callback.on_step():
                    return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, dones)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is dones as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                for idx, done in enumerate(dones):
                    if done:
                        # Update stats
                        num_collected_episodes += 1
                        self._episode_num += 1
                        # Log training infos
                        if log_interval is not None and self._episode_num % log_interval == 0:
                            self._dump_logs()
            callback.on_rollout_end()

            return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _dump_logs(self) -> None:
            """
            Write log.
            """
            assert self.ep_info_buffer is not None
            assert self.ep_success_buffer is not None

            time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
            self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            if self.use_sde:
                self.logger.record("train/std", (self.actor.get_std()).mean().item())

            if len(self.ep_success_buffer) > 0:
                self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
            # Pass the number of timesteps for tensorboard
            self.logger.dump(step=self.num_timesteps)

    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfDreamer],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfDreamer:
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