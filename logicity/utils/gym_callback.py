from stable_baselines3.common.callbacks import CheckpointCallback
from logicity.utils.load import CityLoader
from logicity.utils.gym_wrapper import GymCityWrapper
import numpy as np
import os
import logging
import pickle as pkl
logger = logging.getLogger(__name__)

def make_env(simulation_config, episode_cache=None, return_cache=False): 
    # Unpack arguments from simulation_config and pass them to CityLoader
    city, cached_observation = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    env = GymCityWrapper(city)
    if return_cache: 
        return env, cached_observation
    else:
        return env

class EvalCheckpointCallback(CheckpointCallback):
    def __init__(self, exp_name, simulation_config, episode_data, eval_freq=50000, *args, **kwargs):
        super(EvalCheckpointCallback, self).__init__(*args, **kwargs)
        self.eval_freq = eval_freq
        self.exp_name = exp_name
        self.best_mean_reward = -np.inf
        self.simulation_config = simulation_config
        with open(episode_data, "rb") as f:
            self.episode_data = pkl.load(f)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        # Perform evaluation at specified intervals
        if self.n_calls % self.eval_freq == 0:
            rewards_list = []
            success = []
            for ts in list(self.episode_data.keys()):  # Number of episodes for evaluation
                logger.info("Evaluating episode {}...".format(ts))
                episode_cache = self.episode_data[ts]
                if "label_info" in episode_cache:
                    logger.info("Episode label: {}".format(episode_cache["label_info"]))
                eval_env = make_env(self.simulation_config, episode_cache, False)
                obs = eval_env.init()
                episode_rewards = 0
                step = 0
                done = False
                while not done:
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(int(action))
                    if info["Fail"][0]:
                        episode_rewards += reward
                        break
                    episode_rewards += reward
                    step += 1
                if info["success"]:
                    logger.info("Episode {} success.".format(ts))
                    success.append(1)
                else:
                    logger.info("Episode {} failed.".format(ts))
                    success.append(0)
                rewards_list.append(episode_rewards)
                logger.info("Episode {} achieved a score of {}".format(ts, episode_rewards))


            mean_reward = np.mean(rewards_list)
            sr = np.mean(success)
            logger.info(f"Step: {self.n_calls} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")

            # Log the mean reward
            with open(os.path.join(self.save_path, "{}_eval_rewards.txt".format(self.exp_name)), "a") as file:
                file.write(f"Step: {self.n_calls} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")

            # Update the best model if current mean reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("{}/best_model.zip".format(self.save_path))

        return True
