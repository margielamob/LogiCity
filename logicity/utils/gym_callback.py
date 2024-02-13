from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

class EvalCheckpointCallback(CheckpointCallback):
    def __init__(self, eval_env, exp_name, eval_freq=50000, *args, **kwargs):
        super(EvalCheckpointCallback, self).__init__(*args, **kwargs)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.exp_name = exp_name
        self.best_mean_reward = -np.inf

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
            for episode in range(100):  # Number of episodes for evaluation
                obs = self.eval_env.reset()
                episode_rewards = 0
                step = 0
                done = False
                while not done:
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _info = self.eval_env.step(action)
                    episode_rewards += reward
                    step += 1
                rewards_list.append(episode_rewards)

            mean_reward = np.mean(rewards_list)
            logger.info(f"Step: {self.n_calls} - Mean Reward: {mean_reward}")

            # Log the mean reward
            with open(os.path.join(self.save_path, "{}_eval_rewards.txt".format(self.exp_name)), "a") as file:
                file.write(f"Step: {self.n_calls} - Mean Reward: {mean_reward}\n")

            # Update the best model if current mean reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("{}/best_model.zip".format(self.save_path))

        return True
