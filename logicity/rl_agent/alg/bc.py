import numpy as np
import pickle as pkl
import io
import pathlib
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
import torch.nn.functional as F
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from logicity.rl_agent.policy import build_policy

class BehavioralCloning:
    def __init__(self, policy, env, policy_kwargs, 
                 num_traj,
                 expert_demonstrations,
                 optimizer,
                 device="cuda",
                 batch_size=64,
                 tensorboard_log=None):

        self.policy = build_policy[policy](env, **policy_kwargs)
        self.optimizer = self.build_optimizer(optimizer)
        expert_data = self.load_expert_data(expert_demonstrations, num_traj)
        self.build_dataloader(expert_data, batch_size)
        # setup tensorboard
        self.tensorboard_log = tensorboard_log
        self.device = device
        self.loss = nn.MSELoss()
        self.policy.to(self.device)


    def convert_listofrollouts(self, paths, num_traj, concat_rew=True):
        """
            Take a list of rollout dictionaries
            and return separate arrays,
            where each array is a concatenation of that array from across the rollouts
        """
        observations = []
        actions = []
        next_observations = []
        rewards = []
        for path in paths[:num_traj]:
            for step in path:
                observations.append(step["state"])
                actions.append(step["action"])
                next_observations.append(step["next_state"])
                rewards.append(step["reward"])
        observations = np.array(observations)
        actions = np.array(actions)
        next_observations = np.array(next_observations)
        rewards = np.array(rewards)

        return observations, actions, rewards, next_observations

    def build_dataloader(self, expert_data, batch_size):
        observations, actions = expert_data["observations"], expert_data["actions"]
        # split train and validation data, random shuffle
        ratio = 0.95
        n = len(observations)
        split = int(n * ratio)
        indices = np.random.permutation(n)
        train_indices, val_indices = indices[:split], indices[split:]
        train_observations, train_actions = observations[train_indices], actions[train_indices]
        val_observations, val_actions = observations[val_indices], actions[val_indices]
        train_dataset = TensorDataset(torch.tensor(train_observations), torch.tensor(train_actions))
        val_dataset = TensorDataset(torch.tensor(val_observations), torch.tensor(val_actions))
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        return

    def build_optimizer(self, optimizer_config):
        optimizer = getattr(torch.optim, optimizer_config["type"])
        return optimizer(self.policy.parameters(), **optimizer_config["args"])
    
    def load_expert_data(self, expert_data_path, num_traj):
        with open(expert_data_path, 'rb') as f:
            data = pkl.load(f)
        observations, actions, _, _ = self.convert_listofrollouts(data, num_traj)
        return {
            "observations": observations,
            "actions": actions
        }
    
    def learn(self, total_timesteps, callback, tb_log_name):
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of episodes before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: the trained model
        """

        # setup tensorboard
        if self.tensorboard_log is not None:
            self.writer = SummaryWriter(log_dir=self.tensorboard_log + "/" + tb_log_name)

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            tb_log_name,
        )
        # use the callback to signal the start of the training
        self.policy.train()
        callback.on_training_start(locals(), globals())
        step = 0
        while step < total_timesteps:
            for batch in self.train_loader:
                observations, actions = batch
                observations = observations.to(self.device).float()
                actions = actions.to(self.device).float()
                self.optimizer.zero_grad()
                action_logits, _ = self.policy(observations)
                action_probs = F.softmax(action_logits, dim=-1)
                loss = self.loss(action_probs, actions)
                loss.backward()
                self.optimizer.step()
                step += 1
                # use the callback to signal the training step
                callback.on_step()
                if self.tensorboard_log is not None:
                    self.writer.add_scalar("Loss", loss, step)

        callback.on_training_end()
        return self

    
    def predict(self, observation, deterministic=False):
        if self.policy.training:
            self.policy.eval()
        observation = torch.tensor(observation).to(self.device).float()
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            action_logits, _ = self.policy(observation)
            action = F.softmax(action_logits, dim=-1)
        
        return action.cpu().numpy(), None
    
    def _setup_learn(
        self,
        total_timesteps: int,
        callback = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback
    
    def _init_callback(
        self,
        callback,
        progress_bar: bool = False,
    ):
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback
    
    def load(self, load_path):
        """
        Load the model from a zip-file

        :param load_path: the path to the zip file
        """
        data, params = load_from_zip_file(load_path)
        self.__dict__.update(params)
        self.policy.load_state_dict(data["policy"])
        self.optimizer.load_state_dict(data["optimizer"])

    def save(self, save_path):
        """
        Save the current parameters to file

        :param save_path: The path to the file
        """
        data = {
            "policy": self.policy,
            "optimizer": self.optimizer
        }
        save_to_zip_file(save_path, data)