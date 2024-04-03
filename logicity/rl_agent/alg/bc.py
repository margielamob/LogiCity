import numpy as np
import pickle as pkl
import io
import pathlib
import time
import torch
import torch.nn as nn
from abc import ABC
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file, recursive_getattr, recursive_setattr
from typing import TypeVar

import torch.nn.functional as F
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from logicity.rl_agent.policy import build_policy

import logging
logger = logging.getLogger(__name__)

SelfBehavioralCloning = TypeVar("SelfBehavioralCloning", bound="BehavioralCloning")

class BehavioralCloning(ABC):
    def __init__(self, policy, env, policy_kwargs, 
                 num_traj,
                 expert_demonstrations,
                 optimizer,
                 device="cuda",
                 batch_size=64,
                 tensorboard_log=None,
                 log_interval=10):
        self.policy_class = policy
        self.policy_kwargs = policy_kwargs
        self.num_traj = num_traj
        self.expert_demonstrations = expert_demonstrations
        self.optimizer_config = optimizer
        self.policy = build_policy[policy](env, **policy_kwargs)
        self.optimizer = self.build_optimizer(optimizer)
        expert_data = self.load_expert_data(expert_demonstrations, num_traj)
        self.build_dataloader(expert_data, batch_size)
        # setup tensorboard
        self.tensorboard_log = tensorboard_log
        self.log_interval = log_interval
        self.device = device
        # discrete action space
        self.loss = nn.CrossEntropyLoss()
        self.policy.to(self.device)


    def convert_listofrollouts(self, paths, num_traj):
        """
            Take a list of rollout dictionaries
            and return separate arrays,
            where each array is a concatenation of that array from across the rollouts
        """
        observations = []
        actions = []
        next_observations = []
        rewards = []
        logger.info("Loaded {} trajectories".format(num_traj))
        for path in paths[:num_traj]:
            for step in path:
                observations.append(step["state"])
                actions.append(step["action"])
                next_observations.append(step["next_state"])
                rewards.append(step["reward"])
        logger.info("Loaded {} steps".format(len(observations)))
        observations = np.array(observations)
        actions = np.array(actions)
        next_observations = np.array(next_observations)
        rewards = np.array(rewards)
        self.num_timesteps = 0

        return observations, actions, rewards, next_observations

    def build_dataloader(self, expert_data, batch_size):
        observations, actions = expert_data["observations"], expert_data["actions"]
        # split train and validation data, random shuffle
        train_observations, train_actions = observations, actions
        train_dataset = TensorDataset(torch.tensor(train_observations), torch.tensor(train_actions))
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
            logger.info(f"Tensorboard log saved at {self.tensorboard_log}/{tb_log_name}")

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            tb_log_name,
        )
        # use the callback to signal the start of the training
        self.policy.train()
        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:
            for batch in self.train_loader:
                observations, actions = batch
                observations = observations.to(self.device).float()
                actions = actions.to(self.device).long()
                self.optimizer.zero_grad()
                action_logits, _ = self.policy(observations)
                loss = self.loss(action_logits, actions)
                loss.backward()
                self.optimizer.step()
                self.num_timesteps += 1
                # use the callback to signal the training step
                callback.on_step()
                if self.tensorboard_log is not None and self.num_timesteps % self.log_interval == 0:
                    self.writer.add_scalar("Loss", loss, self.num_timesteps)
                    logger.info(f"Step: {self.num_timesteps}/{total_timesteps}, Loss: {loss}")
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
            action = action.argmax(dim=-1)
        
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
    
    @classmethod
    def load(
        cls: SelfBehavioralCloning,
        path,
        env,
        device="cuda",
        **kwargs
    ):


        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            **kwargs
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
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

        return model
    
    def get_parameters(self):
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params


    def save(
        self,
        path,
        ):
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        exclude = []
        exclude = set(exclude).union(self._excluded_save_params())


        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)

    def _get_torch_save_params(self):
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["policy"]

        return state_dicts, []
    
    def set_parameters(
        self,
        load_path_or_dict,
        exact_match,
        device="auto",
    ):
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, torch.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])  # type: ignore[arg-type]
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )
        

    def _excluded_save_params(self):
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "writer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]