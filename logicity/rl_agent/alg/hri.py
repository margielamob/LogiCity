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
from logicity.rl_agent.policy.hri.Utils import get_unifs

import logging
logger = logging.getLogger(__name__)

SelfHRI = TypeVar("SelfHRI", bound="HRI")

class HRI(ABC):
    def __init__(self, policy, env, policy_kwargs):
        self.policy_class = policy
        self.policy_kwargs = policy_kwargs
        self.policy = build_policy[policy](env, **policy_kwargs)
        self.policy.to(self.device)

    
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
    
    @classmethod
    def load(
        self,
        path,
        device="cuda"
    ):
        self.policy.load_state_dict(torch.load(path))