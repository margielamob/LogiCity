import pickle
import copy
import time

import logging
logger = logging.getLogger(__name__)

class Random:
    def __init__(self, env):
        """
        Random policy for the expert policy.
        """
        self.env = env

    def predict(self, observation, deterministic=True):
        """
        Randomly take the action to take in the environment.
        """
        action = self.env.action_space.sample()
        return action, None