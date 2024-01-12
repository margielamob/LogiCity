import torch
import torch.nn as nn
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Define your CNN here and make sure it's on the correct device
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 61 * 61, features_dim),
            nn.ReLU()
        ).to(device)

    def forward(self, observations):
        observations = observations.to(device)
        return self.cnn(observations)

# Define the policy with the custom feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
