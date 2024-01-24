import torch
import torch.nn as nn
from torchvision.models import resnet18
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Define networks for 2D map and position data
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 64)
        self.pos_extractor = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        self.merge_layer = nn.Linear(128, features_dim)

    def forward(self, observations):
        map_obs = observations["map"]
        pos_data = observations["position"]
        map_features = self.resnet(map_obs)
        pos_features = self.pos_extractor(pos_data)
        merged = torch.cat([map_features, pos_features], dim=1)
        return self.merge_layer(merged)
    
class MLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(MLPFeatureExtractor, self).__init__(observation_space, features_dim)

        # Assuming the observation space is a Box and has a shape attribute
        n_input_features = observation_space.shape[0]

        # Define the MLP layers
        self.net = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )

    def forward(self, observations):
        return self.net(observations)