import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim)

        # Assuming the observation space is a Box and has a shape attribute
        n_input_features = observation_space.shape[0]
        self.embedding = nn.Linear(n_input_features, features_dim)
        self.transformer = nn.Transformer(d_model=features_dim, num_encoder_layers=3, num_decoder_layers=3, 
                                          dim_feedforward=1024, batch_first=True)

    def forward(self, observations):
        embed = self.embedding(observations)
        return self.transformer(src=embed, tgt=embed)
    
class TransformerPolicy(nn.Module):
    def __init__(self, gym_env, features_extractor_class, features_extractor_kwargs):
        super(TransformerPolicy, self).__init__()

        self.features_extractor = features_extractor_class(gym_env.observation_space, **features_extractor_kwargs)
        # Adjust for discrete action spaces
        if isinstance(gym_env.action_space, gym.spaces.Discrete):
            n_output = gym_env.action_space.n  # Number of discrete actions
        else:
            # This is just a fallback for continuous spaces; adjust as necessary
            n_output = gym_env.action_space.shape[0]

        # Create the output layer
        self.action_net = nn.Linear(self.features_extractor.features_dim, n_output)
        self.value_net = nn.Linear(self.features_extractor.features_dim, 1)

    def forward(self, observations):
        # Extract features
        features = self.features_extractor(observations)
        # Get the action logits
        action_logits = self.action_net(features)
        # Get the value
        values = self.value_net(features)
        return action_logits, values