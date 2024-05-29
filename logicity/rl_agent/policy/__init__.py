from .neural import CNNFeatureExtractor, MLPFeatureExtractor, MlpPolicy
from .gnn import GNNPolicy
from .transformer import TransformerPolicy
from .hri import HriPolicy
from .nlm import NLMPolicy
from .hri_helper import *
from .nlm_helper import *
# Reinforcement learning
from .nlm_ac import ActorCriticNLMPolicy
from .nlm_dqn import DQNNLMPolicy, QNetwork
from .mpc import MPCPolicy
from .dreamer import DreamerPolicy
from .mpces import MPCPolicyES

# Supervised learning
build_policy = {
    "MlpPolicy": MlpPolicy,
    "GNNPolicy": GNNPolicy,
    "TransformerPolicy": TransformerPolicy,
    "HriPolicy": HriPolicy,
    "NLMPolicy": NLMPolicy,
    "ActorCriticNLMPolicy": ActorCriticNLMPolicy,
    "DQNNLMPolicy": DQNNLMPolicy,
    "MPCPolicy": MPCPolicy,
    "DreamerPolicy": DreamerPolicy,
    "MPCPolicyES": MPCPolicyES
}