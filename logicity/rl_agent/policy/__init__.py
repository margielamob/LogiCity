from .neural import CNNFeatureExtractor, MLPFeatureExtractor, MlpPolicy
from .hri import HriPolicy
from .nlm import NLMPolicy
from .hri_helper import *
from .nlm_helper import *

# Supervised learning
build_policy = {
    "MlpPolicy": MlpPolicy,
    "HriPolicy": HriPolicy,
    "NLMPolicy": NLMPolicy
}

# Reinforcement learning
from .nlm_rl import ActorCriticNLMPolicy