from .neural import CNNFeatureExtractor, MLPFeatureExtractor, MlpPolicy
from .hri import HriPolicy
from .nlm import NLMPolicy
from .hri_helper import *
from .nlm_helper import *

build_policy = {
    "MlpPolicy": MlpPolicy,
    "HriPolicy": HriPolicy,
    "NLMPolicy": NLMPolicy
}