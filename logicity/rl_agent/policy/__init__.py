from .neural import CNNFeatureExtractor, MLPFeatureExtractor, MlpPolicy
from .hri import HriPolicy
from .hri_helper import *

build_policy = {
    "MlpPolicy": MlpPolicy,
    "HriPolicy": HriPolicy
}