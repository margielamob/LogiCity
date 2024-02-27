from .neural import CNNFeatureExtractor, MLPFeatureExtractor, MlpPolicy
from .hri import HriPolicy

build_policy = {
    "MlpPolicy": MlpPolicy,
    "HriPolicy": HriPolicy
}