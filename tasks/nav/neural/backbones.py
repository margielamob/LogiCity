import torch
import torchvision
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights

BACKBONES = {
    'resnet50': resnet50(weights=ResNet50_Weights.DEFAULT),
    'resnet101': resnet101(weights=ResNet101_Weights.DEFAULT),
    # Ensure you only load the model once and reuse it.
    # 'dinov2_vits14': torch.hub.load('facebookresearch/dino:main', 'dino_vits14'),
}

def get_backbone(name):
    net = BACKBONES[name]
    if name.startswith('resnet'):
        net.fc = torch.nn.Identity()
    elif 'dinov2' in name:
        # Assuming the DINO model has a classifier attribute
        net.classifier = torch.nn.Identity()
    
    # Now pass a dummy image tensor through the network to infer the feature dimension
    # Create a dummy tensor with the right dimensions: batch_size, channels, height, width.
    # For example, for an image size of 224x224 with 3 channels:
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Move the model to CPU for this operation to avoid unnecessary GPU usage
    net = net.to('cpu')
    
    # Get the output feature vector
    with torch.no_grad():
        feature_vector = net(dummy_input)
    
    # The dimension of the feature vector is the size of its last dimension
    feature_dim = feature_vector.size(1)
    
    return net, feature_dim