import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks.nav.neural.backbones import get_backbone

class NaviNet(nn.Module):
    def __init__(self, backbone, pos_hidden, decoder_hidden, output_size):
        super(NaviNet, self).__init__()
        self.backbone, self.img_dim = get_backbone(backbone)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, pos_hidden),
            nn.ReLU(),
            nn.Linear(pos_hidden, pos_hidden),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(pos_hidden + self.img_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, output_size)
            # Softmax removed since nn.CrossEntropyLoss will be used
        )
        self.output_size = output_size

    def forward(self, img, goal):
        visual_feat = self.backbone(img)
        pos_embedding = self.pos_mlp(goal)
        combined_features = torch.cat([visual_feat, pos_embedding], dim=1)
        action_logits = self.decoder(combined_features)
        return action_logits 