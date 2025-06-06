import torch
import torch.nn as nn
import torch.nn.functional as F

class FEN2EvalCNN(nn.Module):
    def __init__(self, extra_feature_size=15):  # ← make this configurable
        super().__init__()

        # Input: (batch, 12, 8, 8)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),  # (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (128, 8, 8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # (128, 4, 4)
        )

        # Dynamically compute flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 12, 8, 8)
            out = self.conv_layers(dummy_input)
            self.flatten_size = out.view(1, -1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size + extra_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, board_tensor, extras):
        x = self.conv_layers(board_tensor)        # (B, 128, 4, 4)
        x = x.view(x.size(0), -1)                 # Flatten
        x = torch.cat([x, extras], dim=1)
        x = self.fc_layers(x)
        return x.squeeze(1)
