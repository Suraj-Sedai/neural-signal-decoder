import torch
import torch.nn as nn

class SpectralMLP(nn.Module):
    def __init__(self, num_channels, freq_bins, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels * freq_bins, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)