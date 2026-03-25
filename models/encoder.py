"""CNN observation encoder for Minigrid environments.

Minigrid observations are (N, M, 3) tensors where each cell has 3 values:
(object_type, color, state). The default agent view is 7x7.
"""

import torch
import torch.nn as nn


class MinigridEncoder(nn.Module):
    """Encodes Minigrid image observations into flat feature vectors."""

    def __init__(self, obs_shape, channels=(16, 32, 64), hidden_dim=128):
        """
        Args:
            obs_shape: Tuple (H, W, C) of observation shape.
            channels: Conv layer channel sizes.
            hidden_dim: Output feature dimension.
        """
        super().__init__()
        h, w, c = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_size = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
        )

        self.output_dim = hidden_dim

    def forward(self, obs):
        """
        Args:
            obs: (batch, H, W, C) uint8 tensor from Minigrid.
        Returns:
            features: (batch, hidden_dim) float tensor.
        """
        # Minigrid gives (H, W, C), conv expects (C, H, W)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        x = obs.permute(0, 3, 1, 2).float() / 10.0
        return self.fc(self.conv(x))
