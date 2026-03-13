"""
Shared CNN backbone for all RL agents.
Processes 96x96 RGB CarRacing-v2 frames.
"""

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    Convolutional feature extractor shared across all agent architectures.

    Input : (B, 3, 96, 96)  – normalised to [0, 1]
    Output: (B, feature_dim) flat feature vector
    """

    feature_dim: int = 256

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # Layer 1: 3 → 32, kernel 8, stride 4  → (B, 32, 23, 23)
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Layer 2: 32 → 64, kernel 4, stride 2  → (B, 64, 10, 10)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Layer 3: 64 → 64, kernel 3, stride 1  → (B, 64, 8, 8)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Observation tensor of shape (B, 3, 96, 96) in [0, 1].
        Returns:
            Feature vector of shape (B, feature_dim).
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.fc(self.conv(x))
