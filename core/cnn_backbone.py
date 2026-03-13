"""
Shared CNN backbone for all RL agents.
Processes 96x96 RGB CarRacing-v2 frames.
"""
import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization of weights to improve training stability.
    Ensures that the initial gradients are well-behaved.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNBackbone(nn.Module):
    """
    Standardized Feature Extractor for CarRacing-v2 Baseline.
    Input: (Batch, 4, 96, 96) - Stacked grayscale frames.
    Output: Flattened vector of 64 * 7 * 7 = 3136 features.
    """

    def __init__(self, input_channels=4):
        super(CNNBackbone, self).__init__()

        self.network = nn.Sequential(
            # Conv 1: Detects low-level features (lines, edges).
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),

            # Conv 2: Detects abstract features (curves, road borders).
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),

            # Conv 3: Refines spatial relationships.
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),

            nn.Flatten(),
        )

    def forward(self, x):
        # Normalization (0-255 -> 0.0-1.0) is handled here to ensure
        # consistency across all agent types.
        return self.network(x / 255.0)