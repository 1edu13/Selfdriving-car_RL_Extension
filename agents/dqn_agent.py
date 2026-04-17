"""
DQN Agent for CarRacing-v2 (discrete action space variant).
Deep Q-Network with experience replay and target network.
"""
import numpy as np
import torch
import torch.nn as nn
import random


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization of weights to improve training stability.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DQNAgent(nn.Module):
    def __init__(self, num_actions=5):
        """
        Deep Q-Network Agent.

        Args:
            num_actions (int): The number of discrete actions.
            For CarRacing-v2, we must discretize the continuous space (e.g., 5 actions:
            0: Do nothing, 1: Accelerate, 2: Brake, 3: Steer Left, 4: Steer Right).
        """
        super(DQNAgent, self).__init__()
        self.num_actions = num_actions

        # 1. Feature Extractor (CNN Backbone) - Identical to PPO Baseline
        # Input: (Batch, 4, 96, 96) - Stacked grayscale frames
        # Output: A flattened vector of spatial features
        self.network = nn.Sequential(
            # Conv 1
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            # Conv 2
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            # Conv 3
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            # Flatten 3D block to 1D vector
            nn.Flatten(),
        )

        # Calculate the size of the CNN output automatically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 96, 96)
            output_dim = self.network(dummy_input).shape[1]

        # 2. Q-Network Head
        # Evaluates the Q-value Q(s, a) for every possible discrete action simultaneously.
        # Input: The output of the CNN feature extractor
        # Output: A vector of size `num_actions` representing expected future rewards.
        self.q_head = nn.Sequential(
            layer_init(nn.Linear(output_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, self.num_actions), std=0.01),
        )

    def forward(self, x):
        """
        Forward pass to calculate Q-values for all actions.
        Args:
            x: Observation tensor of shape (Batch, 4, 96, 96)
        Returns:
            q_values: Tensor of shape (Batch, num_actions)
        """
        # Normalize input pixel values to [0, 1] range (and ensure it is float)
        hidden = self.network(x.float() / 255.0)
        q_values = self.q_head(hidden)
        return q_values


    def get_action(self, x, epsilon=0.0, device="cpu"):
        """
        Selects an action using the epsilon-greedy exploration strategy.

        Args:
            x: Observation tensor (single environment or batched).
            epsilon (float): Probability of choosing a random action (exploration).
            device (torch.device): Device to ensure random action tensors are correctly placed.

        Returns:
            action: The selected discrete action index.
        """
        if random.random() < epsilon:
            # Explore: Choose a random action
            # Handle batch size if x is batched (e.g., during vectorized data collection)
            batch_size = x.shape[0]
            action = torch.randint(0, self.num_actions, (batch_size,), device=device)
        else:
            # Exploit: Choose the action with the highest Q-value
            with torch.no_grad():
                q_values = self.forward(x)
                action = torch.argmax(q_values, dim=1)

        return action