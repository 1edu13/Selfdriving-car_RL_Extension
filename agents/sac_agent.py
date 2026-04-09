"""
SAC Agent for CarRacing-v2.
Soft Actor-Critic (SAC) implementation.

SAC is an off-policy algorithm built on the maximum entropy reinforcement learning framework.
Unlike TD3, the Policy (Actor) is stochastic and aims to maximize a trade-off between
expected return (rewards) and entropy (randomness in the policy).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Constants to keep the Standard Deviation of our policy distribution mathematically stable
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization of weights to improve training stability.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, action_dim=3):
        """
        Stochastic Actor Network for SAC.
        Instead of outputting a definite action (like TD3), it outputs a Probability 
        Distribution (Mean and Standard Deviation) for each action dimension.
        """
        super(Actor, self).__init__()
        
        # 1. Feature Extractor (CNN Backbone)
        # Extracts visual features from the (Batch, 4, 96, 96) grayscale stacked frames
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 96, 96)
            output_dim = self.network(dummy_input).shape[1]

        # 2. Fully connected hidden layers
        self.fc_1 = nn.Sequential(
            layer_init(nn.Linear(output_dim, 512)),
            nn.ReLU(),
        )
        
        # 3. Two separate outputs: The Mean (mu) and the Log-Standard Deviation (log_std)
        self.fc_mean = layer_init(nn.Linear(512, action_dim))
        self.fc_log_std = layer_init(nn.Linear(512, action_dim))

    def forward(self, x):
        """
        Calculates the mean and log_std of the action distribution given a state.
        Usually only used indirectly; `get_action` is the primary method to call.
        """
        hidden = self.network(x / 255.0)
        fc_out = self.fc_1(hidden)
        
        mean = self.fc_mean(fc_out)
        log_std = self.fc_log_std(fc_out)
        
        # Clamp log_std to avoid numerical instability (e.g. exploding variance or divide-by-zero)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def get_action(self, x, deterministic=False):
        """
        Samples an action from the policy distribution and calculates its log probability.
        
        Args:
            x: State tensor
            deterministic: If True, bypasses random sampling and just uses the Mean (used during testing)
            
        Returns:
            action: The squashed action in range [-1, 1]
            log_prob: The log probability of the sampled action (for entropy calculation)
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()
        
        # Create a Normal Gaussian Distribution
        normal = Normal(mean, std)
        
        if deterministic:
            # For inference/testing, we just use the most probable action (the mean)
            x_t = mean
        else:
            # rsample() samples with the "reparameterization trick", meaning the computational 
            # graph remains connected so we can backpropagate gradients through the sampling process!
            x_t = normal.rsample()  
            
        # SAC requires actions to be bounded, so we apply a Tanh squashing function.
        # This constrains the infinite Gaussian distribution to a finite [-1, 1] range.
        action = torch.tanh(x_t)
        
        # Because we squashed the distribution with Tanh, we MUST correct the probability density.
        # This is done using the Jacobian of the Tanh transformation.
        log_prob = normal.log_prob(x_t)
        # log(1 - tanh(x)^2) gives the correction. We add epsilon to prevent log(0) if action is close to -1 or 1.
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        
        # Sum the log probabilities across all action dimensions to get the total log_prob for this step
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, action_dim=3):
        """
        Critic Network for SAC.
        Exactly like TD3, it uses a Twin Q-Network architecture to fight overestimation bias.
        It estimates Q(state, action) -> Value.
        """
        super(Critic, self).__init__()
        
        # ------------------------ Q-Network 1 ------------------------
        self.network_1 = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 96, 96)
            cnn_out_dim = self.network_1(dummy).shape[1]
            
        self.q1_head = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim + action_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1))
        )
        
        # ------------------------ Q-Network 2 ------------------------
        self.network_2 = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.q2_head = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim + action_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1))
        )

    def forward(self, state, action):
        """
        Computes the Q-Values for both Twin networks given a state-action pair.
        """
        state_norm = state / 255.0
        
        # Critic 1
        feat_1 = self.network_1(state_norm)
        q1_in = torch.cat([feat_1, action], dim=1)
        q1 = self.q1_head(q1_in)
        
        # Critic 2
        feat_2 = self.network_2(state_norm)
        q2_in = torch.cat([feat_2, action], dim=1)
        q2 = self.q2_head(q2_in)
        
        return q1, q2
