"""
TD3 Agent for CarRacing-v2.
Twin Delayed Deep Deterministic Policy Gradient (TD3).

This file contains the PyTorch implementation of the TD3 agent.
TD3 is an actor-critic algorithm that addresses the overestimation bias in DDPG 
by using two Q-networks and delaying the actor updates.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization of weights to improve training stability.
    This helps the network maintain gradient variance during backpropagation.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, action_dim=3):
        """
        Actor Network for continuous action spaces.
        It maps the current state to a continuous action vector.
        
        Args:
            action_dim (int): The number of dimensions in the continuous action space.
                              For CarRacing-v2, it's 3 (steering, gas, brake).
        """
        super(Actor, self).__init__()
        
        # 1. Feature Extractor (CNN Backbone)
        # Extracts spatial features from the stacked grayscale frames.
        # Input: (Batch, 4, 96, 96)
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(), # Flattens the 3D tensor into a 1D vector
        )

        # Automatically compute the size of the feature vector coming out of the CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 96, 96)
            output_dim = self.network(dummy_input).shape[1]

        # 2. Action Head
        # A fully connected network that maps extracted features to action values.
        # The output activation is Tanh to bound the actions between -1 and 1.
        self.action_head = nn.Sequential(
            layer_init(nn.Linear(output_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01),
            nn.Tanh() # Tanh ensures the output is symmetrically bounded in [-1, 1]
        )

    def forward(self, x):
        """
        Computes the action to take given the current state.
        
        Args:
            x: State tensor of shape (Batch, 4, 96, 96)
            
        Returns:
            scaled_action: The properly bounded continuous action for the environment.
        """
        # Normalize pixel values to [0, 1]
        hidden = self.network(x / 255.0)
        action = self.action_head(hidden)
        
        # Action Scaling for CarRacing-v2:
        # The environment expects:
        # Steering: [-1, 1]
        # Gas: [0, 1]
        # Brake: [0, 1]
        # Since Tanh outputs values in [-1, 1], we adjust the ranges for gas and brake.
        
        steer = action[:, 0:1] # Already in [-1, 1]
        gas = (action[:, 1:2] + 1.0) / 2.0 # Transformed from [-1, 1] to [0, 1]
        brake = (action[:, 2:3] + 1.0) / 2.0 # Transformed from [-1, 1] to [0, 1]
        
        scaled_action = torch.cat([steer, gas, brake], dim=1)
        return scaled_action


class Critic(nn.Module):
    def __init__(self, action_dim=3):
        """
        Critic Network with twin architecture (Two Q-Networks).
        TD3 uses two critics to estimate the Q-value and takes the minimum of both to 
        reduce overestimation bias.
        
        Args:
            action_dim (int): The number of dimensions in the continuous action space.
        """
        super(Critic, self).__init__()
        
        # ------------------------ Q-Network 1 ------------------------
        # Feature extractor for Critic 1
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
            dummy_input = torch.zeros(1, 4, 96, 96)
            cnn_out_dim = self.network_1(dummy_input).shape[1]
            
        # Value head for Critic 1
        # The input is the concatenation of the state features AND the continuous action
        self.q1_head = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim + action_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1)) # Outputs a single Q-Value
        )
        
        # ------------------------ Q-Network 2 ------------------------
        # Feature extractor for Critic 2 (independent weights)
        self.network_2 = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Value head for Critic 2
        self.q2_head = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim + action_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1)) # Outputs a single Q-Value
        )

    def forward(self, state, action):
        """
        Computes both Q-Values for a given state-action pair.
        
        Args:
            state: State tensor of shape (Batch, 4, 96, 96)
            action: Action tensor of shape (Batch, action_dim)
            
        Returns:
            q1, q2: The Q-values estimated by network 1 and network 2.
        """
        state_norm = state / 255.0
        
        # Forward pass for Critic 1
        feat_1 = self.network_1(state_norm)
        # Concatenating state features with actions (s, a) -> Q(s, a)
        q1_in = torch.cat([feat_1, action], dim=1)
        q1 = self.q1_head(q1_in)
        
        # Forward pass for Critic 2
        feat_2 = self.network_2(state_norm)
        # Concatenating state features with actions
        q2_in = torch.cat([feat_2, action], dim=1)
        q2 = self.q2_head(q2_in)
        
        return q1, q2

    def q1(self, state, action):
        """
        Computes only the Q-Value for the first critic network.
        This is used during the Actor network update, as we only need one Q-function's 
        gradient to update the policy.
        """
        state_norm = state / 255.0
        feat_1 = self.network_1(state_norm)
        q1_in = torch.cat([feat_1, action], dim=1)
        return self.q1_head(q1_in)
