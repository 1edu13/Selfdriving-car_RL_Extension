"""
PPO Agent for CarRacing-v2
Proximal Policy Optimization with continuous action space.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from core.cnn_backbone import CNNBackbone, layer_init # Our new shared module

class PPOAgent(nn.Module):
    def __init__(self, envs):
        super(PPOAgent, self).__init__()
        self.network = CNNBackbone(input_channels=4) #

        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 96, 96)
            self.output_dim = self.network.network(dummy_input).shape[1]

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.output_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        ) #

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.output_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
        ) #

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape))) #

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)
