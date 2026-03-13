"""
TD3 Agent for CarRacing-v2.
Twin Delayed Deep Deterministic Policy Gradient (TD3).
"""

import copy
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.cnn_backbone import CNNBackbone


class Actor(nn.Module):
    """Deterministic policy network."""

    def __init__(self, action_dim: int = 3):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class Critic(nn.Module):
    """Twin Q-networks (Q1, Q2)."""

    def __init__(self, action_dim: int = 3):
        super().__init__()
        self.backbone = CNNBackbone()
        feat = self.backbone.feature_dim + action_dim
        self.q1 = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        features = self.backbone(x)
        sa = torch.cat([features, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        sa = torch.cat([features, a], dim=-1)
        return self.q1(sa)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent.

    Args:
        action_dim: Continuous action dimension.
        lr_actor: Learning rate for the actor.
        lr_critic: Learning rate for the critic.
        gamma: Discount factor.
        tau: Soft target update coefficient.
        policy_noise: Std of Gaussian noise added to target actions.
        noise_clip: Clip range for target policy noise.
        policy_delay: Frequency of actor and target updates.
        buffer_size: Replay buffer capacity.
        batch_size: Mini-batch size.
        device: Torch device string.
    """

    def __init__(
        self,
        action_dim: int = 3,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.device = torch.device(device)
        self._steps = 0

        self.actor = Actor(action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _soft_update(self, net: nn.Module, target: nn.Module):
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, noise: float = 0.1) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().squeeze(0)
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
        return action.clip(-1, 1)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def store(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self._steps += 1
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size
        )

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Target policy smoothing
        noise = (torch.randn_like(actions_t) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        next_actions = (self.actor_target(next_obs_t) + noise).clamp(-1, 1)

        # Critic update
        with torch.no_grad():
            q1_t, q2_t = self.critic_target(next_obs_t, next_actions)
            q_target = rewards_t + self.gamma * torch.min(q1_t, q2_t) * (1 - dones_t)

        q1, q2 = self.critic(obs_t, actions_t)
        critic_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(
            q2, q_target
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {"critic_loss": critic_loss.item()}

        # Delayed actor update
        if self._steps % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(obs_t, self.actor(obs_t)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            metrics["actor_loss"] = actor_loss.item()

        return metrics

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.eval()
