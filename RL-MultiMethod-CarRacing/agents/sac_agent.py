"""
SAC Agent for CarRacing-v2.
Soft Actor-Critic with automatic entropy tuning.
"""

import copy
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from core.cnn_backbone import CNNBackbone

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    """Stochastic Gaussian policy."""

    def __init__(self, action_dim: int = 3):
        super().__init__()
        self.backbone = CNNBackbone()
        self.mean_layer = nn.Linear(self.backbone.feature_dim, action_dim)
        self.log_std_layer = nn.Linear(self.backbone.feature_dim, action_dim)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True), torch.tanh(mean)


class Critic(nn.Module):
    """Twin soft Q-networks."""

    def __init__(self, action_dim: int = 3):
        super().__init__()
        self.backbone1 = CNNBackbone()
        self.backbone2 = CNNBackbone()
        feat = self.backbone1.feature_dim + action_dim
        self.q1 = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        sa1 = torch.cat([self.backbone1(x), a], dim=-1)
        sa2 = torch.cat([self.backbone2(x), a], dim=-1)
        return self.q1(sa1), self.q2(sa2)


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


class SACAgent:
    """
    Soft Actor-Critic agent with automatic entropy tuning.

    Args:
        action_dim: Continuous action dimension.
        lr: Learning rate (shared for actor and critics).
        gamma: Discount factor.
        tau: Soft target update coefficient.
        alpha: Initial entropy temperature (overridden if auto_tune=True).
        auto_tune: Automatically tune entropy temperature.
        buffer_size: Replay buffer capacity.
        batch_size: Mini-batch size.
        device: Torch device string.
    """

    def __init__(
        self,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune: bool = True,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = Actor(action_dim).to(self.device)
        self.critic = Critic(action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.auto_tune = auto_tune
        if auto_tune:
            self.target_entropy = -action_dim  # heuristic
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        self.replay_buffer = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, mean = self.actor.sample(obs_t)
        out = mean if deterministic else action
        return out.cpu().numpy().squeeze(0)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def store(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size
        )

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs_t)
            q1_t, q2_t = self.critic_target(next_obs_t, next_actions)
            min_q = torch.min(q1_t, q2_t) - self.alpha * next_log_probs
            q_target = rewards_t + self.gamma * min_q * (1 - dones_t)

        q1, q2 = self.critic(obs_t, actions_t)
        critic_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(
            q2, q_target
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        pi, log_pi, _ = self.actor.sample(obs_t)
        q1_pi, q2_pi = self.critic(obs_t, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
        }

        # Entropy temperature update – reuse log_pi from the actor update above
        if self.auto_tune:
            alpha_loss = -(
                self.log_alpha * (log_pi.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            metrics["alpha_loss"] = alpha_loss.item()

        self._soft_update()
        return metrics

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": self.log_alpha if self.auto_tune else None,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        if self.auto_tune and ckpt.get("log_alpha") is not None:
            self.log_alpha.data.copy_(ckpt["log_alpha"].data)
            self.alpha = self.log_alpha.exp().item()
        self.actor.eval()
