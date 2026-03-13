"""
PPO Agent for CarRacing-v2
Proximal Policy Optimization with continuous action space.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

from core.cnn_backbone import CNNBackbone


class ActorCritic(nn.Module):
    """Combined Actor-Critic network using a shared CNN backbone."""

    def __init__(self, action_dim: int = 3):
        super().__init__()
        self.backbone = CNNBackbone()
        feature_dim = self.backbone.feature_dim

        # Actor head (policy)
        self.actor_mean = nn.Linear(feature_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        mean = torch.tanh(self.actor_mean(features))
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(features)
        return mean, std, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.critic(features)

    def get_action(self, x: torch.Tensor):
        mean, std, value = self.forward(x)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action.clamp(-1, 1), log_prob, value


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Args:
        action_dim: Dimension of the continuous action space.
        lr: Learning rate.
        gamma: Discount factor.
        clip_eps: PPO clipping epsilon.
        n_epochs: Number of update epochs per rollout.
        device: Torch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.device = torch.device(device)

        self.policy = ActorCritic(action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Rollout buffer helpers
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray):
        """Select an action given a single observation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(obs_t)
        return (
            action.cpu().numpy().squeeze(0),
            log_prob.cpu().item(),
            value.cpu().item(),
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, rollout: dict) -> dict:
        """
        Update policy using a collected rollout buffer.

        Args:
            rollout: dict with keys 'obs', 'actions', 'log_probs',
                     'returns', 'advantages'.

        Returns:
            dict with 'policy_loss', 'value_loss', 'entropy'.
        """
        obs = torch.FloatTensor(rollout["obs"]).to(self.device)
        actions = torch.FloatTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        returns = torch.FloatTensor(rollout["returns"]).to(self.device)
        advantages = torch.FloatTensor(rollout["advantages"]).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.n_epochs):
            mean, std, values = self.policy(obs)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1).mean()

            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()

        for k in metrics:
            metrics[k] /= self.n_epochs
        return metrics

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
