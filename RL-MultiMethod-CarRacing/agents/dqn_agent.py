"""
DQN Agent for CarRacing-v2 (discrete action space variant).
Deep Q-Network with experience replay and target network.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.cnn_backbone import CNNBackbone


class QNetwork(nn.Module):
    """Q-network that maps observations to Q-values for each discrete action."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = nn.Linear(self.backbone.feature_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for discrete CarRacing action spaces.

    The continuous actions are discretised into a finite set defined by
    *action_space* (list of numpy arrays).

    Args:
        action_space: List of discrete actions (each a numpy array).
        lr: Learning rate.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_end: Final exploration rate.
        epsilon_decay: Number of steps over which to anneal epsilon.
        buffer_size: Replay buffer capacity.
        batch_size: Mini-batch size for updates.
        target_update_freq: Steps between target network syncs.
        device: Torch device string.
    """

    # Default discrete action set for CarRacing-v2
    DEFAULT_ACTIONS = [
        np.array([0.0, 0.0, 0.0]),   # no-op
        np.array([-1.0, 0.0, 0.0]),  # steer left
        np.array([1.0, 0.0, 0.0]),   # steer right
        np.array([0.0, 1.0, 0.0]),   # accelerate
        np.array([0.0, 0.0, 0.8]),   # brake
    ]

    def __init__(
        self,
        action_space=None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 100_000,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        device: str = "cpu",
    ):
        self.action_space = action_space or self.DEFAULT_ACTIONS
        self.n_actions = len(self.action_space)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self._steps = 0

        self.q_net = QNetwork(self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Epsilon-greedy action selection."""
        self._steps += 1
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
        )

        if random.random() < self.epsilon:
            idx = random.randrange(self.n_actions)
        else:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                idx = self.q_net(obs_t).argmax(dim=1).item()
        return self.action_space[idx].copy()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def store(self, obs, action_idx, reward, next_obs, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(obs, action_idx, reward, next_obs, done)

    def update(self) -> dict:
        """Sample a mini-batch and perform a gradient update."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size
        )

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(obs_t).gather(1, actions_t)

        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(1, keepdim=True)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss.item(), "epsilon": self.epsilon}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.q_net.eval()
