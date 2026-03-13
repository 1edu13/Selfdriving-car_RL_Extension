"""
Utility functions: environment wrappers and device helpers.
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from typing import Optional


def get_device(force_cpu: bool = False) -> torch.device:
    """Return the best available torch device."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransposeObservation(gym.ObservationWrapper):
    """Convert HWC numpy observation to CHW torch-compatible format."""

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs shape: (H, W, C) → (C, H, W)
        return np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0


class NormalizeReward(gym.RewardWrapper):
    """Clip rewards to [-1, 1]."""

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, -1.0, 1.0))


def make_env(
    env_id: str = "CarRacing-v2",
    render_mode: Optional[str] = None,
    continuous: bool = True,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Create and wrap a CarRacing environment.

    Args:
        env_id: Gymnasium environment ID.
        render_mode: Rendering mode ('human', 'rgb_array', or None).
        continuous: Whether to use a continuous action space.
        seed: Optional random seed.

    Returns:
        A wrapped Gymnasium environment.
    """
    env = gym.make(
        env_id,
        render_mode=render_mode,
        continuous=continuous,
    )
    env = TransposeObservation(env)
    env = NormalizeReward(env)
    if seed is not None:
        env.reset(seed=seed)
    return env
