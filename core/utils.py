"""
Utility functions: environment wrappers and device helpers.
"""
import gymnasium as gym
import torch
from gymnasium.wrappers import GrayScaleObservation, FrameStack


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        if capture_video and idx == 0:
            # Videos will now be saved in results/videos/
            env = gym.wrappers.RecordVideo(env, f"results/videos/{run_name}")

        env = GrayScaleObservation(env, keep_dim=False)  #
        env = FrameStack(env, 4)  #
        env.action_space.seed(seed + idx)
        return env

    return thunk


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  #