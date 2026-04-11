import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import GrayScaleObservation, FrameStack

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Wrapper to convert discrete actions from DQN into continuous actions for CarRacing-v2.
    Action mapping:
    0: Do nothing
    1: Steer Left
    2: Steer Right
    3: Accelerate (Gas)
    4: Brake
    """
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        # 5 discrete actions
        self.action_space = gym.spaces.Discrete(5)
        
        # Mapping to [steering, gas, brake]
        self._action_mapping = {
            0: np.array([0.0, 0.0, 0.0], dtype=np.float32),   # Do nothing
            1: np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # Steer Left
            2: np.array([1.0, 0.0, 0.0], dtype=np.float32),   # Steer Right
            3: np.array([0.0, 1.0, 0.0], dtype=np.float32),   # Gas
            4: np.array([0.0, 0.0, 0.8], dtype=np.float32)    # Brake
        }

    def action(self, act):
        # Convert the integer action back to the continuous array
        return self._action_mapping[int(act)]

def make_env(env_id, seed, idx, capture_video, run_name, is_discrete=False):
    """
    Utility function to create and configure the environment.

    Args:
        env_id (str): The environment ID (e.g., "CarRacing-v2").
        seed (int): Global seed for reproducibility.
        idx (int): Index of the environment (for vectorized environments).
        capture_video (bool): Whether to save videos of the agent driving.
        run_name (str): Name of the experiment for video saving.
        is_discrete (bool): If True, applies the DiscreteActionWrapper for DQN.
    """
    def thunk():
        # Initialize the environment
        env = gym.make(env_id, render_mode="rgb_array")

        # Wrapper to record videos (optional, usually for evaluation or the first env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"results/videos/{run_name}")

        # Apply discretization if using DQN
        if is_discrete:
            env = DiscreteActionWrapper(env)

        # 1. Grayscale Conversion
        env = GrayScaleObservation(env, keep_dim=False)

        # 2. Frame Stacking
        env = FrameStack(env, 4)

        # Seed the environment for reproducibility
        env.action_space.seed(seed + idx)

        return env

    return thunk

def get_device():
    """Returns the best available device (CUDA GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info():
    """
    Returns a dictionary with hardware information for training diagnostics.
    Used by the master pipeline script to display a summary before training.
    """
    import platform
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "os": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
    else:
        info["gpu_name"] = "N/A (CPU mode)"
        info["gpu_vram_gb"] = 0

    return info