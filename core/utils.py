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

class FrameSkipWrapper(gym.Wrapper):
    """
    Action-repeat wrapper: repeats the agent's chosen action for `skip` consecutive
    physics frames, summing the rewards. Returns the observation from the LAST frame.

    This is the single most effective speedup for CPU-bottlenecked RL training:
    the environment's physics simulation and rendering (CPU) is the bottleneck,
    not the neural network forward pass (GPU). By repeating actions, we cut
    the number of expensive env.step() calls by `skip`x.

    For CarRacing-v2 at 50 FPS physics:
      - skip=1 -> agent decides every frame    (slowest, most precise)
      - skip=2 -> agent decides every 2 frames (balance)
      - skip=4 -> agent decides every 4 frames (4x faster, recommended for training)
    """
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env(env_id, seed, idx, capture_video, run_name, is_discrete=False, frame_skip=4):
    """
    Utility function to create and configure the environment.

    Args:
        env_id (str): The environment ID (e.g., "CarRacing-v2").
        seed (int): Global seed for reproducibility.
        idx (int): Index of the environment (for vectorized environments).
        capture_video (bool): Whether to save videos of the agent driving.
        run_name (str): Name of the experiment for video saving.
        is_discrete (bool): If True, applies the DiscreteActionWrapper for DQN.
        frame_skip (int): Frames to repeat each action. Default=4 for max training speed.
                          Higher = faster training, coarser control.
    """
    def thunk():
        # Initialize the environment (rgb_array = no window, render into array only)
        env = gym.make(env_id, render_mode="rgb_array")

        # Wrapper to record videos (optional, usually for evaluation or the first env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"results/videos/{run_name}")

        # Apply discretization if using DQN
        if is_discrete:
            env = DiscreteActionWrapper(env)

        # Frame-skip: repeat each action for `frame_skip` physics steps.
        # This is the primary CPU bottleneck reduction — CarRacing renders every frame
        # even in rgb_array mode. skip=4 cuts env.step() calls by 4x.
        if frame_skip > 1:
            env = FrameSkipWrapper(env, skip=frame_skip)

        # 1. Grayscale: reduce 96x96x3 -> 96x96x1 (less data to move CPU->GPU)
        env = GrayScaleObservation(env, keep_dim=False)

        # 2. Frame Stacking: stack 4 consecutive frames for temporal information
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