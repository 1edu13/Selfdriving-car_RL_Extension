import os
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from core.utils import make_env, get_device


class DQNEvaluator:
    """
    Evaluator for DQN models on CarRacing-v2.
    Captures metrics, videos, and generates a brief summary of the discrete agent's performance.
    """

    def __init__(self, model_path: str, num_episodes: int = 10, seed: int = 42):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.seed = seed
        self.device = get_device()
        self.model_name = os.path.basename(model_path).replace(".pth", "")

        # Output directories setup
        self.output_dir = Path("results/metrics") / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(self):
        print(f"Loading DQN Model from: {self.model_path}")

        # 1. Initialize Network
        agent = DQNAgent(num_actions=5).to(self.device)
        agent.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # Set to evaluation mode (disables dropout/batchnorm if any were used, though our CNN doesn't use them)
        agent.eval()

        # 2. Environment setup
        # CRITICAL: is_discrete=True must be set to map continuous physics to our 5 DQN actions
        env_func = make_env(
            env_id="CarRacing-v2",
            seed=self.seed,
            idx=0,
            capture_video=True,
            run_name=f"{self.model_name}_eval",
            is_discrete=True
        )
        env = env_func()

        episode_rewards = []
        episode_lengths = []

        # 3. Evaluation Loop
        for ep in range(self.num_episodes):
            obs, _ = env.reset(seed=self.seed + ep)
            done = False
            ep_reward = 0
            ep_steps = 0

            while not done:
                # Format observation for the network: (Batch, Channels, Height, Width)
                obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(self.device)

                # Action Selection: Epsilon is 0.0 (Pure exploitation, no random actions)
                with torch.no_grad():
                    action_tensor = agent.get_action(obs_tensor, epsilon=0.0, device=self.device)
                action = action_tensor.cpu().numpy()[0]

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                ep_reward += reward
                ep_steps += 1
                obs = next_obs

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_steps)
            print(f"Episode {ep + 1}/{self.num_episodes} - Reward: {ep_reward:.2f} - Steps: {ep_steps}")

        env.close()

        # 4. Statistics Calculation
        stats = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths))
        }

        print("\n" + "=" * 40)
        print("DQN EVALUATION RESULTS")
        print("=" * 40)
        for k, v in stats.items():
            print(f"{k}: {v:.2f}")
        print("=" * 40)

        # 5. Save results to disk for comparative analysis later
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(stats, f, indent=4)

        return stats


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Update this path once your train_dqn.py script saves a model checkpoint
    MODEL_PATH = "models/dqn_baseline/dqn_step_10000.pth"
    if os.path.exists(MODEL_PATH):
        evaluator = DQNEvaluator(model_path=MODEL_PATH, num_episodes=10)
        evaluator.run_evaluation()
    else:
        print(f"❌ Error: Model not found at {MODEL_PATH}.")
        print("Please ensure you have trained the agent and updated the MODEL_PATH variable.")