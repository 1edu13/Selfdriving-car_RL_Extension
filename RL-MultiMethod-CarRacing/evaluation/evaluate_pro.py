"""
evaluate_pro.py – Robust individual model evaluation.

Runs a trained agent for multiple episodes and saves:
  - A JSON summary with statistics (mean, std, min, max reward)
  - An optional MP4 video of the best episode
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.utils import make_env, get_device


def load_agent(method: str, model_path: str, device: str):
    """Load an agent by method name and model path."""
    if method == "ppo":
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(device=device)
        agent.load(model_path)
        return agent
    if method == "dqn":
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(device=device)
        agent.load(model_path)
        return agent
    if method == "td3":
        from agents.td3_agent import TD3Agent
        agent = TD3Agent(device=device)
        agent.load(model_path)
        return agent
    if method == "sac":
        from agents.sac_agent import SACAgent
        agent = SACAgent(device=device)
        agent.load(model_path)
        return agent
    raise ValueError(f"Unknown method: {method!r}")


def select_action(agent, obs: np.ndarray, method: str) -> np.ndarray:
    """Dispatch action selection based on method."""
    if method == "ppo":
        action, _, _ = agent.select_action(obs)
        return action
    if method == "dqn":
        return agent.select_action(obs)
    if method == "td3":
        return agent.select_action(obs, noise=0.0)
    if method == "sac":
        return agent.select_action(obs, deterministic=True)
    raise ValueError(f"Unknown method: {method!r}")


class RobustEvaluator:
    """Evaluate a single trained model over multiple episodes."""

    def __init__(self, agent, method: str, n_episodes: int = 10, seed: int = 0):
        self.agent = agent
        self.method = method
        self.n_episodes = n_episodes
        self.seed = seed

    def run(self, save_video: bool = False, video_dir: str = "results/evaluation_results"):
        env = make_env(
            render_mode="rgb_array" if save_video else None,
            seed=self.seed,
        )

        rewards, lengths = [], []
        best_reward = -float("inf")
        best_frames = []

        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=self.seed + ep)
            ep_reward, ep_steps = 0.0, 0
            frames = []

            while True:
                if save_video:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

                action = select_action(self.agent, obs, self.method)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_steps += 1

                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            lengths.append(ep_steps)
            print(f"  Episode {ep + 1:3d}: reward={ep_reward:.2f}, steps={ep_steps}")

            if ep_reward > best_reward:
                best_reward = ep_reward
                best_frames = frames

        env.close()

        results = {
            "method": self.method,
            "n_episodes": self.n_episodes,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_steps": float(np.mean(lengths)),
            "rewards": rewards,
        }

        if save_video and best_frames:
            self._save_video(best_frames, video_dir)

        return results

    def _save_video(self, frames: list, video_dir: str):
        try:
            import imageio
            os.makedirs(video_dir, exist_ok=True)
            path = os.path.join(video_dir, f"{self.method}_best_episode.mp4")
            imageio.mimsave(path, frames, fps=30)
            print(f"  Video saved to {path}")
        except ImportError:
            print("  imageio not installed – skipping video save.")


def evaluate(args):
    device = str(get_device(force_cpu=args.cpu))
    agent = load_agent(args.method, args.model_path, device)

    evaluator = RobustEvaluator(
        agent, method=args.method, n_episodes=args.n_episodes, seed=args.seed
    )
    print(f"Evaluating {args.method.upper()} model: {args.model_path}")
    results = evaluator.run(save_video=args.save_video)

    print(
        f"\nResults: mean={results['mean_reward']:.2f} ± {results['std_reward']:.2f} "
        f"[{results['min_reward']:.2f}, {results['max_reward']:.2f}]"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, f"{args.method}_evaluation_{int(time.time())}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained RL agent")
    p.add_argument("--method", required=True, choices=["ppo", "dqn", "td3", "sac"])
    p.add_argument("--model_path", required=True, help="Path to saved .pth file")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--output_dir", default="results/evaluation_results")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
