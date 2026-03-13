"""
Training script for the DQN agent on CarRacing-v2 (discrete actions).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.dqn_agent import DQNAgent
from core.utils import make_env, get_device


def train(args):
    device = get_device(force_cpu=args.cpu)
    env = make_env(seed=args.seed, continuous=False)

    agent = DQNAgent(
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        device=str(device),
    )

    os.makedirs(args.save_dir, exist_ok=True)

    total_steps = 0
    best_ep_reward = -float("inf")

    for episode in range(1, args.total_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        t0 = time.time()

        while True:
            action = agent.select_action(obs)
            action_idx = next(
                i for i, a in enumerate(agent.action_space) if np.allclose(a, action)
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(obs, action_idx, reward, next_obs, float(done))
            metrics = agent.update()

            ep_reward += reward
            total_steps += 1
            obs = next_obs

            if done:
                break

        elapsed = time.time() - t0
        print(
            f"[DQN] ep {episode:4d} | "
            f"reward={ep_reward:.2f} | "
            f"epsilon={agent.epsilon:.3f} | "
            f"steps={total_steps} | "
            f"time={elapsed:.1f}s"
        )

        if ep_reward > best_ep_reward:
            best_ep_reward = ep_reward
            agent.save(os.path.join(args.save_dir, "dqn_best.pth"))

        if episode % args.save_freq == 0:
            agent.save(os.path.join(args.save_dir, f"dqn_ep{episode}.pth"))

    env.close()
    print(f"Training complete. Best reward: {best_ep_reward:.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on CarRacing-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_episodes", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_end", type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=int, default=100_000)
    p.add_argument("--buffer_size", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--target_update_freq", type=int, default=1_000)
    p.add_argument("--save_dir", type=str, default="models/dqn_results")
    p.add_argument("--save_freq", type=int, default=100)
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
