"""
Training script for the SAC agent on CarRacing-v2.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.sac_agent import SACAgent
from core.utils import make_env, get_device


def train(args):
    device = get_device(force_cpu=args.cpu)
    env = make_env(seed=args.seed, continuous=True)

    agent = SACAgent(
        action_dim=3,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_tune=args.auto_tune,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
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
            if total_steps < args.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, float(done))
            metrics = agent.update()

            ep_reward += reward
            total_steps += 1
            obs = next_obs

            if done:
                break

        elapsed = time.time() - t0
        if metrics:
            loss_str = (
                f"critic_loss={metrics.get('critic_loss', 0):.4f} | "
                f"actor_loss={metrics.get('actor_loss', 0):.4f} | "
                f"alpha={metrics.get('alpha', 0):.4f}"
            )
        else:
            loss_str = "warmup"

        print(
            f"[SAC] ep {episode:4d} | "
            f"reward={ep_reward:.2f} | "
            f"{loss_str} | "
            f"steps={total_steps} | "
            f"time={elapsed:.1f}s"
        )

        if ep_reward > best_ep_reward:
            best_ep_reward = ep_reward
            agent.save(os.path.join(args.save_dir, "sac_best.pth"))

        if episode % args.save_freq == 0:
            agent.save(os.path.join(args.save_dir, f"sac_ep{episode}.pth"))

    env.close()
    print(f"Training complete. Best reward: {best_ep_reward:.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="Train SAC on CarRacing-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_episodes", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--auto_tune", action="store_true", default=True)
    p.add_argument("--warmup_steps", type=int, default=10_000)
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save_dir", type=str, default="models/sac_results")
    p.add_argument("--save_freq", type=int, default=100)
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
