"""
Training script for the PPO agent on CarRacing-v2.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.ppo_agent import PPOAgent
from core.utils import make_env, get_device


def collect_rollout(env, agent: PPOAgent, steps: int, gamma: float, lam: float):
    """Collect a rollout of *steps* transitions using GAE."""
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

    obs, _ = env.reset()
    for _ in range(steps):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(log_prob)
        rew_buf.append(reward)
        val_buf.append(value)
        done_buf.append(float(done))

        obs = next_obs if not done else env.reset()[0]

    # GAE returns and advantages
    returns, advantages = [], []
    gae = 0.0
    next_val = agent.policy.get_value(
        torch.FloatTensor(obs).unsqueeze(0)
    ).item()

    for t in reversed(range(steps)):
        mask = 1.0 - done_buf[t]
        next_v = val_buf[t + 1] if t + 1 < steps else next_val
        delta = rew_buf[t] + gamma * next_v * mask - val_buf[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + val_buf[t])

    return {
        "obs": np.array(obs_buf, dtype=np.float32),
        "actions": np.array(act_buf, dtype=np.float32),
        "log_probs": np.array(logp_buf, dtype=np.float32).reshape(-1, 1),
        "returns": np.array(returns, dtype=np.float32).reshape(-1, 1),
        "advantages": np.array(advantages, dtype=np.float32).reshape(-1, 1),
    }


def train(args):
    device = get_device(force_cpu=args.cpu)
    env = make_env(seed=args.seed)

    agent = PPOAgent(
        action_dim=3,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        n_epochs=args.n_epochs,
        device=str(device),
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_reward = -float("inf")

    for iteration in range(1, args.total_iterations + 1):
        t0 = time.time()
        rollout = collect_rollout(env, agent, args.rollout_steps, args.gamma, args.lam)
        metrics = agent.update(rollout)
        elapsed = time.time() - t0

        ep_rewards = rollout["returns"].mean().item()
        print(
            f"[PPO] iter {iteration:4d} | "
            f"mean_return={ep_rewards:.2f} | "
            f"policy_loss={metrics['policy_loss']:.4f} | "
            f"value_loss={metrics['value_loss']:.4f} | "
            f"entropy={metrics['entropy']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if ep_rewards > best_reward:
            best_reward = ep_rewards
            agent.save(os.path.join(args.save_dir, "ppo_best.pth"))

        if iteration % args.save_freq == 0:
            agent.save(os.path.join(args.save_dir, f"ppo_iter{iteration}.pth"))

    env.close()
    print(f"Training complete. Best return: {best_reward:.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on CarRacing-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_iterations", type=int, default=500)
    p.add_argument("--rollout_steps", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--save_dir", type=str, default="models/ppo_baseline")
    p.add_argument("--save_freq", type=int, default=50)
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
