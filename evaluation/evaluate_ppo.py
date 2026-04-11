"""
==============================================================================
PPO EVALUATION -- CarRacing-v2 (On-Policy, Continuous Actions)
==============================================================================
Evaluates the trained PPO agent: records videos, prints per-episode stats,
and saves a CSV with all results.

Usage: python evaluation/evaluate_ppo.py
"""

import sys, os, time, csv
import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.utils import make_env
from gymnasium.wrappers import GrayScaleObservation, FrameStack
from agents.ppo_agent import PPOAgent

# =====================================================================
# CONFIGURATION -- Edit these paths directly
# =====================================================================
MODEL_PATH = "models/ppo_baseline/ppo_final.pth"
NUM_EPISODES = 20
SEED = 42
RECORD_VIDEO = True
# =====================================================================


def evaluate_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(PROJECT_ROOT, MODEL_PATH)

    # --- Header ---
    print()
    print("=" * 64)
    print("  PPO EVALUATION -- CarRacing-v2 (On-Policy)".center(64))
    print("=" * 64)
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Device:      {device}")
    print(f"  Episodes:    {NUM_EPISODES}")
    print(f"  Record:      {RECORD_VIDEO}")
    print("=" * 64)
    print()

    # --- Load Agent ---
    # PPO needs envs for init (to read action_space shape)
    tmp_envs = gym.vector.SyncVectorEnv(
        [make_env("CarRacing-v2", 0, 0, False, "tmp", frame_skip=1)]
    )
    agent = PPOAgent(tmp_envs)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device).eval()
    tmp_envs.close()
    print(f"  [OK] Agent loaded ({sum(p.numel() for p in agent.parameters()):,} params)")
    print()

    # --- Results dirs ---
    results_dir = os.path.join(PROJECT_ROOT, "results", "evaluation", "ppo")
    video_dir = os.path.join(results_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # --- Run ---
    print(f"  {'Ep':>4s} | {'Reward':>9s} | {'Steps':>7s} | {'Time':>7s} | Status")
    print(f"  {'-'*4} | {'-'*9} | {'-'*7} | {'-'*7} | {'-'*8}")

    all_rewards, all_steps, all_times = [], [], []

    for ep in range(NUM_EPISODES):
        env = gym.make("CarRacing-v2", render_mode="rgb_array")
        if RECORD_VIDEO:
            env = gym.wrappers.RecordVideo(env, video_dir,
                name_prefix=f"ppo_ep{ep:02d}", episode_trigger=lambda x: True)
        env = GrayScaleObservation(env, keep_dim=False)
        env = FrameStack(env, 4)
        env.action_space.seed(SEED + ep)

        obs, _ = env.reset(seed=SEED + ep)
        done, total_reward, steps = False, 0.0, 0
        t0 = time.time()

        while not done:
            obs_t = torch.as_tensor(np.array(obs)[np.newaxis], dtype=torch.float32, device=device)
            with torch.no_grad():
                # PPO: use mean action directly (no sampling for evaluation)
                hidden = agent.network(obs_t)
                action = agent.actor_mean(hidden).cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        elapsed = time.time() - t0
        all_rewards.append(total_reward)
        all_steps.append(steps)
        all_times.append(elapsed)

        tag = "[OK]" if total_reward > 0 else "[LOW]" if total_reward > -50 else "[FAIL]"
        print(f"  {ep+1:>4d} | {total_reward:>9.1f} | {steps:>7d} | {elapsed:>6.1f}s | {tag}")
        env.close()

    # --- Summary ---
    r = np.array(all_rewards)
    print()
    print("=" * 64)
    print("  PPO EVALUATION SUMMARY".center(64))
    print("=" * 64)
    print(f"  Reward Mean:      {r.mean():.1f}  (+/- {r.std():.1f})")
    print(f"  Reward Min/Max:   {r.min():.1f} / {r.max():.1f}")
    print(f"  Median:           {np.median(r):.1f}")
    print(f"  Avg Steps:        {np.mean(all_steps):.0f}")
    print(f"  Success (>0):     {(r>0).sum()}/{NUM_EPISODES} ({(r>0).mean()*100:.0f}%)")
    print(f"  High Score (>300):{(r>300).sum()}/{NUM_EPISODES} ({(r>300).mean()*100:.0f}%)")
    print("=" * 64)

    # --- CSV ---
    csv_path = os.path.join(results_dir, "ppo_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "steps", "time_s"])
        for i in range(NUM_EPISODES):
            w.writerow([i+1, f"{all_rewards[i]:.2f}", all_steps[i], f"{all_times[i]:.2f}"])
    print(f"\n  [SAVE] CSV:    {csv_path}")
    if RECORD_VIDEO:
        print(f"  [SAVE] Videos: {video_dir}/")
    print()


if __name__ == "__main__":
    evaluate_ppo()
