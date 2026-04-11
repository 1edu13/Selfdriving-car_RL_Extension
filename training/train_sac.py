"""
Training script for the SAC agent on CarRacing-v2.

SAC (Soft Actor-Critic) is an off-policy algorithm built on Maximum Entropy RL.
It maximizes both expected return AND the policy's entropy (randomness), which leads to:
  - Exceptional exploration through principled entropy maximization
  - Robust policies that don't collapse to a single deterministic behavior
  - Automatic temperature tuning via the learned alpha parameter

Optimized for: NVIDIA RTX 3050 (4GB VRAM) | AMD Ryzen 7 4800H | 32GB RAM
"""

import sys
import os

# Ensure project root is in path when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import gymnasium as gym
import random
import copy
import time
from collections import deque

from agents.sac_agent import Actor, Critic
from core.utils import make_env, get_device


class ReplayBuffer:
    """
    Experience Replay Buffer. Store and sample steps.
    For SAC, we store the NORMALIZED [-1, 1] actions in the buffer, not the environment-scaled ones.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def scale_action_for_env(normalized_action):
    """
    Converts a standard [-1, 1] action vector (from our Actor) into the specialized
    boundaries required by CarRacing-v2.
    """
    env_action = np.copy(normalized_action)
    env_action[1] = (env_action[1] + 1.0) / 2.0  # Gas mapping
    env_action[2] = (env_action[2] + 1.0) / 2.0  # Brake mapping
    return env_action


def print_header(device, use_amp, hp):
    """Prints a formatted training configuration header."""
    print()
    print("=" * 64)
    print("  SAC TRAINING -- CarRacing-v2 (Max Entropy)".center(64))
    print("=" * 64)
    print(f"  Device:         {device}")
    print(f"  AMP (FP16):     {'Enabled' if use_amp else 'Disabled'}")
    print(f"  Total Steps:    {hp['total_timesteps']:,}")
    print(f"  Batch Size:     {hp['batch_size']}")
    print(f"  Buffer Size:    {hp['buffer_capacity']:,}")
    print(f"  Frame Skip:     {hp['frame_skip']} (action repeated {hp['frame_skip']}x per step)")
    print(f"  Learning Rate:  {hp['learning_rate']}")
    print(f"  Gamma:          {hp['gamma']}  |  Tau: {hp['tau']}")
    print(f"  Target Entropy: {hp['target_entropy']}")
    print(f"  Warmup:         {hp['start_training_step']:,} random steps")
    print(f"  Checkpoints:    Every {hp['save_freq']:,} steps")
    print(f"  Resume:         {hp['resume']}")
    print("=" * 64)
    print()


def train_sac():
    # =====================================================================
    # HYPERPARAMETERS -- Optimized for RTX 3050 (4GB VRAM) + 32GB RAM
    # =====================================================================
    run_name = "sac_baseline"
    env_id = "CarRacing-v2"
    seed = 42

    total_timesteps = 1_500_000
    save_freq = 100_000
    log_freq = 10_000
    resume_from_checkpoint = False

    learning_rate = 3e-4
    buffer_capacity = 200_000
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    start_training_step = 25_000
    target_entropy = -3.0

    # =====================================================================
    # SETUP
    # =====================================================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    device_type = device.type
    use_amp = (device_type == "cuda")

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True

    print_header(device, use_amp, {
        'total_timesteps': total_timesteps, 'batch_size': batch_size,
        'buffer_capacity': buffer_capacity, 'learning_rate': learning_rate,
        'gamma': gamma, 'tau': tau, 'target_entropy': target_entropy,
        'start_training_step': start_training_step,
        'save_freq': save_freq, 'resume': resume_from_checkpoint,
        'frame_skip': 2,
    })

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, run_name, is_discrete=False)]
    )

    actor = Actor(action_dim=3).to(device)
    critic = Critic(action_dim=3).to(device)

    start_step = 0
    if resume_from_checkpoint and os.path.exists(f"models/{run_name}"):
        chkp_files = [f for f in os.listdir(f"models/{run_name}") if f.startswith("sac_actor_step_")]
        if chkp_files:
            latest_step = max([int(f.split("_step_")[1].split(".pth")[0]) for f in chkp_files])
            start_step = latest_step
            actor.load_state_dict(torch.load(f"models/{run_name}/sac_actor_step_{latest_step}.pth", map_location=device))
            critic.load_state_dict(torch.load(f"models/{run_name}/sac_critic_step_{latest_step}.pth", map_location=device))
            print(f"  >> Checkpoint found! Resuming from step {latest_step:,}\n")

    critic_target = copy.deepcopy(critic)
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate)

    scaler = GradScaler(enabled=use_amp)
    buffer = ReplayBuffer(buffer_capacity)

    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    obs, _ = envs.reset(seed=seed)
    obs = np.array(obs)

    episode_rewards = []
    current_ep_reward = 0
    episode_count = 0

    recent_critic_losses = []
    recent_actor_losses = []
    recent_alphas = []
    recent_entropies = []
    train_start_time = time.time()

    print("  Step         | Crit. L  | Act. L   | Alpha  | Entropy | Buffer  | Ep Rew  | Avg(10)")
    print("  " + "-" * 90)

    for global_step in range(start_step, total_timesteps):
        # 1. Select Action
        if global_step < start_training_step:
            action_np_normalized = envs.action_space.sample()[0]
        else:
            with torch.no_grad(), autocast(enabled=use_amp):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action_tensor, _ = actor.get_action(obs_tensor)
                action_np_normalized = action_tensor.cpu().numpy()[0]

        action_env = scale_action_for_env(action_np_normalized)

        # 2. Environment Step
        next_obs, rewards, terminations, truncations, infos = envs.step([action_env])
        next_obs = np.array(next_obs)
        dones = np.logical_or(terminations, truncations)

        current_ep_reward += rewards[0]
        if dones[0]:
            episode_rewards.append(current_ep_reward)
            episode_count += 1
            avg_rew = np.mean(episode_rewards[-10:])
            alpha_val = log_alpha.exp().item()
            print(f"  {global_step:>13,} | {'':>8s} | {'':>8s} | {alpha_val:>6.3f} | "
                  f"{'':>7s} | {len(buffer):>7,} | {current_ep_reward:>7.1f} | {avg_rew:>7.1f}")
            current_ep_reward = 0

        # Store NORMALIZED action in buffer
        buffer.push(obs[0], action_np_normalized, rewards[0], next_obs[0], dones[0])
        obs = next_obs

        # 3. Learning Phase
        if global_step >= start_training_step and len(buffer) >= batch_size:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)

            b_obs = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
            b_actions = torch.as_tensor(b_actions, dtype=torch.float32, device=device)
            b_rewards = torch.as_tensor(b_rewards, device=device).unsqueeze(1)
            b_next_obs = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
            b_dones = torch.as_tensor(b_dones, device=device).unsqueeze(1)

            alpha = log_alpha.exp().detach()

            # --- Critic Update ---
            with torch.no_grad(), autocast(enabled=use_amp):
                next_action, next_log_prob = actor.get_action(b_next_obs)
                target_q1, target_q2 = critic_target(b_next_obs, next_action)
                target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                target_q = b_rewards + gamma * target_q * (1 - b_dones)

            with autocast(enabled=use_amp):
                current_q1, current_q2 = critic(b_obs, b_actions)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            critic_optimizer.zero_grad()
            scaler.scale(critic_loss).backward()
            scaler.unscale_(critic_optimizer)
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            scaler.step(critic_optimizer)
            scaler.update()

            # --- Actor Update ---
            with autocast(enabled=use_amp):
                curr_action, curr_log_prob = actor.get_action(b_obs)
                curr_q1, curr_q2 = critic(b_obs, curr_action)
                curr_q = torch.min(curr_q1, curr_q2)
                actor_loss = (alpha * curr_log_prob - curr_q).mean()

            actor_optimizer.zero_grad()
            scaler.scale(actor_loss).backward()
            scaler.step(actor_optimizer)
            scaler.update()

            # --- Alpha Update ---
            alpha_loss = -(log_alpha * (curr_log_prob + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # --- Soft Updates ---
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Track metrics
            recent_critic_losses.append(critic_loss.item())
            recent_actor_losses.append(actor_loss.item())
            recent_alphas.append(alpha.item())
            recent_entropies.append(-curr_log_prob.mean().item())

        # 4. Periodic metrics log
        if global_step > 0 and global_step % log_freq == 0 and recent_critic_losses:
            elapsed = time.time() - train_start_time
            sps = (global_step - start_step) / max(elapsed, 1)
            ms_per_step = 1000 / max(sps, 1)
            avg_cl = np.mean(recent_critic_losses[-500:])
            avg_al = np.mean(recent_actor_losses[-500:])
            avg_alpha = np.mean(recent_alphas[-500:])
            avg_ent = np.mean(recent_entropies[-500:])
            avg_rew = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            pct = 100 * global_step / total_timesteps
            print(f"  {global_step:>13,} | {avg_cl:>8.4f} | {avg_al:>8.4f} | {avg_alpha:>6.3f} | "
                  f"{avg_ent:>7.2f} | {len(buffer):>7,} | {'':>7s} | {avg_rew:>7.1f}  "
                  f"[{pct:.1f}% {sps:,.0f}sps {ms_per_step:.2f}ms/step]")

        # 5. Checkpoint
        if global_step > 0 and global_step % save_freq == 0:
            os.makedirs(f"models/{run_name}", exist_ok=True)
            torch.save(actor.state_dict(), f"models/{run_name}/sac_actor_step_{global_step}.pth")
            torch.save(critic.state_dict(), f"models/{run_name}/sac_critic_step_{global_step}.pth")
            print(f"  >> [SAVE] Checkpoint: models/{run_name}/sac_*_step_{global_step}.pth")

    # Final save
    os.makedirs(f"models/{run_name}", exist_ok=True)
    torch.save(actor.state_dict(), f"models/{run_name}/sac_actor_final.pth")
    torch.save(critic.state_dict(), f"models/{run_name}/sac_critic_final.pth")
    envs.close()

    total_time = time.time() - train_start_time
    print()
    print("=" * 64)
    print("  SAC TRAINING COMPLETE".center(64))
    print("=" * 64)
    print(f"  Total Steps:    {total_timesteps:,}")
    print(f"  Total Time:     {total_time/3600:.1f} hours")
    print(f"  Episodes:       {episode_count}")
    if episode_rewards:
        print(f"  Best Reward:    {max(episode_rewards):.1f}")
        print(f"  Final Avg(10):  {np.mean(episode_rewards[-10:]):.1f}")
    if recent_alphas:
        print(f"  Final Alpha:    {np.mean(recent_alphas[-100:]):.4f}")
    if recent_entropies:
        print(f"  Final Entropy:  {np.mean(recent_entropies[-100:]):.2f}")
    print(f"  Model Saved:    models/{run_name}/sac_*_final.pth")
    print("=" * 64)
    print()


if __name__ == "__main__":
    train_sac()