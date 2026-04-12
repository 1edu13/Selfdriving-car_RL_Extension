"""
Training script for the TD3 agent on CarRacing-v2.

TD3 (Twin Delayed DDPG) is an actor-critic algorithm for continuous control.
It addresses DDPG's overestimation bias through three key innovations:
  1. Twin Q-Networks -- takes the minimum of two critics to reduce overestimation
  2. Delayed Policy Updates -- updates the actor less frequently than the critic
  3. Target Policy Smoothing -- adds noise to target actions for smoother value estimates

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

from agents.td3_agent import Actor, Critic
from core.utils import make_env, get_device


class ReplayBuffer:
    """
    Experience Replay Buffer for continuous action spaces.
    Stores transitions (state, action, reward, next_state, done) and samples
    random batches to break the temporal correlation of data for stable training.
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


def print_header(device, use_amp, hp):
    """Prints a formatted training configuration header."""
    print()
    print("=" * 64)
    print("  TD3 TRAINING -- CarRacing-v2 (Continuous Actions)".center(64))
    print("=" * 64)
    print(f"  Device:         {device}")
    print(f"  AMP (FP16):     {'Enabled' if use_amp else 'Disabled'}")
    print(f"  Total Steps:    {hp['total_timesteps']:,}")
    print(f"  Batch Size:     {hp['batch_size']}")
    print(f"  Buffer Size:    {hp['buffer_capacity']:,}")
    print(f"  Frame Skip:     {hp['frame_skip']} (action repeated {hp['frame_skip']}x per step)")
    print(f"  Gradient Steps: {hp['gradient_steps']} updates per env step")
    print(f"  Learning Rate:  {hp['learning_rate']}")
    print(f"  Gamma:          {hp['gamma']}  |  Tau: {hp['tau']}")
    print(f"  Expl. Noise:    {hp['exploration_noise']}  |  Policy Noise: {hp['policy_noise']}")
    print(f"  Policy Delay:   {hp['policy_delay']} critic updates per actor update")
    print(f"  Warmup:         {hp['start_training_step']:,} random steps")
    print(f"  Checkpoints:    Every {hp['save_freq']:,} steps")
    print(f"  Resume:         {hp['resume']}")
    print("=" * 64)
    print()


def train_td3():
    # =====================================================================
    # HYPERPARAMETERS -- Optimized for RTX 3050 (4GB VRAM) + 32GB RAM
    # =====================================================================
    run_name = "td3_baseline"
    env_id = "CarRacing-v2"
    seed = 42

    total_timesteps = 600_000
    save_freq = 100_000
    log_freq = 10_000
    resume_from_checkpoint = False

    learning_rate = 3e-4
    buffer_capacity = 200_000
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    start_training_step = 25_000
    gradient_steps = 1                # GPU updates per env step (1:1 ratio with single env)

    exploration_noise = 0.1
    policy_noise = 0.2
    noise_clip = 0.5
    policy_delay = 2

    action_low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

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
        'gamma': gamma, 'tau': tau, 'exploration_noise': exploration_noise,
        'policy_noise': policy_noise, 'policy_delay': policy_delay,
        'start_training_step': start_training_step,
        'save_freq': save_freq, 'resume': resume_from_checkpoint,
        'frame_skip': 4, 'gradient_steps': gradient_steps,
    })

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, run_name, is_discrete=False)]
    )

    actor = Actor(action_dim=3).to(device)
    critic = Critic(action_dim=3).to(device)

    start_step = 0
    if resume_from_checkpoint and os.path.exists(f"models/{run_name}"):
        chkp_files = [f for f in os.listdir(f"models/{run_name}") if f.startswith("td3_actor_step_")]
        if chkp_files:
            latest_step = max([int(f.split("_step_")[1].split(".pth")[0]) for f in chkp_files])
            start_step = latest_step
            actor.load_state_dict(torch.load(f"models/{run_name}/td3_actor_step_{latest_step}.pth", map_location=device))
            critic.load_state_dict(torch.load(f"models/{run_name}/td3_critic_step_{latest_step}.pth", map_location=device))
            print(f"  >> Checkpoint found! Resuming from step {latest_step:,}\n")

    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
    scaler_critic = GradScaler(enabled=use_amp)
    scaler_actor = GradScaler(enabled=use_amp)

    buffer = ReplayBuffer(buffer_capacity)

    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    obs, _ = envs.reset(seed=seed)
    obs = np.array(obs)

    episode_rewards = []
    current_ep_reward = 0
    episode_count = 0

    recent_critic_losses = deque(maxlen=1000)
    recent_actor_losses = deque(maxlen=1000)
    train_start_time = time.time()

    print("  Step         | Critic L | Actor L  | Buffer  | Ep Reward | Avg(10)  | Progress")
    print("  " + "-" * 82)

    for global_step in range(start_step, total_timesteps):
        # 1. Select Action
        if global_step < start_training_step:
            action_np = envs.action_space.sample()
        else:
            with torch.no_grad(), autocast(enabled=use_amp):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action_tensor = actor(obs_tensor)
                noise = torch.normal(0, exploration_noise, size=action_tensor.shape, device=device)
                action_tensor = action_tensor + noise
                action_np = action_tensor.cpu().numpy()
                action_np = np.clip(action_np, action_low, action_high)

        # 2. Environment Step
        next_obs, rewards, terminations, truncations, infos = envs.step(action_np)
        next_obs = np.asarray(next_obs)
        dones = np.logical_or(terminations, truncations)

        current_ep_reward += rewards[0]
        if dones[0]:
            episode_rewards.append(current_ep_reward)
            episode_count += 1
            avg_rew = np.mean(episode_rewards[-10:])
            print(f"  {global_step:>13,} | {'':>8s} | {'':>8s} | {len(buffer):>7,} | "
                  f"{current_ep_reward:>9.1f} | {avg_rew:>8.1f} |")
            current_ep_reward = 0

        # 3. Store in Buffer
        buffer.push(obs[0], action_np[0], rewards[0], next_obs[0], dones[0])
        obs = next_obs

        # 4. Network Updates -- gradient_steps updates per env step
        if global_step >= start_training_step and len(buffer) >= batch_size:
            for grad_step in range(gradient_steps):
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)

                b_obs = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
                b_actions = torch.as_tensor(b_actions, dtype=torch.float32, device=device)
                b_rewards = torch.as_tensor(b_rewards, device=device).unsqueeze(1)
                b_next_obs = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
                b_dones = torch.as_tensor(b_dones, device=device).unsqueeze(1)

                # --- Critic Update ---
                with torch.no_grad(), autocast(enabled=use_amp):
                    next_action = actor_target(b_next_obs)
                    noise = torch.normal(0, policy_noise, size=next_action.shape, device=device)
                    noise = torch.clamp(noise, -noise_clip, noise_clip)
                    smoothed_next_action = next_action + noise
                    t_low = torch.as_tensor(action_low, device=device)
                    t_high = torch.as_tensor(action_high, device=device)
                    smoothed_next_action = torch.max(torch.min(smoothed_next_action, t_high), t_low)
                    target_q1, target_q2 = critic_target(b_next_obs, smoothed_next_action)
                    target_q = torch.min(target_q1, target_q2)
                    target_q = b_rewards + gamma * target_q * (1 - b_dones)

                with autocast(enabled=use_amp):
                    current_q1, current_q2 = critic(b_obs, b_actions)
                    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                critic_optimizer.zero_grad()
                scaler_critic.scale(critic_loss).backward()
                scaler_critic.unscale_(critic_optimizer)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                scaler_critic.step(critic_optimizer)
                scaler_critic.update()

                recent_critic_losses.append(critic_loss.item())

                # --- Delayed Actor Update (every policy_delay critic updates) ---
                if grad_step % policy_delay == 0:
                    with autocast(enabled=use_amp):
                        actor_loss = -critic.q1(b_obs, actor(b_obs)).mean()

                    actor_optimizer.zero_grad()
                    scaler_actor.scale(actor_loss).backward()
                    scaler_actor.step(actor_optimizer)
                    scaler_actor.update()

                    recent_actor_losses.append(actor_loss.item())

                    # Soft Updates
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 5. Periodic metrics log
        if global_step > 0 and global_step % log_freq == 0 and recent_critic_losses:
            elapsed = time.time() - train_start_time
            sps = (global_step - start_step) / max(elapsed, 1)
            ms_per_step = 1000 / max(sps, 1)
            avg_cl = np.mean(recent_critic_losses)
            avg_al = np.mean(recent_actor_losses) if recent_actor_losses else 0
            avg_rew = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            pct = 100 * global_step / total_timesteps
            print(f"  {global_step:>13,} | {avg_cl:>8.4f} | {avg_al:>8.4f} | {len(buffer):>7,} | "
                  f"{'':>9s} | {avg_rew:>8.1f} | {pct:>5.1f}% {sps:,.0f}sps {ms_per_step:.2f}ms/step")

        # 6. Save Checkpoint
        if global_step > 0 and global_step % save_freq == 0:
            os.makedirs(f"models/{run_name}", exist_ok=True)
            torch.save(actor.state_dict(), f"models/{run_name}/td3_actor_step_{global_step}.pth")
            torch.save(critic.state_dict(), f"models/{run_name}/td3_critic_step_{global_step}.pth")
            print(f"  >> [SAVE] Checkpoint: models/{run_name}/td3_*_step_{global_step}.pth")

    # Final save
    os.makedirs(f"models/{run_name}", exist_ok=True)
    torch.save(actor.state_dict(), f"models/{run_name}/td3_actor_final.pth")
    torch.save(critic.state_dict(), f"models/{run_name}/td3_critic_final.pth")
    envs.close()

    total_time = time.time() - train_start_time
    print()
    print("=" * 64)
    print("  TD3 TRAINING COMPLETE".center(64))
    print("=" * 64)
    print(f"  Total Steps:    {total_timesteps:,}")
    print(f"  Total Time:     {total_time/3600:.1f} hours")
    print(f"  Episodes:       {episode_count}")
    if episode_rewards:
        print(f"  Best Reward:    {max(episode_rewards):.1f}")
        print(f"  Final Avg(10):  {np.mean(episode_rewards[-10:]):.1f}")
    if recent_critic_losses:
        print(f"  Final Crit. L:  {np.mean(recent_critic_losses[-100:]):.4f}")
    print(f"  Model Saved:    models/{run_name}/td3_*_final.pth")
    print("=" * 64)
    print()


if __name__ == "__main__":
    train_td3()
