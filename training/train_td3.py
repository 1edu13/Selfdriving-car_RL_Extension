"""
Training script for the TD3 agent on CarRacing-v2.

TD3 (Twin Delayed DDPG) is an actor-critic algorithm for continuous control.
It addresses DDPG's overestimation bias through three key innovations:
  1. Twin Q-Networks — takes the minimum of two critics to reduce overestimation
  2. Delayed Policy Updates — updates the actor less frequently than the critic
  3. Target Policy Smoothing — adds noise to target actions for smoother value estimates

Optimized for: NVIDIA RTX 3050 (4GB VRAM) | AMD Ryzen 7 4800H | 32GB RAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
import os
import copy
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
        """Saves a transition step into the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Samples a randomized batch from the stored transitions."""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def train_td3():
    # =====================================================================
    # HYPERPARAMETERS — Optimized for RTX 3050 (4GB VRAM) + 32GB RAM
    # =====================================================================
    run_name = "td3_baseline"
    env_id = "CarRacing-v2"
    seed = 42

    # Training length & checkpointing
    total_timesteps = 1_500_000       # 1.5M steps — TD3 is sample-efficient for continuous control
    save_freq = 100_000               # Save checkpoint every 100K steps
    resume_from_checkpoint = False    # Set to True to resume from the latest checkpoint

    # Network hyperparameters
    learning_rate = 3e-4              # Standard LR for Actor-Critic methods
    buffer_capacity = 200_000         # 200K transitions — fits easily in 32GB RAM
    batch_size = 128                  # Optimized for 4GB VRAM with twin CNN critics
    gamma = 0.99                      # Discount factor for future rewards
    tau = 0.005                       # Soft update rate (Polyak averaging) for target networks
    start_training_step = 25_000      # Random exploration warmup phase

    # TD3-specific parameters
    exploration_noise = 0.1           # Gaussian noise σ added to actions during data collection
    policy_noise = 0.2                # Target policy smoothing noise σ
    noise_clip = 0.5                  # Clip target policy noise to [-0.5, 0.5]
    policy_delay = 2                  # Update actor every 2 critic updates (core TD3 feature)

    # CarRacing-v2 continuous action bounds: [steering, gas, brake]
    action_low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # =====================================================================
    # SETUP — Device, Environment, Networks
    # =====================================================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    device_type = device.type
    use_amp = (device_type == "cuda")
    print(f"🖥️  Using device: {device} | AMP enabled: {use_amp}")

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Environment — continuous actions (is_discrete=False)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, run_name, is_discrete=False)]
    )

    # Initialize Actor and Twin Critic Networks
    actor = Actor(action_dim=3).to(device)
    critic = Critic(action_dim=3).to(device)

    # --- Checkpoint Loading System ---
    start_step = 0
    if resume_from_checkpoint and os.path.exists(f"models/{run_name}"):
        chkp_files = [f for f in os.listdir(f"models/{run_name}") if f.startswith("td3_actor_step_")]
        if chkp_files:
            latest_step = max([int(f.split("_step_")[1].split(".pth")[0]) for f in chkp_files])
            start_step = latest_step
            actor.load_state_dict(torch.load(f"models/{run_name}/td3_actor_step_{latest_step}.pth", map_location=device))
            critic.load_state_dict(torch.load(f"models/{run_name}/td3_critic_step_{latest_step}.pth", map_location=device))
            print(f"\n✅ Checkpoint TD3 found! Resuming training from step {latest_step}...\n")

    # Create target networks as exact copies (for Polyak-averaged Bellman updates)
    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)

    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    # AMP GradScaler for mixed-precision training
    scaler = torch.amp.GradScaler(enabled=use_amp)

    buffer = ReplayBuffer(buffer_capacity)

    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    obs, _ = envs.reset(seed=seed)
    obs = np.array(obs)

    episode_rewards = []
    current_ep_reward = 0

    for global_step in range(start_step, total_timesteps):
        # 1. Select Action — random exploration during warmup, then policy + noise
        if global_step < start_training_step:
            action_np = envs.action_space.sample()
        else:
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action_tensor = actor(obs_tensor)

                # Add Gaussian exploration noise
                noise = torch.normal(0, exploration_noise, size=action_tensor.shape, device=device)
                action_tensor = action_tensor + noise
                action_np = action_tensor.cpu().numpy()

                # Clip to environment bounds
                action_np = np.clip(action_np, action_low, action_high)

        # 2. Environment Step
        next_obs, rewards, terminations, truncations, infos = envs.step(action_np)
        next_obs = np.array(next_obs)
        dones = np.logical_or(terminations, truncations)

        current_ep_reward += rewards[0]
        if dones[0]:
            episode_rewards.append(current_ep_reward)
            print(f"Step: {global_step:>8,} | Ep Reward: {current_ep_reward:.2f}")
            current_ep_reward = 0

        # 3. Store transition in Replay Buffer
        buffer.push(obs[0], action_np[0], rewards[0], next_obs[0], dones[0])
        obs = next_obs

        # 4. Network Updates — after warmup with sufficient buffer data
        if global_step >= start_training_step and len(buffer) >= batch_size:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)

            b_obs = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
            b_actions = torch.as_tensor(b_actions, dtype=torch.float32, device=device)
            b_rewards = torch.as_tensor(b_rewards, device=device).unsqueeze(1)
            b_next_obs = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
            b_dones = torch.as_tensor(b_dones, device=device).unsqueeze(1)

            # --- Update Critic (Every Step) ---
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # Target Policy Smoothing: add clipped noise to target actor's actions
                next_action = actor_target(b_next_obs)
                noise = torch.normal(0, policy_noise, size=next_action.shape, device=device)
                noise = torch.clamp(noise, -noise_clip, noise_clip)
                smoothed_next_action = next_action + noise

                # Clip smoothed action to valid bounds
                t_low = torch.as_tensor(action_low, device=device)
                t_high = torch.as_tensor(action_high, device=device)
                smoothed_next_action = torch.max(torch.min(smoothed_next_action, t_high), t_low)

                # Twin Q-targets: take the minimum to fight overestimation (core TD3 feature)
                target_q1, target_q2 = critic_target(b_next_obs, smoothed_next_action)
                target_q = torch.min(target_q1, target_q2)

                # Bellman target
                target_q = b_rewards + gamma * target_q * (1 - b_dones)

            # Compute critic loss with AMP
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                current_q1, current_q2 = critic(b_obs, b_actions)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # Backward pass with GradScaler
            critic_optimizer.zero_grad()
            scaler.scale(critic_loss).backward()
            scaler.unscale_(critic_optimizer)
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            scaler.step(critic_optimizer)
            scaler.update()

            # --- Delayed Actor Update (every `policy_delay` steps) ---
            if global_step % policy_delay == 0:
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    # Actor loss: maximize Q-value → minimize -Q(s, π(s))
                    actor_loss = -critic.q1(b_obs, actor(b_obs)).mean()

                actor_optimizer.zero_grad()
                scaler.scale(actor_loss).backward()
                scaler.step(actor_optimizer)
                scaler.update()

                # --- Soft Updates (Polyak Averaging) ---
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 5. Save Checkpoint every 100K steps
        if global_step > 0 and global_step % save_freq == 0:
            os.makedirs(f"models/{run_name}", exist_ok=True)
            torch.save(actor.state_dict(), f"models/{run_name}/td3_actor_step_{global_step}.pth")
            torch.save(critic.state_dict(), f"models/{run_name}/td3_critic_step_{global_step}.pth")
            print(f"💾 Checkpoint TD3 saved at step {global_step:,}")

    # Save final model
    os.makedirs(f"models/{run_name}", exist_ok=True)
    torch.save(actor.state_dict(), f"models/{run_name}/td3_actor_final.pth")
    torch.save(critic.state_dict(), f"models/{run_name}/td3_critic_final.pth")
    envs.close()
    print(f"\n✅ TD3 Training Complete — {total_timesteps:,} steps | Final model saved.")


if __name__ == "__main__":
    train_td3()
