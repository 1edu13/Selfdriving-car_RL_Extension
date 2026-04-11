"""
Training script for the SAC agent on CarRacing-v2.

SAC (Soft Actor-Critic) is an off-policy algorithm built on Maximum Entropy RL.
It maximizes both expected return AND the policy's entropy (randomness), which leads to:
  - Exceptional exploration through principled entropy maximization
  - Robust policies that don't collapse to a single deterministic behavior
  - Automatic temperature tuning via the learned alpha parameter

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

from agents.sac_agent import Actor, Critic
from core.utils import make_env, get_device


class ReplayBuffer:
    """
    Experience Replay Buffer. Store and sample steps.
    For SAC, we store the NORMALIZED [-1, 1] actions in the buffer, not the environment-scaled ones.
    This keeps the mathematical integration with the tanh distributions perfectly clean.
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
    - Steering stays [-1, 1]
    - Gas is mapped to [0, 1]
    - Brake is mapped to [0, 1]
    """
    env_action = np.copy(normalized_action)
    env_action[1] = (env_action[1] + 1.0) / 2.0  # Gas mapping
    env_action[2] = (env_action[2] + 1.0) / 2.0  # Brake mapping
    return env_action


def train_sac():
    # =====================================================================
    # HYPERPARAMETERS — Optimized for RTX 3050 (4GB VRAM) + 32GB RAM
    # =====================================================================
    run_name = "sac_baseline"
    env_id = "CarRacing-v2"
    seed = 42

    # Training length & checkpointing
    total_timesteps = 1_500_000       # 1.5M steps — SAC is very sample-efficient with entropy exploration
    save_freq = 100_000               # Save checkpoint every 100K steps
    resume_from_checkpoint = False    # Set to True to resume from the latest checkpoint

    # Network hyperparameters
    learning_rate = 3e-4              # Standard LR for Actor-Critic methods
    buffer_capacity = 200_000         # 200K transitions — 32GB RAM handles this easily
    batch_size = 128                  # Optimized for 4GB VRAM with twin CNN critics
    gamma = 0.99                      # Discount factor
    tau = 0.005                       # Soft update rate for target critic
    start_training_step = 25_000      # Random exploration warmup

    # SAC Auto-Tuning Entropy
    # Target entropy heuristic: -dim(A) = -3 for 3D action space (steering, gas, brake)
    target_entropy = -3.0

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

    # Environment — continuous actions
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, run_name, is_discrete=False)]
    )

    # Initialize Actor (Stochastic Policy) and Twin Critic
    actor = Actor(action_dim=3).to(device)
    critic = Critic(action_dim=3).to(device)

    # --- Checkpoint Loading System ---
    start_step = 0
    if resume_from_checkpoint and os.path.exists(f"models/{run_name}"):
        chkp_files = [f for f in os.listdir(f"models/{run_name}") if f.startswith("sac_actor_step_")]
        if chkp_files:
            latest_step = max([int(f.split("_step_")[1].split(".pth")[0]) for f in chkp_files])
            start_step = latest_step
            actor.load_state_dict(torch.load(f"models/{run_name}/sac_actor_step_{latest_step}.pth", map_location=device))
            critic.load_state_dict(torch.load(f"models/{run_name}/sac_critic_step_{latest_step}.pth", map_location=device))
            print(f"\n✅ Checkpoint SAC found! Resuming training from step {latest_step}...\n")

    # Target Critic for Bellman updates (SAC doesn't use a Target Actor)
    critic_target = copy.deepcopy(critic)

    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    # Learnable entropy temperature (alpha)
    # log_alpha is optimized to ensure alpha stays positive when exponentiated
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate)

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
        # 1. Select Action — random during warmup, stochastic policy after
        if global_step < start_training_step:
            action_np_normalized = envs.action_space.sample()
        else:
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action_tensor, _ = actor.get_action(obs_tensor)
                action_np_normalized = action_tensor.cpu().numpy()[0]

        # Convert [-1, 1] actions into CarRacing-v2 format
        action_env = scale_action_for_env(action_np_normalized)

        # 2. Environment Step
        next_obs, rewards, terminations, truncations, infos = envs.step([action_env])
        next_obs = np.array(next_obs)
        dones = np.logical_or(terminations, truncations)

        current_ep_reward += rewards[0]
        if dones[0]:
            episode_rewards.append(current_ep_reward)
            print(f"Step: {global_step:>8,} | Ep Reward: {current_ep_reward:.2f}")
            current_ep_reward = 0

        # CRITICAL: Store NORMALIZED [-1,1] actions in buffer (not env-scaled)
        # The Actor operates in squashed space, so the Critic must learn over that same space
        buffer.push(obs[0], action_np_normalized, rewards[0], next_obs[0], dones[0])
        obs = next_obs

        # 3. Neural Network Learning Phase
        if global_step >= start_training_step and len(buffer) >= batch_size:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)

            b_obs = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
            b_actions = torch.as_tensor(b_actions, dtype=torch.float32, device=device)
            b_rewards = torch.as_tensor(b_rewards, device=device).unsqueeze(1)
            b_next_obs = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
            b_dones = torch.as_tensor(b_dones, device=device).unsqueeze(1)

            # Dynamic alpha (temperature)
            alpha = log_alpha.exp().detach()

            # --- Update Critic (Q-Networks) ---
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # Sample NEXT actions from current policy
                next_action, next_log_prob = actor.get_action(b_next_obs)

                # Twin Q-targets with entropy penalty
                target_q1, target_q2 = critic_target(b_next_obs, next_action)
                target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob

                # Soft Q-Target: r + γ * (Q_target - α * log_π)
                target_q = b_rewards + gamma * target_q * (1 - b_dones)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                current_q1, current_q2 = critic(b_obs, b_actions)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            critic_optimizer.zero_grad()
            scaler.scale(critic_loss).backward()
            scaler.unscale_(critic_optimizer)
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            scaler.step(critic_optimizer)
            scaler.update()

            # --- Update Actor (Policy) ---
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                curr_action, curr_log_prob = actor.get_action(b_obs)
                curr_q1, curr_q2 = critic(b_obs, curr_action)
                curr_q = torch.min(curr_q1, curr_q2)

                # Actor loss: maximize Q-value AND maximize entropy
                # = minimize (α * log_π(a|s) - Q(s,a))
                actor_loss = (alpha * curr_log_prob - curr_q).mean()

            actor_optimizer.zero_grad()
            scaler.scale(actor_loss).backward()
            scaler.step(actor_optimizer)
            scaler.update()

            # --- Update Temperature (Alpha) ---
            # Auto-tune α to match target entropy: if entropy < target → increase α (explore more)
            alpha_loss = -(log_alpha * (curr_log_prob + target_entropy).detach()).mean()

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # --- Soft Updates (Polyak Averaging) ---
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 4. Checkpoint Saving every 100K steps
        if global_step > 0 and global_step % save_freq == 0:
            os.makedirs(f"models/{run_name}", exist_ok=True)
            torch.save(actor.state_dict(), f"models/{run_name}/sac_actor_step_{global_step}.pth")
            torch.save(critic.state_dict(), f"models/{run_name}/sac_critic_step_{global_step}.pth")
            print(f"💾 Checkpoint SAC saved at step {global_step:,}")

    # Save final model
    os.makedirs(f"models/{run_name}", exist_ok=True)
    torch.save(actor.state_dict(), f"models/{run_name}/sac_actor_final.pth")
    torch.save(critic.state_dict(), f"models/{run_name}/sac_critic_final.pth")
    envs.close()
    print(f"\n✅ SAC Training Complete — {total_timesteps:,} steps | Final model saved.")


if __name__ == "__main__":
    train_sac()