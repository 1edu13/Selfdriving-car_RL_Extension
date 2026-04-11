"""
Training script for the DQN agent on CarRacing-v2.

DQN (Deep Q-Network) discretizes the continuous action space of CarRacing into 5 actions
and learns a value function Q(s, a) that estimates the expected future reward for each
state-action pair. The agent then always picks the action with the highest Q-value.

Optimized for: NVIDIA RTX 3050 (4GB VRAM) | AMD Ryzen 7 4800H | 32GB RAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import os
from collections import deque

from agents.dqn_agent import DQNAgent
from core.utils import make_env, get_device


class ReplayBuffer:
    """
    Experience Replay Buffer for off-policy learning.
    Stores transitions and samples random mini-batches to break temporal correlations.
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


def train_dqn():
    # =====================================================================
    # HYPERPARAMETERS — Optimized for RTX 3050 (4GB VRAM) + 32GB RAM
    # =====================================================================
    run_name = "dqn_baseline"
    env_id = "CarRacing-v2"
    seed = 42

    # Training length & checkpointing
    total_timesteps = 2_000_000       # 2M steps — sufficient for DQN on CarRacing with discrete actions
    save_freq = 100_000               # Save checkpoint every 100K steps
    resume_from_checkpoint = False    # Set to True to resume from the latest checkpoint

    # Network hyperparameters
    learning_rate = 1e-4              # Lower LR for better stability with DQN
    buffer_capacity = 200_000         # 200K transitions — fits easily in 32GB RAM
    batch_size = 128                  # Optimized for 4GB VRAM with single CNN forward/backward
    gamma = 0.99                      # Discount factor
    target_update_freq = 5000         # Hard update target network every 5K steps
    start_training_step = 50_000      # Random warmup phase to fill buffer with diverse data

    # Epsilon-Greedy exploration parameters
    epsilon_start = 1.0               # Start fully random
    epsilon_end = 0.05                # Minimum exploration rate
    epsilon_decay = 500_000           # Linear decay over 500K steps

    # =====================================================================
    # SETUP — Device, Environment, Networks
    # =====================================================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    device_type = device.type  # "cuda" or "cpu" — used for AMP autocast
    use_amp = (device_type == "cuda")
    print(f"🖥️  Using device: {device} | AMP enabled: {use_amp}")

    # Enable cuDNN benchmark for fixed-size inputs (96x96) — finds fastest convolution algorithm
    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Environment setup — DQN requires discrete actions (is_discrete=True)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, run_name, is_discrete=True)]
    )

    # Initialize Policy and Target Networks
    policy_net = DQNAgent(num_actions=5).to(device)
    target_net = DQNAgent(num_actions=5).to(device)

    # --- Checkpoint Loading System ---
    start_step = 0
    if resume_from_checkpoint and os.path.exists(f"models/{run_name}"):
        chkp_files = [f for f in os.listdir(f"models/{run_name}") if f.startswith("dqn_step_")]
        if chkp_files:
            latest_step = max([int(f.split("_step_")[1].split(".pth")[0]) for f in chkp_files])
            start_step = latest_step
            policy_net.load_state_dict(torch.load(f"models/{run_name}/dqn_step_{latest_step}.pth", map_location=device))
            print(f"\n✅ Checkpoint DQN found! Resuming training from step {latest_step}...\n")

    # Synchronize target network with policy network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is strictly for inference

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # AMP GradScaler — scales gradients to prevent underflow in FP16
    # When enabled=False (CPU), all scaler methods become transparent no-ops
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
        # 1. Epsilon Decay — Linear decay from 1.0 to 0.05 over 500K steps
        epsilon = max(epsilon_end, epsilon_start - global_step / epsilon_decay)

        # 2. Select Action using epsilon-greedy strategy
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
            action = policy_net.get_action(obs_tensor, epsilon, device)
        action_np = action.cpu().numpy()

        # 3. Environment Step
        next_obs, rewards, terminations, truncations, infos = envs.step(action_np)
        next_obs = np.array(next_obs)
        dones = np.logical_or(terminations, truncations)

        current_ep_reward += rewards[0]
        if dones[0]:
            episode_rewards.append(current_ep_reward)
            print(f"Step: {global_step:>8,} | Epsilon: {epsilon:.3f} | Ep Reward: {current_ep_reward:.2f}")
            current_ep_reward = 0

        # 4. Store transition in Replay Buffer
        buffer.push(obs[0], action_np[0], rewards[0], next_obs[0], dones[0])
        obs = next_obs

        # 5. Train — Only after warmup and with sufficient buffer data
        if global_step >= start_training_step and len(buffer) >= batch_size:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)

            b_obs = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
            b_actions = torch.as_tensor(b_actions, dtype=torch.int64, device=device).unsqueeze(1)
            b_rewards = torch.as_tensor(b_rewards, device=device)
            b_next_obs = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
            b_dones = torch.as_tensor(b_dones, device=device)

            # Forward pass with AMP autocast (FP16 on GPU for ~2x speedup)
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # Compute Q(s_t, a) — the Q-value for the action we actually took
                q_values = policy_net(b_obs)
                state_action_values = q_values.gather(1, b_actions).squeeze(1)

                # Compute V(s_{t+1}) using frozen Target Network (no gradients)
                with torch.no_grad():
                    next_q_values = target_net(b_next_obs)
                    max_next_q_values = next_q_values.max(1)[0]

                # Bellman Equation: Q_target = r + γ * max_a' Q_target(s', a') * (1 - done)
                expected_state_action_values = b_rewards + (gamma * max_next_q_values * (1 - b_dones))

                # MSE Loss between predicted and target Q-values
                loss = loss_fn(state_action_values, expected_state_action_values)

            # Backward pass with AMP GradScaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # Gradient clipping for stability
            scaler.step(optimizer)
            scaler.update()

        # 6. Hard Update Target Network — Copy policy weights to target every N steps
        if global_step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 7. Save Checkpoint every 100K steps
        if global_step > 0 and global_step % save_freq == 0:
            os.makedirs(f"models/{run_name}", exist_ok=True)
            torch.save(policy_net.state_dict(), f"models/{run_name}/dqn_step_{global_step}.pth")
            print(f"💾 Checkpoint DQN saved at step {global_step:,}")

    # Save final model
    os.makedirs(f"models/{run_name}", exist_ok=True)
    torch.save(policy_net.state_dict(), f"models/{run_name}/dqn_final.pth")
    envs.close()
    print(f"\n✅ DQN Training Complete — {total_timesteps:,} steps | Final model saved.")


if __name__ == "__main__":
    train_dqn()