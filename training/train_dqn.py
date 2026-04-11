"""
Training script for the DQN agent on CarRacing-v2.

DQN (Deep Q-Network) discretizes the continuous action space of CarRacing into 5 actions
and learns a value function Q(s, a) that estimates the expected future reward for each
state-action pair. The agent then always picks the action with the highest Q-value.

Optimized for: NVIDIA RTX 3050 (4GB VRAM) | AMD Ryzen 7 4800H | 32GB RAM
"""

import sys
import os

# Ensure project root is in path when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import gymnasium as gym
import random
import time
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


def print_header(device, use_amp, hparams):
    """Prints a formatted training configuration header."""
    print()
    print("=" * 64)
    print("  DQN TRAINING -- CarRacing-v2 (Discrete Actions)".center(64))
    print("=" * 64)
    print(f"  Device:         {device}")
    print(f"  AMP (FP16):     {'Enabled' if use_amp else 'Disabled'}")
    print(f"  Total Steps:    {hparams['total_timesteps']:,}")
    print(f"  Batch Size:     {hparams['batch_size']}")
    print(f"  Buffer Size:    {hparams['buffer_capacity']:,}")
    print(f"  Frame Skip:     {hparams['frame_skip']} (action repeated {hparams['frame_skip']}x per step)")
    print(f"  Learning Rate:  {hparams['learning_rate']}")
    print(f"  Gamma:          {hparams['gamma']}")
    print(f"  Epsilon Decay:  {hparams['epsilon_decay']:,} steps")
    print(f"  Target Update:  Every {hparams['target_update_freq']:,} steps")
    print(f"  Warmup:         {hparams['start_training_step']:,} random steps")
    print(f"  Checkpoints:    Every {hparams['save_freq']:,} steps")
    print(f"  Resume:         {hparams['resume']}")
    print("=" * 64)
    print()


def train_dqn():
    # =====================================================================
    # HYPERPARAMETERS -- Optimized for RTX 3050 (4GB VRAM) + 32GB RAM
    # =====================================================================
    run_name = "dqn_baseline"
    env_id = "CarRacing-v2"
    seed = 42

    # Training length & checkpointing
    total_timesteps = 2_000_000       # 2M steps -- sufficient for DQN on CarRacing with discrete actions
    save_freq = 100_000               # Save checkpoint every 100K steps
    log_freq = 10_000                 # Print training metrics every 10K steps
    resume_from_checkpoint = False    # Set to True to resume from the latest checkpoint

    # Network hyperparameters
    learning_rate = 1e-4              # Lower LR for better stability with DQN
    buffer_capacity = 200_000         # 200K transitions -- fits easily in 32GB RAM
    batch_size = 256                  # Optimized for 4GB VRAM with single CNN forward/backward
    gamma = 0.99                      # Discount factor
    target_update_freq = 5000         # Hard update target network every 5K steps
    start_training_step = 50_000      # Random warmup phase to fill buffer with diverse data

    # Epsilon-Greedy exploration parameters
    epsilon_start = 1.0               # Start fully random
    epsilon_end = 0.05                # Minimum exploration rate
    epsilon_decay = 500_000           # Linear decay over 500K steps

    # =====================================================================
    # SETUP -- Device, Environment, Networks
    # =====================================================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    device_type = device.type
    use_amp = (device_type == "cuda")

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Print configuration header
    print_header(device, use_amp, {
        'total_timesteps': total_timesteps, 'batch_size': batch_size,
        'buffer_capacity': buffer_capacity, 'learning_rate': learning_rate,
        'gamma': gamma, 'epsilon_decay': epsilon_decay,
        'target_update_freq': target_update_freq,
        'start_training_step': start_training_step,
        'save_freq': save_freq, 'resume': resume_from_checkpoint,
        'frame_skip': 2,
    })

    # Environment setup -- DQN requires discrete actions (is_discrete=True)
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
            print(f"  >> Checkpoint found! Resuming from step {latest_step:,}\n")

    # Synchronize target network with policy network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
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

    # Metric tracking for periodic logging
    recent_losses = []
    recent_q_values = []
    train_start_time = time.time()
    last_log_step = start_step

    print("  Step         | Epsilon | Loss     | Avg Q   | Buffer  | Ep Reward | Avg(10)")
    print("  " + "-" * 80)

    for global_step in range(start_step, total_timesteps):
        # 1. Epsilon Decay
        epsilon = max(epsilon_end, epsilon_start - global_step / epsilon_decay)

        # 2. Select Action
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad(), autocast(enabled=use_amp):
            action = policy_net.get_action(obs_tensor, epsilon, device)
        action_np = action.cpu().numpy()

        # 3. Environment Step
        next_obs, rewards, terminations, truncations, infos = envs.step(action_np)
        next_obs = np.array(next_obs)
        dones = np.logical_or(terminations, truncations)

        current_ep_reward += rewards[0]
        if dones[0]:
            episode_rewards.append(current_ep_reward)
            episode_count += 1
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  {global_step:>13,} | {epsilon:.3f}   | "
                  f"{'---':>8s} | {'---':>7s} | {len(buffer):>7,} | "
                  f"{current_ep_reward:>9.1f} | {avg_reward:>7.1f}")
            current_ep_reward = 0

        # 4. Store in Buffer
        buffer.push(obs[0], action_np[0], rewards[0], next_obs[0], dones[0])
        obs = next_obs

        # 5. Train
        if global_step >= start_training_step and len(buffer) >= batch_size:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)

            b_obs = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
            b_actions = torch.as_tensor(b_actions, dtype=torch.int64, device=device).unsqueeze(1)
            b_rewards = torch.as_tensor(b_rewards, device=device)
            b_next_obs = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
            b_dones = torch.as_tensor(b_dones, device=device)

            with autocast(enabled=use_amp):
                q_values = policy_net(b_obs)
                state_action_values = q_values.gather(1, b_actions).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_net(b_next_obs)
                    max_next_q_values = next_q_values.max(1)[0]

                expected_state_action_values = b_rewards + (gamma * max_next_q_values * (1 - b_dones))
                loss = loss_fn(state_action_values, expected_state_action_values)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            recent_losses.append(loss.item())
            recent_q_values.append(q_values.mean().item())

        # 6. Update Target Network
        if global_step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 7. Periodic training metrics log
        if global_step > 0 and global_step % log_freq == 0 and recent_losses:
            elapsed = time.time() - train_start_time
            steps_per_sec = (global_step - start_step) / max(elapsed, 1)
            ms_per_step = 1000 / max(steps_per_sec, 1)
            avg_loss = np.mean(recent_losses[-1000:])
            avg_q = np.mean(recent_q_values[-1000:])
            avg_rew = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            pct = 100 * global_step / total_timesteps

            print(f"  {global_step:>13,} | {epsilon:.3f}   | "
                  f"{avg_loss:>8.4f} | {avg_q:>7.2f} | {len(buffer):>7,} | "
                  f"{'':>9s} | {avg_rew:>7.1f}  "
                  f"[{pct:>5.1f}% | {steps_per_sec:,.0f} sps | {ms_per_step:.2f} ms/step]")

        # 8. Save Checkpoint
        if global_step > 0 and global_step % save_freq == 0:
            os.makedirs(f"models/{run_name}", exist_ok=True)
            torch.save(policy_net.state_dict(), f"models/{run_name}/dqn_step_{global_step}.pth")
            print(f"  >> [SAVE] Checkpoint saved: models/{run_name}/dqn_step_{global_step}.pth")

    # Save final model
    os.makedirs(f"models/{run_name}", exist_ok=True)
    torch.save(policy_net.state_dict(), f"models/{run_name}/dqn_final.pth")
    envs.close()

    total_time = time.time() - train_start_time
    print()
    print("=" * 64)
    print("  DQN TRAINING COMPLETE".center(64))
    print("=" * 64)
    print(f"  Total Steps:    {total_timesteps:,}")
    print(f"  Total Time:     {total_time/3600:.1f} hours")
    print(f"  Episodes:       {episode_count}")
    if episode_rewards:
        print(f"  Best Reward:    {max(episode_rewards):.1f}")
        print(f"  Final Avg(10):  {np.mean(episode_rewards[-10:]):.1f}")
    print(f"  Model Saved:    models/{run_name}/dqn_final.pth")
    print("=" * 64)
    print()


if __name__ == "__main__":
    train_dqn()