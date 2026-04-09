"""
Training script for the TD3 agent on CarRacing-v2.

This file orchestrates the training loop for the Twin Delayed DDPG algorithm.
It handles environment interactions, experience replay, and the delayed network 
updates that characterize TD3.
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
    # --- Hyperparameters Config ---
    run_name = "td3_baseline"
    env_id = "CarRacing-v2"
    seed = 42
    
    # TD3 standard hyperparameters
    total_timesteps = 300000        # TOTAL training steps (Reducido a 300k para prueba)
    save_freq = 50000               # Cada cuántos pasos guarda un checkpoint
    resume_from_checkpoint = True   # Si encuentra un checkpoint previo, continua desde él
    learning_rate = 3e-4            # Learning rate for both Actor and Critic
    buffer_capacity = 100000        # Replay buffer size
    batch_size = 256                # Size of the training mini-batch
    gamma = 0.99                    # Discount factor for future rewards
    tau = 0.005                     # Soft update rate for target networks
    
    # Exploration and Noise parameters
    exploration_noise = 0.1         # Standard deviation of Gaussian noise added to action for exploration
    policy_noise = 0.2              # Target policy smoothing noise standard deviation
    noise_clip = 0.5                # Clip max/min target policy noise
    policy_delay = 2                # In TD3, the actor is updated less frequently (every 2 steps)
    start_training_step = 25000     # Random exploration phase before learning begins
    
    # Define action bounds matching CarRacing continuous actions
    action_low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # --- Setup ---
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    # Set up the environment. Note: is_discrete=False because TD3 uses continuous actions.
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, run_name, is_discrete=False)]
    )

    # 1. Initialize Actor and Critic Networks
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
            print(f"\n✅ ¡Checkpoint encontrado! Continuando el entrenamiento desde el paso {latest_step}...\n")
            
    actor_target = copy.deepcopy(actor) # Create target network as exact copy
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)

    critic_target = copy.deepcopy(critic) # Create target network as exact copy
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    buffer = ReplayBuffer(buffer_capacity)

    # --- Training Loop ---
    obs, _ = envs.reset(seed=seed)
    obs = np.array(obs)

    episode_rewards = []
    current_ep_reward = 0

    for global_step in range(start_step, total_timesteps):
        
        # 2. Select Action (Exploration vs Exploitation)
        # For the first 'start_training_step' steps, sample purely random actions
        # to ensure the buffer has diverse data before updating the networks.
        if global_step < start_training_step:
            action_np = envs.action_space.sample() # Random continuous action
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                
                # Get the deterministic action from the actor network
                action_tensor = actor(obs_tensor)
                
                # Add Gaussian noise for exploration during training
                noise = torch.normal(0, exploration_noise, size=action_tensor.shape).to(device)
                action_tensor = action_tensor + noise
                
                action_np = action_tensor.cpu().numpy()
                
                # Ensure the noisy action respects environment limits
                action_np = np.clip(action_np, action_low, action_high)

        # 3. Step Environment
        next_obs, rewards, terminations, truncations, infos = envs.step(action_np)
        next_obs = np.array(next_obs)
        dones = np.logical_or(terminations, truncations)

        current_ep_reward += rewards[0]
        if dones[0]:
            episode_rewards.append(current_ep_reward)
            print(f"Step: {global_step} | Ep Reward: {current_ep_reward:.2f}")
            current_ep_reward = 0

        # 4. Store the transition in the Replay Buffer
        buffer.push(obs[0], action_np[0], rewards[0], next_obs[0], dones[0])
        obs = next_obs

        # 5. Network Updates (Learning Phase)
        # We only start updating if we have passed the initial warmup AND we have enough experiences in buffer
        if global_step >= start_training_step and len(buffer) >= batch_size:
            # Sample a mini-batch of transitions
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)

            b_obs = torch.tensor(b_obs, dtype=torch.float32).to(device)
            b_actions = torch.tensor(b_actions, dtype=torch.float32).to(device)
            b_rewards = torch.tensor(b_rewards).unsqueeze(1).to(device)
            b_next_obs = torch.tensor(b_next_obs, dtype=torch.float32).to(device)
            b_dones = torch.tensor(b_dones).unsqueeze(1).to(device)

            # --- Update Critic (Every Step) ---
            with torch.no_grad():
                # Target Policy Smoothing: Add clipped noise to the target actor's next action.
                # This makes the value estimation more robust and prevents exploitation of peaks in the Q-function.
                next_action = actor_target(b_next_obs)
                noise = torch.normal(0, policy_noise, size=next_action.shape).to(device)
                noise = torch.clamp(noise, -noise_clip, noise_clip)
                smoothed_next_action = next_action + noise
                
                # Clip the smoothed action to ensure it remains within valid bounds
                t_low = torch.tensor(action_low).to(device)
                t_high = torch.tensor(action_high).to(device)
                smoothed_next_action = torch.max(torch.min(smoothed_next_action, t_high), t_low)

                # Twin Q-Network: Evaluate the next state-action pair using both target critics
                target_q1, target_q2 = critic_target(b_next_obs, smoothed_next_action)
                
                # Take the minimum of both Q-values to combat overestimation bias (TD3 core feature)
                target_q = torch.min(target_q1, target_q2)
                
                # Compute the Bellman target value
                target_q = b_rewards + gamma * target_q * (1 - b_dones)

            # Get current Q-value estimates from both critics
            current_q1, current_q2 = critic(b_obs, b_actions)

            # Calculate MSE loss for both critics comparing to the Bellman target
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # Optimize Critic networks
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # --- Delayed Actor Update ---
            # TD3 delays the actor update to ensure the critic has stabilized before optimizing the policy
            if global_step % policy_delay == 0:
                
                # Actor objective is to maximize the expected Q-value.
                # We calculate the action proposed by our current actor, then evaluate it using Critic 1.
                # We use only Critic 1 (q1) for the actor update gradient (it's sufficient).
                # We use negative Q-value because Optimizers minimize loss
                actor_loss = -critic.q1(b_obs, actor(b_obs)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # --- Soft Updates of Target Networks ---
                # Slowly blend weights of local networks into target networks using Polyak averaging (tau)
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 6. Save Model Checkpoints
        if global_step > 0 and global_step % save_freq == 0:
            os.makedirs(f"models/{run_name}", exist_ok=True)
            # Both actor and critic are saved for continuation or evaluation
            torch.save(actor.state_dict(), f"models/{run_name}/td3_actor_step_{global_step}.pth")
            torch.save(critic.state_dict(), f"models/{run_name}/td3_critic_step_{global_step}.pth")
            print(f"💾 Checkpoint guardado temporalmente en el paso {global_step}")

if __name__ == "__main__":
    train_td3()
