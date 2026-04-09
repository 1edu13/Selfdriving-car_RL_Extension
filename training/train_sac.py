"""
Training script for the SAC agent on CarRacing-v2.

This file manages the advanced Maximum Entropy training loop of Soft Actor-Critic.
SAC maximizes both expected return AND the policy's entropy (randomness), 
which leads to exceptional exploration and robust policies.
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
    # --- Hyperparameters Config ---
    run_name = "sac_baseline"
    env_id = "CarRacing-v2"
    seed = 42
    
    # Checkpoints & Length
    total_timesteps = 300000        # Prueba de entrenamiento
    save_freq = 50000
    resume_from_checkpoint = True
    
    # Network hyperparameters
    learning_rate = 3e-4
    buffer_capacity = 100000
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    start_training_step = 25000     # Allow buffer to fill before training
    
    # Auto-Tuning Entropy (Alpha) Hyperparameters
    # SAC attempts to find the perfect balance between exploration (entropy) and exploitation (Q-value)
    # We define a target entropy that the algorithm will try to match during training.
    # A standard heuristic for target entropy is `-dim(A)` (negative action dimension)
    target_entropy = -3.0  
    
    # --- Initialization Setup ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    # Vector environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, run_name, is_discrete=False)]
    )

    # 1. Initialize Actor and Critic Networks
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
            print(f"\n✅ ¡Checkpoint SAC encontrado! Continuando el entrenamiento desde el paso {latest_step}...\n")

    # Target Critic network for Bellman updates (SAC doesn't use a Target Actor)
    critic_target = copy.deepcopy(critic)

    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    # We learn the entropy temperature (alpha) using its own separate optimizer
    # log_alpha is used to ensure alpha always remains positive when exponentiated
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate)

    buffer = ReplayBuffer(buffer_capacity)

    # --- Main Training Loop ---
    obs, _ = envs.reset(seed=seed)
    obs = np.array(obs)

    episode_rewards = []
    current_ep_reward = 0

    for global_step in range(start_step, total_timesteps):
        
        # 2. Select Action
        if global_step < start_training_step:
            # Completely random initial exploration
            action_np_normalized = envs.action_space.sample() 
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                
                # Sample the stochastic policy
                # Returns the normalized action [-1, 1] mathematically squashed by Tanh
                action_tensor, _ = actor.get_action(obs_tensor)
                action_np_normalized = action_tensor.cpu().numpy()[0]

        # Convert the [-1, 1] actions into the format the game actually expects
        action_env = scale_action_for_env(action_np_normalized)

        # 3. Step Environment
        # Note: We send the scaled action to the game, but...
        next_obs, rewards, terminations, truncations, infos = envs.step([action_env])
        next_obs = np.array(next_obs)
        dones = np.logical_or(terminations, truncations)

        current_ep_reward += rewards[0]
        if dones[0]:
            episode_rewards.append(current_ep_reward)
            print(f"Step: {global_step} | Ep Reward: {current_ep_reward:.2f}")
            current_ep_reward = 0

        # ... We MUST push the NORMALIZED action to the Replay Buffer.
        # This is incredibly CRITICAL. The Actor works in a squashed [-1, 1] space, so the
        # Critic must learn Q-values over the squashed space, not the environment space!
        buffer.push(obs[0], action_np_normalized, rewards[0], next_obs[0], dones[0])
        obs = next_obs

        # 4. Neural Network Learning Phase
        if global_step >= start_training_step and len(buffer) >= batch_size:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)

            b_obs = torch.tensor(b_obs, dtype=torch.float32).to(device)
            b_actions = torch.tensor(b_actions, dtype=torch.float32).to(device)
            b_rewards = torch.tensor(b_rewards).unsqueeze(1).to(device)
            b_next_obs = torch.tensor(b_next_obs, dtype=torch.float32).to(device)
            b_dones = torch.tensor(b_dones).unsqueeze(1).to(device)

            # Get the current dynamic alpha (Temperature factor)
            alpha = log_alpha.exp().detach()

            # --- Update Critic (Q-Networks) ---
            with torch.no_grad():
                # Sample NEXT actions and their log-probabilities from current policy
                next_action, next_log_prob = actor.get_action(b_next_obs)
                
                # Evaluate Bellman Target using Twin Critics
                target_q1, target_q2 = critic_target(b_next_obs, next_action)
                target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                
                # Soft Q-Target = Reward + Discount * (Value of next state - Entropy value)
                target_q = b_rewards + gamma * target_q * (1 - b_dones)

            # Get current Q-values given our observed Replay actions
            current_q1, current_q2 = critic(b_obs, b_actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # --- Update Actor (Policy) ---
            # Evaluate CURRENT state through the Actor to compute the loss map
            curr_action, curr_log_prob = actor.get_action(b_obs)
            
            # The Critic evaluates these newly proposed actions
            curr_q1, curr_q2 = critic(b_obs, curr_action)
            curr_q = torch.min(curr_q1, curr_q2)

            # Loss for the actor is carefully modeled:
            # We want to MAXIMIZE Expected Q-Value (`curr_q` needs to be high -> negative loss)
            # AND MAXIMIZE Entropy (`curr_log_prob` needs to be broad -> low log prob)
            actor_loss = (alpha * curr_log_prob - curr_q).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # --- Update Temperature (Alpha) ---
            # Automatically tune alpha to match our heuristic target entropy.
            # If the current entropy is lower than target, log_alpha becomes more positive (pushes for exploration)
            alpha_loss = -(log_alpha * (curr_log_prob + target_entropy).detach()).mean()

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # --- Soft Updates ---
            # Slowly blend local critic weights into target critic network
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 5. Checkpoint Saving
        if global_step > 0 and global_step % save_freq == 0:
            os.makedirs(f"models/{run_name}", exist_ok=True)
            torch.save(actor.state_dict(), f"models/{run_name}/sac_actor_step_{global_step}.pth")
            torch.save(critic.state_dict(), f"models/{run_name}/sac_critic_step_{global_step}.pth")
            print(f"💾 Checkpoint SAC guardado en el paso {global_step}")

if __name__ == "__main__":
    train_sac()