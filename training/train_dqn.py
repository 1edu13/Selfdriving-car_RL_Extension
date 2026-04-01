import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import os
from collections import deque

from agents.dqn_agent import DQNAgent # Assuming you saved the previous agent here
from core.utils import make_env, get_device

class ReplayBuffer:
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
    # --- Hyperparameters ---
    run_name = "dqn_baseline"
    env_id = "CarRacing-v2"
    seed = 42
    total_timesteps = 15000
    learning_rate = 1e-4
    buffer_capacity = 50000
    batch_size = 128
    gamma = 0.99
    target_update_freq = 1000  # Steps before updating target network
    start_training_step = 10000 # Wait before training to fill buffer

    # Epsilon-Greedy parameters
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 100000 # Decay epsilon over this many steps

    # --- Setup ---
    device = get_device()
    print(f"Using device: {device}")

    # Setup environment (Using SyncVectorEnv for compatibility, though DQN often uses single envs)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, False, run_name, is_discrete=True)]
    )

    # Initialize Networks
    policy_net = DQNAgent(num_actions=5).to(device)
    target_net = DQNAgent(num_actions=5).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Target network is strictly for inference

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    buffer = ReplayBuffer(buffer_capacity)

    # --- Training Loop ---
    obs, _ = envs.reset(seed=seed)
    obs = np.array(obs) # Ensure numpy array

    episode_rewards = []
    current_ep_reward = 0

    for global_step in range(total_timesteps):
        # 1. Epsilon Decay Calculation
        epsilon = max(epsilon_end, epsilon_start - global_step / epsilon_decay)

        # 2. Select Action
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action = policy_net.get_action(obs_tensor, epsilon, device)
        action_np = action.cpu().numpy()

        # 3. Environment Step
        next_obs, rewards, terminations, truncations, infos = envs.step(action_np)
        next_obs = np.array(next_obs)
        dones = np.logical_or(terminations, truncations)

        current_ep_reward += rewards[0]
        if dones[0]:
            episode_rewards.append(current_ep_reward)
            print(f"Step: {global_step} | Epsilon: {epsilon:.2f} | Ep Reward: {current_ep_reward:.2f}")
            current_ep_reward = 0

        # 4. Store in Buffer
        buffer.push(obs[0], action_np[0], rewards[0], next_obs[0], dones[0])
        obs = next_obs

        # 5. Train
        if len(buffer) > start_training_step:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)

            b_obs = torch.tensor(b_obs, dtype=torch.float32).to(device)
            b_actions = torch.tensor(b_actions, dtype=torch.int64).unsqueeze(1).to(device)
            b_rewards = torch.tensor(b_rewards).to(device)
            b_next_obs = torch.tensor(b_next_obs, dtype=torch.float32).to(device)
            b_dones = torch.tensor(b_dones).to(device)

            # Compute Q(s_t, a)
            q_values = policy_net(b_obs)
            state_action_values = q_values.gather(1, b_actions).squeeze(1)

            # Compute V(s_{t+1}) for all next states using Target Network
            with torch.no_grad():
                next_q_values = target_net(b_next_obs)
                max_next_q_values = next_q_values.max(1)[0]

            # Compute expected Q values (Bellman Equation)
            # If state is terminal (done), the expected future reward is 0
            expected_state_action_values = b_rewards + (gamma * max_next_q_values * (1 - b_dones))

            # Compute Huber loss or MSE loss
            loss = loss_fn(state_action_values, expected_state_action_values)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(policy_net.parameters(), 100) # Gradient clipping
            optimizer.step()

        # 6. Update Target Network
        if global_step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 7. Save Model
        if global_step > 0 and global_step % 10000 == 0:
            os.makedirs("models/dqn_baseline", exist_ok=True)
            torch.save(policy_net.state_dict(), f"models/dqn_baseline/dqn_step_{global_step}.pth")
            print(f"Model saved at step {global_step}")

if __name__ == "__main__":
    train_dqn()