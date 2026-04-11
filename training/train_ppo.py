"""
Training script for the PPO agent on CarRacing-v2.

PPO (Proximal Policy Optimization) is an on-policy algorithm that uses clipped
surrogate objectives to ensure stable policy updates. Key features:
  - On-policy: learns from freshly collected rollout data (no replay buffer)
  - Clipped objective: prevents destructively large policy updates
  - GAE (Generalized Advantage Estimation): reduces variance in advantage estimates
  - Vectorized environments: collects data from multiple parallel environments

On-policy methods require ~2-3x more samples than off-policy, hence 3M total timesteps.

Optimized for: NVIDIA RTX 3050 (4GB VRAM) | AMD Ryzen 7 4800H | 32GB RAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os

from agents.ppo_agent import PPOAgent
from core.utils import make_env, get_device


def train_ppo():
    # =====================================================================
    # HYPERPARAMETERS — Optimized for RTX 3050 (4GB VRAM) + 32GB RAM
    # =====================================================================
    run_name = "ppo_baseline"
    env_id = "CarRacing-v2"
    seed = 42

    # Training length
    total_timesteps = 3_000_000       # 3M steps — on-policy needs more samples than off-policy

    # PPO core parameters
    learning_rate = 3e-4
    num_envs = 4                      # 4 parallel envs (reduced from 8 for VRAM constraints)
    num_steps = 1024                  # Steps per env per rollout
    anneal_lr = True                  # Linear LR decay
    gamma = 0.99                      # Discount factor
    gae_lambda = 0.95                 # GAE smoothing parameter
    num_minibatches = 16              # Split batch into 16 minibatches
    update_epochs = 10                # PPO optimization epochs per rollout
    norm_adv = True                   # Normalize advantages per minibatch
    clip_coef = 0.2                   # PPO clipping coefficient
    ent_coef = 0.01                   # Entropy bonus coefficient (encourages exploration)
    vf_coef = 0.5                     # Value function loss coefficient
    max_grad_norm = 0.5               # Gradient clipping

    # Derived sizes
    batch_size = int(num_envs * num_steps)       # 4 * 1024 = 4096
    minibatch_size = int(batch_size // num_minibatches)  # 4096 / 16 = 256

    # Checkpoint frequency — save approximately every 100K steps
    # Each update processes batch_size steps, so: 100K / 4096 ≈ 25 updates
    save_freq_updates = max(1, 100_000 // batch_size)  # = 24 updates ≈ ~98K steps

    # =====================================================================
    # SETUP — Device, Environment, Networks
    # =====================================================================
    device = get_device()
    device_type = device.type
    use_amp = (device_type == "cuda")
    print(f"🖥️  Using device: {device} | AMP enabled: {use_amp}")

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(f"models/{run_name}", exist_ok=True)
    os.makedirs("results/videos", exist_ok=True)

    # Vectorized environment — SyncVectorEnv for Windows stability
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i, i, capture_video=False, run_name=run_name) for i in range(num_envs)]
    )

    # Initialize PPO Agent
    agent = PPOAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # AMP GradScaler for mixed-precision
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # --- Rollout Storage Buffers ---
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Start training
    global_step = 0
    next_obs = torch.Tensor(envs.reset(seed=seed)[0]).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size

    print(f"🚀 Starting PPO Training... Total Updates: {num_updates} | Batch: {batch_size} | Minibatch: {minibatch_size}")

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ===== Phase 1: Rollout Collection =====
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(terminations | truncations).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"Step: {global_step:>8,} | Ep Reward: {info['episode']['r']:.2f}")

        # ===== Phase 2: GAE (Generalized Advantage Estimation) =====
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # ===== Phase 3: PPO Optimization =====
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        mb_advantages = b_advantages[mb_inds]
                        if norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Clipped surrogate objective
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value function loss
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Entropy bonus
                    entropy_loss = entropy.mean()

                    # Combined loss
                    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                # Backward pass with AMP
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        # ===== Save Checkpoint every ~100K steps =====
        if update % save_freq_updates == 0:
            save_path = f"models/{run_name}/ppo_step_{global_step}.pth"
            torch.save(agent.state_dict(), save_path)
            print(f"💾 Checkpoint PPO saved at step {global_step:,}")

    # Save final model
    torch.save(agent.state_dict(), f"models/{run_name}/ppo_final.pth")
    envs.close()
    print(f"\n✅ PPO Training Complete — {total_timesteps:,} steps | Final model saved.")


if __name__ == "__main__":
    train_ppo()