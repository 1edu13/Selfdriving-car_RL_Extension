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
import time

from agents.ppo_agent import PPOAgent
from core.utils import make_env, get_device


def print_header(device, use_amp, hp):
    """Prints a formatted training configuration header."""
    print()
    print("=" * 64)
    print("  PPO TRAINING -- CarRacing-v2 (On-Policy)".center(64))
    print("=" * 64)
    print(f"  Device:         {device}")
    print(f"  AMP (FP16):     {'Enabled' if use_amp else 'Disabled'}")
    print(f"  Total Steps:    {hp['total_timesteps']:,}")
    print(f"  Num Envs:       {hp['num_envs']}")
    print(f"  Steps/Rollout:  {hp['num_steps']} (Batch: {hp['batch_size']:,})")
    print(f"  Minibatch:      {hp['minibatch_size']} x {hp['num_minibatches']} minibatches")
    print(f"  Update Epochs:  {hp['update_epochs']}")
    print(f"  Learning Rate:  {hp['learning_rate']}  (Annealing: {hp['anneal_lr']})")
    print(f"  Gamma:          {hp['gamma']}  |  GAE Lambda: {hp['gae_lambda']}")
    print(f"  Clip Coef:      {hp['clip_coef']}  |  Ent Coef: {hp['ent_coef']}")
    print(f"  Total Updates:  {hp['num_updates']}")
    print(f"  Frame Skip:     {hp['frame_skip']} (action repeated {hp['frame_skip']}x per step)")
    print(f"  Checkpoints:    Every ~{hp['save_freq_steps']:,} steps")
    print("=" * 64)
    print()


def train_ppo():
    # =====================================================================
    # HYPERPARAMETERS -- Optimized for RTX 3050 (4GB VRAM) + 32GB RAM
    # =====================================================================
    run_name = "ppo_baseline"
    env_id = "CarRacing-v2"
    seed = 42

    total_timesteps = 3_000_000

    learning_rate = 3e-4
    num_envs = 4
    num_steps = 1024
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 16
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    batch_size = int(num_envs * num_steps)       # 4096
    minibatch_size = int(batch_size // num_minibatches)  # 256
    save_freq_updates = max(1, 100_000 // batch_size)  # ~25 updates

    # =====================================================================
    # SETUP
    # =====================================================================
    device = get_device()
    device_type = device.type
    use_amp = (device_type == "cuda")

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(f"models/{run_name}", exist_ok=True)
    os.makedirs("results/videos", exist_ok=True)

    num_updates = total_timesteps // batch_size

    print_header(device, use_amp, {
        'total_timesteps': total_timesteps, 'num_envs': num_envs,
        'num_steps': num_steps, 'batch_size': batch_size,
        'minibatch_size': minibatch_size, 'num_minibatches': num_minibatches,
        'update_epochs': update_epochs, 'learning_rate': learning_rate,
        'anneal_lr': anneal_lr, 'gamma': gamma, 'gae_lambda': gae_lambda,
        'clip_coef': clip_coef, 'ent_coef': ent_coef,
        'num_updates': num_updates,
        'save_freq_steps': save_freq_updates * batch_size,
        'frame_skip': 2,
    })

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i, i, capture_video=False, run_name=run_name) for i in range(num_envs)]
    )

    agent = PPOAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    scaler = GradScaler(enabled=use_amp)

    # Rollout Storage
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset(seed=seed)[0]).to(device)
    next_done = torch.zeros(num_envs).to(device)

    # Metric tracking
    episode_rewards = []
    train_start_time = time.time()

    print("  Update   | Steps      | PG Loss  | V Loss   | Entropy | LR       | Avg Rew | Progress")
    print("  " + "-" * 90)

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        else:
            lrnow = learning_rate

        # ===== Phase 1: Rollout Collection =====
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad(), autocast(enabled=use_amp):
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
                        episode_rewards.append(float(info['episode']['r']))

        # ===== Phase 2: GAE =====
        with torch.no_grad(), autocast(enabled=use_amp):
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

        # Track losses for this update
        update_pg_losses = []
        update_v_losses = []
        update_entropies = []

        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                with autocast(enabled=use_amp):
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        mb_advantages = b_advantages[mb_inds]
                        if norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()

                    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                update_pg_losses.append(pg_loss.item())
                update_v_losses.append(v_loss.item())
                update_entropies.append(entropy_loss.item())

        # ===== Per-Update Logging =====
        avg_pg = np.mean(update_pg_losses)
        avg_vl = np.mean(update_v_losses)
        avg_ent = np.mean(update_entropies)
        avg_rew = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        pct = 100 * global_step / total_timesteps
        elapsed = time.time() - train_start_time
        sps = global_step / max(elapsed, 1)
        ms_per_step = 1000 / max(sps, 1)

        print(f"  {update:>8d} | {global_step:>10,} | {avg_pg:>8.4f} | {avg_vl:>8.4f} | "
              f"{avg_ent:>7.3f} | {lrnow:>8.1e} | {avg_rew:>7.1f} | {pct:>5.1f}% {sps:,.0f}sps {ms_per_step:.2f}ms/step")

        # ===== Save Checkpoint =====
        if update % save_freq_updates == 0:
            save_path = f"models/{run_name}/ppo_step_{global_step}.pth"
            torch.save(agent.state_dict(), save_path)
            print(f"  >> [SAVE] Checkpoint: {save_path}")

    # Final save
    torch.save(agent.state_dict(), f"models/{run_name}/ppo_final.pth")
    envs.close()

    total_time = time.time() - train_start_time
    print()
    print("=" * 64)
    print("  PPO TRAINING COMPLETE".center(64))
    print("=" * 64)
    print(f"  Total Steps:    {total_timesteps:,}")
    print(f"  Total Time:     {total_time/3600:.1f} hours")
    print(f"  Total Updates:  {num_updates}")
    print(f"  Episodes:       {len(episode_rewards)}")
    if episode_rewards:
        print(f"  Best Reward:    {max(episode_rewards):.1f}")
        print(f"  Final Avg(10):  {np.mean(episode_rewards[-10:]):.1f}")
    print(f"  Model Saved:    models/{run_name}/ppo_final.pth")
    print("=" * 64)
    print()


if __name__ == "__main__":
    train_ppo()