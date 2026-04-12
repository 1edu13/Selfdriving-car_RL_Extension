"""
==============================================================================
EVALUATION SCRIPT -- Individual Agent Assessment for CarRacing-v2
==============================================================================

This script evaluates a single trained RL agent on CarRacing-v2, generating:
  1. Quantitative statistics (reward mean/std/min/max, tiles visited, episode length)
  2. Recorded videos of the agent driving (saved as .mp4)
  3. A per-episode results CSV for later analysis

Usage:
  python evaluation/evaluate_agent.py --agent dqn --model models/dqn_baseline/dqn_final.pth
  python evaluation/evaluate_agent.py --agent td3 --model models/td3_baseline/td3_actor_final.pth
  python evaluation/evaluate_agent.py --agent sac --model models/sac_baseline/sac_actor_final.pth
  python evaluation/evaluate_agent.py --agent ppo --model models/ppo_baseline/ppo_final.pth
  python evaluation/evaluate_agent.py --agent dqn --model auto   (auto-finds latest checkpoint)

Options:
  --episodes   Number of evaluation episodes (default: 20)
  --record     Record video of all episodes (default: True)
  --seed       Random seed (default: 42)
  --render     Show live rendering window (default: False)
"""

import sys
import os
import argparse
import time
import csv
import numpy as np
import torch
import gymnasium as gym

# Ensure project root is in path for direct execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.utils import make_env, get_device


# =========================================================================
# AGENT LOADING
# =========================================================================

def load_agent(agent_type, model_path, device):
    """
    Loads the appropriate agent architecture and weights.
    
    Args:
        agent_type: One of 'dqn', 'td3', 'sac', 'ppo'
        model_path: Path to the .pth weights file
        device: torch device
        
    Returns:
        agent: The loaded agent model in eval mode
        is_discrete: Whether this agent uses discrete actions
    """
    agent_type = agent_type.lower()

    if agent_type == "dqn":
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(num_actions=5)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.to(device).eval()
        return agent, True

    elif agent_type == "td3":
        from agents.td3_agent import Actor as TD3Actor
        agent = TD3Actor(action_dim=3)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.to(device).eval()
        return agent, False

    elif agent_type == "sac":
        from agents.sac_agent import Actor as SACActor
        agent = SACActor(action_dim=3)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.to(device).eval()
        return agent, False

    elif agent_type == "ppo":
        # PPO requires the env to init (for action space shape)
        # We create a temporary env just for initialization
        tmp_envs = gym.vector.SyncVectorEnv(
            [make_env("CarRacing-v2", 0, 0, False, "tmp", frame_skip=1)]
        )
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(tmp_envs)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.to(device).eval()
        tmp_envs.close()
        return agent, False

    else:
        raise ValueError(f"Unknown agent type: '{agent_type}'. Use: dqn, td3, sac, ppo")


def auto_find_model(agent_type):
    """
    Automatically finds the best model checkpoint for the given agent type.
    Searches for *_final.pth first, then falls back to the latest step checkpoint.
    """
    agent_type = agent_type.lower()
    
    # Map agent type to expected directory and file patterns
    dir_map = {
        "dqn": "models/dqn_baseline",
        "td3": "models/td3_baseline",
        "sac": "models/sac_baseline",
        "ppo": "models/ppo_baseline",
    }
    
    final_map = {
        "dqn": "dqn_final.pth",
        "td3": "td3_actor_final.pth",
        "sac": "sac_actor_final.pth",
        "ppo": "ppo_final.pth",
    }
    
    step_prefix_map = {
        "dqn": "dqn_step_",
        "td3": "td3_actor_step_",
        "sac": "sac_actor_step_",
        "ppo": "ppo_step_",
    }
    
    model_dir = os.path.join(PROJECT_ROOT, dir_map[agent_type])
    
    if not os.path.exists(model_dir):
        print(f"  [ERROR] Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Try final model first
    final_path = os.path.join(model_dir, final_map[agent_type])
    if os.path.exists(final_path):
        return final_path
    
    # Fall back to latest step checkpoint
    prefix = step_prefix_map[agent_type]
    step_files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pth")]
    
    if not step_files:
        print(f"  [ERROR] No model files found in: {model_dir}")
        sys.exit(1)
    
    # Extract step numbers and find the latest
    steps = []
    for f in step_files:
        try:
            step_num = int(f.replace(prefix, "").replace(".pth", ""))
            steps.append((step_num, f))
        except ValueError:
            continue
    
    if not steps:
        print(f"  [ERROR] Could not parse step numbers from files in: {model_dir}")
        sys.exit(1)
    
    steps.sort(key=lambda x: x[0], reverse=True)
    latest_file = steps[0][1]
    return os.path.join(model_dir, latest_file)


# =========================================================================
# ACTION SELECTION
# =========================================================================

def select_action(agent, agent_type, obs_tensor, device):
    """
    Selects a deterministic action from the agent for evaluation.
    
    Returns:
        action_np: numpy action array ready for env.step()
    """
    agent_type = agent_type.lower()

    with torch.no_grad():
        if agent_type == "dqn":
            # DQN: greedy action (epsilon=0)
            action = agent.get_action(obs_tensor, epsilon=0.0, device=device)
            return action.cpu().numpy()

        elif agent_type == "td3":
            # TD3: deterministic forward pass (already scales actions internally)
            action = agent(obs_tensor)
            return action.cpu().numpy()

        elif agent_type == "sac":
            # SAC: deterministic mode (uses mean, no sampling)
            action, _ = agent.get_action(obs_tensor, deterministic=True)
            # Scale from [-1,1] to CarRacing ranges [steer, gas, brake]
            action_np = action.cpu().numpy()
            action_np[0, 1] = (action_np[0, 1] + 1.0) / 2.0  # Gas: [-1,1] -> [0,1]
            action_np[0, 2] = (action_np[0, 2] + 1.0) / 2.0  # Brake: [-1,1] -> [0,1]
            return action_np

        elif agent_type == "ppo":
            # PPO: use the mean action (no sampling for evaluation)
            hidden = agent.network(obs_tensor)
            action_mean = agent.actor_mean(hidden)
            return action_mean.cpu().numpy()


# =========================================================================
# EVALUATION LOOP
# =========================================================================

def evaluate(agent_type, model_path, num_episodes=20, seed=42, record=True, render=False):
    """
    Main evaluation function.
    """
    device = get_device()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    agent_upper = agent_type.upper()
    print()
    print("=" * 64)
    print(f"  {agent_upper} EVALUATION -- CarRacing-v2".center(64))
    print("=" * 64)
    print(f"  Agent Type:     {agent_upper}")
    print(f"  Model:          {model_path}")
    print(f"  Device:         {device}")
    print(f"  Episodes:       {num_episodes}")
    print(f"  Record Video:   {record}")
    print(f"  Seed:           {seed}")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # Load Agent
    # ------------------------------------------------------------------
    print(f"  Loading {agent_upper} agent...")
    agent, is_discrete = load_agent(agent_type, model_path, device)
    param_count = sum(p.numel() for p in agent.parameters())
    print(f"  [OK] Agent loaded ({param_count:,} parameters)")
    print()

    # ------------------------------------------------------------------
    # Results directory
    # ------------------------------------------------------------------
    results_dir = os.path.join(PROJECT_ROOT, "results", "evaluation", agent_type.lower())
    os.makedirs(results_dir, exist_ok=True)
    
    video_dir = os.path.join(results_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Environment setup (NO frame_skip for fair evaluation)
    # ------------------------------------------------------------------
    render_mode = "human" if render else "rgb_array"

    # ------------------------------------------------------------------
    # Run Episodes
    # ------------------------------------------------------------------
    print(f"  {'Episode':>8s} | {'Reward':>9s} | {'Steps':>7s} | {'Time (s)':>9s} | Status")
    print(f"  {'-' * 8} | {'-' * 9} | {'-' * 7} | {'-' * 9} | {'-' * 12}")

    episode_rewards = []
    episode_steps = []
    episode_times = []

    for ep in range(num_episodes):
        # Create fresh env per episode (cleanest approach, allows video per episode)
        if record:
            env = gym.make("CarRacing-v2", render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env, video_dir,
                name_prefix=f"{agent_type}_ep{ep:02d}",
                episode_trigger=lambda x: True  # Record every episode
            )
        else:
            env = gym.make("CarRacing-v2", render_mode=render_mode)

        # Apply same wrappers as training (but NO frame_skip for eval accuracy)
        if is_discrete:
            from core.utils import DiscreteActionWrapper
            env = DiscreteActionWrapper(env)

        from gymnasium.wrappers import GrayScaleObservation, FrameStack
        env = GrayScaleObservation(env, keep_dim=False)
        env = FrameStack(env, 4)
        env.action_space.seed(seed + ep)

        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        step_count = 0
        ep_start = time.time()

        while not done:
            # Prepare observation tensor
            obs_array = np.array(obs)
            if obs_array.ndim == 3:
                obs_array = obs_array[np.newaxis, ...]  # Add batch dim: (1, 4, 96, 96)
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)

            # Select action
            action_np = select_action(agent, agent_type, obs_tensor, device)

            # Flatten action for non-vectorized env
            if action_np.ndim > 1:
                action_np = action_np[0]
            
            # For DQN, action is a scalar integer
            if is_discrete:
                action_env = int(action_np[0]) if isinstance(action_np, np.ndarray) else int(action_np)
            else:
                action_env = action_np

            obs, reward, terminated, truncated, info = env.step(action_env)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        ep_time = time.time() - ep_start
        episode_rewards.append(total_reward)
        episode_steps.append(step_count)
        episode_times.append(ep_time)

        # Status indicator
        status = "[OK]" if total_reward > 0 else "[LOW]" if total_reward > -50 else "[FAIL]"

        print(f"  {ep + 1:>8d} | {total_reward:>9.1f} | {step_count:>7d} | {ep_time:>9.2f} | {status}")

        env.close()

    # ------------------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------------------
    rewards = np.array(episode_rewards)
    steps_arr = np.array(episode_steps)
    times_arr = np.array(episode_times)

    print()
    print("=" * 64)
    print(f"  {agent_upper} EVALUATION SUMMARY".center(64))
    print("=" * 64)
    print(f"  Episodes:        {num_episodes}")
    print(f"  Reward Mean:     {rewards.mean():.1f}")
    print(f"  Reward Std:      {rewards.std():.1f}")
    print(f"  Reward Min:      {rewards.min():.1f}")
    print(f"  Reward Max:      {rewards.max():.1f}")
    print(f"  Reward Median:   {np.median(rewards):.1f}")
    print(f"  Avg Steps:       {steps_arr.mean():.0f}")
    print(f"  Avg Time/Episode:{times_arr.mean():.2f}s")
    print(f"  Success Rate:    {(rewards > 0).sum()}/{num_episodes} "
          f"({(rewards > 0).mean() * 100:.0f}%)")
    print(f"  High Score Rate: {(rewards > 300).sum()}/{num_episodes} "
          f"({(rewards > 300).mean() * 100:.0f}%)")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(results_dir, f"{agent_type}_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "time_seconds"])
        for i in range(num_episodes):
            writer.writerow([i + 1, f"{episode_rewards[i]:.2f}", 
                           episode_steps[i], f"{episode_times[i]:.2f}"])

    print(f"\n  [SAVE] Results CSV:  {csv_path}")
    if record:
        print(f"  [SAVE] Videos:       {video_dir}/")
    print()

    return rewards


# =========================================================================
# CLI ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL agent on CarRacing-v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/evaluate_agent.py --agent td3 --model models/td3_baseline/td3_actor_final.pth
  
        """
    )
    parser.add_argument("--agent", type=str, required=True,
                        choices=["dqn", "td3", "sac", "ppo"],
                        help="Agent type to evaluate")
    parser.add_argument("--model", type=str, default="auto",
                        help="Path to model .pth file, or 'auto' to find the latest")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--record", action="store_true", default=True,
                        help="Record videos (default: True)")
    parser.add_argument("--no-record", action="store_false", dest="record",
                        help="Disable video recording")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Show live render window")

    args = parser.parse_args()

    # Auto-find model if needed
    if args.model == "auto":
        args.model = auto_find_model(args.agent)
        print(f"  [AUTO] Found model: {args.model}")

    # Validate model path
    if not os.path.exists(args.model):
        print(f"  [ERROR] Model file not found: {args.model}")
        sys.exit(1)

    evaluate(
        agent_type=args.agent,
        model_path=args.model,
        num_episodes=args.episodes,
        seed=args.seed,
        record=args.record,
        render=args.render,
    )
