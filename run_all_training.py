"""
CARRACING-V2 -- FULL TRAINING PIPELINE
=======================================

Master orchestration script that trains all 4 RL agents sequentially
with a single command. Each model is run as an independent subprocess
to ensure complete GPU memory cleanup between training sessions.

Training Order: DQN (2M) -> TD3 (1.5M) -> SAC (1.5M) -> PPO (3M)
Total: 8M timesteps

Optimized for: NVIDIA RTX 3050 (4GB VRAM) | AMD Ryzen 7 4800H | 32GB RAM

Usage:
    python run_all_training.pyf
"""

import subprocess
import sys
import time
import os
from datetime import datetime, timedelta

# =====================================================================
# CONFIGURATION -- Comment out any model to skip it
# =====================================================================
MODELS_TO_TRAIN = [
    {
        "name": "DQN",
        "script": "training/train_dqn.py",
        "timesteps": "2,000,000",
        "description": "Deep Q-Network (discrete actions)",
    },
    {
        "name": "TD3",
        "script": "training/train_td3.py",
        "timesteps": "1,500,000",
        "description": "Twin Delayed DDPG (continuous)",
    },
    {
        "name": "SAC",
        "script": "training/train_sac.py",
        "timesteps": "1,500,000",
        "description": "Soft Actor-Critic (max entropy)",
    },
    {
        "name": "PPO",
        "script": "training/train_ppo.py",
        "timesteps": "3,000,000",
        "description": "Proximal Policy Optimization (on-policy)",
    },
]


def get_hardware_summary():
    """Detects and returns hardware information for the training header."""
    try:
        import torch
        info = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            info["gpu_vram"] = f"{vram_bytes / (1024**3):.1f} GB"
            info["cuda_version"] = torch.version.cuda
        else:
            info["gpu_name"] = "N/A (CPU mode)"
            info["gpu_vram"] = "N/A"
            info["cuda_version"] = "N/A"
        return info
    except ImportError:
        return {
            "torch": "NOT INSTALLED",
            "cuda_available": False,
            "gpu_name": "UNKNOWN",
            "gpu_vram": "UNKNOWN",
            "cuda_version": "UNKNOWN",
        }


def format_duration(seconds):
    """Converts seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def print_header(hw_info):
    """Prints a formatted header with hardware info and training plan."""
    total_steps = sum(
        int(m["timesteps"].replace(",", "")) for m in MODELS_TO_TRAIN
    )

    print()
    print("=" * 64)
    print("   CARRACING-V2 -- FULL TRAINING PIPELINE".center(64))
    print("=" * 64)
    print(f"  GPU:   {hw_info['gpu_name']}")
    print(f"  VRAM:  {hw_info['gpu_vram']}  |  CUDA: {hw_info['cuda_version']}  |  PyTorch: {hw_info['torch']}")
    print(f"  Total: {total_steps:,} timesteps across {len(MODELS_TO_TRAIN)} models")
    print("-" * 64)

    for i, model in enumerate(MODELS_TO_TRAIN, 1):
        line = f"  [{i}/{len(MODELS_TO_TRAIN)}] {model['name']:4s} -- {model['timesteps']:>11s} steps  ({model['description']})"
        # Truncate if too long
        if len(line) > 64:
            line = line[:61] + "..."
        print(line)

    print("-" * 64)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Started at: {timestamp}")
    print("=" * 64)
    print()


def train_model(index, total, model_config, project_root):
    """
    Trains a single model as a subprocess.
    Returns a dict with timing info and success status.
    """
    name = model_config["name"]
    script = model_config["script"]
    script_path = os.path.join(project_root, script)

    if not os.path.exists(script_path):
        print(f"  [ERROR] Script not found: {script_path}")
        return {"name": name, "success": False, "duration": 0, "error": "Script not found"}

    separator = "-" * 64
    print(separator)
    print(f"  [START] [{index}/{total}] Starting {name} Training...")
    print(f"     Script: {script}")
    print(f"     Steps:  {model_config['timesteps']}")
    print(separator)

    start_time = time.time()

    try:
        # Run training script as a subprocess
        # This ensures complete GPU memory cleanup between models
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            # Stream output directly to console in real-time
            stdout=None,  # Inherit parent stdout (prints to console)
            stderr=subprocess.STDOUT,
            timeout=None,  # No timeout -- training can take hours
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n  [OK] {name} completed successfully in {format_duration(elapsed)}")
            return {"name": name, "success": True, "duration": elapsed, "error": None}
        else:
            print(f"\n  [FAIL] {name} failed with return code {result.returncode}")
            return {"name": name, "success": False, "duration": elapsed, "error": f"Return code: {result.returncode}"}

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n  [WARN] {name} interrupted by user after {format_duration(elapsed)}")
        raise  # Re-raise to stop the entire pipeline
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  [FAIL] {name} crashed: {str(e)}")
        return {"name": name, "success": False, "duration": elapsed, "error": str(e)}


def print_summary(results, total_start_time):
    """Prints a final summary table of all training results."""
    total_elapsed = time.time() - total_start_time

    print()
    print("=" * 64)
    print("   TRAINING PIPELINE -- FINAL SUMMARY".center(64))
    print("=" * 64)

    for r in results:
        status = "[OK]  " if r["success"] else "[FAIL]"
        duration_str = format_duration(r["duration"]) if r["duration"] > 0 else "N/A"
        line = f"  {status} {r['name']:4s} -- {duration_str}"
        if r["error"]:
            line += f"  (Error: {r['error'][:30]})"
        print(line)

    print("-" * 64)

    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"  Total time:  {format_duration(total_elapsed)}")
    print(f"  Successful:  {successful}/{len(results)}")
    if failed > 0:
        print(f"  Failed:      {failed}/{len(results)}")
    print(f"  Finished at: {timestamp}")
    print("=" * 64)

    if successful == len(results):
        print("\n  All models trained successfully! Check models/ for checkpoints.")
    else:
        print(f"\n  WARNING: {failed} model(s) failed. Check the output above for details.")
    print()


def main():
    # Determine project root (directory where this script lives)
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Detect hardware
    hw_info = get_hardware_summary()

    # Safety check
    if not hw_info["cuda_available"]:
        print("\n[WARNING] No CUDA GPU detected! Training will run on CPU (much slower).")
        print("   Make sure you have the CUDA version of PyTorch installed:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        response = input("\n   Continue on CPU? [y/N]: ").strip().lower()
        if response != "y":
            print("   Aborted.")
            return

    # Print beautiful header
    print_header(hw_info)

    # Create output directories
    for model in MODELS_TO_TRAIN:
        run_name = model["name"].lower() + "_baseline"
        os.makedirs(f"models/{run_name}", exist_ok=True)

    # Train all models sequentially
    results = []
    total_start = time.time()

    try:
        for i, model in enumerate(MODELS_TO_TRAIN, 1):
            result = train_model(i, len(MODELS_TO_TRAIN), model, project_root)
            results.append(result)

            # Brief pause between models to let GPU cool down
            if i < len(MODELS_TO_TRAIN):
                print("\n  [WAIT] Pausing 10 seconds for GPU memory cleanup...\n")
                time.sleep(10)

    except KeyboardInterrupt:
        print("\n\n  [STOP] Pipeline interrupted by user (Ctrl+C).")
        print("     Partial results will be shown below.")

    # Print final summary
    if results:
        print_summary(results, total_start)


if __name__ == "__main__":
    main()
