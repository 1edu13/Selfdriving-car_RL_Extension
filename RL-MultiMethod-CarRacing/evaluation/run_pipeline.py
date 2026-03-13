"""
run_pipeline.py – Master orchestration script.

Runs the full evaluation pipeline for all trained models:
  1. Individual evaluation (evaluate_pro.py) for each available model.
  2. Comparative analysis (compare_models.py) across all methods.

Usage:
    python evaluation/run_pipeline.py [--models_root models/] [--n_episodes 10]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIRS = {
    "ppo": BASE_DIR / "models" / "ppo_baseline",
    "dqn": BASE_DIR / "models" / "dqn_results",
    "td3": BASE_DIR / "models" / "td3_results",
    "sac": BASE_DIR / "models" / "sac_results",
}

BEST_CHECKPOINT = {
    "ppo": "ppo_best.pth",
    "dqn": "dqn_best.pth",
    "td3": "td3_best.pth",
    "sac": "sac_best.pth",
}


def find_model(method: str) -> Optional[Path]:
    """Return path to the best checkpoint for *method*, or None if not found."""
    model_dir = MODEL_DIRS[method]
    best = model_dir / BEST_CHECKPOINT[method]
    if best.exists():
        return best
    # Fallback: pick the latest checkpoint in the directory
    checkpoints = sorted(model_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def run_evaluation(method: str, model_path: Path, args) -> bool:
    """Run evaluate_pro.py for a single method."""
    cmd = [
        sys.executable,
        str(BASE_DIR / "evaluation" / "evaluate_pro.py"),
        "--method", method,
        "--model_path", str(model_path),
        "--n_episodes", str(args.n_episodes),
        "--seed", str(args.seed),
        "--output_dir", str(BASE_DIR / "results" / "evaluation_results"),
    ]
    if args.save_video:
        cmd.append("--save_video")
    if args.cpu:
        cmd.append("--cpu")

    print(f"\n{'='*60}")
    print(f"Evaluating {method.upper()} | model: {model_path}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def run_comparison(args):
    """Run compare_models.py."""
    cmd = [
        sys.executable,
        str(BASE_DIR / "evaluation" / "compare_models.py"),
        "--results_dir", str(BASE_DIR / "results" / "evaluation_results"),
        "--output_dir", str(BASE_DIR / "results" / "comparison_analysis"),
    ]
    print(f"\n{'='*60}")
    print("Running comparative analysis…")
    print(f"{'='*60}")
    subprocess.run(cmd, check=False)


def main(args):
    evaluated = []
    skipped = []

    for method in args.methods:
        model_path = find_model(method)
        if model_path is None:
            print(f"[SKIP] No checkpoint found for {method.upper()}")
            skipped.append(method)
            continue
        success = run_evaluation(method, model_path, args)
        if success:
            evaluated.append(method)
        else:
            print(f"[WARN] Evaluation failed for {method.upper()}")

    if len(evaluated) >= 2:
        run_comparison(args)
    else:
        print(
            f"\n[INFO] Need at least 2 evaluated models for comparison "
            f"(got {len(evaluated)}). Skipping comparative analysis."
        )

    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  Evaluated : {evaluated}")
    print(f"  Skipped   : {skipped}")
    print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser(description="Full evaluation pipeline")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["ppo", "dqn", "td3", "sac"],
        choices=["ppo", "dqn", "td3", "sac"],
    )
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
