"""
compare_models.py – Multi-model comparative analysis and plotting.

Loads evaluation JSON results produced by evaluate_pro.py and generates
summary statistics and comparison charts.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class ComparativeAnalysis:
    """
    Load per-method evaluation results and produce comparison plots.

    Args:
        results_dir: Directory containing evaluation JSON files.
        output_dir: Directory where charts are saved.
    """

    METHOD_COLORS = {
        "ppo": "#4C72B0",
        "dqn": "#DD8452",
        "td3": "#55A868",
        "sac": "#C44E52",
    }

    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Dict] = {}

    def load(self):
        """Load all JSON result files from *results_dir*."""
        json_files = list(self.results_dir.glob("*_evaluation_*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No evaluation JSON files found in {self.results_dir}"
            )
        for path in json_files:
            with open(path) as f:
                result = json.load(f)
            method = result["method"]
            # Keep the most recent result per method
            if method not in self.data or path.stat().st_mtime > self.data[method].get(
                "_mtime", 0
            ):
                self.data[method] = result
                self.data[method]["_mtime"] = path.stat().st_mtime
        print(f"Loaded results for: {sorted(self.data.keys())}")

    def summary_table(self) -> str:
        """Return a formatted text summary table."""
        rows = [f"{'Method':<8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}"]
        rows.append("-" * 52)
        for method, res in sorted(self.data.items()):
            rows.append(
                f"{method.upper():<8} "
                f"{res['mean_reward']:>10.2f} "
                f"{res['std_reward']:>10.2f} "
                f"{res['min_reward']:>10.2f} "
                f"{res['max_reward']:>10.2f}"
            )
        return "\n".join(rows)

    def plot_bar_comparison(self):
        """Bar chart of mean ± std rewards per method."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed – skipping bar chart.")
            return

        methods = sorted(self.data.keys())
        means = [self.data[m]["mean_reward"] for m in methods]
        stds = [self.data[m]["std_reward"] for m in methods]
        colors = [self.METHOD_COLORS.get(m, "#888888") for m in methods]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            [m.upper() for m in methods],
            means,
            yerr=stds,
            capsize=6,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_title("Mean Reward Comparison (±1 std)", fontsize=14)
        ax.set_ylabel("Episode Reward")
        ax.set_xlabel("Method")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = self.output_dir / "reward_bar_comparison.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Bar chart saved to {out_path}")

    def plot_reward_distributions(self):
        """Box plot of reward distributions per method."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed – skipping box plot.")
            return

        methods = sorted(self.data.keys())
        reward_lists = [self.data[m].get("rewards", []) for m in methods]

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(
            reward_lists,
            labels=[m.upper() for m in methods],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(self.METHOD_COLORS.get(method, "#888888"))

        ax.set_title("Reward Distribution per Method", fontsize=14)
        ax.set_ylabel("Episode Reward")
        ax.set_xlabel("Method")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = self.output_dir / "reward_distributions.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Box plot saved to {out_path}")

    def run(self):
        """Full analysis pipeline."""
        self.load()
        print("\n" + self.summary_table() + "\n")
        self.plot_bar_comparison()
        self.plot_reward_distributions()

        # Save aggregated summary JSON
        summary = {
            m: {k: v for k, v in res.items() if not k.startswith("_")}
            for m, res in self.data.items()
        }
        out_path = self.output_dir / "comparison_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Comparison summary saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare RL method evaluation results")
    p.add_argument(
        "--results_dir",
        default="results/evaluation_results",
        help="Directory with evaluation JSON files",
    )
    p.add_argument(
        "--output_dir",
        default="results/comparison_analysis",
        help="Directory to save charts and summary",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analysis = ComparativeAnalysis(args.results_dir, args.output_dir)
    analysis.run()
