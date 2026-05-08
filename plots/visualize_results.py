#!/usr/bin/env python3
"""
Visualize blicket trial results from results.jsonl files.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: Path) -> list[dict]:
    """Load trial results from a JSONL file."""
    trials = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def parse_rule_response(response: str, true_rule: str) -> bool:
    """Check if rule type response matches true rule."""
    if not response:
        return False
    resp_lower = response.lower().strip()
    if "conjunctive" in resp_lower and "disjunctive" not in resp_lower:
        return true_rule == "conjunctive"
    if "disjunctive" in resp_lower and "conjunctive" not in resp_lower:
        return true_rule == "disjunctive"
    return False


def visualize(trials: list[dict], output_path: Path):
    """Create visualization of trial results."""
    if not trials:
        print("No trials to visualize.")
        return

    n_trials = len(trials)
    qa_accuracy = [t["num_correct"] / t["num_questions"] if t["num_questions"] > 0 else 0 for t in trials]
    rule_correct = [parse_rule_response(t.get("rule_type_response", ""), t["true_rule"]) for t in trials]
    num_steps = [t["num_steps"] for t in trials]
    turn_machine_on = [t["turn_machine_on"] for t in trials]
    api_errors = [t.get("num_traj_api_errors", 0) + t.get("num_ans_api_errors", 0) for t in trials]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Q&A accuracy distribution
    ax = axes[0, 0]
    ax.hist(qa_accuracy, bins=np.linspace(0, 1.01, 6), edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(qa_accuracy), color="red", linestyle="--", label=f"Mean: {np.mean(qa_accuracy):.2%}")
    ax.set_xlabel("Q&A Accuracy (correct / total questions)")
    ax.set_ylabel("Number of trials")
    ax.set_title("Blicket Q&A Accuracy Distribution")
    ax.legend()

    # 2. Rule type correctness
    ax = axes[0, 1]
    rule_correct_pct = np.mean(rule_correct) * 100
    ax.bar(["Correct", "Incorrect"], [np.sum(rule_correct), n_trials - np.sum(rule_correct)], color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Number of trials")
    ax.set_title(f"Rule Type Identification ({rule_correct_pct:.1f}% correct)")

    # 3. Steps per trial
    ax = axes[1, 0]
    ax.hist(num_steps, bins=range(0, max(num_steps) + 2), edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(num_steps), color="red", linestyle="--", label=f"Mean: {np.mean(num_steps):.1f}")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Number of trials")
    ax.set_title("Exploration Steps per Trial")
    ax.legend()

    # 4. Cumulative Q&A accuracy over trials
    ax = axes[1, 1]
    cum_correct = np.cumsum([t["num_correct"] for t in trials])
    cum_total = np.cumsum([t["num_questions"] for t in trials])
    cum_accuracy = np.where(cum_total > 0, cum_correct / cum_total, 0)
    ax.plot(range(1, n_trials + 1), cum_accuracy, "b-", linewidth=2)
    ax.fill_between(range(1, n_trials + 1), cum_accuracy, alpha=0.3)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Cumulative Q&A accuracy")
    ax.set_title("Cumulative Q&A Accuracy Over Trials")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Blicket Task Results (n={n_trials} trials)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize blicket trial results")
    parser.add_argument("results", type=Path, help="Path to results.jsonl or results directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output figure path")
    args = parser.parse_args()

    if args.results.is_file():
        results_files = [args.results]
    else:
        results_files = list(args.results.glob("**/results*.jsonl"))

    if not results_files:
        print(f"No results files found in {args.results}")
        return

    # Use most recent if multiple
    results_path = max(results_files, key=lambda p: p.stat().st_mtime)
    trials = load_results(results_path)
    print(f"Loaded {len(trials)} trials from {results_path}")

    if args.output is None:
        args.output = results_path.parent / "results_visualization.png"
    else:
        args.output = Path(args.output)

    visualize(trials, args.output)


if __name__ == "__main__":
    main()
