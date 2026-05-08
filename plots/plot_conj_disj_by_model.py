#!/usr/bin/env python3
"""Plot conjunctive vs disjunctive accuracies for each model.

Metrics:
- strict object identification accuracy (all object questions correct)
- rule type accuracy
- joint accuracy (both strict object and rule type correct)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_overrides(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("- ") or "=" not in line:
            continue
        k, v = line[2:].split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_trials(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def rule_correct(rule_type_response: str, true_rule: str) -> float:
    txt = str(rule_type_response or "").lower()
    if "conjunctive" in txt and "disjunctive" not in txt:
        return 1.0 if true_rule == "conjunctive" else 0.0
    if "disjunctive" in txt and "conjunctive" not in txt:
        return 1.0 if true_rule == "disjunctive" else 0.0
    return 0.0


def mean_or_zero(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--model-regex",
        type=str,
        default=r"^(qwen|gemma)",
        help="Only include models whose names match this regex.",
    )
    parser.add_argument(
        "--require-num-trials",
        type=int,
        default=32,
        help="Only include runs whose override num_trials equals this value.",
    )
    parser.add_argument(
        "--exp-tag",
        type=str,
        default="",
        help="Only include runs with exp_tag=<value> in overrides.yaml.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("results/model_conj_disj_object_rule_joint_accuracy.png"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/model_conj_disj_object_rule_joint_accuracy.json"),
    )
    parser.add_argument("--wandb", action="store_true", help="Log summary and plot to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="blicket-text-llm")
    parser.add_argument("--wandb-run-name", type=str, default="")
    args = parser.parse_args()

    obj_acc: Dict[str, Dict[str, List[float]]] = {}
    rule_acc: Dict[str, Dict[str, List[float]]] = {}
    joint_acc: Dict[str, Dict[str, List[float]]] = {}

    for results_file in sorted(args.results_root.glob("**/results_*.jsonl")):
        run_dir = results_file.parent
        overrides = parse_overrides(run_dir / ".hydra" / "overrides.yaml")
        if str(overrides.get("num_trials", "")) != str(args.require_num_trials):
            continue
        if args.exp_tag and overrides.get("exp_tag", "") != args.exp_tag:
            continue

        model = overrides.get("agent.model", "unknown").replace("ollama/", "")
        if not re.search(args.model_regex, model):
            continue

        trials = load_trials(results_file)
        for t in trials:
            true_rule = str(t.get("true_rule", ""))
            if true_rule not in ("conjunctive", "disjunctive"):
                continue

            n_q = max(int(t.get("num_questions", 0)), 1)
            n_correct = int(t.get("num_correct", 0))
            obj = 1.0 if n_correct == n_q else 0.0

            racc = rule_correct(t.get("rule_type_response", ""), true_rule)
            jacc = 1.0 if (obj == 1.0 and racc == 1.0) else 0.0

            obj_acc.setdefault(model, {"conjunctive": [], "disjunctive": []})[true_rule].append(obj)
            rule_acc.setdefault(model, {"conjunctive": [], "disjunctive": []})[true_rule].append(racc)
            joint_acc.setdefault(model, {"conjunctive": [], "disjunctive": []})[true_rule].append(jacc)

    if not obj_acc:
        raise SystemExit(f"No matching trials found under {args.results_root}")

    models = sorted(obj_acc.keys())
    x = np.arange(len(models))
    width = 0.35

    obj_conj = [mean_or_zero(obj_acc[m]["conjunctive"]) for m in models]
    obj_disj = [mean_or_zero(obj_acc[m]["disjunctive"]) for m in models]
    rule_conj = [mean_or_zero(rule_acc[m]["conjunctive"]) for m in models]
    rule_disj = [mean_or_zero(rule_acc[m]["disjunctive"]) for m in models]
    joint_conj = [mean_or_zero(joint_acc[m]["conjunctive"]) for m in models]
    joint_disj = [mean_or_zero(joint_acc[m]["disjunctive"]) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

    axes[0].bar(x - width / 2, obj_conj, width, label="Conjunctive")
    axes[0].bar(x + width / 2, obj_disj, width, label="Disjunctive")
    axes[0].set_title("Object ID accuracy (strict)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - width / 2, rule_conj, width, label="Conjunctive")
    axes[1].bar(x + width / 2, rule_disj, width, label="Disjunctive")
    axes[1].set_title("Rule type accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    axes[2].bar(x - width / 2, joint_conj, width, label="Conjunctive")
    axes[2].bar(x + width / 2, joint_disj, width, label="Disjunctive")
    axes[2].set_title("Joint (object + rule) accuracy")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20, ha="right")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend()

    title = f"Model Accuracy (n={args.require_num_trials} per rule): Conjunctive vs Disjunctive"
    if args.exp_tag:
        title += f" | exp_tag={args.exp_tag}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160, bbox_inches="tight")
    print(f"Saved plot: {args.out_png}")

    summary = {}
    for i, m in enumerate(models):
        summary[m] = {
            "conjunctive": {
                "object_accuracy_strict": obj_conj[i],
                "rule_accuracy": rule_conj[i],
                "joint_accuracy": joint_conj[i],
                "n_trials": len(obj_acc[m]["conjunctive"]),
            },
            "disjunctive": {
                "object_accuracy_strict": obj_disj[i],
                "rule_accuracy": rule_disj[i],
                "joint_accuracy": joint_disj[i],
                "n_trials": len(obj_acc[m]["disjunctive"]),
            },
        }
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary: {args.out_json}")

    if args.wandb:
        try:
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                name=(args.wandb_run_name or None),
                config={
                    "results_root": str(args.results_root),
                    "model_regex": args.model_regex,
                    "require_num_trials": args.require_num_trials,
                    "exp_tag": args.exp_tag,
                },
            )
            table = wandb.Table(
                columns=[
                    "model",
                    "rule",
                    "object_accuracy_strict",
                    "rule_accuracy",
                    "joint_accuracy",
                    "n_trials",
                ]
            )
            for i, m in enumerate(models):
                table.add_data(
                    m,
                    "conjunctive",
                    obj_conj[i],
                    rule_conj[i],
                    joint_conj[i],
                    len(obj_acc[m]["conjunctive"]),
                )
                table.add_data(
                    m,
                    "disjunctive",
                    obj_disj[i],
                    rule_disj[i],
                    joint_disj[i],
                    len(obj_acc[m]["disjunctive"]),
                )
            wandb.log(
                {
                    "model_conj_disj_metrics": table,
                    "model_conj_disj_plot": wandb.Image(str(args.out_png)),
                }
            )
            run.finish()
            print("Logged metrics to Weights & Biases.")
        except Exception as e:
            print(f"W&B logging skipped due to error: {e}")


if __name__ == "__main__":
    main()
