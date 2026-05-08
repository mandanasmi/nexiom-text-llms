#!/usr/bin/env python3
"""Visualize Blicket results across agents/models from Hydra run folders."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _load_overrides(path: Path) -> Dict[str, str]:
    """Parse Hydra overrides.yaml lines like '- key=value'."""
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        kv = line[2:]
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _load_hydra_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_results(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rule_correct(rule_type_response: str, true_rule: str) -> int:
    if not rule_type_response:
        return 0
    r = str(rule_type_response).lower()
    if "conjunctive" in r and "disjunctive" not in r:
        return int(true_rule == "conjunctive")
    if "disjunctive" in r and "conjunctive" not in r:
        return int(true_rule == "disjunctive")
    return 0


def collect_runs(results_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for results_path in sorted(results_root.glob("**/results_*.jsonl")):
        run_dir = results_path.parent
        overrides = _load_overrides(run_dir / ".hydra" / "overrides.yaml")
        cfg = _load_hydra_config(run_dir / ".hydra" / "config.yaml")
        trials = _load_results(results_path)
        if not trials:
            continue

        agent_name = overrides.get("agent", str(cfg.get("agent", {}).get("_target_", "unknown")))
        model_name = overrides.get("agent.model", str(cfg.get("agent", {}).get("model", "unknown")))
        label = f"{agent_name}\n{model_name}"

        qa_success = []
        rule_acc = []
        joint_success = []
        steps = []
        api_errors = []
        for t in trials:
            n_q = max(int(t.get("num_questions", 0)), 1)
            n_correct = int(t.get("num_correct", 0))
            # Strict metric: success only if all object Q&A are correct.
            qa_ok = 1.0 if n_correct == n_q else 0.0
            rule_ok = float(_rule_correct(t.get("rule_type_response", ""), t.get("true_rule", "")))
            qa_success.append(qa_ok)
            rule_acc.append(rule_ok)
            # Joint metric requested: both object Q&A and hypothesis/rule correct.
            joint_success.append(1.0 if (qa_ok == 1.0 and rule_ok == 1.0) else 0.0)
            steps.append(float(t.get("num_steps", 0)))
            api_errors.append(float(t.get("num_traj_api_errors", 0)) + float(t.get("num_ans_api_errors", 0)))

        rows.append(
            {
                "run_dir": str(run_dir),
                "label": label,
                "agent": agent_name,
                "model": model_name,
                "qa_acc_mean": float(np.mean(qa_success)),
                "rule_acc_mean": float(np.mean(rule_acc)),
                "joint_acc_mean": float(np.mean(joint_success)),
                "steps_mean": float(np.mean(steps)),
                "api_errors_mean": float(np.mean(api_errors)),
                "n_trials": len(trials),
            }
        )
    return rows


def aggregate_by_key(rows: List[Dict], key_field: str) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for r in rows:
        key_value = str(r.get(key_field, "unknown"))
        grouped.setdefault(key_value, []).append(r)

    agg: List[Dict] = []
    for key_value, items in grouped.items():
        agg.append(
            {
                "label": key_value,
                "qa_acc_mean": float(np.mean([x["qa_acc_mean"] for x in items])),
                "rule_acc_mean": float(np.mean([x["rule_acc_mean"] for x in items])),
                "joint_acc_mean": float(np.mean([x["joint_acc_mean"] for x in items])),
                "steps_mean": float(np.mean([x["steps_mean"] for x in items])),
                "api_errors_mean": float(np.mean([x["api_errors_mean"] for x in items])),
                "n_runs": len(items),
                "n_trials_total": int(np.sum([x["n_trials"] for x in items])),
            }
        )
    agg.sort(key=lambda x: x["label"])
    return agg


def plot(agg: List[Dict], out_png: Path):
    labels = [x["label"] for x in agg]
    qa = [x["qa_acc_mean"] for x in agg]
    rule = [x["rule_acc_mean"] for x in agg]
    joint = [x["joint_acc_mean"] for x in agg]
    steps = [x["steps_mean"] for x in agg]

    x = np.arange(len(labels))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].bar(x, qa)
    axs[0, 0].set_title("Strict Q&A success rate (all object answers correct)")
    axs[0, 0].set_ylim(0, 1.0)
    axs[0, 0].set_ylabel("accuracy")

    axs[0, 1].bar(x, joint)
    axs[0, 1].set_title("Joint success (all objects + rule hypothesis)")
    axs[0, 1].set_ylim(0, 1.0)
    axs[0, 1].set_ylabel("accuracy")

    axs[1, 0].bar(x, steps)
    axs[1, 0].set_title("Mean exploration steps")
    axs[1, 0].set_ylabel("steps")

    axs[1, 1].bar(x, rule)
    axs[1, 1].set_title("Mean rule-type accuracy")
    axs[1, 1].set_ylim(0, 1.0)
    axs[1, 1].set_ylabel("accuracy")

    for ax in axs.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Blicket Results Across Agents/Models", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_summary(agg: List[Dict], out_json: Path):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-png", type=Path, default=Path("results/results_across_agents.png"))
    parser.add_argument("--out-json", type=Path, default=Path("results/results_across_agents_summary.json"))
    parser.add_argument(
        "--group-by",
        choices=["label", "model", "agent"],
        default="label",
        help="How to aggregate runs before plotting.",
    )
    args = parser.parse_args()

    rows = collect_runs(args.results_root)
    if not rows:
        raise SystemExit(f"No results files found under {args.results_root}")
    agg = aggregate_by_key(rows, args.group_by)
    plot(agg, args.out_png)
    save_summary(agg, args.out_json)
    print(f"Saved plot: {args.out_png}")
    print(f"Saved summary: {args.out_json}")


if __name__ == "__main__":
    main()
