#!/usr/bin/env python3
"""Bar plot: object accuracy and rule inference accuracy by model."""

import argparse
import json
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


def rule_correct(rule_type_response: str, true_rule: str) -> float:
    txt = str(rule_type_response or "").lower()
    if "conjunctive" in txt and "disjunctive" not in txt:
        return 1.0 if true_rule == "conjunctive" else 0.0
    if "disjunctive" in txt and "conjunctive" not in txt:
        return 1.0 if true_rule == "disjunctive" else 0.0
    return 0.0


def load_trials(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-png", type=Path, default=Path("results/model_object_rule_bar.png"))
    parser.add_argument("--out-json", type=Path, default=Path("results/model_object_rule_bar_summary.json"))
    args = parser.parse_args()

    per_model_obj: Dict[str, List[float]] = {}
    per_model_rule: Dict[str, List[float]] = {}

    for results_file in sorted(args.results_root.glob("**/results_*.jsonl")):
        run_dir = results_file.parent
        overrides = parse_overrides(run_dir / ".hydra" / "overrides.yaml")
        model = overrides.get("agent.model", "unknown")
        # remove "ollama/" for x-axis readability
        model = model.replace("ollama/", "")

        trials = load_trials(results_file)
        if not trials:
            continue

        for t in trials:
            n_q = max(int(t.get("num_questions", 0)), 1)
            n_correct = int(t.get("num_correct", 0))
            obj_acc = n_correct / n_q
            rule_acc = rule_correct(t.get("rule_type_response", ""), t.get("true_rule", ""))
            per_model_obj.setdefault(model, []).append(obj_acc)
            per_model_rule.setdefault(model, []).append(rule_acc)

    if not per_model_obj:
        raise SystemExit(f"No results found under {args.results_root}")

    models = sorted(per_model_obj.keys())
    obj_means = [float(np.mean(per_model_obj[m])) for m in models]
    rule_means = [float(np.mean(per_model_rule.get(m, [0.0]))) for m in models]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, obj_means, width, label="Object accuracy (mean correct/4)")
    ax.bar(x + width / 2, rule_means, width, label="Rule inference accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Object and Rule Accuracy by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        m: {
            "object_accuracy_mean": obj_means[i],
            "rule_accuracy_mean": rule_means[i],
            "n_trials": len(per_model_obj[m]),
        }
        for i, m in enumerate(models)
    }
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved plot: {args.out_png}")
    print(f"Saved summary: {args.out_json}")


if __name__ == "__main__":
    main()
