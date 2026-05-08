"""Stacked-bar outcome breakdown (both wrong / rule✓ obj✗ / rule✗ obj✓ / both✓).

Outputs:
  - results/human_data/outcomes_human_active.png  (humans, 2 bars per rule type)
  - results/eig_analysis/outcomes_llm_recent_all_models.png  (LLMs, one bar per
    model in each of two subplots: Conjunctive | Disjunctive)
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

REPO = "/network/scratch/s/samieima/projects/blicket-text-llm"

RESULTS_BY_MODEL = {
    "o4-mini": [
        f"{REPO}/results/20260428_232945_9398595",
        f"{REPO}/results/20260428_232945_9398596",
        f"{REPO}/results/20260428_232946_9398597",
        f"{REPO}/results/20260428_232945_9398598",
    ],
    "gpt-5": [
        f"{REPO}/results/20260428_221343_9398102",
        f"{REPO}/results/20260428_221342_9398103",
        f"{REPO}/results/20260428_221343_9398104",
        f"{REPO}/results/20260428_221342_9398105",
    ],
    "gpt-5-mini": [
        f"{REPO}/results/20260428_233913_9398665",
        f"{REPO}/results/20260428_233912_9398666",
        f"{REPO}/results/20260428_233912_9398667",
        f"{REPO}/results/20260428_233913_9398668",
    ],
    "gemini-2.5-flash": [
        f"{REPO}/results/20260428_222010_9398172",
        f"{REPO}/results/20260428_222009_9398173",
        f"{REPO}/results/20260428_222009_9398174",
        f"{REPO}/results/20260428_222009_9398175",
    ],
    "deepseek-reasoner": [
        f"{REPO}/results/20260428_231811_9398537",
        f"{REPO}/results/20260428_231812_9398538",
        f"{REPO}/results/20260428_231812_9398539",
        f"{REPO}/results/20260428_231811_9398540",
    ],
    "deepseek-chat": [
        f"{REPO}/results/20260429_094520_9403286",
        f"{REPO}/results/20260429_094519_9403287",
        f"{REPO}/results/20260429_094520_9403288",
        f"{REPO}/results/20260429_094521_9403289",
    ],
}
HUMAN_DATA_PATH = f"{REPO}/results/human_data/human_data_active_explore.json"
RULES = ("conjunctive", "disjunctive")

OUTCOMES = [
    ("both wrong",                    "#d62728"),
    ("rule correct & objects wrong",  "#ff7f0e"),
    ("rule wrong & objects correct",  "#1f77b4"),
    ("both correct",                  "#2ca02c"),
]
LABELS = [o[0] for o in OUTCOMES]
COLORS = [o[1] for o in OUTCOMES]


def classify(rule_ok: bool, obj_ok: bool) -> int:
    if not rule_ok and not obj_ok:
        return 0
    if rule_ok and not obj_ok:
        return 1
    if not rule_ok and obj_ok:
        return 2
    return 3


def load_llm_trials(results_dirs):
    for results_dir in results_dirs:
        if not os.path.isdir(results_dir):
            continue
        for fname in os.listdir(results_dir):
            if fname.startswith("results") and fname.endswith(".jsonl"):
                with open(os.path.join(results_dir, fname)) as f:
                    for line in f:
                        try:
                            yield json.loads(line.replace("NaN", "null"))
                        except json.JSONDecodeError:
                            continue


def collect_llm_counts():
    """Return dict[model][rule] -> np.array(4) of outcome counts."""
    counts = defaultdict(lambda: {r: np.zeros(4, dtype=int) for r in RULES})
    for model, dirs in RESULTS_BY_MODEL.items():
        for t in load_llm_trials(dirs):
            rule = str(t.get("true_rule", "")).strip().lower()
            if rule not in RULES:
                continue
            obj_ok = bool(t.get("all_nexioms_correct"))
            rule_ok = bool(t.get("rule_type_correct"))
            counts[model][rule][classify(rule_ok, obj_ok)] += 1
    return counts


def collect_human_counts():
    counts = {r: np.zeros(4, dtype=int) for r in RULES}
    top = {r: np.zeros(4, dtype=int) for r in RULES}
    if not os.path.exists(HUMAN_DATA_PATH):
        return counts, top
    with open(HUMAN_DATA_PATH) as f:
        data = json.load(f)
    for sess in data.values():
        if not isinstance(sess, dict):
            continue
        mg = sess.get("main_game") or {}
        rule = str(mg.get("rule", "")).strip().lower()
        if rule not in RULES:
            continue
        true_b = set(mg.get("true_blicket_indices") or [])
        chosen = set(mg.get("user_chosen_blickets") or [])
        obj_ok = chosen == true_b
        rt = str(mg.get("rule_type", "")).strip().lower()
        rule_ok = rt.startswith(rule)
        counts[rule][classify(rule_ok, obj_ok)] += 1
        if rule_ok and obj_ok:
            top[rule][3] += 1
    return counts, top


def to_pct(c):
    n = c.sum()
    return c.astype(float) / n * 100.0 if n > 0 else c.astype(float)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def stacked_bar(ax, x, pct, width, min_draw, label_first=False):
    draw = pct.copy()
    nz = draw > 0
    draw[nz] = np.maximum(draw[nz], min_draw)
    if draw.sum() > 0:
        draw = draw / draw.sum() * 100.0
    bottom = 0.0
    for j, (lbl, col) in enumerate(zip(LABELS, COLORS)):
        h = draw[j]
        if h <= 0:
            continue
        ax.bar(x, h, width, bottom=bottom, color=col,
               edgecolor="white", linewidth=1.0,
               label=lbl if label_first else None)
        ax.text(x, bottom + h / 2, f"{pct[j]:.0f}%",
                ha="center", va="center", fontsize=15,
                color="white", fontweight="bold")
        bottom += h


def plot_human(human_counts, out_path, min_draw=14.0):
    fig, ax = plt.subplots(figsize=(7, 6))
    xs = np.arange(len(RULES))
    for i, rt in enumerate(RULES):
        pct = to_pct(human_counts[rt])
        n = int(human_counts[rt].sum())
        print(f"human / {rt}: n={n} -> " + ", ".join(f"{l}={p:.1f}%" for l, p in zip(LABELS, pct)))
        stacked_bar(ax, xs[i], pct, 0.55, min_draw, label_first=(i == 0))
    ax.set_xticks(xs)
    ax.set_xticklabels(["Conjunctive", "Disjunctive"])
    ax.set_ylabel("Percentage of participants (%)")
    ax.set_ylim(0, 100)
    _despine(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def plot_llm(llm_counts, out_path, min_draw=10.0, human_counts=None, human_top=None):
    models = list(llm_counts.keys())

    # Per-rule sort by green ("both correct") happens below.
    width = 0.7
    base, ext = os.path.splitext(out_path)
    extras = []
    if human_counts: extras.append("average human adults")
    if human_top:    extras.append("top human")
    for rule in RULES:
        def get_c(m, r=rule):
            if m == "average human adults": return human_counts[r]
            if m == "top human":            return human_top[r]
            return llm_counts[m][r]
        def green(m):
            c = get_c(m); n = c.sum()
            return c[3] / n if n else 0.0
        tiebreak = {"deepseek-reasoner": 0, "gpt-5": 1, "top human": 2}
        groups = sorted(list(models) + extras,
                        key=lambda m: (green(m), tiebreak.get(m, -1)))
        xs = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(groups) + 3), 6))
        for i, m in enumerate(groups):
            c = get_c(m)
            pct = to_pct(c)
            n = int(c.sum())
            print(f"{m} / {rule}: n={n} -> " +
                  ", ".join(f"{l}={p:.1f}%" for l, p in zip(LABELS, pct)))
            stacked_bar(ax, xs[i], pct, width, min_draw, label_first=False)
        ax.set_xticks(xs)
        ax.set_xticklabels(groups, rotation=30, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage of trials (%)")
        _despine(ax)
        proxies = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
        fig.legend(proxies, LABELS, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        out_p = f"{base}_{rule}{ext}"
        plt.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.savefig(os.path.splitext(out_p)[0] + ".pdf", bbox_inches="tight")
        plt.savefig(os.path.splitext(out_p)[0] + ".svg", bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-human",
                    default=f"{REPO}/results/human_data/outcomes_human_active.png")
    ap.add_argument("--out-llm",
                    default=f"{REPO}/results/eig_analysis/outcomes_llm_recent_all_models.png")
    ap.add_argument("--skip-human", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    human_counts, human_top = collect_human_counts()
    if not args.skip_human:
        plot_human(human_counts, args.out_human)
    if not args.skip_llm:
        plot_llm(collect_llm_counts(), args.out_llm,
                 human_counts=human_counts, human_top=human_top)


if __name__ == "__main__":
    main()
