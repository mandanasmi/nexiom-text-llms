"""Information-gain curves for the recent runs (same model set as
plot_accuracy_recent_all_models.py / plot_hypotheses_remaining_recent_all_models.py).
Conjunctive and disjunctive trials are plotted in separate figures.

Usage:
    python plot_info_gain_recent_all_models.py --metric cum
    python plot_info_gain_recent_all_models.py --metric step

Metrics:
    cum  -- cumulative info gain (bits) = log2(H0) - log2(max(remaining, 1))
    step -- per-step info gain (bits/test) = log2(remaining[t-1]) - log2(remaining[t])
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_hypotheses_remaining_recent_all_models import (
    RUNS_BY_MODEL, MODEL_COLOR, HUMAN_COLOR, BEST_HUMAN_COLOR, BASELINE_COLOR,
    OUT_DIR, H0, collect, load_human_tests_by_rule, simulate_random_baseline,
    trial_remaining_curve, pick_best_humans,
)

LOG2H0 = float(np.log2(H0))


def cumulative_info_gain(remain_mat: np.ndarray) -> np.ndarray:
    return LOG2H0 - np.log2(np.clip(remain_mat, 1, None))


def per_step_info_gain(remain_mat: np.ndarray) -> np.ndarray:
    log_h = np.log2(np.clip(remain_mat, 1, None))
    return log_h[:, :-1] - log_h[:, 1:]


def remain_matrix(per_trial, rule, max_tests):
    mats = [trial_remaining_curve(t, max_tests)
            for _, _, tr, t in per_trial if tr == rule]
    if not mats:
        return None
    return np.stack(mats, axis=0)


def human_remain_matrix(entries, max_tests):
    if not entries:
        return None
    return np.stack([trial_remaining_curve(t, max_tests) for t, _, _ in entries], axis=0)


METRICS = {
    "cum": {
        "fn": cumulative_info_gain,
        "x_offset": 0,
        "ylabel": "cumulative info gain (bits)",
        "axhline": LOG2H0,
        "ylim": (0, LOG2H0 + 0.3),
        "xlim_start": 0,
        "legend_loc": "lower right",
        "out_template": "info_gain_cum_recent_all_models_{rule}.png",
    },
    "step": {
        "fn": per_step_info_gain,
        "x_offset": 1,
        "ylabel": "info gain per test (bits)",
        "axhline": 0,
        "ylim": None,
        "xlim_start": 1,
        "legend_loc": "upper right",
        "out_template": "info_gain_per_step_recent_all_models_{rule}.png",
    },
}


def plot_rule(rule, by_model, max_tests, human_by_rule, cfg):
    fn = cfg["fn"]
    x = np.arange(cfg["x_offset"], max_tests + 1) if cfg["x_offset"] == 0 \
        else np.arange(1, max_tests + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    base = simulate_random_baseline(rule, max_tests)
    ig = fn(base)
    m = ig.mean(axis=0); sem = ig.std(axis=0, ddof=1) / np.sqrt(ig.shape[0])
    ax.plot(x, m, color=BASELINE_COLOR, linestyle=":", linewidth=1.6,
            label="random baseline")
    ax.fill_between(x, m - sem, m + sem, color=BASELINE_COLOR, alpha=0.12)

    for label in RUNS_BY_MODEL.keys():
        mat = remain_matrix(by_model.get(label, []), rule, max_tests)
        if mat is None:
            continue
        ig = fn(mat)
        m = ig.mean(axis=0); sem = ig.std(axis=0, ddof=1) / np.sqrt(ig.shape[0])
        c = MODEL_COLOR[label]
        ax.plot(x, m, color=c, linewidth=2.0, marker="o", markersize=3.5,
                label=f"{label}  (n={mat.shape[0]})")
        ax.fill_between(x, m - sem, m + sem, color=c, alpha=0.13)

    h_mat = human_remain_matrix(human_by_rule.get(rule, []), max_tests)
    if h_mat is not None:
        ig = fn(h_mat)
        m = ig.mean(axis=0); sem = ig.std(axis=0, ddof=1) / np.sqrt(ig.shape[0])
        ax.plot(x, m, color=HUMAN_COLOR, linewidth=2.0, marker="s", markersize=4,
                label=f"average human adults  (n={h_mat.shape[0]})")
        ax.fill_between(x, m - sem, m + sem, color=HUMAN_COLOR, alpha=0.15)

    best_list = pick_best_humans(human_by_rule.get(rule, []), max_tests)
    if best_list:
        mat = np.stack([c for _, c in best_list], axis=0)
        ig = fn(mat)
        m = ig.mean(axis=0)
        sem = (ig.std(axis=0, ddof=1) / np.sqrt(ig.shape[0])
               if ig.shape[0] > 1 else np.zeros_like(m))
        ax.plot(x, m, color=BEST_HUMAN_COLOR, linewidth=1.8,
                linestyle="-", marker="s", markersize=4,
                label=f"top human explorers (n={mat.shape[0]})")
        ax.fill_between(x, m - sem, m + sem, color=BEST_HUMAN_COLOR, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(cfg["axhline"], color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("test number")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(f"{rule}")
    if cfg["ylim"] is not None:
        ax.set_ylim(*cfg["ylim"])
    ax.set_xlim(cfg["xlim_start"], max_tests)
    ax.legend(loc=cfg["legend_loc"], fontsize=7.5, frameon=False,
              borderpad=0.4, handlelength=1.8)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, cfg["out_template"].format(rule=rule))
    fig.savefig(out, dpi=150)
    fig.savefig(os.path.splitext(out)[0] + ".svg")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=list(METRICS), required=True,
                        help="cum: cumulative info gain; step: per-step info gain")
    args = parser.parse_args()
    cfg = METRICS[args.metric]

    by_model = collect()
    human_by_rule = load_human_tests_by_rule()
    model_max = max(
        (max((len(t) for _, _, _, t in pt), default=0) for pt in by_model.values()),
        default=0,
    )
    human_max = max((len(t) for v in human_by_rule.values() for t, _, _ in v), default=0)
    max_tests = max(model_max, human_max, 1)
    for rule in ("conjunctive", "disjunctive"):
        plot_rule(rule, by_model, max_tests, human_by_rule, cfg)


if __name__ == "__main__":
    main()
