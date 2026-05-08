"""Hypotheses-remaining and cumulative info-gain curves for the recent models
(4 seeds × 12 trials each, temperature=0.0). Style matches
plot_hypotheses_remaining_o4_deepseek_gpt5.py.
"""
import json
import os
import random
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

REPO = "/network/scratch/s/samieima/projects/blicket-text-llm"

RUNS_BY_MODEL = {
    "gemma3:27b": [
        f"{REPO}/results/20260429_001427_9398844",
        f"{REPO}/results/20260429_001427_9398845",
        f"{REPO}/results/20260429_002045_9398846",
        f"{REPO}/results/20260429_002045_9398847",
    ],
    "gpt-4o": [
        f"{REPO}/results/20260429_105510_9403903",
        f"{REPO}/results/20260429_105510_9403904",
        f"{REPO}/results/20260429_105510_9403905",
        f"{REPO}/results/20260429_105823_9403906",
    ],
    "deepseek-chat": [
        f"{REPO}/results/20260429_094520_9403286",
        f"{REPO}/results/20260429_094519_9403287",
        f"{REPO}/results/20260429_094520_9403288",
        f"{REPO}/results/20260429_094521_9403289",
    ],
    "o4-mini": [
        f"{REPO}/results/20260428_232945_9398595",
        f"{REPO}/results/20260428_232945_9398596",
        f"{REPO}/results/20260428_232946_9398597",
        f"{REPO}/results/20260428_232945_9398598",
    ],
    "deepseek-reasoner": [
        f"{REPO}/results/20260428_231811_9398537",
        f"{REPO}/results/20260428_231812_9398538",
        f"{REPO}/results/20260428_231812_9398539",
        f"{REPO}/results/20260428_231811_9398540",
    ],
    "gemini-2.5-flash": [
        f"{REPO}/results/20260428_222010_9398172",
        f"{REPO}/results/20260428_222009_9398173",
        f"{REPO}/results/20260428_222009_9398174",
        f"{REPO}/results/20260428_222009_9398175",
    ],
    "gpt-5-mini": [
        f"{REPO}/results/20260428_233913_9398665",
        f"{REPO}/results/20260428_233912_9398666",
        f"{REPO}/results/20260428_233912_9398667",
        f"{REPO}/results/20260428_233913_9398668",
    ],
    "gpt-5": [
        f"{REPO}/results/20260428_221343_9398102",
        f"{REPO}/results/20260428_221342_9398103",
        f"{REPO}/results/20260428_221343_9398104",
        f"{REPO}/results/20260428_221342_9398105",
    ],
}

MODEL_COLOR = {
    "gemma3:27b":        "#8c564b",
    "gpt-4o":            "#1f77b4",
    "deepseek-chat":     "#7f7f7f",
    "o4-mini":           "#ff7f0e",
    "deepseek-reasoner": "#2ca02c",
    "gemini-2.5-flash":  "#d62728",
    "gpt-5-mini":        "#9467bd",
    "gpt-5":             "#17becf",
}
RULE_MARKER = {"conjunctive": "s", "disjunctive": "o"}
RULE_LINESTYLE = {"conjunctive": "-", "disjunctive": "--"}
BASELINE_STYLE = {
    "conjunctive": {"color": "#555555", "linestyle": "-",  "alpha": 0.7, "label": "Random Baseline (Conj)"},
    "disjunctive": {"color": "#555555", "linestyle": "--", "alpha": 0.7, "label": "Random Baseline (Disj)"},
}

OBJECTS = (1, 2, 3, 4)
SUBSETS = [frozenset(c) for k in range(1, len(OBJECTS) + 1) for c in combinations(OBJECTS, k)]
HYPOTHESES = [(B, rule) for B in SUBSETS for rule in ("conj", "disj")]
H0 = len(HYPOTHESES)
LOG2_H0 = np.log2(H0)
OUT_DIR = f"{REPO}/results/eig_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def predict(h, S):
    B, rule = h
    return B.issubset(S) if rule == "conj" else bool(B & S)


def simulate_random_baseline(rule_type, max_tests, n_seeds=48):
    rk = "conj" if rule_type == "conjunctive" else "disj"
    pair_truths = [(B, rk) for B in SUBSETS if len(B) == 2]
    runs = []
    for seed in range(n_seeds):
        true_h = pair_truths[seed % len(pair_truths)]
        rng = random.Random(seed)
        current_H = list(HYPOTHESES)
        counts = [H0]
        for _ in range(max_tests):
            k = rng.randint(1, len(OBJECTS))
            S = frozenset(rng.sample(OBJECTS, k))
            outcome = predict(true_h, S)
            current_H = [h for h in current_H if predict(h, S) == outcome]
            counts.append(len(current_H))
        runs.append(counts)
    return np.array(runs)


def parse_trial_tests(rows):
    tests = []
    main_rows = [r for r in rows if r.get("phase") == "main" and "question" not in r]
    main_rows.sort(key=lambda r: r.get("steps", 10**9))
    for r in main_rows:
        if r.get("action") != "test":
            continue
        gs = r.get("game_state") or {}
        ts = gs.get("true_state") or []
        if len(ts) < 5:
            continue
        S = frozenset(idx + 1 for idx in range(4) if ts[idx])
        light = bool(ts[-1])
        tests.append((S, light))
    return tests


def trial_remaining_curve(tests, max_tests):
    counts = [H0]
    H = list(HYPOTHESES)
    for (S, o) in tests:
        H = [h for h in H if predict(h, S) == o]
        counts.append(len(H))
    while len(counts) < max_tests + 1:
        counts.append(counts[-1])
    return np.array(counts[: max_tests + 1])


def load_meta(run_dir):
    out = {}
    for fn in os.listdir(run_dir):
        if fn.startswith("results_") and fn.endswith(".jsonl"):
            with open(os.path.join(run_dir, fn)) as f:
                for line in f:
                    r = json.loads(line.replace("NaN", "null"))
                    out[int(r["trial_idx"])] = r.get("true_rule")
    return out


def collect_per_model_curves(max_tests):
    curves = {}  # (model, rule) -> matrix (n_trials, max_tests+1)
    for model, dirs in RUNS_BY_MODEL.items():
        per_rule = {"conjunctive": [], "disjunctive": []}
        for d in dirs:
            if not os.path.isdir(d):
                continue
            meta = load_meta(d)
            for fn in sorted(os.listdir(d)):
                if not fn.startswith("action_log_trial-"):
                    continue
                idx = int(fn.split("trial-")[1].split("_")[0])
                rule = str(meta.get(idx, "")).strip().lower()
                if rule not in ("conjunctive", "disjunctive"):
                    continue
                with open(os.path.join(d, fn)) as f:
                    rows = [json.loads(l) for l in f]
                tests = parse_trial_tests(rows)
                per_rule[rule].append(trial_remaining_curve(tests, max_tests))
        for rule, mats in per_rule.items():
            if mats:
                curves[(model, rule)] = np.stack(mats, axis=0)
    return curves


def find_max_tests():
    n = 0
    for dirs in RUNS_BY_MODEL.values():
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.startswith("action_log_trial-"):
                    continue
                with open(os.path.join(d, fn)) as f:
                    rows = [json.loads(l) for l in f]
                n = max(n, len(parse_trial_tests(rows)))
    return n


def plot_panel(curves, baselines, max_tests, *, ylabel, title, transform, out_path,
               y_lower=None, y_upper=None, hline=None):
    x = np.arange(max_tests + 1)
    fig, ax = plt.subplots(figsize=(10, 6.2))

    for rule in ("conjunctive", "disjunctive"):
        arr = transform(baselines[rule])
        m = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        st = BASELINE_STYLE[rule]
        ax.plot(x, m, **st, linewidth=1.5)
        ax.fill_between(x, m - sem, m + sem, color=st["color"], alpha=0.10)

    for model in RUNS_BY_MODEL:
        for rule in ("conjunctive", "disjunctive"):
            if (model, rule) not in curves:
                continue
            arr = transform(curves[(model, rule)])
            m = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            color = MODEL_COLOR[model]
            ax.plot(x, m, color=color, marker=RULE_MARKER[rule], markersize=4,
                    linestyle=RULE_LINESTYLE[rule], linewidth=1.8,
                    label=f"{model} · {rule}")
            ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("test number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_lower is not None or y_upper is not None:
        ax.set_ylim(y_lower, y_upper)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


def main():
    max_tests = find_max_tests()
    print(f"max tests across all trials = {max_tests}")
    curves = collect_per_model_curves(max_tests)
    baselines = {r: simulate_random_baseline(r, max_tests) for r in ("conjunctive", "disjunctive")}

    plot_panel(curves, baselines, max_tests,
               ylabel="hypotheses remaining (mean ± SEM)",
               title="hypotheses remaining — recent models, T=0.0 (4 seeds × 12 trials)",
               transform=lambda H: H,
               out_path=os.path.join(OUT_DIR, "hypotheses_remaining_recent_models.png"),
               y_lower=0, y_upper=H0 + 1, hline=1)

    # Cumulative info gain in bits: log2(H0) - log2(H_t).
    plot_panel(curves, baselines, max_tests,
               ylabel="cumulative info gain (bits, mean ± SEM)",
               title="cumulative info gain — recent models, T=0.0 (4 seeds × 12 trials)",
               transform=lambda H: LOG2_H0 - np.log2(np.clip(H, 1, None)),
               out_path=os.path.join(OUT_DIR, "info_gain_cum_recent_models.png"),
               y_lower=0, y_upper=LOG2_H0 + 0.3, hline=LOG2_H0)


if __name__ == "__main__":
    main()
