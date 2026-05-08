"""Hypotheses-remaining curves for the recent runs used in
plot_accuracy_recent_all_models.py. Conjunctive and disjunctive trials are
plotted in separate figures. Mean +/- SEM across trials of each rule, with a
random-sampling baseline and human curves overlaid.
"""
import json
import os
import random
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

REPO = "/network/scratch/s/samieima/projects/blicket-text-llm"

RUNS_BY_MODEL = {
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
    "gemini-3.1-pro-preview": [
        f"{REPO}/results/20260426_203002_9372363",
    ],
}

MODEL_COLOR = {
    "o4-mini":           "#ff7f0e",
    "gpt-5":             "#17becf",
    "gpt-5-mini":        "#9467bd",
    "gemini-2.5-flash":  "#d62728",
    "deepseek-reasoner": "#2ca02c",
    "deepseek-chat":     "#7f7f7f",
    "gemini-3.1-pro-preview": "#1f77b4",
    "gemma3:27b":        "#8c564b",
    "gpt-4o":            "#1f77b4",
}
HUMAN_COLOR = "#000000"
BEST_HUMAN_COLOR = "#f4d35e"
BASELINE_COLOR = "#888888"

HUMAN_DATA_PATH = f"{REPO}/results/human_data/human_data_active_explore.json"
OUT_DIR = f"{REPO}/results/eig_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_TEMPLATE = os.path.join(
    OUT_DIR, "hypotheses_remaining_recent_all_models_{rule}.png"
)

OBJECTS = (1, 2, 3, 4)
SUBSETS = [frozenset(c) for k in range(1, len(OBJECTS) + 1) for c in combinations(OBJECTS, k)]
HYPOTHESES = [(B, rule) for B in SUBSETS for rule in ("conj", "disj")]
H0 = len(HYPOTHESES)
assert H0 == 30


def predict(h, S):
    B, rule = h
    return B.issubset(S) if rule == "conj" else bool(B & S)


def simulate_random_baseline(rule_type, max_tests, n_seeds=48):
    rk = "conj" if rule_type == "conjunctive" else "disj"
    pair_truths = [(B, rk) for B in SUBSETS if len(B) == 2]
    results_h = []
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
        results_h.append(counts)
    return np.array(results_h)


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


def load_results_metadata(results_dir):
    out = {}
    if not os.path.isdir(results_dir):
        return out
    for fn in os.listdir(results_dir):
        if fn.startswith("results_") and fn.endswith(".jsonl"):
            with open(os.path.join(results_dir, fn)) as f:
                for line in f:
                    try:
                        r = json.loads(line.replace("NaN", "null"))
                    except json.JSONDecodeError:
                        continue
                    out[int(r["trial_idx"])] = r.get("true_rule")
    return out


def collect():
    """For each model, returns list of (run_dir, trial_idx, true_rule, tests)."""
    by_model = {}
    for label, run_dirs in RUNS_BY_MODEL.items():
        per_trial = []
        for run_dir in run_dirs:
            if not os.path.isdir(run_dir):
                continue
            meta = load_results_metadata(run_dir)
            for fn in sorted(os.listdir(run_dir)):
                if not fn.startswith("action_log_trial-"):
                    continue
                idx = int(fn.split("trial-")[1].split("_")[0])
                with open(os.path.join(run_dir, fn)) as f:
                    rows = []
                    for l in f:
                        try:
                            rows.append(json.loads(l))
                        except json.JSONDecodeError:
                            continue
                tests = parse_trial_tests(rows)
                per_trial.append((run_dir, idx, meta.get(idx), tests))
        by_model[label] = per_trial
    return by_model


def load_human_tests_by_rule(path=HUMAN_DATA_PATH):
    out = {"conjunctive": [], "disjunctive": []}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        data = json.load(f)
    for sess in data.values():
        if not isinstance(sess, dict):
            continue
        rounds = (sess.get("config") or {}).get("rounds") or []
        if not rounds or rounds[0].get("num_objects") != 4:
            continue
        rule = str(rounds[0].get("rule", "")).lower()
        if rule not in out:
            continue
        mg = sess.get("main_game") or {}
        actions = mg.get("user_test_actions") or []
        tests = []
        for a in actions:
            if a.get("action_type") != "test":
                continue
            objs = a.get("objects_tested") or []
            S = frozenset(int(i) + 1 for i in objs)
            light = bool(a.get("machine_state_after"))
            tests.append((S, light))
        if not tests:
            continue
        chosen = set(mg.get("user_chosen_blickets") or [])
        truth = set(mg.get("true_blicket_indices") or [])
        rt = str(mg.get("rule_type", "")).lower()
        tr = str(mg.get("true_rule", rule)).lower()
        fully_correct = (chosen == truth) and rt.startswith(tr)
        out[rule].append((tests, fully_correct, frozenset(truth)))
    return out


def pick_best_humans(human_entries, max_tests):
    by_combo = {}
    for tests, fully_correct, truth in human_entries:
        if not fully_correct:
            continue
        curve = trial_remaining_curve(tests, max_tests)
        min_rem = int(curve.min())
        first_min = int(np.argmax(curve == min_rem))
        key = (first_min, min_rem, len(tests))
        prev = by_combo.get(truth)
        if prev is None or key < prev[0]:
            by_combo[truth] = (key, tests, curve)
    return [(t, c) for _, t, c in by_combo.values()]


def plot_rule(rule, by_model, max_tests, human_by_rule):
    x = np.arange(max_tests + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    base = simulate_random_baseline(rule, max_tests)
    m = base.mean(axis=0)
    sem = base.std(axis=0, ddof=1) / np.sqrt(base.shape[0])
    ax.plot(x, m, color=BASELINE_COLOR, linestyle=":", linewidth=1.6,
            label="random baseline")
    ax.fill_between(x, m - sem, m + sem, color=BASELINE_COLOR, alpha=0.12)

    for label in RUNS_BY_MODEL.keys():
        per_trial = by_model.get(label, [])
        mats = [trial_remaining_curve(t, max_tests)
                for _, _, tr, t in per_trial if tr == rule]
        if not mats:
            continue
        mat = np.stack(mats, axis=0)
        m = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        c = MODEL_COLOR[label]
        ax.plot(x, m, color=c, linewidth=2.0, marker="o", markersize=3.5,
                label=f"{label}  (n={mat.shape[0]})")
        ax.fill_between(x, m - sem, m + sem, color=c, alpha=0.13)

    human_tests_list = human_by_rule.get(rule, [])
    best_list = pick_best_humans(human_tests_list, max_tests) if human_tests_list else []
    if best_list:
        mat = np.stack([c for _, c in best_list], axis=0)
        m = mat.mean(axis=0)
        sem = (mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
               if mat.shape[0] > 1 else np.zeros_like(m))
        ax.plot(x, m, color=BEST_HUMAN_COLOR, linewidth=1.8,
                linestyle="-", marker="s", markersize=4,
                label=f"top human explorers (n={mat.shape[0]})")
        ax.fill_between(x, m - sem, m + sem, color=BEST_HUMAN_COLOR, alpha=0.2)
    if human_tests_list:
        mats = [trial_remaining_curve(t, max_tests) for t, _, _ in human_tests_list]
        mat = np.stack(mats, axis=0)
        m = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ax.plot(x, m, color=HUMAN_COLOR, linewidth=2.0, marker="s", markersize=4,
                label=f"average human adults  (n={mat.shape[0]})")
        ax.fill_between(x, m - sem, m + sem, color=HUMAN_COLOR, alpha=0.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(1, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("test number")
    ax.set_ylabel("hypotheses remaining")
    ax.set_title(f"{rule}")
    ax.set_ylim(0, H0 + 1)
    ax.set_xlim(0, max_tests)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False,
              borderpad=0.4, handlelength=1.8)
    fig.tight_layout()

    out = OUT_TEMPLATE.format(rule=rule)
    fig.savefig(out, dpi=150)
    fig.savefig(os.path.splitext(out)[0] + ".svg")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    by_model = collect()
    human_by_rule = load_human_tests_by_rule()
    model_max = max(
        (max((len(t) for _, _, _, t in pt), default=0) for pt in by_model.values()),
        default=0,
    )
    human_max = max((len(t) for v in human_by_rule.values() for t, _, _ in v), default=0)
    max_tests = max(model_max, human_max, 1)
    for rule in ("conjunctive", "disjunctive"):
        plot_rule(rule, by_model, max_tests, human_by_rule)


if __name__ == "__main__":
    main()
