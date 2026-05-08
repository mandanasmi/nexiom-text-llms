"""Among incorrect blicket selections, how many of the 4 object answers were
wrong (Hamming distance to the true blicket set)? Compares humans vs the LLMs
in results/final_llms/.

A trial is a 'mistake' if num_wrong >= 1. We plot, per agent, the conditional
distribution P(num_wrong = k | num_wrong >= 1) for k in {1, 2, 3, 4}.
"""
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from plot_hypotheses_remaining_recent_all_models import (
    RUNS_BY_MODEL, MODEL_COLOR,
)

REPO = "/network/scratch/s/samieima/projects/blicket-text-llm"
OUT_DIR = f"{REPO}/results/figures/camera_ready"
HUMAN_PATH = f"{REPO}/results/human_data/human_data_active_explore.json"
RULES = ("conjunctive", "disjunctive")


def llm_mistakes():
    """Returns model -> list of num_wrong (0..4) per trial."""
    out = {}
    for model, run_dirs in RUNS_BY_MODEL.items():
        wrongs = []
        for d in run_dirs:
            d = d.replace("/results/", "/results/final_llms/")
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not (fn.startswith("results_") and fn.endswith(".jsonl")):
                    continue
                with open(os.path.join(d, fn)) as f:
                    for line in f:
                        try:
                            r = json.loads(line.replace("NaN", "null"))
                        except json.JSONDecodeError:
                            continue
                        if str(r.get("true_rule", "")).lower() not in RULES:
                            continue
                        nq = r.get("num_questions")
                        nc = r.get("num_correct")
                        if nq != 4 or nc is None:
                            continue
                        wrongs.append(int(nq - nc))
        out[model] = wrongs
    return out


def human_mistakes():
    if not os.path.exists(HUMAN_PATH):
        return []
    with open(HUMAN_PATH) as f:
        data = json.load(f)
    wrongs = []
    for sess in data.values():
        if not isinstance(sess, dict):
            continue
        rounds = (sess.get("config") or {}).get("rounds") or []
        if not rounds or rounds[0].get("num_objects") != 4:
            continue
        if str(rounds[0].get("rule", "")).lower() not in RULES:
            continue
        mg = sess.get("main_game") or {}
        chosen = set(mg.get("user_chosen_blickets") or [])
        truth = set(mg.get("true_blicket_indices") or [])
        # Hamming over 4-object universe = symmetric difference size.
        wrongs.append(len(chosen ^ truth))
    return wrongs


def cond_dist(wrongs):
    """Conditional dist over k=1..4 given num_wrong>=1, plus mistake rate."""
    nonzero = [w for w in wrongs if w >= 1]
    n_total = len(wrongs)
    n_mis = len(nonzero)
    c = Counter(nonzero)
    pct = np.array([100.0 * c.get(k, 0) / max(n_mis, 1) for k in (1, 2, 3, 4)])
    counts = np.array([c.get(k, 0) for k in (1, 2, 3, 4)], dtype=int)
    return pct, counts, n_total, n_mis


def main():
    llm = llm_mistakes()
    hum = human_mistakes()

    agents = [("human", hum)] + [(m, llm[m]) for m in RUNS_BY_MODEL.keys()]

    pct_mat, counts_mat, ns = [], [], []
    for name, w in agents:
        pct, cnt, n_tot, n_mis = cond_dist(w)
        pct_mat.append(pct)
        counts_mat.append(cnt)
        ns.append((n_tot, n_mis))
        rate = 100.0 * n_mis / max(n_tot, 1)
        print(f"{name}: n_trials={n_tot}, n_mistakes={n_mis} "
              f"({rate:.1f}%), k=1/2/3/4 counts={cnt.tolist()} "
              f"pct={pct.round(1).tolist()}")
    pct_mat = np.array(pct_mat)

    K_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}
    K_LABEL = {1: "1 wrong", 2: "2 wrong", 3: "3 wrong", 4: "4 wrong"}

    # Stacked bars: one bar per agent, segments = k=1..4.
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(agents))
    bottom = np.zeros(len(agents))
    for j, k in enumerate((1, 2, 3, 4)):
        vals = pct_mat[:, j]
        ax.bar(x, vals, bottom=bottom, color=K_COLORS[k],
               edgecolor="white", linewidth=0.5, label=K_LABEL[k])
        for i, v in enumerate(vals):
            if v >= 4:
                ax.text(x[i], bottom[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9, color="white")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n(n={ns[i][1]}/{ns[i][0]})"
                        for i, (name, _) in enumerate(agents)],
                       rotation=20, ha="right")
    ax.set_ylabel("% of mistake trials")
    ax.set_ylim(0, 100)
    ax.set_title("Among incorrect trials, how many of the 4 object answers were wrong?")
    ax.legend(title="# wrong objects", loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"mistake_breakdown.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    # Correlation view: scatter of mistake-rate (overall) vs avg num_wrong | mistake.
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, w) in enumerate(agents):
        n_tot, n_mis = ns[i]
        rate = 100.0 * n_mis / max(n_tot, 1)
        nonzero = [v for v in w if v >= 1]
        avgw = float(np.mean(nonzero)) if nonzero else 0.0
        color = "#000000" if name == "human" else MODEL_COLOR.get(name, "#444")
        ax.scatter(rate, avgw, color=color, s=80, edgecolor="white", zorder=3)
        ax.annotate(name, (rate, avgw), xytext=(5, 5),
                    textcoords="offset points", fontsize=10)
    ax.set_xlabel("mistake rate (% of trials with any wrong answer)")
    ax.set_ylabel("avg # objects wrong | mistake")
    ax.set_ylim(1, 4)
    ax.grid(alpha=0.3)
    ax.set_title("Mistake rate vs depth of mistake")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"mistake_breakdown_scatter.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def per_object_wrong_by_model():
    """For each model, fraction of main-phase per-object answers that were wrong,
    broken out by object id (1..4), averaged over seeds (run dirs)."""
    out = {}
    for model, run_dirs in RUNS_BY_MODEL.items():
        per_seed = []  # list of (4,) arrays of wrong-rates
        for d in run_dirs:
            d = d.replace("/results/", "/results/final_llms/")
            if not os.path.isdir(d):
                continue
            wrong = np.zeros(4)
            total = np.zeros(4)
            # only count trials whose true_rule is conjunctive/disjunctive
            res_path = None
            for fn in os.listdir(d):
                if fn.startswith("results_") and fn.endswith(".jsonl"):
                    res_path = os.path.join(d, fn)
                    break
            keep_trials = set()
            if res_path:
                with open(res_path) as f:
                    for line in f:
                        try:
                            r = json.loads(line.replace("NaN", "null"))
                        except json.JSONDecodeError:
                            continue
                        if (str(r.get("true_rule", "")).lower() in RULES
                                and r.get("num_questions") == 4):
                            keep_trials.add(int(r["trial_idx"]))
            for fn in os.listdir(d):
                if not (fn.startswith("action_log_trial-") and fn.endswith(".jsonl")):
                    continue
                # extract trial idx
                try:
                    ti = int(fn.split("trial-")[1].split("_")[0])
                except (ValueError, IndexError):
                    continue
                if ti not in keep_trials:
                    continue
                with open(os.path.join(d, fn)) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if r.get("phase") != "main":
                            continue
                        q = str(r.get("question", ""))
                        if not q.startswith("Is object "):
                            continue
                        try:
                            oid = int(q.split("Is object ")[1].split(" ")[0])
                        except (ValueError, IndexError):
                            continue
                        if not 1 <= oid <= 4:
                            continue
                        a = str(r.get("answer", "")).strip().lower()
                        t = str(r.get("true_answer", "")).strip().lower()
                        if a not in ("true", "false") or t not in ("true", "false"):
                            continue
                        total[oid - 1] += 1
                        if a != t:
                            wrong[oid - 1] += 1
            if total.sum() > 0:
                rates = np.where(total > 0, wrong / np.maximum(total, 1), 0.0)
                per_seed.append(rates)
        if per_seed:
            arr = np.stack(per_seed)  # (n_seeds, 4)
            out[model] = (arr.mean(0) * 100, arr.std(0) * 100, arr.shape[0])
    return out


def plot_per_object_wrong():
    data = per_object_wrong_by_model()
    if not data:
        print("no per-object data")
        return
    models = list(data.keys())
    n_m = len(models)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / n_m
    x = np.arange(4)
    for i, m in enumerate(models):
        mean, std, n = data[m]
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(x + offset, mean, width, yerr=std,
               color=MODEL_COLOR.get(m, "#444"),
               edgecolor="white", linewidth=0.5,
               label=f"{m} (n_seeds={n})", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"object {k}" for k in (1, 2, 3, 4)])
    ax.set_ylabel("% answered incorrectly (mean ± SD over seeds)")
    ax.set_title("Per-object error rate by model (averaged over seeds)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"mistake_breakdown_per_object.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
    # print summary
    for m, (mean, std, n) in data.items():
        worst = int(np.argmax(mean)) + 1
        print(f"{m}: per-object wrong% = {mean.round(1).tolist()} "
              f"(worst=object {worst})")


if __name__ == "__main__":
    main()
    plot_per_object_wrong()
