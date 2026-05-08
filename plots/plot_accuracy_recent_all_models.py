"""Main-phase accuracy across recent active runs for multiple models, split by
ground-truth rule. Aggregates across 4 seeds per model. Style mirrors
results/figures/exploration_complete/accuracy_all.png.

Includes deepseek-chat alongside the prior set.
"""

import csv
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

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
OUT_DIR = f"{REPO}/results/eig_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_COLORS = {
    "o4-mini": "#ff7f0e",
    "gpt-5": "#17becf",
    "gpt-5-mini": "#9467bd",
    "gemini-2.5-flash": "#d62728",
    "deepseek-reasoner": "#2ca02c",
    "deepseek-chat": "#7f7f7f",
    "gemma3:27b": "#8c564b",
    "gpt-4o": "#1f77b4",
    "human": "#000000",
    "best human active explorer": "#f4d35e",
}
HUMAN_DATA_PATH = f"{REPO}/results/human_data/human_data_active_explore.json"
RULES = ("conjunctive", "disjunctive")


def load_trials(results_dirs):
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


def _human_test_curve_min_step(actions):
    from itertools import combinations
    objects = (1, 2, 3, 4)
    subsets = [frozenset(c) for k in range(1, 5) for c in combinations(objects, k)]
    H = [(B, r) for B in subsets for r in ("conj", "disj")]

    def predict(h, S):
        B, r = h
        return B.issubset(S) if r == "conj" else bool(B & S)
    tests = [(frozenset(int(i) + 1 for i in (a.get("objects_tested") or [])),
              bool(a.get("machine_state_after")))
             for a in actions if a.get("action_type") == "test"]
    cur = list(H)
    counts = [len(H)]
    for S, o in tests:
        cur = [h for h in cur if predict(h, S) == o]
        counts.append(len(cur))
    if not tests:
        return None
    mn = min(counts)
    return (counts.index(mn), mn, len(tests))


def load_human(agg, best=False):
    if not os.path.exists(HUMAN_DATA_PATH):
        return
    with open(HUMAN_DATA_PATH) as f:
        data = json.load(f)
    by_rule = {"conjunctive": [], "disjunctive": []}
    for sess in data.values():
        if not isinstance(sess, dict):
            continue
        rounds = (sess.get("config") or {}).get("rounds") or []
        if not rounds or rounds[0].get("num_objects") != 4:
            continue
        rule = str(rounds[0].get("rule", "")).lower()
        if rule not in RULES:
            continue
        mg = sess.get("main_game") or {}
        true_b = set(mg.get("true_blicket_indices") or [])
        chosen = set(mg.get("user_chosen_blickets") or [])
        n_obj = 4
        obj_correct = sum(1 for i in range(n_obj) if (i in true_b) == (i in chosen))
        all_ok = chosen == true_b
        rt = str(mg.get("rule_type", "")).lower()
        tr = str(mg.get("true_rule", rule)).lower()
        rule_ok = rt.startswith(tr)
        if best:
            if not (all_ok and rule_ok):
                continue
            actions = mg.get("user_test_actions") or []
            key = _human_test_curve_min_step(actions)
            if key is None:
                continue
            by_rule[rule].append((key, frozenset(true_b), mg))
        else:
            a = agg[("human", rule)]
            a["obj_correct"] += obj_correct
            a["obj_total"] += n_obj
            a["all_correct_trials"] += int(all_ok)
            a["rule_correct_trials"] += int(rule_ok)
            a["full_correct_trials"] += int(all_ok and rule_ok)
            a["num_trials"] += 1
    if best:
        for rule, lst in by_rule.items():
            lst.sort(key=lambda x: x[0])
            best_per_combo = {}
            for _, truth, mg in lst:
                if truth not in best_per_combo:
                    best_per_combo[truth] = mg
            for mg in best_per_combo.values():
                true_b = set(mg.get("true_blicket_indices") or [])
                chosen = set(mg.get("user_chosen_blickets") or [])
                obj_correct = sum(1 for i in range(4) if (i in true_b) == (i in chosen))
                all_ok = chosen == true_b
                rt = str(mg.get("rule_type", "")).lower()
                tr = str(mg.get("true_rule", rule)).lower()
                rule_ok = rt.startswith(tr)
                a = agg[("best human active explorer", rule)]
                a["obj_correct"] += obj_correct
                a["obj_total"] += 4
                a["all_correct_trials"] += int(all_ok)
                a["rule_correct_trials"] += int(rule_ok)
                a["full_correct_trials"] += int(all_ok and rule_ok)
                a["num_trials"] += 1


def main():
    agg = defaultdict(lambda: {"obj_correct": 0, "obj_total": 0,
                                "all_correct_trials": 0, "rule_correct_trials": 0,
                                "full_correct_trials": 0, "num_trials": 0})
    for model, dirs in RESULTS_BY_MODEL.items():
        for t in load_trials(dirs):
            rule = str(t.get("true_rule", "")).strip().lower()
            if rule not in RULES:
                continue
            key = (model, rule)
            agg[key]["obj_correct"] += int(t.get("num_correct", 0))
            agg[key]["obj_total"] += int(t.get("num_questions", 0))
            all_ok = bool(t.get("all_nexioms_correct"))
            rule_ok = bool(t.get("rule_type_correct"))
            agg[key]["all_correct_trials"] += int(all_ok)
            agg[key]["rule_correct_trials"] += int(rule_ok)
            agg[key]["full_correct_trials"] += int(all_ok and rule_ok)
            agg[key]["num_trials"] += 1
    load_human(agg, best=False)
    load_human(agg, best=True)

    def overall(model):
        num = den = 0
        for rule in RULES:
            a = agg[(model, rule)]
            num += a["full_correct_trials"]
            den += a["num_trials"]
        return num / den if den else 0.0

    def conj_score(model):
        a = agg[(model, "conjunctive")]
        return (a["full_correct_trials"] / a["num_trials"]) if a["num_trials"] else 0.0

    # Sort by conjunctive full-hypothesis (the discriminating rule);
    # break ties by overall full-hypothesis.
    model_groups = sorted(RESULTS_BY_MODEL.keys(),
                          key=lambda m: (conj_score(m), overall(m)))
    human_groups = sorted(["human", "best human active explorer"],
                          key=lambda m: (conj_score(m), overall(m)))
    all_groups = model_groups + human_groups

    print(f"{'model':<22} {'rule':<13} {'trials':>6} {'obj/obj':>10} {'all':>6} {'rule':>6} {'full':>6}")
    for model in all_groups:
        for rule in RULES:
            a = agg[(model, rule)]
            if a["num_trials"] == 0:
                continue
            po = a["obj_correct"] / a["obj_total"] if a["obj_total"] else float("nan")
            pa = a["all_correct_trials"] / a["num_trials"]
            pr = a["rule_correct_trials"] / a["num_trials"]
            pf = a["full_correct_trials"] / a["num_trials"]
            print(f"{model:<22} {rule:<13} {a['num_trials']:>6} {po:>10.3f} {pa:>6.2f} {pr:>6.2f} {pf:>6.2f}")

    csv_path = os.path.join(OUT_DIR, "main_accuracy_recent_all_models.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "true_rule", "num_trials",
                    "object_correct", "object_total", "object_accuracy",
                    "all_objects_correct_rate", "rule_accuracy", "full_hypothesis_rate"])
        for model in all_groups:
            for rule in RULES:
                a = agg[(model, rule)]
                if a["num_trials"] == 0:
                    continue
                po = a["obj_correct"] / a["obj_total"] if a["obj_total"] else float("nan")
                pa = a["all_correct_trials"] / a["num_trials"]
                pr = a["rule_correct_trials"] / a["num_trials"]
                pf = a["full_correct_trials"] / a["num_trials"]
                w.writerow([model, rule, a["num_trials"],
                            a["obj_correct"], a["obj_total"], f"{po:.4f}",
                            f"{pa:.4f}", f"{pr:.4f}", f"{pf:.4f}"])
    print(f"wrote {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5), sharey=False)
    panels = [
        ("all objects correct", "all"),
        ("rule", "rule"),
        ("full hypothesis", "full"),
    ]
    random_baseline = {"all": 1 / 16, "rule": 0.5, "full": 1 / 32}
    width = 0.30
    human_set = {"human", "best human active explorer"}
    HUMAN_GAP = 0.6  # extra spacing (in bar-widths) between LLMs and humans

    def metric_score(model, metric, rule):
        a = agg[(model, rule)]
        if a["num_trials"] == 0:
            return 0.0
        if metric == "all":
            k = a["all_correct_trials"]
        elif metric == "rule":
            k = a["rule_correct_trials"]
        else:
            k = a["full_correct_trials"]
        return k / a["num_trials"]

    n_models = len(RESULTS_BY_MODEL) + len(human_set)
    # Reserve a column slot per model + a gap before humans.
    slots_per_cluster = n_models + HUMAN_GAP
    cluster_w = (slots_per_cluster - 1) * width
    x = np.arange(len(RULES)) * (cluster_w + 0.8)

    for ax, (title, metric) in zip(axes, panels):
        seen_labels = set()
        # The "rule" panel's legend at upper-left crowds the conjunctive cluster;
        # nudge that panel's clusters to the right.
        x_shift = cluster_w * 0.25 if metric == "rule" else cluster_w * 0.05
        x_panel = x + x_shift
        for r_idx, rule in enumerate(RULES):
            # Sort independently per (panel, rule): LLMs then humans, each by metric.
            # Sort LLMs by metric, but pin gpt-5 as the rightmost (highest) LLM
            # — useful when several top models tie at 1.0 and gpt-5 would
            # otherwise be ordered by the dict's iteration order.
            llm_order = sorted(
                RESULTS_BY_MODEL.keys(),
                key=lambda m: (m == "gpt-5", metric_score(m, metric, rule)),
            )
            human_order = sorted(human_set,
                                 key=lambda m: metric_score(m, metric, rule))
            ordered = llm_order + list(human_order)

            # Per-cluster offsets with gap before humans.
            offsets = []
            slot = 0
            for m in ordered:
                if m == human_order[0]:
                    slot += HUMAN_GAP
                offsets.append(slot)
                slot += 1
            center = (offsets[0] + offsets[-1]) / 2
            offsets = [(o - center) * width for o in offsets]

            for i, model in enumerate(ordered):
                a = agg[(model, rule)]
                if a["num_trials"] == 0:
                    p, sem, lab = 0.0, 0.0, "n=0"
                else:
                    if metric == "all":
                        k, n = a["all_correct_trials"], a["num_trials"]
                    elif metric == "rule":
                        k, n = a["rule_correct_trials"], a["num_trials"]
                    else:
                        k, n = a["full_correct_trials"], a["num_trials"]
                    p = k / n
                    sem = math.sqrt(p * (1 - p) / n)
                    lab = f"{p:.2f}\n±{sem:.2f}"
                lo = min(p, sem); hi = min(1 - p, sem)
                display_label = (
                    "human adults" if model == "human"
                    else "top 6 human explorer" if model == "best human active explorer"
                    else model
                )
                # Avoid duplicate legend entries (we draw each model twice — once per rule).
                legend_label = display_label if display_label not in seen_labels else None
                if legend_label is not None:
                    seen_labels.add(display_label)
                bar = ax.bar(x_panel[r_idx] + offsets[i], p, width, label=legend_label,
                             color=MODEL_COLORS.get(model, "#444444"), alpha=0.85,
                             yerr=[[lo], [hi]], capsize=3,
                             error_kw={"elinewidth": 1, "ecolor": "black"})
                ax.annotate(lab,
                            xy=(bar[0].get_x() + bar[0].get_width() / 2, p + hi),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x_panel)
        # Lock xlim using the unshifted geometry so x_shift visibly moves the
        # rule panel's clusters to the right (otherwise autoscaling re-centers).
        ax.set_xlim(x[0] - cluster_w / 2 - 0.5,
                    x[-1] + cluster_w / 2 + 0.5 + x_shift)
        ax.set_xticklabels(RULES)
        ax.set_ylim(0, 1.18)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(title)
        ax.axhline(1.0, color="gray", ls="--", lw=0.6)
        ax.axhline(random_baseline[metric], color="red", ls=":", lw=1.2,
                   label="random baseline" if metric == "all" else None)
        ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "main_accuracy_recent_all_models.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
