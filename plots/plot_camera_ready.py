"""Camera-ready versions of the 5 main figures (now 7 — accuracy split into
3 separate figures, one per metric):

  - main_accuracy_all_objects.png/.pdf
  - main_accuracy_rule.png/.pdf
  - main_accuracy_full_hypothesis.png/.pdf
  - hypotheses_remaining_conjunctive.png/.pdf
  - hypotheses_remaining_disjunctive.png/.pdf
  - info_gain_cum_conjunctive.png/.pdf
  - info_gain_cum_disjunctive.png/.pdf

Same data sources as the recent_all_models scripts (no gpt-4o, no gemma3:27b).
"""
import json
import math
import os
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_hypotheses_remaining_recent_all_models import (
    RUNS_BY_MODEL, MODEL_COLOR, HUMAN_COLOR, BEST_HUMAN_COLOR, BASELINE_COLOR,
    H0, collect, load_human_tests_by_rule, simulate_random_baseline,
    trial_remaining_curve, pick_best_humans,
)

REPO = "/network/scratch/s/samieima/projects/blicket-text-llm"
OUT_DIR = f"{REPO}/results/figures/camera_ready"
os.makedirs(OUT_DIR, exist_ok=True)

LOG2H0 = float(np.log2(H0))
RULES = ("conjunctive", "disjunctive")
HUMAN_LABELS = {"human": "average human adults", "best human active explorer": "top human"}
HUMAN_BAR_COLOR = {"human": "#000000", "best human active explorer": "#f4d35e"}

# --- styling ----------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 600,
})


def save(fig, basename):
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(OUT_DIR, f"{basename}.{ext}")
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


# --- accuracy ---------------------------------------------------------------
def load_trials(d):
    if not os.path.isdir(d):
        return
    for fn in os.listdir(d):
        if fn.startswith("results_") and fn.endswith(".jsonl"):
            with open(os.path.join(d, fn)) as f:
                for line in f:
                    try:
                        yield json.loads(line.replace("NaN", "null"))
                    except json.JSONDecodeError:
                        continue


HUMAN_DATA_PATH = f"{REPO}/results/human_data/human_data_active_explore.json"


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


def collect_accuracy():
    agg = defaultdict(lambda: {"all": 0, "rule": 0, "full": 0, "n": 0})
    for model, dirs in RUNS_BY_MODEL.items():
        for d in dirs:
            for t in load_trials(d):
                rule = str(t.get("true_rule", "")).strip().lower()
                if rule not in RULES:
                    continue
                a = bool(t.get("all_nexioms_correct"))
                r = bool(t.get("rule_type_correct"))
                key = (model, rule)
                agg[key]["all"] += int(a)
                agg[key]["rule"] += int(r)
                agg[key]["full"] += int(a and r)
                agg[key]["n"] += 1

    if os.path.exists(HUMAN_DATA_PATH):
        with open(HUMAN_DATA_PATH) as f:
            hdata = json.load(f)
        best_pool = {"conjunctive": [], "disjunctive": []}
        for sess in hdata.values():
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
            all_ok = chosen == true_b
            rt = str(mg.get("rule_type", "")).lower()
            tr = str(mg.get("true_rule", rule)).lower()
            rule_ok = rt.startswith(tr)
            agg[("human", rule)]["all"] += int(all_ok)
            agg[("human", rule)]["rule"] += int(rule_ok)
            agg[("human", rule)]["full"] += int(all_ok and rule_ok)
            agg[("human", rule)]["n"] += 1
            if all_ok and rule_ok:
                actions = mg.get("user_test_actions") or []
                key = _human_test_curve_min_step(actions)
                if key is not None:
                    best_pool[rule].append((key, frozenset(true_b), mg))
        # one best explorer per truth combo
        for rule, lst in best_pool.items():
            lst.sort(key=lambda x: x[0])
            best = {}
            for _, truth, mg in lst:
                if truth not in best:
                    best[truth] = mg
            for mg in best.values():
                true_b = set(mg.get("true_blicket_indices") or [])
                chosen = set(mg.get("user_chosen_blickets") or [])
                all_ok = chosen == true_b
                rt = str(mg.get("rule_type", "")).lower()
                tr = str(mg.get("true_rule", rule)).lower()
                rule_ok = rt.startswith(tr)
                a = agg[("best human active explorer", rule)]
                a["all"] += int(all_ok)
                a["rule"] += int(rule_ok)
                a["full"] += int(all_ok and rule_ok)
                a["n"] += 1
    return agg


def plot_accuracy_panel(agg, metric, ylabel, basename):
    HUMAN_GROUPS = ["human", "best human active explorer"]
    MODELS = list(RUNS_BY_MODEL.keys())
    random_baseline = {"all": 1 / 16, "rule": 0.5, "full": 1 / 32}[metric]

    def score(model, rule):
        a = agg[(model, rule)]
        return (a[metric] / a["n"]) if a["n"] else 0.0

    # sort: LLMs by metric (with gpt-5 pinned right), then humans by metric.
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    n_groups = len(MODELS) + len(HUMAN_GROUPS)
    width = 2.4
    HUMAN_GAP = 0.0
    cluster_w = (n_groups - 1 + HUMAN_GAP) * width
    x_centers = np.arange(len(RULES)) * (cluster_w + 13.0)

    seen = set()
    for r_idx, rule in enumerate(RULES):
        mixed = MODELS + ["human"]
        llm_order = sorted(mixed, key=lambda m: (m == "gpt-5", score(m, rule)))
        human_order = ["best human active explorer"]
        ordered = llm_order + human_order
        offsets = []
        slot = 0.0
        for m in ordered:
            if m == human_order[0]:
                slot += HUMAN_GAP
            offsets.append(slot)
            slot += 1.25
        center = (offsets[0] + offsets[-1]) / 2
        offsets = [(o - center) * width for o in offsets]

        for i, model in enumerate(ordered):
            a = agg[(model, rule)]
            if a["n"] == 0:
                continue
            p = a[metric] / a["n"]
            sem = math.sqrt(p * (1 - p) / a["n"])
            lo = min(p, sem); hi = min(1 - p, sem)
            display = HUMAN_LABELS.get(model, model)
            label = display if display not in seen else None
            if label is not None:
                seen.add(display)
            color = HUMAN_BAR_COLOR.get(model, MODEL_COLOR.get(model, "#444"))
            ax.bar(x_centers[r_idx] + offsets[i], p, width, label=label,
                   color=color, alpha=0.92,
                   yerr=[[lo], [hi]], capsize=2.5,
                   error_kw={"elinewidth": 0.8, "ecolor": "black"},
                   linewidth=0)
            ax.annotate(f"{p:.2f}",
                        xy=(x_centers[r_idx] + offsets[i], p + hi),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12)

    ax.axhline(random_baseline, color="red", ls=":", lw=2.2)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([r.capitalize() for r in RULES], fontsize=17)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(ylabel)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_order = list(MODELS) + HUMAN_GROUPS
    handles = [Line2D([0], [0], color="red", ls=":", lw=2.2)]
    labels = ["random baseline"]
    for m in legend_order:
        c = HUMAN_BAR_COLOR.get(m, MODEL_COLOR.get(m, "#444"))
        handles.append(Patch(facecolor=c, alpha=0.92))
        labels.append(HUMAN_LABELS.get(m, m))
    # Arrange legend as 4 / 3 / 2 across three rows in a fixed order.
    # Row 0: random baseline, gpt-5-mini, deepseek-chat, o4-mini
    # Row 1: gemini-2.5-flash, gpt-5, deepseek-reasoner
    # Row 2: humans
    label_to_handle = dict(zip(labels, handles))
    def _pair(name):
        display = HUMAN_LABELS.get(name, name)
        return (label_to_handle[display], display)
    row0 = [(label_to_handle["random baseline"], "random baseline"),
            _pair("gpt-5-mini"), _pair("deepseek-chat"), _pair("o4-mini")]
    row1 = [_pair("gemini-2.5-flash"), _pair("gpt-5"), _pair("deepseek-reasoner")]
    row2 = [_pair(g) for g in HUMAN_GROUPS]
    # Stack three centered single-row legends so each row is centered
    # independently (4/3/2 layout). Use figure-level legends so they sit
    # in figure coords above the axes.
    fig.subplots_adjust(top=0.72)
    y_anchors = [0.97, 0.91, 0.85]
    for row, y in zip([row0, row1, row2], y_anchors):
        hs = [p[0] for p in row]
        ls = [p[1] for p in row]
        fig.legend(hs, ls, loc="center",
                   bbox_to_anchor=(0.5, y),
                   frameon=False, ncol=len(row), handlelength=1.3,
                   columnspacing=1.2, handletextpad=0.5, fontsize=16,
                   borderaxespad=0.1)
    save(fig, basename)


# --- line-plot helpers ------------------------------------------------------
MODEL_COLOR_LINES = MODEL_COLOR


def cumulative_info_gain(remain_mat):
    return LOG2H0 - np.log2(np.clip(remain_mat, 1, None))


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


def _line_panel(ax, x, mats_by_label, baseline_curve, ylabel, ylim, hline=None,
                legend_loc="upper right", legend_bbox_x=1.02, legend_bbox_y=None,
                legend_top_horizontal=False):
    m, sem = baseline_curve
    ax.plot(x, m, color=BASELINE_COLOR, ls=":", lw=2.4, label="random baseline")
    ax.fill_between(x, m - sem, m + sem, color=BASELINE_COLOR, alpha=0.10)
    for label, (mat_mean, mat_sem, color, ls, marker, n) in mats_by_label.items():
        ax.plot(x, mat_mean, color=color, lw=2.4, ls=ls, marker=marker,
                markersize=8.0, label=label)
        ax.fill_between(x, mat_mean - mat_sem, mat_mean + mat_sem,
                        color=color, alpha=0.13)
    if hline is not None:
        ax.axhline(hline, color="grey", ls=":", lw=0.6, alpha=0.7)
    ax.set_ylim(*ylim)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=23)
    fig_ = ax.figure
    fig_.canvas.draw()
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    if xlabels and xlabels[0] in ("0", "$\\mathdefault{0}$"):
        xlabels[0] = ""
        ax.set_xticklabels(xlabels)
    if False and legend_top_horizontal:
        # 4 / 3 / 2 stacked top-centered legend matching the accuracy plots.
        h_by_label = {h.get_label(): h for h in ax.get_lines()}
        def _pair(name, display=None):
            display = display or name
            return (h_by_label[name], display)
        row0 = [_pair("random baseline"), _pair("gpt-5-mini"),
                _pair("deepseek-chat")]
        row1 = [_pair("gemini-2.5-flash"), _pair("deepseek-reasoner"),
                _pair("o4-mini")]
        row2_src = [("gpt-5", "gpt-5"),
                    ("human adults", "average human adults"),
                    ("top human", "top human")]
        row2 = [(_pair(n, d)) for n, d in row2_src if n in h_by_label]
        fig = ax.figure
        fig.subplots_adjust(top=0.70)
        fig.canvas.draw()
        pos = ax.get_position()
        cx = pos.x0 + pos.width / 2
        y_anchors = [1.16, 1.08, 1.00]
        for row, y in zip([row0, row1, row2], y_anchors):
            if not row:
                continue
            hs = [p[0] for p in row]
            ls = [p[1] for p in row]
            fig.legend(hs, ls, loc="center", bbox_to_anchor=(cx, y),
                       frameon=False, ncol=len(row), handlelength=2.8,
                       columnspacing=1.2, handletextpad=0.5, fontsize=22,
                       markerscale=1.8, borderaxespad=0.1)
    else:
        pass


def plot_lines(rule, by_model, max_tests, human_by_rule, kind):
    """kind in {'remain','cum_ig'}"""
    x = np.arange(max_tests + 1)

    base_remain = simulate_random_baseline(rule, max_tests)
    base = base_remain if kind == "remain" else cumulative_info_gain(base_remain)
    base_m = base.mean(axis=0)
    base_sem = base.std(axis=0, ddof=1) / np.sqrt(base.shape[0])

    series = {}
    for label in RUNS_BY_MODEL:
        mat = remain_matrix(by_model.get(label, []), rule, max_tests)
        if mat is None:
            continue
        data = mat if kind == "remain" else cumulative_info_gain(mat)
        m = data.mean(axis=0)
        sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
        series[label] = (m, sem, MODEL_COLOR_LINES[label], "-", "o", mat.shape[0])

    h_mat = human_remain_matrix(human_by_rule.get(rule, []), max_tests)
    if h_mat is not None:
        data = h_mat if kind == "remain" else cumulative_info_gain(h_mat)
        m = data.mean(axis=0)
        sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
        series["human adults"] = (m, sem, HUMAN_COLOR, "-", "s", h_mat.shape[0])
    best_list = pick_best_humans(human_by_rule.get(rule, []), max_tests)
    if best_list:
        mat = np.stack([c for _, c in best_list], axis=0)
        data = mat if kind == "remain" else cumulative_info_gain(mat)
        m = data.mean(axis=0)
        sem = (data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
               if data.shape[0] > 1 else np.zeros_like(m))
        series["top human"] = (m, sem, BEST_HUMAN_COLOR, "-", "s", mat.shape[0])

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    inset_zoom = None  # (x_lo, x_hi, y_lo, y_hi, axes_bounds)
    if kind == "remain":
        _line_panel(ax, x, series, (base_m, base_sem),
                    ylabel="hypotheses remaining",
                    ylim=(0, H0 + 1), hline=1, legend_loc="upper right",
                    legend_bbox_x=0.98,
                    legend_top_horizontal=True)
        basename = f"hypotheses_remaining_{rule}"
        if rule == "disjunctive":
            inset_zoom = (1.5, 4.5, 0, 12, (0.42, 0.42, 0.48, 0.47))
        elif rule == "conjunctive":
            inset_zoom = (4.5, 9, 0, 10, (0.50, 0.50, 0.48, 0.47))
    else:
        bbox_x = 1.08 if rule == "conjunctive" else 0.98
        bbox_y = 0.005 if rule == "conjunctive" else None
        if rule == "disjunctive":
            inset_zoom = (3.5, 6.5, 3.5, LOG2H0 + 0.05, (0.50, 0.05, 0.48, 0.47))
        elif rule == "conjunctive":
            inset_zoom = (5.5, 10, 3, 5, (0.50, 0.05, 0.48, 0.47))
        _line_panel(ax, x, series, (base_m, base_sem),
                    ylabel="cumulative info gain",
                    ylim=(0, LOG2H0 + 0.3), hline=LOG2H0,
                    legend_loc="lower right",
                    legend_bbox_x=bbox_x, legend_bbox_y=bbox_y,
                    legend_top_horizontal=True)
        basename = f"info_gain_cum_{rule}"

    if inset_zoom is not None:
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        x_lo, x_hi, y_lo, y_hi, bounds = inset_zoom
        axins = ax.inset_axes(bounds)
        # baseline
        axins.plot(x, base_m, color=BASELINE_COLOR, ls=":", lw=1.2)
        for label, (m, sem, color, ls, marker, n) in series.items():
            axins.plot(x, m, color=color, lw=1.4, ls=ls, marker=marker,
                       markersize=2.8)
            axins.fill_between(x, m - sem, m + sem, color=color, alpha=0.13)
        axins.set_xlim(x_lo, x_hi)
        axins.set_ylim(y_lo, y_hi)
        axins.set_xticks([])
        axins.set_yticks([])
        for s in axins.spines.values():
            s.set_linewidth(0.6)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none",
                   ec="0.5", lw=0.6, ls="--")

    fig.tight_layout()
    save(fig, basename)


def save_lines_legend(by_model, human_by_rule, max_tests):
    """Standalone single-line legend SVG for the hypothesis-remaining plot."""
    from matplotlib.lines import Line2D
    rule = "conjunctive"
    handles = [Line2D([0], [0], color=BASELINE_COLOR, ls=":", lw=2.4,
                      label="random baseline")]
    for label in RUNS_BY_MODEL:
        mat = remain_matrix(by_model.get(label, []), rule, max_tests)
        if mat is None:
            continue
        handles.append(Line2D([0], [0], color=MODEL_COLOR_LINES[label],
                              lw=2.4, marker="o", markersize=8.0, label=label))
    if human_remain_matrix(human_by_rule.get(rule, []), max_tests) is not None:
        handles.append(Line2D([0], [0], color=HUMAN_COLOR, lw=2.4,
                              marker="s", markersize=8.0,
                              label="average human adults"))
    if pick_best_humans(human_by_rule.get(rule, []), max_tests):
        handles.append(Line2D([0], [0], color=BEST_HUMAN_COLOR, lw=2.4,
                              marker="s", markersize=8.0, label="top human"))
    fig = plt.figure(figsize=(22, 0.9))
    fig.legend(handles=handles, loc="center", ncol=len(handles),
               frameon=False, handlelength=2.8, columnspacing=1.4,
               handletextpad=0.5, fontsize=22, markerscale=1.0)
    save(fig, "hypotheses_remaining_legend")


def main():
    agg = collect_accuracy()
    plot_accuracy_panel(agg, "all", "All-objects-correct accuracy",
                        "main_accuracy_all_objects")
    plot_accuracy_panel(agg, "rule", "Rule-correct accuracy",
                        "main_accuracy_rule")
    plot_accuracy_panel(agg, "full", "Full Hypothesis Accuracy",
                        "main_accuracy_full_hypothesis")

    by_model = collect()
    human_by_rule = load_human_tests_by_rule()
    model_max = max(
        (max((len(t) for _, _, _, t in pt), default=0) for pt in by_model.values()),
        default=0,
    )
    human_max = max((len(t) for v in human_by_rule.values() for t, _, _ in v), default=0)
    max_tests = max(model_max, human_max, 1)

    for rule in RULES:
        plot_lines(rule, by_model, max_tests, human_by_rule, kind="remain")
        plot_lines(rule, by_model, max_tests, human_by_rule, kind="cum_ig")

    save_lines_legend(by_model, human_by_rule, max_tests)


if __name__ == "__main__":
    main()
