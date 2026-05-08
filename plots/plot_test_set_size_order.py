"""Per-model distribution of test-set sizes (|S|=1,2,3,4) across test order
1..16 in the main exploration phase, averaged over all trials & 4 seeds in
results/final_llms/.
"""
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from plot_hypotheses_remaining_recent_all_models import (
    RUNS_BY_MODEL, MODEL_COLOR, parse_trial_tests, load_results_metadata,
    load_human_tests_by_rule, pick_best_humans, trial_remaining_curve,
)
import json

REPO = "/network/scratch/s/samieima/projects/blicket-text-llm"
OUT_DIR = f"{REPO}/results/figures/camera_ready"
HUMAN_PATH = f"{REPO}/results/human_data/human_data_active_explore.json"
RULES = ("conjunctive", "disjunctive")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_TESTS = 16
SIZES = (1, 2, 3, 4)
SIZE_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}
SIZE_LABEL = {1: "single (|S|=1)", 2: "pair (|S|=2)",
              3: "triple (|S|=3)", 4: "all four (|S|=4)"}


def collect_size_by_step():
    """Returns model -> array shape (MAX_TESTS, len(SIZES)) of percentages."""
    out = {}
    for model, run_dirs in RUNS_BY_MODEL.items():
        # counts[step][size] across all trials for this model
        counts = np.zeros((MAX_TESTS, len(SIZES)), dtype=float)
        totals = np.zeros(MAX_TESTS, dtype=float)
        for run_dir in run_dirs:
            run_dir = run_dir.replace("/results/", "/results/final_llms/")
            if not os.path.isdir(run_dir):
                continue
            for fn in sorted(os.listdir(run_dir)):
                if not fn.startswith("action_log_trial-"):
                    continue
                with open(os.path.join(run_dir, fn)) as f:
                    rows = []
                    for l in f:
                        try:
                            rows.append(json.loads(l))
                        except json.JSONDecodeError:
                            continue
                tests = parse_trial_tests(rows)
                for step, (S, _) in enumerate(tests[:MAX_TESTS]):
                    sz = len(S)
                    if sz < 1 or sz > 4:
                        continue
                    counts[step, SIZES.index(sz)] += 1
                    totals[step] += 1
        pct = np.zeros_like(counts)
        for s in range(MAX_TESTS):
            if totals[s] > 0:
                pct[s] = 100.0 * counts[s] / totals[s]
        out[model] = (pct, totals)
    return out


def collect_size_by_step_by_rule():
    """model -> rule -> (pct (MAX_TESTS, 4), totals (MAX_TESTS,))."""
    out = {}
    for model, run_dirs in RUNS_BY_MODEL.items():
        by_rule = {r: (np.zeros((MAX_TESTS, len(SIZES))),
                       np.zeros(MAX_TESTS)) for r in RULES}
        for run_dir in run_dirs:
            run_dir = run_dir.replace("/results/", "/results/final_llms/")
            if not os.path.isdir(run_dir):
                continue
            meta = load_results_metadata(run_dir)
            for fn in sorted(os.listdir(run_dir)):
                if not fn.startswith("action_log_trial-"):
                    continue
                try:
                    ti = int(fn.split("trial-")[1].split("_")[0])
                except (ValueError, IndexError):
                    continue
                rule = str(meta.get(ti, "")).lower()
                if rule not in RULES:
                    continue
                with open(os.path.join(run_dir, fn)) as f:
                    rows = []
                    for l in f:
                        try:
                            rows.append(json.loads(l))
                        except json.JSONDecodeError:
                            continue
                tests = parse_trial_tests(rows)
                counts, totals = by_rule[rule]
                for step, (S, _) in enumerate(tests[:MAX_TESTS]):
                    sz = len(S)
                    if sz < 1 or sz > 4:
                        continue
                    counts[step, SIZES.index(sz)] += 1
                    totals[step] += 1
        rule_out = {}
        for rule, (counts, totals) in by_rule.items():
            pct = np.zeros_like(counts)
            for s in range(MAX_TESTS):
                if totals[s] > 0:
                    pct[s] = 100.0 * counts[s] / totals[s]
            rule_out[rule] = (pct, totals)
        out[model] = rule_out
    return out


def collect_human_size_by_step_by_rule(top_only=False):
    out = {r: (np.zeros((MAX_TESTS, len(SIZES))),
               np.zeros(MAX_TESTS)) for r in RULES}
    if top_only:
        human_by_rule = load_human_tests_by_rule()
        for rule in RULES:
            entries = human_by_rule.get(rule, [])
            if not entries:
                continue
            best = pick_best_humans(entries, MAX_TESTS)
            counts, totals = out[rule]
            for tests, _curve in best:
                for step, (S, _light) in enumerate(tests[:MAX_TESTS]):
                    sz = len(S)
                    if sz < 1 or sz > 4:
                        continue
                    counts[step, SIZES.index(sz)] += 1
                    totals[step] += 1
    elif os.path.exists(HUMAN_PATH):
        with open(HUMAN_PATH) as f:
            data = json.load(f)
        for sess in data.values():
            if not isinstance(sess, dict):
                continue
            mg = sess.get("main_game") or {}
            cfg = mg.get("config") or {}
            if cfg.get("num_objects") != 4:
                continue
            rule = str(cfg.get("rule", "")).lower()
            if rule not in RULES:
                continue
            counts, totals = out[rule]
            for step, a in enumerate((mg.get("user_test_actions") or [])[:MAX_TESTS]):
                if a.get("action_type") != "test":
                    continue
                sz = len(a.get("objects_tested") or [])
                if sz < 1 or sz > 4:
                    continue
                counts[step, SIZES.index(sz)] += 1
                totals[step] += 1
    rule_out = {}
    for rule, (counts, totals) in out.items():
        pct = np.zeros_like(counts)
        for s in range(MAX_TESTS):
            if totals[s] > 0:
                pct[s] = 100.0 * counts[s] / totals[s]
        rule_out[rule] = (pct, totals)
    return rule_out


def collect_human_size_by_step(top_only=False):
    """Returns (pct (MAX_TESTS, 4), totals (MAX_TESTS,)) from human_data.

    If top_only=True, restrict to sessions where the user correctly identified
    the blicket set (user_chosen_blickets == true_blicket_indices).
    """
    counts = np.zeros((MAX_TESTS, len(SIZES)), dtype=float)
    totals = np.zeros(MAX_TESTS, dtype=float)
    if top_only:
        human_by_rule = load_human_tests_by_rule()
        for rule in RULES:
            for tests, _curve in pick_best_humans(
                    human_by_rule.get(rule, []), MAX_TESTS):
                for step, (S, _light) in enumerate(tests[:MAX_TESTS]):
                    sz = len(S)
                    if sz < 1 or sz > 4:
                        continue
                    counts[step, SIZES.index(sz)] += 1
                    totals[step] += 1
    elif os.path.exists(HUMAN_PATH):
        with open(HUMAN_PATH) as f:
            data = json.load(f)
        for sess in data.values():
            if not isinstance(sess, dict):
                continue
            mg = sess.get("main_game") or {}
            cfg = mg.get("config") or {}
            if cfg.get("num_objects") != 4:
                continue
            if str(cfg.get("rule", "")).lower() not in RULES:
                continue
            actions = mg.get("user_test_actions") or []
            for step, a in enumerate(actions[:MAX_TESTS]):
                if a.get("action_type") != "test":
                    continue
                sz = len(a.get("objects_tested") or [])
                if sz < 1 or sz > 4:
                    continue
                counts[step, SIZES.index(sz)] += 1
                totals[step] += 1
    pct = np.zeros_like(counts)
    for s in range(MAX_TESTS):
        if totals[s] > 0:
            pct[s] = 100.0 * counts[s] / totals[s]
    return pct, totals


def plot_grid(data, human=None, human_top=None):
    models = list(RUNS_BY_MODEL.keys())
    series = list(models)
    extras = {}
    if human is not None:
        series.append("human avg")
        extras["human avg"] = human
    if human_top is not None:
        series.append("human top")
        extras["human top"] = human_top
    n = len(series)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.2 * rows),
                             sharey=True)
    axes = np.array(axes).reshape(-1)
    x = np.arange(1, MAX_TESTS + 1)
    for i, model in enumerate(series):
        ax = axes[i]
        pct, totals = (extras[model] if model in extras else data[model])
        bottom = np.zeros(MAX_TESTS)
        for j, sz in enumerate(SIZES):
            ax.bar(x, pct[:, j], bottom=bottom,
                   color=SIZE_COLORS[sz], label=SIZE_LABEL[sz],
                   edgecolor="white", linewidth=0.4, width=0.85)
            bottom += pct[:, j]
        if model in extras:
            n_max = int(totals.max()) if totals.size else 0
            n_min = int(totals[totals > 0].min()) if (totals > 0).any() else 0
            title = f"{model} n={n_max}→{n_min}"
        else:
            title = model
        ax.set_title(title)
        ax.set_xlabel("test index")
        ax.set_ylabel("% of trials")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in x], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_xlim(0.4, MAX_TESTS + 0.6)
        for s in range(MAX_TESTS):
            if totals[s] == 0:
                ax.text(x[s], 50, "n/a", ha="center", va="center",
                        fontsize=7, color="#888")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Distribution of test-set size by test order (final_llms, all trials × 4 seeds)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"test_set_size_by_order.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def plot_lines(data):
    """Line plot: % of single-object tests vs test index, one line per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    x = np.arange(1, MAX_TESTS + 1)
    for ax, sz in zip(axes, (1, 2)):
        for model in RUNS_BY_MODEL:
            pct, _ = data[model]
            ax.plot(x, pct[:, SIZES.index(sz)], marker="o", markersize=4,
                    color=MODEL_COLOR.get(model, "#333"), label=model)
        ax.set_title(f"% tests with {SIZE_LABEL[sz]}")
        ax.set_xlabel("test index")
        ax.set_ylabel("%")
        ax.set_xticks(x)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"test_set_size_lines.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def plot_grouped_bars(data, human=None):
    """Three-category bar plot (single / pair / multiple>=3) where each category
    has its own base color and each model gets a shade of that color."""
    import matplotlib.colors as mcolors

    CATS = [
        ("single",   [1],    "#1f77b4"),
        ("pair",     [2],    "#2ca02c"),
        ("multiple", [3, 4], "#d62728"),
    ]
    models = list(RUNS_BY_MODEL.keys())
    series = (["human"] if human is not None else []) + models
    n_series = len(series)

    # Build per-category, per-series percentage arrays of shape (n_series, MAX_TESTS).
    cat_pct = {}
    for name, sizes, _ in CATS:
        arr = np.zeros((n_series, MAX_TESTS))
        for mi, m in enumerate(series):
            pct, _ = (human if m == "human" else data[m])
            for sz in sizes:
                arr[mi] += pct[:, SIZES.index(sz)]
        cat_pct[name] = arr

    # Shades per model for each category (light -> dark across models).
    def shades(base_hex, n):
        base_rgb = np.array(mcolors.to_rgb(base_hex))
        # Mix from 0.75 (light) -> 0.0 (full color)
        out = []
        for k in range(n):
            t = 0.75 - 0.75 * (k / max(n - 1, 1))  # 0.75..0
            rgb = base_rgb + (1 - base_rgb) * t
            out.append(rgb)
        return out

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    x = np.arange(1, MAX_TESTS + 1)
    bar_w = 0.8 / n_series

    legend_handles_models = None
    for ax, (name, _sizes, base) in zip(axes, CATS):
        sh = shades(base, len(models))
        for mi, m in enumerate(series):
            offset = (mi - (n_series - 1) / 2) * bar_w
            if m == "human":
                ax.bar(x + offset, cat_pct[name][mi], width=bar_w,
                       facecolor="white", edgecolor="black", linewidth=0.6,
                       hatch="///", label=m)
            else:
                ax.bar(x + offset, cat_pct[name][mi], width=bar_w,
                       color=sh[mi - (1 if human is not None else 0)],
                       edgecolor="white", linewidth=0.3, label=m)
        ax.set_title(f"{name} ({'|S|=1' if name=='single' else '|S|=2' if name=='pair' else '|S|>=3'})",
                     loc="left", color=base, fontweight="bold")
        ax.set_ylabel("% of tests")
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.3)
        if legend_handles_models is None:
            legend_handles_models = ax.get_legend_handles_labels()
    axes[-1].set_xlabel("test index")

    # Two legends: model shades (using neutral grey shades) + category colors.
    from matplotlib.patches import Patch
    cat_patches = [Patch(facecolor=base, label=name) for name, _, base in CATS]
    grey_sh = shades("#444444", len(models))
    model_patches = []
    if human is not None:
        h_totals = human[1]
        n_max = int(h_totals.max()) if h_totals.size else 0
        n_min = int(h_totals[h_totals > 0].min()) if (h_totals > 0).any() else 0
        model_patches.append(Patch(
            facecolor="white", edgecolor="black", hatch="///",
            label=f"human n={n_max}→{n_min} sessions"))
    model_patches += [Patch(facecolor=grey_sh[i], label=m)
                      for i, m in enumerate(models)]
    leg1 = fig.legend(handles=cat_patches, title="category",
                      loc="upper left", bbox_to_anchor=(0.88, 0.97),
                      frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=model_patches, title="model (shade, light→dark)",
               loc="upper left", bbox_to_anchor=(0.88, 0.78),
               frameon=False, fontsize=9)

    fig.suptitle("Test-set size distribution by test order", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.87, 0.97))
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"test_set_size_grouped_bars.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def plot_grouped_bars_by_rule(data_by_rule, human_by_rule=None,
                              human_top_by_rule=None):
    """Like plot_grouped_bars, but with conjunctive vs disjunctive side-by-side
    (3 size-categories × 2 rules = 6 subplots)."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    CATS = [
        ("single",   [1],    "#1f77b4"),
        ("pair",     [2],    "#2ca02c"),
        ("multiple", [3, 4], "#d62728"),
    ]
    models = list(RUNS_BY_MODEL.keys())
    use_human = human_by_rule is not None
    use_human_top = human_top_by_rule is not None
    human_series = []
    if use_human:
        human_series.append("human avg")
    if use_human_top:
        human_series.append("human top")
    human_data_map = {}
    if use_human:
        human_data_map["human avg"] = human_by_rule
    if use_human_top:
        human_data_map["human top"] = human_top_by_rule
    human_hatch = {"human avg": "///", "human top": "xxx"}
    series = human_series + models
    n_series = len(series)

    def shades(base_hex, n):
        base_rgb = np.array(mcolors.to_rgb(base_hex))
        out = []
        for k in range(n):
            t = 0.75 - 0.75 * (k / max(n - 1, 1))
            out.append(base_rgb + (1 - base_rgb) * t)
        return out

    fig, axes = plt.subplots(len(CATS), len(RULES),
                             figsize=(8 * len(RULES), 3.0 * len(CATS)),
                             sharex=True, sharey=True)
    x = np.arange(1, MAX_TESTS + 1)
    bar_w = 0.8 / n_series

    for ci, (cat_name, sizes, base) in enumerate(CATS):
        sh = shades(base, len(models))
        for ri, rule in enumerate(RULES):
            ax = axes[ci, ri]
            for mi, m in enumerate(series):
                pct, _ = (human_data_map[m][rule] if m in human_data_map
                          else data_by_rule[m][rule])
                vals = np.zeros(MAX_TESTS)
                for sz in sizes:
                    vals += pct[:, SIZES.index(sz)]
                offset = (mi - (n_series - 1) / 2) * bar_w
                if m in human_data_map:
                    ax.bar(x + offset, vals, width=bar_w, facecolor="white",
                           edgecolor="black", linewidth=0.6,
                           hatch=human_hatch[m], label=m)
                else:
                    ax.bar(x + offset, vals, width=bar_w,
                           color=sh[mi - len(human_series)],
                           edgecolor="white", linewidth=0.3, label=m)
            if ci == 0:
                ax.set_title(rule, fontweight="bold")
            if ri == 0:
                ax.set_ylabel(f"{cat_name}\n% of tests", color=base,
                              fontweight="bold")
            ax.set_ylim(0, 100)
            ax.set_xticks(x)
            ax.grid(axis="y", alpha=0.3)
        axes[-1, ri].set_xlabel("test index")
    for ri in range(len(RULES)):
        axes[-1, ri].set_xlabel("test index")

    cat_patches = [Patch(facecolor=base, label=name) for name, _, base in CATS]
    grey_sh = shades("#444444", len(models))
    model_patches = []
    for hname in human_series:
        hbr = human_data_map[hname]
        h_max = max(int(hbr[r][1].max()) for r in RULES)
        h_min_nz = []
        for r in RULES:
            t = hbr[r][1]
            if (t > 0).any():
                h_min_nz.append(int(t[t > 0].min()))
        h_min = min(h_min_nz) if h_min_nz else 0
        model_patches.append(Patch(
            facecolor="white", edgecolor="black", hatch=human_hatch[hname],
            label=f"{hname} n={h_max}→{h_min}"))
    model_patches += [Patch(facecolor=grey_sh[i], label=m)
                      for i, m in enumerate(models)]
    leg1 = fig.legend(handles=cat_patches, title="category",
                      loc="upper left", bbox_to_anchor=(0.92, 0.97),
                      frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=model_patches, title="series",
               loc="upper left", bbox_to_anchor=(0.92, 0.78),
               frameon=False, fontsize=9)

    fig.suptitle("Test-set size distribution by test order — conjunctive vs disjunctive",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.91, 0.97))
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"test_set_size_grouped_bars_by_rule.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def _mean_size_row(pct, totals):
    """Per-step mean |S| from the (MAX_TESTS, 4) % matrix; NaN where totals==0."""
    sizes = np.array(SIZES, dtype=float)
    row = (pct / 100.0) @ sizes
    row[totals == 0] = np.nan
    return row


def plot_mean_size_heatmap(data, human=None, human_top=None,
                           data_by_rule=None, human_by_rule=None,
                           human_top_by_rule=None):
    """Heatmap: rows=series, cols=test index, color=mean |S|.

    If *_by_rule args are given, render conjunctive and disjunctive side by side
    in addition to the overall heatmap.
    """
    series_specs = []
    if human_top is not None:
        series_specs.append(("human top", human_top))
    if human is not None:
        series_specs.append(("human avg", human))
    for m in RUNS_BY_MODEL:
        series_specs.append((m, data[m]))

    def _matrix(get):
        rows, labels = [], []
        for name, payload in series_specs:
            pct, totals = get(name, payload)
            rows.append(_mean_size_row(pct, totals))
            labels.append(name)
        return np.vstack(rows), labels

    panels = [("all trials", lambda n, p: p)]
    if data_by_rule is not None:
        for rule in RULES:
            def getter(n, p, _rule=rule):
                if n == "human avg" and human_by_rule is not None:
                    return human_by_rule[_rule]
                if n == "human top" and human_top_by_rule is not None:
                    return human_top_by_rule[_rule]
                return data_by_rule[n][_rule]
            panels.append((rule, getter))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
    })
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(6.4 * len(panels), 0.55 * len(series_specs) + 2.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#dddddd")
    x = np.arange(1, MAX_TESTS + 1)
    im = None
    for ax, (title, getter) in zip(axes, panels):
        mat, labels = _matrix(getter)
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=1.0, vmax=4.0,
                       interpolation="nearest")
        ax.set_xticks(np.arange(MAX_TESTS))
        ax.set_xticklabels([str(t) for t in x])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("test index")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=10,
                        color="white" if v > 2.5 else "black")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("mean test-set size |S|")
    fig.suptitle("Mean test-set size by test order")
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"test_set_size_heatmap.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def plot_dominant_strategy_heatmap(data, human=None, human_top=None,
                                   data_by_rule=None, human_by_rule=None,
                                   human_top_by_rule=None):
    """Like plot_modal_strategy_heatmap, but cell opacity encodes the share of
    trials choosing the modal |S|. Solid = locked-in strategy, pale = mixed."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
    })
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    series_specs = []
    if human_top is not None:
        series_specs.append(("human top", human_top))
    if human is not None:
        series_specs.append(("human avg", human))
    for m in RUNS_BY_MODEL:
        series_specs.append((m, data[m]))

    panels = [("all trials", lambda n, p: p)]
    if data_by_rule is not None:
        for rule in RULES:
            def getter(n, p, _rule=rule):
                if n == "human avg" and human_by_rule is not None:
                    return human_by_rule[_rule]
                if n == "human top" and human_top_by_rule is not None:
                    return human_top_by_rule[_rule]
                return data_by_rule[n][_rule]
            panels.append((rule, getter))

    # Map share in [0.25, 1.0] to alpha in [0.15, 1.0] (with 4 sizes, 0.25 = uniform).
    def _alpha(share):
        lo, hi = 0.25, 1.0
        s = max(min(share, hi), lo)
        return 0.15 + (s - lo) / (hi - lo) * 0.85

    def _rgba_grid(getter):
        rows_rgba, labels, share_rows = [], [], []
        for name, payload in series_specs:
            pct, totals = getter(name, payload)
            row_rgba = np.ones((MAX_TESTS, 4))
            shares = np.full(MAX_TESTS, np.nan)
            for s in range(MAX_TESTS):
                if totals[s] == 0:
                    row_rgba[s] = mcolors.to_rgba("#dddddd")
                    continue
                idx = int(np.argmax(pct[s]))
                share = pct[s, idx] / 100.0
                base = np.array(mcolors.to_rgb(SIZE_COLORS[SIZES[idx]]))
                a = _alpha(share)
                blended = base * a + np.array([1, 1, 1]) * (1 - a)
                row_rgba[s, :3] = blended
                row_rgba[s, 3] = 1.0
                shares[s] = share
            rows_rgba.append(row_rgba)
            share_rows.append(shares)
            labels.append(name)
        return np.stack(rows_rgba, axis=0), share_rows, labels

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5.8 * len(panels),
                                      0.55 * len(series_specs) + 2.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(1, MAX_TESTS + 1)
    for ax, (title, getter) in zip(axes, panels):
        rgba, shares, labels = _rgba_grid(getter)
        ax.imshow(rgba, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(MAX_TESTS))
        ax.set_xticklabels([str(t) for t in x])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("test index")
        ax.set_title(title)
        for i, srow in enumerate(shares):
            for j, sh in enumerate(srow):
                if np.isnan(sh):
                    continue
                ax.text(j, i, f"{int(round(sh * 100))}", ha="center",
                        va="center", fontsize=8,
                        color="white" if _alpha(sh) > 0.6 else "black")

    cat_handles = [Patch(facecolor=SIZE_COLORS[s], label=SIZE_LABEL[s])
                   for s in SIZES]
    opacity_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=(0.85, 0.85, 0.85, 1.0), markersize=14,
               label="low agreement (~25%)"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=(0.55, 0.55, 0.55, 1.0), markersize=14,
               label="medium (~60%)"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=(0.15, 0.15, 0.15, 1.0), markersize=14,
               label="locked-in (~100%)"),
    ]
    leg1 = fig.legend(handles=cat_handles, title="dominant |S|",
                      loc="upper right", bbox_to_anchor=(1.0, 0.95),
                      frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=opacity_handles, title="opacity = share of trials",
               loc="upper right", bbox_to_anchor=(1.0, 0.55),
               frameon=False)
    fig.suptitle("Dominant test-set size by test order (opacity = agreement)")
    fig.tight_layout(rect=(0, 0, 0.82, 0.95))
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(OUT_DIR, f"test_set_size_dominant_heatmap.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def plot_modal_strategy_heatmap(data, human=None, human_top=None,
                                data_by_rule=None, human_by_rule=None,
                                human_top_by_rule=None):
    """Heatmap of the modal test-set size (argmax over |S|=1..4) per series and
    test index. One panel for all trials, plus per-rule panels if given."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
    })
    series_specs = []
    if human_top is not None:
        series_specs.append(("human top", human_top))
    if human is not None:
        series_specs.append(("human avg", human))
    for m in RUNS_BY_MODEL:
        series_specs.append((m, data[m]))

    def _modal_row(pct, totals):
        row = np.full(MAX_TESTS, np.nan)
        for s in range(MAX_TESTS):
            if totals[s] == 0:
                continue
            row[s] = SIZES[int(np.argmax(pct[s]))]
        return row

    panels = [("all trials", lambda n, p: p)]
    if data_by_rule is not None:
        for rule in RULES:
            def getter(n, p, _rule=rule):
                if n == "human avg" and human_by_rule is not None:
                    return human_by_rule[_rule]
                if n == "human top" and human_top_by_rule is not None:
                    return human_top_by_rule[_rule]
                return data_by_rule[n][_rule]
            panels.append((rule, getter))

    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    colors = [SIZE_COLORS[s] for s in SIZES]
    cmap = ListedColormap(colors)
    cmap.set_bad("#dddddd")
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5.6 * len(panels),
                                      0.5 * len(series_specs) + 2.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(1, MAX_TESTS + 1)
    for ax, (title, getter) in zip(axes, panels):
        rows, labels = [], []
        for name, payload in series_specs:
            pct, totals = getter(name, payload)
            rows.append(_modal_row(pct, totals))
            labels.append(name)
        mat = np.ma.masked_invalid(np.vstack(rows))
        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm,
                  interpolation="nearest")
        ax.set_xticks(np.arange(MAX_TESTS))
        ax.set_xticklabels([str(t) for t in x])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("test index")
        ax.set_title(title)

    legend_handles = [Patch(facecolor=SIZE_COLORS[s], label=SIZE_LABEL[s])
                      for s in SIZES]
    fig.legend(handles=legend_handles, title="modal |S|",
               loc="center right", bbox_to_anchor=(1.0, 0.5),
               frameon=False)
    fig.suptitle("Modal test-set size by test order")
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"test_set_size_modal_heatmap.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def plot_mean_size_lines_per_rule(data_by_rule, human_by_rule=None,
                                  human_top_by_rule=None):
    """One figure per rule: mean |S| vs test index, one line per series."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
    })
    SERIES_COLOR = {"human top": "#f4d35e", "human avg": "#000000"}
    series_specs = []
    if human_top_by_rule is not None:
        series_specs.append(("human top", human_top_by_rule))
    if human_by_rule is not None:
        series_specs.append(("human avg", human_by_rule))
    for m in RUNS_BY_MODEL:
        series_specs.append((m, data_by_rule[m]))

    x = np.arange(1, MAX_TESTS + 1)
    for rule in RULES:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for name, payload in series_specs:
            color = SERIES_COLOR.get(name, MODEL_COLOR.get(name, "#444"))
            pct, totals = payload[rule]
            row = _mean_size_row(pct, totals)
            ax.plot(x, row, color=color, marker="o", markersize=5,
                    linewidth=2.0, label=name)
        ax.set_xlabel("test index")
        ax.set_ylabel(r"mean test-set size $|S|$")
        ax.set_xticks(x)
        ax.set_ylim(0.8, 4.2)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["1 (single)", "2 (pair)", "3 (triple)", "4 (all)"])
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, fontsize=12)
        fig.suptitle(f"Mean test-set size over time — {rule}")
        fig.tight_layout(rect=(0, 0, 0.80, 0.95))
        for ext in ("png", "pdf"):
            path = os.path.join(OUT_DIR, f"test_set_size_mean_lines_{rule}.{ext}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)


def plot_mean_size_lines_by_rule(data_by_rule, human_by_rule=None,
                                 human_top_by_rule=None):
    """Single panel: mean |S| vs test index. Solid=conjunctive, dashed=disjunctive.
    One color per series (human top, human avg, each LLM)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
    })
    SERIES_COLOR = {"human top": "#f4d35e", "human avg": "#000000"}
    series_specs = []
    if human_top_by_rule is not None:
        series_specs.append(("human top", human_top_by_rule))
    if human_by_rule is not None:
        series_specs.append(("human avg", human_by_rule))
    for m in RUNS_BY_MODEL:
        series_specs.append((m, data_by_rule[m]))

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(1, MAX_TESTS + 1)
    for name, payload in series_specs:
        color = SERIES_COLOR.get(name, MODEL_COLOR.get(name, "#444"))
        for rule, ls, marker in (("conjunctive", "-", "o"),
                                 ("disjunctive", "--", "s")):
            pct, totals = payload[rule]
            row = _mean_size_row(pct, totals)
            ax.plot(x, row, color=color, linestyle=ls, marker=marker,
                    markersize=4.5, linewidth=1.8, label=f"{name} ({rule})")

    ax.set_xlabel("test index")
    ax.set_ylabel(r"mean test-set size $|S|$")
    ax.set_xticks(x)
    ax.set_ylim(0.8, 4.2)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1 (single)", "2 (pair)", "3 (triple)", "4 (all)"])
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    series_handles = [Line2D([0], [0], color=SERIES_COLOR.get(n, MODEL_COLOR.get(n, "#444")),
                             linewidth=2.5, label=n)
                      for n, _ in series_specs]
    rule_handles = [Line2D([0], [0], color="#444", linestyle="-", marker="o",
                           label="conjunctive"),
                    Line2D([0], [0], color="#444", linestyle="--", marker="s",
                           label="disjunctive")]
    leg1 = ax.legend(handles=series_handles, title="series",
                     loc="center left", bbox_to_anchor=(1.01, 0.7),
                     frameon=False, fontsize=12, title_fontsize=13)
    ax.add_artist(leg1)
    ax.legend(handles=rule_handles, title="rule",
              loc="center left", bbox_to_anchor=(1.01, 0.25),
              frameon=False, fontsize=12, title_fontsize=13)

    fig.suptitle("Mean test-set size over time")
    fig.tight_layout(rect=(0, 0, 0.82, 0.96))
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"test_set_size_mean_lines_by_rule.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    data = collect_size_by_step()
    human = collect_human_size_by_step()
    human_top = collect_human_size_by_step(top_only=True)
    print(f"human (avg) n by test index 1..{MAX_TESTS}: "
          f"{human[1].astype(int).tolist()} (total tests={int(human[1].sum())})")
    print(f"human (top) n by test index 1..{MAX_TESTS}: "
          f"{human_top[1].astype(int).tolist()} (total tests={int(human_top[1].sum())})")
    plot_grid(data, human=human, human_top=human_top)
    plot_lines(data)
    plot_grouped_bars(data, human=human)
    data_by_rule = collect_size_by_step_by_rule()
    human_by_rule = collect_human_size_by_step_by_rule()
    human_top_by_rule = collect_human_size_by_step_by_rule(top_only=True)
    for r in RULES:
        print(f"human avg {r} n by test index: "
              f"{human_by_rule[r][1].astype(int).tolist()}")
        print(f"human top {r} n by test index: "
              f"{human_top_by_rule[r][1].astype(int).tolist()}")
    plot_grouped_bars_by_rule(data_by_rule, human_by_rule=human_by_rule,
                              human_top_by_rule=human_top_by_rule)
    plot_mean_size_lines_by_rule(data_by_rule, human_by_rule=human_by_rule,
                                 human_top_by_rule=human_top_by_rule)
    plot_mean_size_lines_per_rule(data_by_rule, human_by_rule=human_by_rule,
                                  human_top_by_rule=human_top_by_rule)
    plot_modal_strategy_heatmap(data, human=human, human_top=human_top,
                                data_by_rule=data_by_rule,
                                human_by_rule=human_by_rule,
                                human_top_by_rule=human_top_by_rule)
    plot_dominant_strategy_heatmap(data, human=human, human_top=human_top,
                                   data_by_rule=data_by_rule,
                                   human_by_rule=human_by_rule,
                                   human_top_by_rule=human_top_by_rule)
    plot_mean_size_heatmap(data, human=human, human_top=human_top,
                           data_by_rule=data_by_rule,
                           human_by_rule=human_by_rule,
                           human_top_by_rule=human_top_by_rule)
