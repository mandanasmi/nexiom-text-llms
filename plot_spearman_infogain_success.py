"""Spearman correlation between per-trial information gain (over the main-phase
test sequence) and task success (all_nexioms_correct AND rule_type_correct).

For each agent (LLMs in RUNS_BY_MODEL + humans), we compute:
  - info_gain (bits) = log2(H0) - log2(H_remaining_after_all_tests)
  - success         = 1 if fully correct, else 0
and report Spearman rho overall and split by ground-truth rule.

Outputs a barplot of rho per (agent, rule) with n annotated, and prints a table.
"""
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_hypotheses_remaining_recent_all_models import (
    RUNS_BY_MODEL, MODEL_COLOR, H0, HYPOTHESES, predict,
    parse_trial_tests, load_results_metadata,
    HUMAN_DATA_PATH, load_human_tests_by_rule, pick_best_humans,
)

REPO = "/network/scratch/s/samieima/projects/blicket-text-llm"
OUT_DIR = f"{REPO}/results/figures/camera_ready"
os.makedirs(OUT_DIR, exist_ok=True)
RULES = ("conjunctive", "disjunctive")
HUMAN_COLOR = "#000000"


def trial_mean_remaining(tests):
    """Mean |H_remaining| after each test in this trial (excludes initial H0)."""
    H = list(HYPOTHESES); rem = []
    for (S, o) in tests:
        H = [h for h in H if predict(h, S) == o]
        rem.append(len(H))
        if not H:
            break
    return float(np.mean(rem)) if rem else float(H0)


def trial_num_unique_tests(tests):
    return len({S for S, _ in tests})


def trial_info_gain(tests):
    """Bits of info eliminated over the whole test sequence: log2(H0/|H_end|)."""
    H = list(HYPOTHESES)
    for (S, o) in tests:
        H = [h for h in H if predict(h, S) == o]
        if not H:
            break
    n_end = max(len(H), 1)
    return math.log2(H0) - math.log2(n_end)


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    rx = _ranks(x); ry = _ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if abs(rho) >= 1.0 or n <= 2:
        p = float("nan")
    else:
        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        # two-sided p via normal approx (good enough for n>=20).
        p = math.erfc(abs(t) / math.sqrt(2))
    return rho, p


def _ranks(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a))
    # average ranks for ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def collect_llm_metrics():
    """agent -> list of (rule, info_gain, full, rule_acc, all_obj, per_obj)."""
    out = {m: [] for m in RUNS_BY_MODEL}
    for model, run_dirs in RUNS_BY_MODEL.items():
        for run_dir in run_dirs:
            run_dir_fl = run_dir.replace("/results/", "/results/final_llms/")
            d = run_dir_fl if os.path.isdir(run_dir_fl) else run_dir
            if not os.path.isdir(d):
                continue
            res = {}
            for fn in os.listdir(d):
                if fn.startswith("results_") and fn.endswith(".jsonl"):
                    with open(os.path.join(d, fn)) as f:
                        for line in f:
                            try:
                                r = json.loads(line.replace("NaN", "null"))
                            except json.JSONDecodeError:
                                continue
                            res[int(r["trial_idx"])] = r
            for fn in sorted(os.listdir(d)):
                if not fn.startswith("action_log_trial-"):
                    continue
                try:
                    ti = int(fn.split("trial-")[1].split("_")[0])
                except (ValueError, IndexError):
                    continue
                meta = res.get(ti, {})
                rule = str(meta.get("true_rule", "")).lower()
                if rule not in RULES:
                    continue
                with open(os.path.join(d, fn)) as f:
                    rows = []
                    for l in f:
                        try:
                            rows.append(json.loads(l))
                        except json.JSONDecodeError:
                            continue
                tests = parse_trial_tests(rows)
                ig = trial_info_gain(tests)
                rule_acc = 1 if bool(meta.get("rule_type_correct")) else 0
                all_obj = 1 if bool(meta.get("all_nexioms_correct")) else 0
                full = 1 if (rule_acc and all_obj) else 0
                nq = meta.get("num_questions") or 0
                nc = meta.get("num_correct") or 0
                per_obj = (nc / nq) if nq else 0.0
                out[model].append((rule, ig, full, rule_acc, all_obj, per_obj,
                                   trial_mean_remaining(tests),
                                   trial_num_unique_tests(tests)))
    return out


def collect_human_metrics():
    out = []
    if not os.path.exists(HUMAN_DATA_PATH):
        return out
    with open(HUMAN_DATA_PATH) as f:
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
        tests = []
        for a in (mg.get("user_test_actions") or []):
            if a.get("action_type") != "test":
                continue
            S = frozenset(int(i) + 1 for i in (a.get("objects_tested") or []))
            light = bool(a.get("machine_state_after"))
            tests.append((S, light))
        if not tests:
            continue
        chosen = set(mg.get("user_chosen_blickets") or [])
        truth = set(mg.get("true_blicket_indices") or [])
        rt = str(mg.get("rule_type", "")).lower()
        rule_acc = 1 if rt.startswith(rule) else 0
        all_obj = 1 if chosen == truth else 0
        full = 1 if (rule_acc and all_obj) else 0
        # Per-object accuracy = 1 - hamming/4 over the 4-object universe.
        per_obj = 1.0 - len(chosen ^ truth) / 4.0
        out.append((rule, trial_info_gain(tests), full, rule_acc, all_obj, per_obj,
                    trial_mean_remaining(tests),
                    trial_num_unique_tests(tests)))
    return out


def collect_llm():
    """agent -> list of (rule, info_gain, success)."""
    out = {m: [] for m in RUNS_BY_MODEL}
    for model, run_dirs in RUNS_BY_MODEL.items():
        for run_dir in run_dirs:
            run_dir_fl = run_dir.replace("/results/", "/results/final_llms/")
            d = run_dir_fl if os.path.isdir(run_dir_fl) else run_dir
            if not os.path.isdir(d):
                continue
            res = {}
            for fn in os.listdir(d):
                if fn.startswith("results_") and fn.endswith(".jsonl"):
                    with open(os.path.join(d, fn)) as f:
                        for line in f:
                            try:
                                r = json.loads(line.replace("NaN", "null"))
                            except json.JSONDecodeError:
                                continue
                            res[int(r["trial_idx"])] = r
            for fn in sorted(os.listdir(d)):
                if not fn.startswith("action_log_trial-"):
                    continue
                try:
                    ti = int(fn.split("trial-")[1].split("_")[0])
                except (ValueError, IndexError):
                    continue
                meta = res.get(ti, {})
                rule = str(meta.get("true_rule", "")).lower()
                if rule not in RULES:
                    continue
                with open(os.path.join(d, fn)) as f:
                    rows = []
                    for l in f:
                        try:
                            rows.append(json.loads(l))
                        except json.JSONDecodeError:
                            continue
                tests = parse_trial_tests(rows)
                ig = trial_info_gain(tests)
                succ = (bool(meta.get("all_nexioms_correct"))
                        and bool(meta.get("rule_type_correct")))
                out[model].append((rule, ig, 1 if succ else 0))
    return out


def collect_human():
    out = []
    if not os.path.exists(HUMAN_DATA_PATH):
        return out
    with open(HUMAN_DATA_PATH) as f:
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
        tests = []
        for a in (mg.get("user_test_actions") or []):
            if a.get("action_type") != "test":
                continue
            S = frozenset(int(i) + 1 for i in (a.get("objects_tested") or []))
            light = bool(a.get("machine_state_after"))
            tests.append((S, light))
        if not tests:
            continue
        chosen = set(mg.get("user_chosen_blickets") or [])
        truth = set(mg.get("true_blicket_indices") or [])
        rt = str(mg.get("rule_type", "")).lower()
        succ = (chosen == truth) and rt.startswith(rule)
        out.append((rule, trial_info_gain(tests), 1 if succ else 0))
    return out


def main():
    llm = collect_llm()
    human = collect_human()

    agents = [("human", human)] + [(m, llm[m]) for m in RUNS_BY_MODEL]

    print(f"{'agent':<22}{'rule':<14}{'n':>5}{'mean_ig':>10}"
          f"{'acc':>8}{'rho':>8}{'p':>10}")
    rows_for_plot = []  # (agent, rule, rho, n)
    for name, recs in agents:
        for rule in (None,) + RULES:
            sub = [(ig, s) for r, ig, s in recs if rule is None or r == rule]
            if len(sub) < 3:
                continue
            xs = [ig for ig, _ in sub]; ys = [s for _, s in sub]
            rho, p = spearman(xs, ys)
            label = "all" if rule is None else rule
            print(f"{name:<22}{label:<14}{len(sub):>5}"
                  f"{np.mean(xs):>10.2f}{np.mean(ys):>8.2f}"
                  f"{rho:>+8.2f}{p:>10.3g}")
            rows_for_plot.append((name, label, rho, len(sub), p))

    # Bar plot: 3 groups (all / conjunctive / disjunctive) on x; one bar per agent.
    cats = ["all", "conjunctive", "disjunctive"]
    agent_names = [a for a, _ in agents]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(cats))
    bar_w = 0.8 / len(agent_names)
    for i, name in enumerate(agent_names):
        vals = []; ns = []; ps = []
        for c in cats:
            row = next((r for r in rows_for_plot
                        if r[0] == name and r[1] == c), None)
            vals.append(row[2] if row else np.nan)
            ns.append(row[3] if row else 0)
            ps.append(row[4] if row else float("nan"))
        offset = (i - (len(agent_names) - 1) / 2) * bar_w
        color = HUMAN_COLOR if name == "human" else MODEL_COLOR.get(name, "#444")
        bars = ax.bar(x + offset, vals, width=bar_w, color=color,
                      edgecolor="white", linewidth=0.4, label=name)
        for j, b in enumerate(bars):
            if np.isnan(vals[j]):
                continue
            y = b.get_height()
            star = ""
            if not np.isnan(ps[j]):
                star = "*" if ps[j] < 0.05 else ""
            ax.text(b.get_x() + b.get_width() / 2,
                    y + (0.02 if y >= 0 else -0.04),
                    f"n={ns[j]}{star}",
                    ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=7, rotation=90, color="#222")

    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Spearman ρ  (info gain vs task success)")
    ax.set_ylim(-1, 1)
    ax.set_title("Per-trial info gain ↔ task success  (* = p<0.05, two-sided)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"spearman_infogain_success.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def plot_scatter_agent_rule():
    """Each point = (agent, rule). x = mean per-trial info gain, y = full-hyp acc.
    Reports Spearman correlation across all points."""
    llm = collect_llm()
    human = collect_human()
    agents = [("human", human)] + [(m, llm[m]) for m in RUNS_BY_MODEL]

    points = []  # (agent, rule, mean_ig, acc, n)
    for name, recs in agents:
        for rule in RULES:
            sub = [(ig, s) for r, ig, s in recs if r == rule]
            if not sub:
                continue
            xs = [ig for ig, _ in sub]; ys = [s for _, s in sub]
            points.append((name, rule, float(np.mean(xs)),
                           float(np.mean(ys)), len(sub)))

    xs = np.array([p[2] for p in points])
    ys = np.array([p[3] for p in points])
    rho, pv = spearman(xs, ys)
    # Pearson too, just for reference.
    pear = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 2 else float("nan")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    rule_marker = {"conjunctive": "o", "disjunctive": "^"}
    seen_agents = set(); seen_rules = set()
    for name, rule, mig, acc, n in points:
        color = HUMAN_COLOR if name == "human" else MODEL_COLOR.get(name, "#444")
        ax.scatter(mig, acc, s=120 + 6 * n, color=color,
                   marker=rule_marker[rule], edgecolor="white", linewidth=1.0,
                   alpha=0.9, zorder=3,
                   label=name if name not in seen_agents else None)
        seen_agents.add(name); seen_rules.add(rule)
        ax.annotate(f"{name}\n({rule[:4]}.)", (mig, acc),
                    xytext=(6, 6), textcoords="offset points", fontsize=8)

    # OLS fit line for visual reference.
    if len(xs) >= 2:
        m, b = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xx, m * xx + b, color="#888", linestyle="--", linewidth=1.0,
                zorder=1)

    ax.set_xlabel("mean per-trial information gain (bits)")
    ax.set_ylabel("full-hypothesis accuracy  (all objects ∧ rule)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Each point = (agent × rule).  "
                 f"Spearman ρ = {rho:+.2f} (p={pv:.2g}, n={len(points)});  "
                 f"Pearson r = {pear:+.2f}")
    ax.grid(alpha=0.3)

    # Two legends: agents (color) + rule (marker shape).
    from matplotlib.lines import Line2D
    agent_handles = [Line2D([0], [0], marker="o", linestyle="",
                            color=(HUMAN_COLOR if a == "human"
                                   else MODEL_COLOR.get(a, "#444")),
                            markersize=9, label=a)
                     for a in dict.fromkeys([p[0] for p in points])]
    rule_handles = [Line2D([0], [0], marker=rule_marker[r], linestyle="",
                           color="#444", markersize=9, label=r)
                    for r in RULES]
    leg1 = ax.legend(handles=agent_handles, title="agent",
                     loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     frameon=False, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=rule_handles, title="rule",
              loc="upper left", bbox_to_anchor=(1.01, 0.55),
              frameon=False, fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"spearman_scatter_agent_rule.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    print(f"\nspearman over (agent x rule) points: rho={rho:+.3f} p={pv:.3g} n={len(points)}")
    print(f"pearson r = {pear:+.3f}")


def _place_labels_no_overlap(ax, fig, items, fontsize=8):
    """Greedy label placement with leader lines, in display (pixel) coords.
    items: iterable of (x, y, text). Tries 8 candidate offsets; picks the first
    that doesn't overlap any earlier label or any data point."""
    fig.canvas.draw()
    # Process points in order: top-right (crowded) last so they get more room.
    items = sorted(items, key=lambda it: -(it[0] + it[1]))
    placed_bboxes = []
    pt_disp = [ax.transData.transform((x, y)) for x, y, _ in items]
    candidates_px = [(8, 8), (8, -14), (-50, 8), (-50, -14),
                     (10, 22), (10, -28), (-55, 22), (-55, -28),
                     (20, 0), (-65, 0), (0, 25), (0, -28)]
    for (x, y, text), (px, py) in zip(items, pt_disp):
        chosen = None
        for dx, dy in candidates_px:
            tx, ty = px + dx, py + dy
            # Estimate text bbox: width ~ 6.2 px per char at fs=8, height ~ 11.
            w = 6.2 * len(text); h = 11
            bbox = (tx, ty, tx + w, ty + h)
            ok = True
            for b in placed_bboxes:
                if not (bbox[2] < b[0] or bbox[0] > b[2]
                        or bbox[3] < b[1] or bbox[1] > b[3]):
                    ok = False; break
            if ok:
                # also avoid overlapping any data point disk (radius ~6)
                for (qx, qy) in pt_disp:
                    if (tx <= qx <= tx + w and ty <= qy <= ty + h):
                        ok = False; break
            if ok:
                chosen = (dx, dy, bbox); break
        if chosen is None:
            chosen = (candidates_px[0][0], candidates_px[0][1], None)
        dx, dy, bbox = chosen
        if bbox is not None:
            placed_bboxes.append(bbox)
        ax.annotate(text, xy=(x, y), xytext=(dx, dy),
                    textcoords="offset pixels", fontsize=fontsize, color="#333",
                    arrowprops=dict(arrowstyle="-", color="#999",
                                    linewidth=0.5, shrinkA=0, shrinkB=2))


METRIC_INFO = {
    "full":    ("full-hypothesis accuracy",  2),
    "rule":    ("rule-type accuracy",        3),
    "all_obj": ("all-objects accuracy",      4),
    "per_obj": ("per-object accuracy",       5),
}

XMETRIC_INFO = {
    "info_gain":      ("mean per-trial information gain (bits)", 1),
    "mean_remaining": ("mean #hypotheses remaining per test",    6),
    "num_unique":     ("number of unique tests per trial",       7),
}


def plot_scatter_simple(metric="full", xmetric="info_gain"):
    """Single scatter: 8 agents (6 models + avg human + top human) x 2 rules
    = 16 points. All dots green, orange regression line + 90% CI shading."""
    llm = collect_llm_metrics()
    human = collect_human_metrics()
    human_by_rule_full = load_human_tests_by_rule()

    ylabel, idx = METRIC_INFO[metric]
    xlabel, xidx = XMETRIC_INFO[xmetric]

    avg_pts = {}
    for rule in RULES:
        sub = [(rec[xidx], rec[idx]) for rec in human if rec[0] == rule]
        if sub:
            xs = [a for a, _ in sub]; ys = [b for _, b in sub]
            avg_pts[rule] = (float(np.mean(xs)), float(np.mean(ys)), len(sub))

    # Top human per rule (uses fully-correct trials by construction;
    # rule_acc and all_obj therefore equal 1.0).
    top_pts = {}
    for rule in RULES:
        entries = human_by_rule_full.get(rule, [])
        best = pick_best_humans(entries, max_tests=16)
        if best:
            x_fn = {
                "info_gain":      trial_info_gain,
                "mean_remaining": trial_mean_remaining,
                "num_unique":     trial_num_unique_tests,
            }[xmetric]
            xs_top = [x_fn(tests) for tests, _ in best]
            # for top human, all metrics are 1.0 by construction.
            top_pts[rule] = (float(np.mean(xs_top)), 1.0, len(best))

    points = []
    for m in RUNS_BY_MODEL:
        for rule in RULES:
            sub = [(rec[xidx], rec[idx]) for rec in llm[m] if rec[0] == rule]
            if not sub:
                continue
            xs = [a for a, _ in sub]; ys = [b for _, b in sub]
            points.append((m, rule, float(np.mean(xs)), float(np.mean(ys))))
    for rule, (x, y, _) in avg_pts.items():
        points.append(("avg human", rule, x, y))
    for rule, (x, y, _) in top_pts.items():
        points.append(("top human", rule, x, y))

    xs = np.array([p[2] for p in points])
    ys = np.array([p[3] for p in points])
    rho, pv = spearman(xs, ys)

    # OLS fit + bootstrap 90% CI band.
    rng = np.random.default_rng(0)
    n = len(xs)
    xx = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 100)
    boot = np.zeros((2000, len(xx)))
    for i in range(boot.shape[0]):
        idx = rng.integers(0, n, n)
        m, b = np.polyfit(xs[idx], ys[idx], 1)
        boot[i] = m * xx + b
    lo = np.percentile(boot, 5, axis=0)
    hi = np.percentile(boot, 95, axis=0)
    m, b = np.polyfit(xs, ys, 1)
    yy = m * xx + b

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(xx, lo, hi, color="#ff7f0e", alpha=0.20,
                    label="90% CI (bootstrap)")
    ax.plot(xx, yy, color="#ff7f0e", linewidth=2.0, label="OLS fit")
    ax.scatter(xs, ys, s=85, color="#2ca02c", edgecolor="white",
               linewidth=0.8, zorder=3, label=f"agent × rule (n={n})")
    SHORT = {
        "o4-mini": "o4-m",
        "gpt-5": "gpt5",
        "gpt-5-mini": "gpt5-m",
        "gemini-2.5-flash": "gem-f",
        "deepseek-reasoner": "dsk-r",
        "deepseek-chat": "dsk-c",
        "avg human": "human",
        "top human": "human*",
    }
    RSHORT = {"conjunctive": "c", "disjunctive": "d"}
    _place_labels_no_overlap(
        ax, fig,
        [(x, y, f"{SHORT.get(label, label)} ({RSHORT[rule]})")
         for label, rule, x, y in points])

    rho_txt = f"Spearman ρ = {rho:+.2f}\np = {pv:.2g}"
    ax.text(0.02, 0.98, rho_txt, transform=ax.transAxes,
            fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#888"))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{xlabel} vs. {ylabel}  (agents × rules)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    msuf = "" if metric == "full" else f"_{metric}"
    xsuf = "" if xmetric == "info_gain" else f"_x_{xmetric}"
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(OUT_DIR,
                            f"spearman_scatter_simple{msuf}{xsuf}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
    print(f"\nsimple scatter ({metric} vs {xmetric}): "
          f"n={n}, rho={rho:+.3f}, p={pv:.3g}")


if __name__ == "__main__":
    main()
    plot_scatter_agent_rule()
    for metric in ("full", "rule", "all_obj", "per_obj"):
        plot_scatter_simple(metric)
    plot_scatter_simple("full", "mean_remaining")
    plot_scatter_simple("full", "num_unique")
