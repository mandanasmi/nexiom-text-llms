"""LLM-based classifier for rule_inference_response in results/final_llms,
plus humans' rule_hypothesis from results/human_data/human_data_active_explore.json.

Classifies each free-text rule description as conjunctive / disjunctive / unclear
using gpt-5-mini, then reports pre vs post accuracy and writes a cache + plot.
"""
import json
import os
import glob
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np

from lm_api import get_client

CACHE_PATH = 'results/final_llms/rule_inference_classifications.json'
HUMAN_PATH = 'results/human_data/human_data_active_explore.json'
CLASSIFIER_MODEL = 'gpt-5-mini'

SYSTEM = """You classify free-text descriptions of a causal activation rule.

The rule governs when a machine turns on given a set of objects placed on it.
- "disjunctive": the machine activates whenever ANY single qualifying object is present (one object alone is sufficient).
- "conjunctive": the machine activates only when MULTIPLE specific objects are present together (no single object alone is sufficient).
- "unclear": the description is too vague, contradictory, or absent to decide.

Examples:
"Object 2 alone turns on the machine; objects 1 and 3 don't." -> disjunctive
"The light turns on iff objects 1 and 3 are both on the machine." -> conjunctive
"At least one of objects 2 or 4 activates it." -> disjunctive
"All three nexioms must be present together." -> conjunctive
"The machine turns on when at least three objects are on it." -> conjunctive
"Any even-numbered object turns it on." -> disjunctive
"I think objects 2 and 4 must be active to turn the machine on." -> conjunctive
"Object 3 turns it on by itself." -> disjunctive

Respond with ONLY one word: conjunctive, disjunctive, or unclear."""


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)


def classify_one(client, text):
    r = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": text}],
        max_completion_tokens=2000,
    )
    out = (r.choices[0].message.content or '').strip().lower()
    for label in ('disjunctive', 'conjunctive', 'unclear'):
        if label in out:
            return label
    return 'unclear'


def gather_llm_trials():
    trials = []
    for f in sorted(glob.glob('results/final_llms/*/results_*.jsonl')):
        run_id = os.path.basename(os.path.dirname(f))
        for line in open(f):
            d = json.loads(line)
            text = d.get('rule_inference_response', '') or ''
            trials.append({
                'key': f"{run_id}::{d['model']}::{d['trial_idx']}",
                'group': d['model'],
                'true_rule': d['true_rule'],
                'rule_type_correct': bool(d.get('rule_type_correct')),
                'text': text,
            })
    return trials


def gather_human_trials():
    """Returns trials for `human` (all) and `top human` (fully-correct
    participants, one most-efficient per (rule, true_blicket combo))."""
    trials = []
    if not os.path.exists(HUMAN_PATH):
        return trials
    data = json.load(open(HUMAN_PATH))

    # First pass: build all human entries with metadata for top-human selection.
    entries = []
    for pid, v in data.items():
        mg = v.get('main_game')
        if not isinstance(mg, dict):
            continue
        true_rule = (mg.get('true_rule') or mg.get('rule') or '').lower()
        if true_rule not in ('conjunctive', 'disjunctive'):
            continue
        chosen_text = (mg.get('rule_type') or '').lower()
        if 'conjunctive' in chosen_text:
            chosen_label = 'conjunctive'
        elif 'disjunctive' in chosen_text:
            chosen_label = 'disjunctive'
        else:
            chosen_label = 'unclear'
        chosen_blickets = set(mg.get('user_chosen_blickets') or [])
        truth = frozenset(mg.get('true_blicket_indices') or [])
        rt = str(mg.get('rule_type', '')).lower()
        fully_correct = (chosen_blickets == set(truth)) and rt.startswith(true_rule)
        n_tests = sum(1 for a in (mg.get('user_test_actions') or [])
                      if a.get('action_type') == 'test')
        entries.append({
            'pid': pid, 'true_rule': true_rule, 'chosen_label': chosen_label,
            'fully_correct': fully_correct, 'n_tests': n_tests, 'truth': truth,
            'text': mg.get('rule_hypothesis', '') or '',
        })

    # All humans -> 'human' group (preserve current behavior).
    for e in entries:
        trials.append({
            'key': f"human::{e['pid']}",
            'group': 'human',
            'true_rule': e['true_rule'],
            'rule_type_correct': (e['chosen_label'] == e['true_rule']),
            'text': e['text'],
        })

    # Top human: among fully-correct, pick min n_tests per (rule, truth combo).
    by_combo = {}  # (rule, truth) -> entry
    for e in entries:
        if not e['fully_correct']:
            continue
        key = (e['true_rule'], e['truth'])
        prev = by_combo.get(key)
        if prev is None or e['n_tests'] < prev['n_tests']:
            by_combo[key] = e
    for e in by_combo.values():
        trials.append({
            'key': f"human::{e['pid']}",  # same key — classification reused
            'group': 'top human',
            'true_rule': e['true_rule'],
            'rule_type_correct': (e['chosen_label'] == e['true_rule']),
            'text': e['text'],
        })
    return trials


def classify_all(trials, cache, max_workers=8):
    todo = [t for t in trials if t['key'] not in cache and t['text'].strip()]
    print(f'Total trials: {len(trials)}; cached: {sum(1 for t in trials if t["key"] in cache)}; to classify: {len(todo)}')
    if not todo:
        return
    client = get_client(CLASSIFIER_MODEL)
    done = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(classify_one, client, t['text']): t for t in todo}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                cache[t['key']] = fut.result()
            except Exception as e:
                print(f"  err {t['key']}: {e}")
                cache[t['key']] = 'unclear'
            done += 1
            if done % 25 == 0 or done == len(todo):
                save_cache(cache)
                print(f'  {done}/{len(todo)} ({time.time()-t0:.1f}s)')
    save_cache(cache)


def report(trials, cache):
    stats = defaultdict(lambda: {'n': 0, 'pre': 0, 'post': 0, 'unclear': 0})
    by_rule = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'pre': 0, 'post': 0}))
    for t in trials:
        label = cache.get(t['key'], 'unclear')
        g, tr = t['group'], t['true_rule']
        s = stats[g]
        s['n'] += 1
        s['pre'] += int(label == tr)
        s['post'] += int(t['rule_type_correct'])
        if label == 'unclear':
            s['unclear'] += 1
        by_rule[g][tr]['n'] += 1
        by_rule[g][tr]['pre'] += int(label == tr)
        by_rule[g][tr]['post'] += int(t['rule_type_correct'])

    print(f"\n{'Group':<25} {'N':>4} {'Pre-acc':>8} {'Post-acc':>9} {'Gap':>7} {'Unclear':>8}")
    print('-' * 70)
    for g in sorted(stats, key=lambda k: -stats[k]['pre']/stats[k]['n']):
        s = stats[g]
        pa, qa = s['pre']/s['n'], s['post']/s['n']
        print(f"{g:<25} {s['n']:>4} {pa:>8.1%} {qa:>9.1%} {qa-pa:>+7.1%} {s['unclear']:>8}")
    return stats, by_rule


def plot(stats, by_rule):
    # Order: LLMs by pre-acc descending, then human at the end
    human_set = {'human', 'top human'}
    llm_groups = sorted([g for g in stats if g not in human_set],
                        key=lambda m: -stats[m]['pre']/stats[m]['n'])
    human_groups = [g for g in ['human', 'top human'] if g in stats]
    groups = llm_groups + human_groups
    labels = [g.title() if g in human_set else g for g in groups]
    pre_acc  = [stats[g]['pre']/stats[g]['n']  for g in groups]
    post_acc = [stats[g]['post']/stats[g]['n'] for g in groups]
    human_pre_colors  = {'human': '#e8a87c', 'top human': '#f4d35e'}
    human_post_colors = {'human': '#c44536', 'top human': '#b08900'}
    colors_pre  = ['#888fbf'] * len(llm_groups) + [human_pre_colors[g]  for g in human_groups]
    colors_post = ['#3b5998'] * len(llm_groups) + [human_post_colors[g] for g in human_groups]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    x = np.arange(len(groups)); w = 0.38
    ax.bar(x - w/2, pre_acc,  w, color=colors_pre,  edgecolor='black', linewidth=0.5,
           label='Pre-definition (LLM-judged free-text)')
    ax.bar(x + w/2, post_acc, w, color=colors_post, edgecolor='black', linewidth=0.5,
           label='Post-definition (forced choice)')
    for i, (p, q) in enumerate(zip(pre_acc, post_acc)):
        ax.text(i - w/2, p + 0.01, f'{p:.0%}', ha='center', fontsize=9)
        ax.text(i + w/2, q + 0.01, f'{q:.0%}', ha='center', fontsize=9)
        ax.text(i, max(p, q) + 0.06, f'{q-p:+.0%}', ha='center', fontsize=9,
                color='#444', fontweight='bold')
    if human_groups:
        ax.axvline(len(llm_groups) - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Rule-type accuracy'); ax.set_ylim(0, 1.15)
    ax.set_title(f'Rule identification: pre vs post (LLM judge: {CLASSIFIER_MODEL})')
    ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]; w = 0.2
    cp  = [by_rule[g]['conjunctive']['pre']/max(1, by_rule[g]['conjunctive']['n'])  for g in groups]
    cpo = [by_rule[g]['conjunctive']['post']/max(1, by_rule[g]['conjunctive']['n']) for g in groups]
    dp  = [by_rule[g]['disjunctive']['pre']/max(1, by_rule[g]['disjunctive']['n'])  for g in groups]
    dpo = [by_rule[g]['disjunctive']['post']/max(1, by_rule[g]['disjunctive']['n']) for g in groups]
    ax.bar(x - 1.5*w, cp,  w, label='Conj — Pre',  color='#f4b183')
    ax.bar(x - 0.5*w, cpo, w, label='Conj — Post', color='#c55a11')
    ax.bar(x + 0.5*w, dp,  w, label='Disj — Pre',  color='#a9d18e')
    ax.bar(x + 1.5*w, dpo, w, label='Disj — Post', color='#385723')
    if human_groups:
        ax.axvline(len(llm_groups) - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.1)
    ax.set_title('Pre vs Post accuracy split by true rule type')
    ax.legend(loc='lower right', fontsize=8); ax.grid(axis='y', alpha=0.3)

    n = sum(s['n'] for s in stats.values())
    plt.suptitle(f'Pre- vs Post-definition rule identification (final_llms + humans, N={n}, judge={CLASSIFIER_MODEL})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = 'results/final_llms/rule_inference_gap_llmjudge.png'
    out_svg = 'results/final_llms/rule_inference_gap_llmjudge.svg'
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.savefig(out_svg, bbox_inches='tight')
    print(f'\nSaved: {out_png}\nSaved: {out_svg}')


if __name__ == '__main__':
    cache = load_cache()
    trials = gather_llm_trials() + gather_human_trials()
    classify_all(trials, cache)
    stats, by_rule = report(trials, cache)
    plot(stats, by_rule)
