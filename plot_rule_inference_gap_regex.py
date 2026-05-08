"""Improved regex classifier for rule_inference text, calibrated against the
gpt-5-mini LLM-judge results. Reports pre vs post accuracy and writes a plot
using ONLY the regex labels (no LLM calls)."""
import json
import os
import glob
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# --- Regex patterns (calibrated by inspecting LLM-judge disagreements) ---

DISJ_PATTERNS = [
    # Logical "or" between objects / numbers
    r'\bobject\s+\d+\s+or\s+object\s+\d+',
    r'\b\d+\s+or\s+\d+\s+(?:is|are|on|placed|are options)',
    r'\(or both\)', r'\bor both\b', r'\band/or\b',
    # Quantifier phrases
    r'\bat least one\b', r'\bany (?:one )?of\b', r'\bone or more\b',
    r'\beither\b', r'\bif any\b',
    # Single-object sufficiency
    r'\balone\b', r'\bby itself\b', r'\bon its own\b',
    r'\bregardless of (?:the )?other', r'\bregardless of what',
    r'\beach (?:one )?(?:turn|light|activat|make)',
    r'\beach is a nexiom\b', r'\bseparately or together\b',
    # Single-object exclusivity
    r'\bonly object \d+\b.{0,40}\b(?:turn|light|activat|make|is a nexiom|ever)',
    r'\bobject \d+ only\b', r'\bby object \d+ only\b',
    # Property-based descriptions (each qualifying object alone activates)
    r'\b(?:an?\s+)?even[- ]numbered\b', r'\b(?:an?\s+)?odd[- ]numbered\b',
    r'\bprime[- ]numbered\b', r'\bperfect[- ]?square',
    r'\bonly the (?:even|odd|prime)\b',
    r'\beven numbers?\b', r'\bodd numbers?\b',
    r'\bgreater than \d', r'\bnumbered (?:two or less|less than)',
    r'\bhigher than \d',
    # "Lights up exactly for" / "lights when X is on it"
    r'\blights? up exactly for\b',
    r'\bturns? on if (?:and only if )?object \d+ is\b',
    r'\bturns? on (?:if|when|whenever) (?:object|an?) ',
    r'\biff you place an?\b', r'\biff (?:object|any)\b',
    r'\bplacing (?:object \d+|an?)\b.{0,40}\bturn',
    r'\bputting (?:object \d+|an?)\b.{0,40}\bturn',
    # "Nexiom-as-property" phrasing (any nexiom alone suffices)
    r'\bif (?:a|any) nexiom is on\b', r'\bwhen (?:a|any) nexiom is\b',
    r'\bwhenever (?:a|any) nexiom is\b',
    r'\bnexiom is placed on (?:the )?(?:platform|machine)\b',
    r'\bany objects? (?:in|on) the machine that are nexioms\b',
    r'\bturned on by\b', r'\bswitches? on (?:when|whenever)\b',
    r'\bif the object is a nexiom\b', r'\bwhen the object is a nexiom\b',
    r'\bturn(?:s)? on the machine\.?\s*the other', r'\bdo not (?:activate|turn)',
    r'\bobjects? \d+ and \d+ turn on the machine\b',
]

CONJ_PATTERNS = [
    # Direct "must / both"
    r'\bboth\b', r'\bmust both\b', r'\bare both on\b',
    r'\bboth (?:on (?:the )?machine|present|on it)\b',
    r'\bat the same time\b', r'\bsimultaneously\b', r'\btogether\b',
    r'\bpaired with\b',
    # Negation of single-object sufficiency
    r'\bno single (?:object|nexiom)\b',
    r'\bno individual (?:object|nexiom)\b',
    r'\bneither (?:alone|one alone|on its own)\b',
    r'\bnever lit\b', r'\bnever lights?\b',
    r'\bdoes not turn on the machine but\b',
    r'\bhas to be on .{0,40}with\b',
    # All / every
    r'\ball (?:of )?(?:them|the)\b', r'\ball nexioms?\b',
    r'\ball \d+ (?:objects|nexioms)\b',
    r'\b(?:are|is) all on\b', r'\bare all (?:on|present)\b',
    r'\ball objects? need\b', r'\bonly (?:when|if) all\b', r'\bwhen all\b',
    r'\brequires? all\b', r'\beach of\b', r'\bevery (?:nexiom|object)\b',
    r'\ball (?:three|four|2|3|4) (?:objects|nexioms)\b',
    # Combinations / pairs
    r'\bthe (?:exact )?pair of\b', r'\bthe pair\b.{0,30}\btogether\b',
    r'\bonly the pair\b', r'\bthe combination of\b',
    r'\bonly (?:the )?combination\b', r'\blogical and\b', r'\b\(and\)\b',
    r'\bmust be (?:all|both|present|sequential)\b',
    r'\b(?:two|three|2|3) other objects? (?:on|with)\b',
    r'\bwith two other objects\b',
    # Threshold (multiple required)
    r'\bat least (?:two|three|2|3)\b',
    r'\bat least object \d+ and object \d+\b',
    r'\b(?:3\+?|three|2\+?|two)[- ]?(?:object )?threshold\b',
    r'\bcontribute equally\b',
    # "object N and object M" with conjunction marker
    r'\bobject\s+\d+\s+and\s+object\s+\d+\b.{0,80}\b(?:both|together|simultaneously|at the same time|all on|must)',
    r'\bobjects?\s+\d+\s+and\s+\d+\s+(?:must|are both|together|are needed)',
    r'\b(?:1|2|3|4)\s*\+\s*(?:1|2|3|4)\b',
]


def classify(text):
    if not text:
        return 'unknown'
    t = text.lower()
    d = sum(1 for p in DISJ_PATTERNS if re.search(p, t))
    c = sum(1 for p in CONJ_PATTERNS if re.search(p, t))
    if d > c: return 'disjunctive'
    if c > d: return 'conjunctive'
    if d == c == 0: return 'unknown'
    return 'ambiguous'


def gather():
    """Return list of trial dicts with text, true_rule, group, post correctness."""
    out = []
    for f in sorted(glob.glob('results/final_llms/*/results_*.jsonl')):
        run_id = os.path.basename(os.path.dirname(f))
        for line in open(f):
            d = json.loads(line)
            out.append({
                'key': f"{run_id}::{d['model']}::{d['trial_idx']}",
                'group': d['model'],
                'true_rule': d['true_rule'],
                'rule_type_correct': bool(d.get('rule_type_correct')),
                'text': d.get('rule_inference_response', '') or '',
            })
    hd_path = 'results/human_data/human_data_active_explore.json'
    if not os.path.exists(hd_path):
        return out
    hd = json.load(open(hd_path))
    entries = []
    for pid, v in hd.items():
        mg = v.get('main_game')
        if not isinstance(mg, dict): continue
        tr = (mg.get('true_rule') or mg.get('rule') or '').lower()
        if tr not in ('conjunctive', 'disjunctive'): continue
        chosen_text = (mg.get('rule_type') or '').lower()
        chosen = ('conjunctive' if 'conjunctive' in chosen_text else
                  'disjunctive' if 'disjunctive' in chosen_text else 'unclear')
        chosen_b = set(mg.get('user_chosen_blickets') or [])
        truth = frozenset(mg.get('true_blicket_indices') or [])
        rt = str(mg.get('rule_type','')).lower()
        full_correct = (chosen_b == set(truth)) and rt.startswith(tr)
        n_tests = sum(1 for a in (mg.get('user_test_actions') or [])
                      if a.get('action_type') == 'test')
        entries.append({'pid': pid, 'tr': tr, 'chosen': chosen,
                        'full_correct': full_correct, 'n_tests': n_tests,
                        'truth': truth, 'text': mg.get('rule_hypothesis','') or ''})
    for e in entries:
        out.append({'key': f"human::{e['pid']}", 'group': 'human',
                    'true_rule': e['tr'],
                    'rule_type_correct': (e['chosen'] == e['tr']),
                    'text': e['text']})
    by_combo = {}
    for e in entries:
        if not e['full_correct']: continue
        k = (e['tr'], e['truth'])
        if k not in by_combo or e['n_tests'] < by_combo[k]['n_tests']:
            by_combo[k] = e
    for e in by_combo.values():
        out.append({'key': f"human::{e['pid']}", 'group': 'top human',
                    'true_rule': e['tr'],
                    'rule_type_correct': (e['chosen'] == e['tr']),
                    'text': e['text']})
    return out


def main():
    trials = gather()
    stats = defaultdict(lambda: {'n':0,'pre':0,'post':0,'unknown':0,'amb':0})
    by_rule = defaultdict(lambda: defaultdict(lambda: {'n':0,'pre':0,'post':0}))
    for t in trials:
        label = classify(t['text'])
        g, tr = t['group'], t['true_rule']
        s = stats[g]
        s['n'] += 1
        s['pre'] += int(label == tr)
        s['post'] += int(t['rule_type_correct'])
        if label == 'unknown': s['unknown'] += 1
        if label == 'ambiguous': s['amb'] += 1
        by_rule[g][tr]['n'] += 1
        by_rule[g][tr]['pre'] += int(label == tr)
        by_rule[g][tr]['post'] += int(t['rule_type_correct'])

    print(f"\n{'Group':<25} {'N':>4} {'Pre-acc':>8} {'Post-acc':>9} {'Gap':>7} {'Unk':>4} {'Amb':>4}")
    print('-'*72)
    for g in sorted(stats, key=lambda k: -stats[k]['pre']/stats[k]['n']):
        s = stats[g]; pa, qa = s['pre']/s['n'], s['post']/s['n']
        print(f"{g:<25} {s['n']:>4} {pa:>8.1%} {qa:>9.1%} {qa-pa:>+7.1%} {s['unknown']:>4} {s['amb']:>4}")

    # --- Plot ---
    human_set = {'human', 'top human'}
    llm_groups = sorted([g for g in stats if g not in human_set],
                        key=lambda m: -stats[m]['pre']/stats[m]['n'])
    human_groups = [g for g in ['human','top human'] if g in stats]
    groups = llm_groups + human_groups
    labels = [g.title() if g in human_set else g for g in groups]
    pre_acc  = [stats[g]['pre']/stats[g]['n']  for g in groups]
    post_acc = [stats[g]['post']/stats[g]['n'] for g in groups]
    h_pre  = {'human':'#e8a87c','top human':'#f4d35e'}
    h_post = {'human':'#c44536','top human':'#b08900'}
    colors_pre  = ['#888fbf']*len(llm_groups) + [h_pre[g]  for g in human_groups]
    colors_post = ['#3b5998']*len(llm_groups) + [h_post[g] for g in human_groups]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    x = np.arange(len(groups)); w = 0.38
    ax.bar(x-w/2, pre_acc,  w, color=colors_pre,  edgecolor='black', linewidth=0.5,
           label='Pre-definition (regex-judged free-text)')
    ax.bar(x+w/2, post_acc, w, color=colors_post, edgecolor='black', linewidth=0.5,
           label='Post-definition (forced choice)')
    for i,(p,q) in enumerate(zip(pre_acc, post_acc)):
        ax.text(i-w/2, p+0.01, f'{p:.0%}', ha='center', fontsize=9)
        ax.text(i+w/2, q+0.01, f'{q:.0%}', ha='center', fontsize=9)
        ax.text(i, max(p,q)+0.06, f'{q-p:+.0%}', ha='center', fontsize=9,
                color='#444', fontweight='bold')
    if human_groups:
        ax.axvline(len(llm_groups)-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Rule-type accuracy'); ax.set_ylim(0, 1.15)
    ax.set_title('Rule identification: pre vs post (improved regex classifier)')
    ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]; w = 0.2
    cp  = [by_rule[g]['conjunctive']['pre']/max(1, by_rule[g]['conjunctive']['n'])  for g in groups]
    cpo = [by_rule[g]['conjunctive']['post']/max(1, by_rule[g]['conjunctive']['n']) for g in groups]
    dp  = [by_rule[g]['disjunctive']['pre']/max(1, by_rule[g]['disjunctive']['n'])  for g in groups]
    dpo = [by_rule[g]['disjunctive']['post']/max(1, by_rule[g]['disjunctive']['n']) for g in groups]
    ax.bar(x-1.5*w, cp,  w, label='Conj — Pre',  color='#f4b183')
    ax.bar(x-0.5*w, cpo, w, label='Conj — Post', color='#c55a11')
    ax.bar(x+0.5*w, dp,  w, label='Disj — Pre',  color='#a9d18e')
    ax.bar(x+1.5*w, dpo, w, label='Disj — Post', color='#385723')
    if human_groups:
        ax.axvline(len(llm_groups)-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.1)
    ax.set_title('Pre vs Post accuracy split by true rule type')
    ax.legend(loc='lower right', fontsize=8); ax.grid(axis='y', alpha=0.3)

    n = sum(s['n'] for s in stats.values())
    plt.suptitle(f'Pre- vs Post-definition rule identification (improved regex, N={n})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = 'results/final_llms/rule_inference_gap_regex.png'
    out_svg = 'results/final_llms/rule_inference_gap_regex.svg'
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.savefig(out_svg, bbox_inches='tight')
    print(f'\nSaved: {out_png}\nSaved: {out_svg}')


if __name__ == '__main__':
    main()
