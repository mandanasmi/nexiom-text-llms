"""Export per-trial rule_inference text with LLM judge label and regex label
(plus the matched regex patterns) to a CSV."""
import csv
import glob
import json
import os
import re

CACHE_PATH = 'results/final_llms/rule_inference_classifications.json'
HUMAN_PATH = 'results/human_data/human_data_active_explore.json'
OUT = 'results/final_llms/rule_inference_judgements.csv'

DISJ_PATTERNS = [
    r'\bat least one\b', r'\bany (?:one )?of\b', r'\beither\b', r'\bone or more\b',
    r'\bif any\b', r'\balone\b', r'\bregardless of (?:the )?other',
    r'\bregardless of what', r'\bby itself\b', r'\bon its own\b',
    r'\bsingle (?:object|nexiom)\b', r'\biff you place an?\b',
    r'\biff (?:object|any)\b', r'\bonly object \d+ is a nexiom\b',
    r'\bonly the even', r'\bonly the odd', r'\bany even', r'\bany odd',
]
CONJ_PATTERNS = [
    r'\ball (?:of )?(?:them|the)\b', r'\bboth\b', r'\ball nexioms?\b',
    r'\bonly (?:when|if) all\b', r'\btogether\b', r'\beach of\b',
    r'\brequires? all\b', r'\bwhen all\b', r'\bsimultaneously\b',
    r'\b(?:3\+?|three|2\+?|two)[- ]?(?:object )?threshold\b',
    r'\bat least (?:two|three|2|3)\b',
    r'\ball (?:three|four|2|3|4) (?:objects|nexioms)\b',
    r'\bcontribute equally\b', r'\bmust be (?:all|both)\b',
    r'\bevery (?:nexiom|object)\b',
]


def regex_classify(text):
    if not text:
        return 'unknown', [], []
    t = text.lower()
    matched_d = [p for p in DISJ_PATTERNS if re.search(p, t)]
    matched_c = [p for p in CONJ_PATTERNS if re.search(p, t)]
    d, c = len(matched_d), len(matched_c)
    if d > c: label = 'disjunctive'
    elif c > d: label = 'conjunctive'
    elif d == c == 0: label = 'unknown'
    else: label = 'ambiguous'
    return label, matched_d, matched_c


def main():
    cache = json.load(open(CACHE_PATH))
    rows = []

    for f in sorted(glob.glob('results/final_llms/*/results_*.jsonl')):
        run_id = os.path.basename(os.path.dirname(f))
        for line in open(f):
            d = json.loads(line)
            key = f"{run_id}::{d['model']}::{d['trial_idx']}"
            text = d.get('rule_inference_response', '') or ''
            rgx_label, md, mc = regex_classify(text)
            rows.append({
                'key': key,
                'group': d['model'],
                'trial_idx': d['trial_idx'],
                'true_rule': d['true_rule'],
                'rule_inference_text': text,
                'llm_judge_label': cache.get(key, ''),
                'llm_judge_correct': cache.get(key) == d['true_rule'],
                'regex_label': rgx_label,
                'regex_correct': rgx_label == d['true_rule'],
                'matched_disj_patterns': ' | '.join(md),
                'matched_conj_patterns': ' | '.join(mc),
                'rule_type_response_post': d.get('rule_type_response', ''),
                'rule_type_correct_post': bool(d.get('rule_type_correct')),
            })

    if os.path.exists(HUMAN_PATH):
        hd = json.load(open(HUMAN_PATH))
        for pid, v in hd.items():
            mg = v.get('main_game')
            if not isinstance(mg, dict):
                continue
            tr = (mg.get('true_rule') or mg.get('rule') or '').lower()
            if tr not in ('conjunctive', 'disjunctive'):
                continue
            text = mg.get('rule_hypothesis', '') or ''
            rgx_label, md, mc = regex_classify(text)
            chosen = (mg.get('rule_type') or '').lower()
            if 'conjunctive' in chosen: post = 'conjunctive'
            elif 'disjunctive' in chosen: post = 'disjunctive'
            else: post = ''
            key = f"human::{pid}"
            rows.append({
                'key': key,
                'group': 'human',
                'trial_idx': '',
                'true_rule': tr,
                'rule_inference_text': text,
                'llm_judge_label': cache.get(key, ''),
                'llm_judge_correct': cache.get(key) == tr,
                'regex_label': rgx_label,
                'regex_correct': rgx_label == tr,
                'matched_disj_patterns': ' | '.join(md),
                'matched_conj_patterns': ' | '.join(mc),
                'rule_type_response_post': mg.get('rule_type', ''),
                'rule_type_correct_post': post == tr,
            })

    fields = ['key', 'group', 'trial_idx', 'true_rule', 'rule_inference_text',
              'llm_judge_label', 'llm_judge_correct',
              'regex_label', 'regex_correct',
              'matched_disj_patterns', 'matched_conj_patterns',
              'rule_type_response_post', 'rule_type_correct_post']
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    # Header file with the regex pattern lists for reference
    pat_path = OUT.replace('.csv', '_patterns.txt')
    with open(pat_path, 'w') as f:
        f.write('# Heuristic regex patterns used by the regex classifier\n\n')
        f.write('## DISJUNCTIVE patterns\n')
        for p in DISJ_PATTERNS:
            f.write(f'  {p}\n')
        f.write('\n## CONJUNCTIVE patterns\n')
        for p in CONJ_PATTERNS:
            f.write(f'  {p}\n')

    print(f'Wrote {len(rows)} rows -> {OUT}')
    print(f'Wrote regex pattern reference -> {pat_path}')


if __name__ == '__main__':
    main()
