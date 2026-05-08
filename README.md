# Nexiom: Text-based Causal Discovery with LLMs

Code for evaluating large language models on the **Nexiom** task — a text-based
causal discovery game in which an agent interacts with a set of objects and a
machine, must infer which objects are "nexioms" (causally relevant), and must
identify the rule (conjunctive or disjunctive) that governs the machine.

This repository contains the final experimental setup used to evaluate six LLMs
on the active-exploration + Q&A protocol.

## Installation

```bash
conda create -n nexiom python=3.10
conda activate nexiom
pip install -r requirements.txt
```

## API access

Configure one or more of the following providers depending on which models you
want to run.

- **OpenAI**: `export OPENAI_API_KEY="..."`
- **DeepSeek**: `export DEEPSEEK_API_KEY="..."`
- **Gemini** (OpenAI-compatible endpoint):
  `export GOOGLE_API_KEY="..."` or `export GEMINI_API_KEY="..."`
- **vLLM** (local / remote OpenAI-compatible server, used for open-weight models):
  - Start a vLLM server (default URL: `http://localhost:8000/v1`)
  - Use model names prefixed with `vllm/`, e.g. `vllm/Qwen/Qwen2.5-7B-Instruct`
  - `export VLLM_BASE_URL="http://localhost:8000/v1"`
  - `export VLLM_API_KEY="EMPTY"`

Static price estimates live in `lm_api.py`; verify against current provider pricing.

## Running trials

Single trial with a hosted model:

```bash
HYDRA_FULL_ERROR=1 python run_trials_nexiom.py \
  agent=nexiom_llm \
  agent.model="gpt-4o-2024-08-06" \
  agent.temperature=1.0 \
  num_trials=1 max_actions_per_trial=16 \
  env_kwargs.num_objects=4 env_kwargs.num_blickets=2 \
  env_kwargs.rule="alternate" \
  seed=0
```

Sweep over rules and seeds:

```bash
HYDRA_FULL_ERROR=1 python run_trials_nexiom.py \
  agent=nexiom_llm \
  use_threadpool=True tp_max_workers=16 \
  num_trials=32 max_actions_per_trial=16 \
  agent.model="deepseek-chat" \
  env_kwargs.rule="conjunctive","disjunctive" \
  seed=0,1,2,3 -m
```

Local open-weight model via vLLM:

```bash
./run_nexiom_with_vllm.sh 32 Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct
```

Outputs are written to `results/<YYYYMMDD_HHMMSS>/` (single run) or
`exp_output/<date>/<time>/` (Hydra multirun). Both are gitignored.

## Prompt-variant ablations

`agent/nexiom_llm.yaml` exposes A/B knobs under `agent.prompt_variant`:

- `option_order`: `conj_first` | `disj_first`
- `comprehension_rule`: `default` | `disjunctive` | `conjunctive` | `alternate` | `random`
- `comprehension_phrasing`: `at_least_one` | `neutral`
- `reasoning_cue`: `true` | `false`
- `output_format`: `text` | `json`

Override from the command line, e.g.
`agent.prompt_variant.comprehension_phrasing=neutral`.

## Analysis

Aggregate raw run outputs into DuckDB databases for analysis:

```bash
python process_hypothesis_exps.py \
  results/*/results.jsonl \
  --output_dir processed_output \
  --max_workers 4
```

Final figures are produced by the `plot_*.py` scripts at the repo root, plus
`classify_rule_inference.py` and `export_rule_inference_csv.py` for the
rule-inference analyses.

## Repository layout

```
env/blicket_text.py        Nexiom environment
agent/agents.py            Base agent classes and shared prompt scaffolding
agent/nexiom_llm.py        LLM agent used in all reported experiments
agent/prompt_variants.py   Prompt-variant resolution (ablations)
lm_api.py                  Provider-agnostic LLM client (OpenAI/DeepSeek/Gemini/vLLM)
run_trials_nexiom.py       Hydra entry point for trials
plot_*.py                  Final figure scripts
```
