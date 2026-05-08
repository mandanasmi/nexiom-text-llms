#!/bin/bash
# Launch (or reuse) a local vLLM OpenAI-compatible server and run Nexiom trials.
# Usage:
#   ./run_nexiom_with_vllm.sh [num_trials] [served_model_name] [hf_model_path]
# Example:
#   ./run_nexiom_with_vllm.sh 10 Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_TRIALS="${1:-10}"
SERVED_MODEL_NAME="${2:-Qwen/Qwen2.5-7B-Instruct}"
HF_MODEL_PATH="${3:-$SERVED_MODEL_NAME}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

export VLLM_BASE_URL
export VLLM_API_KEY

# Prefer vllm on PATH; else project .venv; else `uv run vllm` (same env as `uv run python`).
if command -v vllm >/dev/null 2>&1; then
    VLLM_CMD=(vllm)
elif [ -x "${SCRIPT_DIR}/.venv/bin/vllm" ]; then
    VLLM_CMD=("${SCRIPT_DIR}/.venv/bin/vllm")
elif command -v uv >/dev/null 2>&1; then
    VLLM_CMD=(uv run vllm)
else
    VLLM_CMD=()
fi

wait_for_vllm() {
    echo "Waiting for vLLM at $VLLM_BASE_URL ..."
    # First startup can take several minutes (model download + load).
    for i in {1..300}; do
        if curl -s "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
            echo "vLLM is ready."
            return 0
        fi
        sleep 2
    done
    echo "Error: vLLM did not become ready in time."
    exit 1
}

if ! curl -s "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
    if [ ${#VLLM_CMD[@]} -eq 0 ]; then
        echo "Error: vLLM CLI not found."
        echo "Install with: uv pip install vllm  (or pip install vllm)"
        exit 1
    fi

    echo "Starting vLLM server for model: ${HF_MODEL_PATH}"
    "${VLLM_CMD[@]}" serve "${HF_MODEL_PATH}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        > "vllm_local_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm
else
    echo "vLLM is already running at ${VLLM_BASE_URL}"
fi

echo ""
echo "Running Nexiom trials with vLLM model '${SERVED_MODEL_NAME}' (n=${NUM_TRIALS})"
uv run python run_trials_nexiom.py \
    agent=nexiom_llm \
    agent.model="vllm/${SERVED_MODEL_NAME}" \
    num_trials="${NUM_TRIALS}"

if [ -n "${VLLM_PID:-}" ]; then
    echo ""
    echo "Stopping vLLM server (PID ${VLLM_PID}) ..."
    kill "${VLLM_PID}" 2>/dev/null || true
fi

echo "Done. Results are in results/."
