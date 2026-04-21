#!/usr/bin/env bash
# test_autoeval.sh — Run auto_eval (with auto_quant if needed) inside an openClaw container
# Usage:
#   ./test_autoeval.sh                          # uses .env defaults
#   EVAL_TASKS=piqa,hellaswag ./test_autoeval.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load .env ────────────────────────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

CONTAINER="${CONTAINER_NAME:?Set CONTAINER_NAME in .env}"
MODEL_ID="${MODEL_ID:?Set MODEL_ID in .env}"
SCHEME="${SCHEME:-W4A16}"
METHOD="${METHOD:-RTN}"
EXPORT_FORMAT="${EXPORT_FORMAT:-auto_round:auto_gptq}"
DEVICE="${DEVICE:-cuda}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/.openclaw/workspace/quantized}"
TIMEOUT="${TIMEOUT:-7200}"

EVAL_TASKS="${EVAL_TASKS:-piqa}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

MODEL_SLUG="${MODEL_ID//\//_}"
FULL_OUTPUT="${OUTPUT_DIR}/${MODEL_SLUG}-${SCHEME}"

QUANT_SKILL_PATH="/root/.openclaw/workspace/skills/auto_quant/SKILL.md"
EVAL_SKILL_PATH="/root/.openclaw/workspace/skills/auto_eval/SKILL.md"

# ── Step 1: Check if auto_quant already done ─────────────────────────────────
echo "=== Auto-Eval Pipeline ==="
echo "Container : ${CONTAINER}"
echo "Model     : ${MODEL_ID}"
echo "Scheme    : ${SCHEME} / ${METHOD}"
echo "Eval tasks: ${EVAL_TASKS}"
echo "Output    : ${FULL_OUTPUT}"
echo "Device    : ${DEVICE}:${DEVICE_INDEX}"
echo "=========================="

QUANT_STATUS=$(docker exec "${CONTAINER}" bash -c "
  if [[ -f '${FULL_OUTPUT}/summary.json' ]]; then
    python3 -c \"import json; d=json.load(open('${FULL_OUTPUT}/summary.json')); print(d.get('status','unknown'))\"
  else
    echo 'missing'
  fi
" 2>/dev/null || echo "error")

echo "Quant status: ${QUANT_STATUS}"

if [[ "${QUANT_STATUS}" != "success" ]]; then
    echo ""
    echo ">>> Step 1/2: Running auto_quant first ..."
    echo ""

    QUANT_SESSION="autoeval_quant_$$"
    QUANT_PROMPT="You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: ${QUANT_SKILL_PATH}

Model: ${MODEL_ID}
Quantization: ${SCHEME} / ${METHOD}
Export format: ${EXPORT_FORMAT}
Output directory: ${FULL_OUTPUT}
Runtime device: ${DEVICE}:${DEVICE_INDEX}

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.

IMPORTANT — After quantization completes (success or failure), you MUST produce:

${FULL_OUTPUT}/summary.json — structured summary:
{
  \"model_id\": \"${MODEL_ID}\",
  \"scheme\": \"${SCHEME}\",
  \"method\": \"${METHOD}\",
  \"export_format\": \"${EXPORT_FORMAT}\",
  \"device\": \"${DEVICE}:${DEVICE_INDEX}\",
  \"output_dir\": \"${FULL_OUTPUT}\",
  \"status\": \"success\" or \"failed\",
  \"duration_seconds\": <float>,
  \"original_size_mb\": <float or null>,
  \"quantized_size_mb\": <float or null>,
  \"compression_ratio\": <float or null>,
  \"errors\": [<list of error strings>],
  \"solutions\": [<list of solution strings>],
  \"output_files\": [<list of file paths in output_dir>]
}

Write as valid JSON. If quantization fails, still write summary.json with status=failed."

    docker exec "${CONTAINER}" bash -c "
      export http_proxy=\$HTTP_PROXY https_proxy=\$HTTPS_PROXY && \
      openclaw agent --local \
        --session-id '${QUANT_SESSION}' \
        --message '$(echo "$QUANT_PROMPT" | sed "s/'/'\\''/g")' \
        --timeout ${TIMEOUT}
    "

    # Verify quant succeeded
    QUANT_STATUS=$(docker exec "${CONTAINER}" bash -c "
      if [[ -f '${FULL_OUTPUT}/summary.json' ]]; then
        python3 -c \"import json; d=json.load(open('${FULL_OUTPUT}/summary.json')); print(d.get('status','unknown'))\"
      else
        echo 'missing'
      fi
    " 2>/dev/null || echo "error")

    if [[ "${QUANT_STATUS}" != "success" ]]; then
        echo "ERROR: auto_quant failed or did not produce summary.json (status=${QUANT_STATUS})"
        docker exec "${CONTAINER}" bash -c "cat '${FULL_OUTPUT}/summary.json' 2>/dev/null || echo '(not found)'"
        exit 1
    fi

    echo ">>> auto_quant completed successfully."
else
    echo ">>> Quantized model already exists, skipping auto_quant."
fi

# ── Step 2: Run auto_eval ────────────────────────────────────────────────────
echo ""
echo ">>> Step 2/2: Running auto_eval ..."
echo ""

EVAL_SESSION="autoeval_eval_$$"
EVAL_PROMPT="You are an expert in evaluating quantized LLM models.
You MUST follow the skill instructions in: ${EVAL_SKILL_PATH}

Quantized model path: ${FULL_OUTPUT}
Evaluation tasks: ${EVAL_TASKS}
Batch size: ${EVAL_BATCH_SIZE}
Device: ${DEVICE}:${DEVICE_INDEX}

The quantized model was produced by auto_quant with scheme=${SCHEME}, export_format=${EXPORT_FORMAT}.
A venv may already exist at ${FULL_OUTPUT}/venv (created by auto_quant with --system-site-packages).

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If a venv already exists at ${FULL_OUTPUT}/venv, reuse it — just install lm_eval and vllm into it.

IMPORTANT — After evaluation completes, you MUST produce:

${FULL_OUTPUT}/accuracy.json — evaluation results:
{
  \"model_id\": \"${MODEL_ID}\",
  \"model_path\": \"${FULL_OUTPUT}\",
  \"scheme\": \"${SCHEME}\",
  \"device\": \"${DEVICE}:${DEVICE_INDEX}\",
  \"tasks\": {
    \"<task_name>\": {
      \"accuracy\": <float>,
      \"accuracy_stderr\": <float or null>
    }
  },
  \"status\": \"success\" or \"failed\",
  \"duration_seconds\": <float>,
  \"eval_framework\": \"lm_eval+vllm\" or \"lm_eval+hf\" or \"manual\",
  \"errors\": [<list of error strings if any>]
}

The accuracy values MUST be real numbers from actual evaluation runs.
Write as valid JSON. If evaluation fails, still write accuracy.json with status=failed."

docker exec "${CONTAINER}" bash -c "
  export http_proxy=\$HTTP_PROXY https_proxy=\$HTTPS_PROXY && \
  openclaw agent --local \
    --session-id '${EVAL_SESSION}' \
    --message '$(echo "$EVAL_PROMPT" | sed "s/'/'\\''/g")' \
    --timeout ${TIMEOUT}
"

EVAL_EXIT=$?

echo "=========================="
echo "Eval exit code: ${EVAL_EXIT}"
echo ""

# Print results
docker exec "${CONTAINER}" bash -c "
  echo '--- summary.json (quant) ---'
  cat '${FULL_OUTPUT}/summary.json' 2>/dev/null || echo '(not found)'
  echo ''
  echo '--- accuracy.json (eval) ---'
  cat '${FULL_OUTPUT}/accuracy.json' 2>/dev/null || echo '(not found)'
"

echo "=== Auto-Eval Pipeline Done ==="
