#!/usr/bin/env bash
# test_autoquant.sh — Run auto_quant inside an openClaw_config-deployed container
# Usage:
#   ./test_autoquant.sh
#   MODEL_ID=Qwen/Qwen3-0.6B ./test_autoquant.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load .env (same file as run.sh) ──────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

CONTAINER="${CONTAINER_NAME:?Set CONTAINER_NAME in .env}"
MODEL_ID="${MODEL_ID:?Set MODEL_ID in .env}"
SCHEME="${SCHEME:-W4A16}"
METHOD="${METHOD:-RTN}"
EXPORT_FORMAT="${EXPORT_FORMAT:-auto_round:auto_gptq}"
DEVICE="${DEVICE:-xpu}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/.openclaw/workspace/quantized}"
SKILL_PATH="/root/.openclaw/workspace/skills/auto_quant/SKILL.md"
TIMEOUT="${TIMEOUT:-7200}"
SESSION_KEY="test_autoquant_$$"

MODEL_SLUG="${MODEL_ID//\//_}"
FULL_OUTPUT="${OUTPUT_DIR}/${MODEL_SLUG}-${SCHEME}"

PROMPT="You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: ${SKILL_PATH}

Model: ${MODEL_ID}
Quantization: ${SCHEME} / ${METHOD}
Export format: ${EXPORT_FORMAT}
Output directory: ${FULL_OUTPUT}
Runtime device: ${DEVICE}:${DEVICE_INDEX}

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+xpu pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+xpu. Do NOT pip install torch inside the venv.

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

echo "=== Auto-Quant Test ==="
echo "Container : ${CONTAINER}"
echo "Model     : ${MODEL_ID}"
echo "Scheme    : ${SCHEME} / ${METHOD}"
echo "Output    : ${FULL_OUTPUT}"
echo "Device    : ${DEVICE}:${DEVICE_INDEX}"
echo "========================"

docker exec "${CONTAINER}" bash -c "
  export http_proxy=\$HTTP_PROXY https_proxy=\$HTTPS_PROXY && \
  openclaw agent --local \
    --session-id '${SESSION_KEY}' \
    --message '$(echo "$PROMPT" | sed "s/'/'\\''/g")' \
    --timeout ${TIMEOUT}
"

EXIT_CODE=$?

echo "========================"
echo "Exit code: ${EXIT_CODE}"

# Print summary if generated
docker exec "${CONTAINER}" bash -c "
  echo '--- summary.json ---'
  cat '${FULL_OUTPUT}/summary.json' 2>/dev/null || echo '(not found)'
"

echo "=== Auto-Quant Test Done ==="

