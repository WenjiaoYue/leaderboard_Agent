#!/usr/bin/env bash
# run.sh — 一键部署 openclaw XPU 容器
# Usage:
#   ./run.sh              # 首次: build + run  (reads .env if present)
#   ./run.sh --no-build   # 已有镜像，只启动容器
#   CONTAINER_NAME=mybox ./run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load .env (if present) ────────────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    echo "[run.sh] Loading env from $ENV_FILE"
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
fi

IMAGE_NAME="${IMAGE_NAME:-xpu-openclaw:local}"
CONTAINER_NAME="${CONTAINER_NAME:-xpu-openclaw-new}"
BUILD_IMAGE=true

for arg in "$@"; do
    case "$arg" in
        --no-build) BUILD_IMAGE=false ;;
        *) echo "Usage: ./run.sh [--no-build]"; exit 1 ;;
    esac
done

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ "$BUILD_IMAGE" == "true" ]]; then
    echo "[run.sh] Building image: $IMAGE_NAME ..."
    docker build \
        --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
        --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
        -t "$IMAGE_NAME" "$SCRIPT_DIR"
    echo "[run.sh] Build done."
fi

# ── Remove old container with same name ───────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[run.sh] Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[run.sh] Starting container: $CONTAINER_NAME"

docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --device /dev/dri:/dev/dri \
    -e MINIMAX_API_KEY="${MINIMAX_API_KEY:-}" \
    -e HTTP_PROXY="${HTTP_PROXY:-}" \
    -e HTTPS_PROXY="${HTTPS_PROXY:-}" \
    -e http_proxy="${HTTP_PROXY:-}" \
    -e https_proxy="${HTTPS_PROXY:-}" \
    "$IMAGE_NAME" \
    sleep infinity

echo "[run.sh] Container $CONTAINER_NAME is running."
echo "[run.sh] Use: docker exec -it $CONTAINER_NAME bash"
