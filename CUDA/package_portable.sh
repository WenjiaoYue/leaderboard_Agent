#!/usr/bin/env bash
# package_portable.sh
# Export a portable openclaw bundle from a running container.
# Keeps config + workspace/skills, removes history/state.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-cuda-openclaw}"
BUNDLE_DIR="${BUNDLE_DIR:-$SCRIPT_DIR/portable_bundle}"
BUNDLE_HOME="$BUNDLE_DIR/openclaw_home"

echo "[package] Source container: $CONTAINER_NAME"
echo "[package] Output dir: $BUNDLE_DIR"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[package] Container not running: $CONTAINER_NAME"
    exit 1
fi

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_HOME"

# 1) Copy user config and workspace from container
if docker exec "$CONTAINER_NAME" test -f /root/.openclaw/openclaw.json; then
    docker cp "$CONTAINER_NAME":/root/.openclaw/openclaw.json "$BUNDLE_HOME/openclaw.json"
else
    echo "[package] Missing /root/.openclaw/openclaw.json in container"
    exit 1
fi

if docker exec "$CONTAINER_NAME" test -d /root/.openclaw/workspace; then
    docker cp "$CONTAINER_NAME":/root/.openclaw/workspace "$BUNDLE_HOME/workspace"
else
    mkdir -p "$BUNDLE_HOME/workspace"
fi

# 2) Remove history/state directories (no history requirement)
rm -rf \
    "$BUNDLE_HOME/agents" \
    "$BUNDLE_HOME/logs" \
    "$BUNDLE_HOME/completions" \
    "$BUNDLE_HOME/canvas" \
    "$BUNDLE_HOME/workspace-minimax"

# 3) Optional: remove session dirs under workspace if present
find "$BUNDLE_HOME/workspace" -type d \( -name sessions -o -name session -o -name .sessions \) -prune -exec rm -rf {} + 2>/dev/null || true

# 4) Bundle runtime files needed on target machine
cp "$SCRIPT_DIR/Dockerfile" "$BUNDLE_DIR/Dockerfile"
cp "$SCRIPT_DIR/run.sh" "$BUNDLE_DIR/run.sh"
cp "$SCRIPT_DIR/entrypoint.sh" "$BUNDLE_DIR/entrypoint.sh"
cp "$SCRIPT_DIR/.env.template" "$BUNDLE_DIR/.env.template"

chmod +x "$BUNDLE_DIR/run.sh" "$BUNDLE_DIR/entrypoint.sh"

cat > "$BUNDLE_DIR/README_PORTABLE.txt" <<'EOF'
Portable bundle usage on target machine:

1) cd portable_bundle
2) cp .env.template .env  && fill in your values:
     MINIMAX_API_KEY
     HTTP_PROXY/HTTPS_PROXY (if behind corporate proxy)
3) Build and run (reads .env automatically):
     ./run.sh
4) Re-run without rebuilding:
     ./run.sh --no-build

Environment variables are injected at container startup via entrypoint.sh.
No need to manually edit openclaw.json.
EOF

echo "[package] Done. Bundle created at: $BUNDLE_DIR"
