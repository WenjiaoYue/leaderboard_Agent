#!/usr/bin/env bash
# entrypoint.sh — Set up runtime proxy env, then exec CMD.
set -euo pipefail

# ── 1. Persist proxy so ALL shell types inherit it (interactive + docker exec) ─
if [[ -n "${HTTP_PROXY:-}" ]]; then
    cat > /etc/profile.d/proxy.sh <<EOF
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY:-$HTTP_PROXY}"
export HTTP_PROXY="${HTTP_PROXY}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
EOF
    # Also write to /etc/environment for non-login shells (docker exec bash -c)
    cat > /etc/environment <<EOF
http_proxy=${HTTP_PROXY}
https_proxy=${HTTPS_PROXY:-$HTTP_PROXY}
HTTP_PROXY=${HTTP_PROXY}
HTTPS_PROXY=${HTTPS_PROXY:-$HTTP_PROXY}
EOF
    echo "[entrypoint] Proxy written to /etc/profile.d/ and /etc/environment"
fi

# ── 2. Set proxy for current exec chain ──────────────────────────────────────
export http_proxy="${HTTP_PROXY:-$http_proxy}"
export https_proxy="${HTTPS_PROXY:-$https_proxy}"

# Run whatever was passed as CMD (default: bash)
exec "$@"
