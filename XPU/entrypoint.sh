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

# ── 3. Sync runtime API key into openclaw auth profiles ─────────────────────
AUTH_PROFILES="/root/.openclaw/agents/main/agent/auth-profiles.json"
if [[ -n "${MINIMAX_API_KEY:-}" && -f "$AUTH_PROFILES" ]]; then
    python3 - "$AUTH_PROFILES" <<'PY'
import json
import os
import sys

path = sys.argv[1]
key = os.environ["MINIMAX_API_KEY"]

with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

profiles = data.setdefault("profiles", {})
for name in ("minimax-global", "minimax:cn", "minimax:global"):
    profile = profiles.get(name)
    if isinstance(profile, dict) and profile.get("provider") == "minimax":
        profile["key"] = key

with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
    echo "[entrypoint] MINIMAX_API_KEY synced to $AUTH_PROFILES"
fi

# Run whatever was passed as CMD (default: bash)
exec "$@"
