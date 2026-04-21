# Base: NVIDIA CUDA runtime
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ── Build-time proxy (override via --build-arg, empty = no proxy) ────────────
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3-pip python3-dev python3-venv xz-utils git \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js v22.14 (openclaw requires >=22.12) ──────────────────────────────
RUN curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz \
        | tar -xJ --strip-components=1 -C /usr/local

# ── openclaw (npm global + Python package) ──────────────────────────────────
RUN npm install -g openclaw@2026.3.24
RUN pip3 install --no-cache-dir --break-system-packages openclaw==2026.3.20

# ── PyTorch + CUDA support (so venvs with --system-site-packages inherit it) ─
RUN pip3 install --no-cache-dir --break-system-packages \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir --break-system-packages \
        transformers accelerate auto-round datasets

# ── Make venvs inherit system torch+cuda by default ──────────────────────────
RUN pip3 config set global.break-system-packages true

# ── Bake in openclaw config + workspace (skills) ─────────────────────────────
COPY openclaw_home/ /root/.openclaw/

# ── Entrypoint: set up proxy at startup ───────────────────────────────────────
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /root
ENTRYPOINT ["entrypoint.sh"]
CMD ["bash"]
