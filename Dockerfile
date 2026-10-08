# ── Stage 1: build the React frontend ─────────────────────────────────
FROM node:20-slim AS web

WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build


# ── Stage 2: Python runtime (CPU-only torch) ──────────────────────────
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# libgomp1: required by torch / scikit-learn wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first (default Linux wheel pulls ~2.5 GB of CUDA libs), then the rest.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + bundled demo assets
COPY phyglobt/ phyglobt/
COPY server/ server/
COPY --from=web /build/dist/ web/dist/
COPY data/scenes/ data/scenes/
COPY checkpoints/ checkpoints/

ENV DATA_DIR=/app/data/scenes \
    CKPT_DIR=/app/checkpoints

# Hugging Face Spaces (Docker SDK) routes HTTPS traffic to this port.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
