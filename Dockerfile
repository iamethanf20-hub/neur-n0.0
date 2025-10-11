# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System packages (minimal; playwright will bring the rest with --with-deps)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright + Chromium and all OS deps
RUN python -m playwright install --with-deps chromium

# App code
COPY . .

# Recommended envs
ENV HOST=0.0.0.0 \
    PORT=10000 \
    BROWSER=chromium \
    HEADLESS=1 \
    ENABLE_GPT_OSS=0 \
    TRANSFORMERS_CACHE=/models/hf \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

EXPOSE 10000

# Some hosts need --no-sandbox; keep it handy
# If Chromium fails to launch, set PLAYWRIGHT_ARGS="--no-sandbox"
ENV PLAYWRIGHT_ARGS=""

# Use shell form so $PORT expands
CMD sh -c 'uvicorn main:app --host $HOST --port ${PORT:-10000}'
