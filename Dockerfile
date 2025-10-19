# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.cache/huggingface \
    TRANSFORMERS_CACHE=/data/.cache/huggingface/transformers \
    HEADLESS=1 \
    BROWSER_CHANNEL=chrome \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_NO_REMOTE_CODE=0

# Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Install Chrome (base image has system deps already)
RUN playwright install chrome

# App code
COPY . .

# Persistent cache path
RUN mkdir -p /data/.cache/huggingface /data/.cache/huggingface/transformers

# Expose — Render will set $PORT at runtime; keep 8000 as default
EXPOSE 8000

# Healthcheck (optional but handy for local runs)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python - <<'PY' || exit 1
import os, urllib.request, json
url=f"http://127.0.0.1:{os.getenv('PORT','8000')}/healthz"
with urllib.request.urlopen(url, timeout=3) as r:
    d=json.load(r); assert 'model_loaded' in d
PY

# Start (use $PORT, fallback 8000). Use bash to expand env.
CMD ["bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
