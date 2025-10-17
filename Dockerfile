# syntax=docker/dockerfile:1
FROM python:3.11-slim

# -------------------------------
# System deps (incl. browser libs)
# -------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl git ca-certificates \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libxcomposite1 libxrandr2 libxdamage1 libxfixes3 libpango-1.0-0 \
    libgbm1 libasound2 fonts-liberation libgtk-3-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------------------------------
# Python deps (single install step)
# -------------------------------
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Install Playwright deps + Chrome (prefer Chrome channel)
RUN python -m playwright install-deps \
 && python -m playwright install chrome

# -------------------------------
# App code
# -------------------------------
COPY . .

# -------------------------------
# Runtime env (no deprecated vars)
# -------------------------------
ENV HOST=0.0.0.0 \
    PORT=10000 \
    HEADLESS=1 \
    BROWSER_CHANNEL=chrome \
    HF_HOME=/data/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_NO_REMOTE_CODE=0 \
    PYTHONUNBUFFERED=1

# optional: create cache dirs
RUN mkdir -p /data/.cache/huggingface

EXPOSE 10000

# Use your actual module name (your file is app.py with `app`)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
