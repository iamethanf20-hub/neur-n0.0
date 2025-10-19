# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy

WORKDIR /app

# ---- Environment setup ----
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.cache/huggingface \
    TRANSFORMERS_CACHE=/data/.cache/huggingface/transformers \
    HEADLESS=1 \
    BROWSER_CHANNEL=chrome \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_NO_REMOTE_CODE=0

# ---- Install dependencies ----
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Install Chrome for Playwright
RUN playwright install chrome

# ---- Copy application ----
COPY . .

# Persistent Hugging Face cache dirs
RUN mkdir -p /data/.cache/huggingface /data/.cache/huggingface/transformers

# Expose default port (Render sets $PORT automatically)
EXPOSE 8000

# ---- Launch FastAPI ----
CMD ["bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
