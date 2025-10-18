# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Install Chrome via Playwright (no install-deps needed on this image)
RUN playwright install chrome

# App code
COPY . .

# Runtime env
ENV HOST=0.0.0.0 \
    PORT=10000 \
    HEADLESS=1 \
    BROWSER_CHANNEL=chrome \
    HF_HOME=/data/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_NO_REMOTE_CODE=0 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data/.cache/huggingface
EXPOSE 10000

# Launch the FastAPI application defined in main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
