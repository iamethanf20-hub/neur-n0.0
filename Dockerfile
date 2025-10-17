# Use Playwright’s Python base (Ubuntu jammy) that has the correct deps
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

# If your file is app.py with `app` instance:
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
# If it's main.py instead, swap to:
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
