# syntax=docker/dockerfile:1
FROM python:3.11-slim

# =========================================================
# 🧩 System dependencies (minimal base + chromium runtime libs)
# =========================================================
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl git ca-certificates \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libxcomposite1 libxrandr2 libxdamage1 libxfixes3 libpango-1.0-0 \
    libgbm1 libasound2 fonts-liberation libgtk-3-0 \
 && rm -rf /var/lib/apt/lists/*

# =========================================================
# 📂 Working directory
# =========================================================
WORKDIR /app

# =========================================================
# 📦 Python dependencies
# =========================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Ensure a recent Hugging Face stack for custom arch `gpt_oss`
RUN pip install --no-cache-dir \
    "transformers>=4.44.0" \
    "huggingface_hub>=0.24.0" \
    "accelerate>=0.31.0" \
    safetensors einops sentencepiece

# (Optional for 4-bit): uncomment if you plan to use it
# RUN pip install --no-cache-dir bitsandbytes

# =========================================================
# 🌐 Install Playwright + Chromium manually (safe for Render)
# =========================================================
RUN pip install --no-cache-dir playwright && python -m playwright install chromium

# =========================================================
# 🧠 Application code
# =========================================================
COPY . .

# =========================================================
# ⚙️ Environment variables
# =========================================================
ENV HOST=0.0.0.0 \
    PORT=10000 \
    BROWSER=chromium \
    HEADLESS=1 \
    ENABLE_GPT_OSS=1 \
    TRANSFORMERS_CACHE=/models/hf \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TRANSFORMERS_NO_REMOTE_CODE=0 \
    PLAYWRIGHT_ARGS="" \
    PYTHONUNBUFFERED=1

# =========================================================
# 🗂️ Create model cache directories (optional but helps with caching)
# =========================================================
RUN mkdir -p /models/hf /models/torch

EXPOSE 10000

# =========================================================
# 🚀 Start FastAPI with Uvicorn
# =========================================================
CMD sh -c 'uvicorn main:app --host $HOST --port ${PORT:-10000}'
