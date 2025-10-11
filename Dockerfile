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

# =========================================================
# 🌐 Install Playwright + Chromium manually (safe for Render)
# =========================================================
RUN pip install playwright && python -m playwright install chromium

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
    ENABLE_GPT_OSS=0 \
    TRANSFORMERS_CACHE=/models/hf \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    PLAYWRIGHT_ARGS=""

EXPOSE 10000

# =========================================================
# 🚀 Start FastAPI with Uvicorn
# =========================================================
CMD sh -c 'uvicorn main:app --host $HOST --port ${PORT:-10000}'
