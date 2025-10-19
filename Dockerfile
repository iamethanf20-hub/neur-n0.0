# Build on a slim Python base. Works on Render and Cloud Run.
FROM python:3.10-slim

# System deps for Playwright install-deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Install Chrome + its OS dependencies for server environments
# (install-deps makes it portable across distros used by Render/Cloud Run)
RUN python -m playwright install-deps \
 && python -m playwright install chrome

# App code
COPY . /app

# Environment
ENV PORT=8080
ENV HF_HOME=/data/.cache/huggingface

# Expose is optional for Cloud Run; harmless on Render
EXPOSE 8080

# Start server (Render and Cloud Run both set $PORT)
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
