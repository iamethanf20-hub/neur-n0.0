# Use Playwright's official base (all browser deps preinstalled)
FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && python -m playwright install --with-deps chromium

# App code
COPY . /app

# Environment
ENV PORT=8080
# Use a writable, non-root home cache path in this base image
ENV HF_HOME=/home/pwuser/.cache/huggingface

# Informational; Cloud Run ignores EXPOSE but it's fine to keep
EXPOSE 8080

# Start server (Render & Cloud Run both pass $PORT)
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
