# Use official Playwright Python image with browsers preinstalled
# Matches requirements: playwright <1.48
FROM mcr.microsoft.com/playwright/python:v1.47.2-jammy

WORKDIR /app

# System env
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Copy only requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Render provides PORT; default to 8000 for local
ENV PORT=8000 \
    HEADLESS=1 \
    BROWSER=chromium \
    ENABLE_GPT_OSS=0

# Start the FastAPI app via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]

