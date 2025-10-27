# Neur Agent API

A FastAPI service that combines browser automation (Playwright) with an LLM-powered research agent for web scraping, code analysis, and intelligent web search.

## Features

- 🤖 **LLM-Powered Agent**: Uses the `neur-0.0-full` model for intelligent web research
- 🌐 **Browser Automation**: Full Playwright integration for web scraping and interaction
- 🔍 **Smart Search**: Autonomous web search with DuckDuckGo integration
- 💻 **Code Analysis**: AI-powered code fixing and analysis
- 🚀 **Production Ready**: Docker + Cloud Run/Render deployment configs

## Architecture

This service provides three main capabilities:

1. **Browser Control** - Headless browser automation via Playwright
2. **Agent Search** - LLM-driven web research with autonomous search
3. **Code Analysis** - AI-powered code review and fixing

## Quick Start

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-directory>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Run the server
uvicorn main:app --reload --port 8080
```

Visit `http://localhost:8080` to verify the service is running.

### Environment Variables

```bash
# Model Configuration
MODEL_ID=xenon111/neur-0.0-full
MODEL_BASE_ID=openai/gpt-oss-20b
HF_TOKEN=your_huggingface_token  # Optional, for private models

# Browser Settings
HEADLESS=true
BROWSER_CHANNEL=chromium  # or 'chrome'
BROWSER_TIMEOUT_MS=12000

# Generation Settings
GEN_MAX_NEW_TOKENS=512
AGENT_MAX_NEW_TOKENS=512
CODEFIX_MAX_NEW_TOKENS=512
GENERATION_TEMPERATURE=0.0
MAX_AGENT_TURNS=6
GEN_MAX_CONCURRENCY=2

# Cache & Storage
HF_HOME=/home/pwuser/.cache/huggingface
WORKSPACE_ROOT=/tmp/workspace
OFFLOAD_DIR=/tmp/offload

# Performance (optional)
HF_HUB_ENABLE_HF_TRANSFER=1
```

## API Endpoints

### Health & Status

#### `GET /`
Basic health check.

**Response:**
```json
{
  "ok": true,
  "api": "1.4.0"
}
```

#### `GET /healthz`
Detailed health status including model and GPU info.

**Response:**
```json
{
  "playwright": true,
  "model_loaded": true,
  "device": "cuda",
  "gpu": {
    "name": "Tesla T4",
    "total": 16106127360,
    "allocated": 8500000000
  }
}
```

### Agent (LLM + Search)

#### `POST /agent/ask`
Ask a research question. The agent will autonomously search the web and synthesize an answer.

**Request:**
```json
{
  "question": "What are the latest developments in quantum computing?",
  "max_turns": 6
}
```

**Response:**
```json
{
  "answer": "Recent developments include...",
  "sources": [
    "https://example.com/article1",
    "https://example.com/article2"
  ],
  "searches": [
    {
      "query": "quantum computing 2025",
      "results": [
        {
          "title": "Breakthrough in Quantum Computing",
          "url": "https://example.com/article1",
          "snippet": "Scientists have achieved..."
        }
      ]
    }
  ]
}
```

### Code Analysis

#### `POST /codefix/analyze`
Analyze code and get AI-powered fixes.

**Request:**
```json
{
  "code": "def add(a, b):\n    return a + b + c",
  "issue": "NameError: name 'c' is not defined"
}
```

**Response:**
```json
{
  "fix": "def add(a, b):\n    # Fixed: removed undefined variable 'c'\n    return a + b"
}
```

### Browser Automation

#### `POST /browser/session`
Create a new browser session.

**Response:**
```json
{
  "session_id": "140234567890123"
}
```

#### `POST /browser/open?session_id=<id>`
Navigate to a URL.

**Request:**
```json
{
  "url": "https://example.com",
  "wait_until": "domcontentloaded",
  "timeout_ms": 12000
}
```

**Response:**
```json
{
  "session_id": "140234567890123",
  "title": "Example Domain",
  "url": "https://example.com"
}
```

#### `POST /browser/click?session_id=<id>`
Click an element.

**Request:**
```json
{
  "selector": "button.submit",
  "button": "left",
  "click_count": 1,
  "timeout_ms": 5000
}
```

#### `POST /browser/type?session_id=<id>`
Type text into an input.

**Request:**
```json
{
  "selector": "input[name='search']",
  "text": "hello world",
  "delay_ms": 100
}
```

#### `POST /browser/fill?session_id=<id>`
Fill an input instantly.

**Request:**
```json
{
  "selector": "input[name='email']",
  "value": "user@example.com"
}
```

#### `POST /browser/screenshot?session_id=<id>`
Take a screenshot.

**Request:**
```json
{
  "full_page": true,
  "type": "png",
  "quality": 80
}
```

**Response:**
```json
{
  "session_id": "140234567890123",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "mime": "image/png"
}
```

#### `GET /browser/html?session_id=<id>`
Get page HTML.

#### `POST /browser/query_texts?session_id=<id>`
Extract text from elements.

**Request:**
```json
{
  "selector": "h1, h2, h3"
}
```

**Response:**
```json
{
  "session_id": "140234567890123",
  "texts": ["Heading 1", "Heading 2", "Heading 3"]
}
```

#### `DELETE /browser/session?session_id=<id>`
Close a browser session.

## Deployment

### Docker

```bash
# Build
docker build -t neur-agent .

# Run
docker run -p 8080:8080 \
  -e HEADLESS=true \
  -e HF_TOKEN=your_token \
  neur-agent
```

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/neur-n00

# Deploy
gcloud run deploy neur-n00 \
  --image gcr.io/YOUR_PROJECT/neur-n00 \
  --platform managed \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars HEADLESS=true
```

Or use the included `cloudrun.yaml`:

```bash
gcloud run services replace cloudrun.yaml
```

### Render

Simply connect your repository and Render will use `render.yaml` for automatic deployment with persistent disk storage for model caching.

## Architecture Details

### Lazy Loading

The service uses lazy loading for heavy resources:
- Browser launches on first browser API call
- Model loads on first AI endpoint call

This reduces cold start time and memory usage.

### Model Requirements

- **Transformers**: >=4.55.0 (required for GPT-OSS architecture)
- **Accelerate**: >=0.34.2
- **HuggingFace Hub**: >=0.24.6

The model uses `trust_remote_code=True` to load the custom GPT-OSS architecture.

### Agent Loop

The agent uses a think-act-observe loop:

1. **Think**: LLM decides whether to search or answer
2. **Act**: Performs web search if needed
3. **Observe**: Receives search results
4. **Repeat**: Until final answer or max turns reached

## Performance Tuning

### Memory Optimization

```bash
# Enable model offloading for large models
OFFLOAD_DIR=/tmp/offload

# Reduce concurrent generations
GEN_MAX_CONCURRENCY=1
```

### GPU Settings

The service automatically detects GPU capability and uses:
- `bfloat16` for Ampere+ GPUs (compute capability >= 8.0)
- `float16` for older GPUs
- `float32` for CPU

### Browser Performance

```bash
# Use bundled Chromium (faster, more stable)
BROWSER_CHANNEL=chromium

# Or use system Chrome
BROWSER_CHANNEL=chrome
```

## Troubleshooting

### Model Won't Load

Check `/healthz` endpoint for detailed error information:

```bash
curl http://localhost:8080/healthz
```

Common issues:
- Transformers version < 4.55.0
- Missing `trust_remote_code=True` permissions
- Insufficient memory/GPU

### Browser Issues

If browser fails to launch:
- Ensure Playwright browsers are installed: `playwright install chromium`
- Check for sandbox issues in Docker (use `--no-sandbox` flag)
- Verify HEADLESS setting matches your environment

### Out of Memory

Reduce memory usage:
- Lower `GEN_MAX_NEW_TOKENS`
- Set `GEN_MAX_CONCURRENCY=1`
- Enable model offloading with `OFFLOAD_DIR`

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- All endpoints include proper error handling
- Tests pass (when implemented)

## Support

For issues and questions:
- Check `/healthz` endpoint for diagnostics
- Review logs for detailed error messages
- Open an issue on GitHub
