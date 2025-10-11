# Operator + CodeFix API

FastAPI service that drives Playwright for browser automation and exposes an optional code-fix endpoint backed by a local Transformers pipeline.

## Run Locally

- Create and activate a venv, then install deps:
  - `python -m venv .venv && . .venv/Scripts/Activate.ps1` (Windows PowerShell)
  - `pip install -r requirements.txt`
- Start the API: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- Open: `http://localhost:8000/`

Notes
- Playwright needs browsers installed for local runs: `python -m playwright install chromium`
- Optional LLM endpoint is off by default. Enable with `ENABLE_GPT_OSS=1` and configure `GPT_OSS_MODEL`.
- CORS is wide-open for development (`allow_origins=["*"]`). For production, restrict to your domains in `main.py`.

## Deploy on Render (Docker)

This repo includes a Dockerfile based on the official Playwright image, which bundles browsers and dependencies.

1) Push to GitHub
- Initialize git: `git init` (if needed)
- `git add . && git commit -m "Initial commit"`
- Create a GitHub repo and push: `git remote add origin <your-repo-url>` then `git push -u origin main`

2) One-click from Render Blueprint
- In the Render dashboard, click `New` → `Blueprint` and point to this repo’s `render.yaml`.
- Render will build using the Dockerfile and run `uvicorn main:app`.

Config
- The service listens on `PORT` provided by Render.
- Provided env vars (in `render.yaml`): `HEADLESS=1`, `BROWSER=chromium`, `ENABLE_GPT_OSS=0`.
- Health check path: `/`.

## Non-Docker alternative (not recommended for Playwright)

Render’s native Python environment does not include Playwright system dependencies. If you avoid Docker, you must bring your own dependencies, which generally isn’t feasible on Render without a Docker image.

## Troubleshooting

- Large venv: Do not commit `.venv` (it’s ignored via `.gitignore`).
- Windows reserved filename: A file named `CON` can cause git issues on Windows. If you see it locally, remove/rename it before committing.
- Models: Transformers + Torch are heavy. Keep `ENABLE_GPT_OSS=0` unless you need it.

## API

- Root: `GET /` -> `{ ok: true, api: "v1" }`
- Browser: create session `POST /browser/session`, open URL `POST /browser/open`, screenshot `GET /browser/screenshot`, controls: `/browser/eval_js`, `/browser/click`, `/browser/type`, `/browser/press`, `/browser/upload`.
- Code fix: `POST /codefix/analyze` (requires `ENABLE_GPT_OSS=1`).

## License

This project is licensed under the MIT License. See `LICENSE` for details.
