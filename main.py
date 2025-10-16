# main.py
# ---------- Core Web Framework ----------
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from pathlib import Path
import asyncio, os, json, base64, io, shutil, logging
from urllib.parse import quote_plus

# ---------- Operator Tools ----------
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, TimeoutError as PWTimeout

# ---------- AI Model: GPT-OSS-20B (hardcoded fine-tune) ----------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from transformers import BitsAndBytesConfig  # <- uncomment if you want 4-bit

# ---------- App ----------
app = FastAPI(title="Operator + CodeFix API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("uvicorn.error")

# ---------- Fix for hf_transfer ----------
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").lower() in ("1", "true", "yes"):
    try:
        import hf_transfer  # type: ignore
        log.info("[hf_transfer] fast download enabled.")
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        log.warning("[hf_transfer] requested but not installed; disabling fast download.")

# ---------- Globals ----------
_pw = None
_browser: Optional[Browser] = None
_sessions: Dict[str, Page] = {}
model_pipe = None

DEFAULT_TIMEOUT_MS = int(os.getenv("BROWSER_TIMEOUT_MS", "12000"))
HEADLESS = os.getenv("HEADLESS", "true").lower() in ("1","true","yes","on")
BROWSER_CHANNEL = os.getenv("BROWSER_CHANNEL", "chrome")  # "chrome" by default
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", os.getcwd())).resolve()
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "6"))

# === LLM integration additions ===
MODEL_ID = os.getenv("MODEL_ID", "xenon111/neur-0.0")  # override if needed
HF_TOKEN = os.getenv("HF_TOKEN")  # set if your repo is private
USE_CHAT_TEMPLATE = True
GEN_MAX_CONCURRENCY = int(os.getenv("GEN_MAX_CONCURRENCY", "2"))
_gen_sema = asyncio.Semaphore(GEN_MAX_CONCURRENCY)
_last_load_error: Optional[str] = None

# Optional cache paths (great for Render)
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "/data/.cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.getenv("TRANSFORMERS_CACHE", "/data/.cache/huggingface/transformers"))

def _dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32

def _load_model_once():
    """
    Idempotent model loader. Sets global model_pipe or records _last_load_error.
    Runs blocking HF init; call inside asyncio.to_thread.
    """
    global model_pipe, _last_load_error
    if model_pipe is not None:
        return
    try:
        log.info(f"[startup] Loading fine-tuned model: {MODEL_ID}")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN, use_fast=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=_dtype(),
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            device_map="auto",
            torch_dtype=_dtype(),
            trust_remote_code=True,
        )
        model_pipe = pipe
        _last_load_error = None
        log.info("[startup] Model loaded successfully.")
    except Exception as e:
        _last_load_error = f"{type(e).__name__}: {e}"
        model_pipe = None
        log.exception("[startup] Model load failed.")

async def _ensure_model_loaded():
    """Retry loading the model lazily if not yet available."""
    global model_pipe
    if model_pipe is None:
        await asyncio.to_thread(_load_model_once)
        if model_pipe is None:
            raise HTTPException(503, f"Model not loaded. Check startup logs. last_error={_last_load_error}")

# ---------- Models ----------
class CodeFixRequest(BaseModel):
    code: str
    issue: str

class AgentAskRequest(BaseModel):
    question: str
    max_turns: Optional[int] = None

class AgentSearchSnippet(BaseModel):
    title: str
    url: str
    snippet: str

class AgentSearchLog(BaseModel):
    query: str
    results: List[AgentSearchSnippet]

class AgentAskResponse(BaseModel):
    answer: str
    sources: List[str]
    searches: List[AgentSearchLog]

# ---- Browser request models
class OpenBody(BaseModel):
    url: str
    wait_until: str = Field(default="domcontentloaded", description="load|domcontentloaded|networkidle|commit")
    timeout_ms: Optional[int] = None

class EvalJSBody(BaseModel):
    expression: str
    arg: Optional[Any] = None
    timeout_ms: Optional[int] = None

class QueryBody(BaseModel):
    selector: str
    timeout_ms: Optional[int] = None

class ClickBody(BaseModel):
    selector: str
    button: str = "left"
    click_count: int = 1
    timeout_ms: Optional[int] = None
    delay_ms: Optional[int] = None

class TypeBody(BaseModel):
    selector: str
    text: str
    delay_ms: Optional[int] = None
    timeout_ms: Optional[int] = None

class FillBody(BaseModel):
    selector: str
    value: str
    timeout_ms: Optional[int] = None

class UploadBody(BaseModel):
    selector: str
    path: str
    timeout_ms: Optional[int] = None

class WaitForSelectorBody(BaseModel):
    selector: str
    state: str = "visible"
    timeout_ms: Optional[int] = None

class ViewportBody(BaseModel):
    width: int
    height: int

class ScreenshotBody(BaseModel):
    full_page: bool = True
    quality: Optional[int] = None
    type: str = "png"
    selector: Optional[str] = None
    timeout_ms: Optional[int] = None

# ---- Filesystem models ----
class WriteFileBody(BaseModel):
    path: str
    content: str
    mode: str = Field(default="w", description="w to overwrite, a to append")
    mkdirs: bool = True
    encoding: str = "utf-8"

class MkdirBody(BaseModel):
    path: str
    exist_ok: bool = True
    parents: bool = True

# ---------- Health ----------
@app.get("/")
async def root():
    return {"ok": True, "api": "v1.2.0"}

@app.get("/healthz")
async def health():
    return {
        "playwright": bool(_browser is not None),
        "model_loaded": bool(model_pipe is not None),
        "workspace_root": str(WORKSPACE_ROOT),
        "browser_channel": BROWSER_CHANNEL,
        "headless": HEADLESS,
        "model_id": MODEL_ID,
        "last_error": _last_load_error,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

# ---------- Helpers ----------
def _safe_path(rel: str) -> Path:
    p = (WORKSPACE_ROOT / rel).resolve()
    if not str(p).startswith(str(WORKSPACE_ROOT)):
        raise HTTPException(400, "Path escapes WORKSPACE_ROOT")
    return p

def _sid(page: Page) -> str:
    return str(id(page))

async def _ensure_session(session_id: Optional[str]) -> str:
    if session_id and session_id in _sessions:
        return session_id
    if _browser is None:
        raise HTTPException(503, "Browser not ready")
    ctx = await _browser.new_context()
    page = await ctx.new_page()
    sid = _sid(page)
    _sessions[sid] = page
    return sid

async def _get_page(session_id: str) -> Page:
    page = _sessions.get(session_id)
    if not page:
        raise HTTPException(404, "session_id not found")
    return page

# ---------- Startup / Shutdown ----------
@app.on_event("startup")
async def startup():
    global _pw, _browser
    _pw = await async_playwright().start()

    try:
        _browser = await _pw.chromium.launch(headless=HEADLESS, channel=BROWSER_CHANNEL)
        print(f"[startup] Launched browser channel={BROWSER_CHANNEL}")
    except Exception as e:
        print(f"[startup] Chrome launch failed ({e}); falling back to bundled Chromium.")
        try:
            _browser = await _pw.chromium.launch(headless=HEADLESS)
            print("[startup] Launched default Chromium.")
        except Exception as e2:
            print(f"[startup] Failed to launch any browser: {e2}")
            _browser = None

    try:
        await asyncio.to_thread(_load_model_once)
    except Exception:
        pass

@app.on_event("shutdown")
async def shutdown():
    global _browser, _pw
    for sid, page in list(_sessions.items()):
        try:
            await page.context.close()
        except Exception:
            pass
        _sessions.pop(sid, None)
    try:
        if _browser:
            await _browser.close()
    finally:
        _browser = None
    try:
        if _pw:
            await _pw.stop()
    finally:
        _pw = None

# ---------- Code Fix Endpoint ----------
@app.post("/codefix/analyze")
async def analyze_code(req: CodeFixRequest):
    await _ensure_model_loaded()
    prompt = f"""You are a coding assistant. Fix the issue below with minimal changes.
Return ONLY corrected code with short inline comments if needed.

Issue:
{req.issue}

Code:
{req.code}
"""
    try:
        async with _gen_sema:
            out = await asyncio.to_thread(model_pipe, prompt, return_full_text=False)
        text = out[0]["generated_text"]
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")
    return {"fix": text}

# ---------- Local Dev Entry ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
