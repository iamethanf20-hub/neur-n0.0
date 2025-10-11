# ---------- Core Web Framework ----------
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- Async & System Tools ----------
import asyncio, os, base64
from typing import Optional, Dict, List, Any
from pathlib import Path

# ---------- Operator Tools ----------
from playwright.async_api import async_playwright, Page, BrowserContext, Browser

# ---------- AI Model: GPT-OSS-20B ----------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ---------- App ----------
app = FastAPI(title="Operator + CodeFix API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Globals ----------
_pw = None
_browser: Optional[Browser] = None
_sessions: Dict[str, Page] = {}
model_pipe = None
DEFAULT_TIMEOUT_MS = int(os.getenv("BROWSER_TIMEOUT_MS", "10000"))
WORKSPACE_ROOT = Path(os.getcwd()).resolve()

# ---------- Startup / Shutdown ----------
@app.on_event("startup")
async def startup():
    global _pw, _browser, model_pipe
    # Playwright
    _pw = await async_playwright().start()
    # Default browser + headless via env: BROWSER=chrome|chromium|firefox|webkit, HEADLESS=1|0
    browser_name = os.getenv("BROWSER", "chrome").lower()
    headless = os.getenv("HEADLESS", "1").lower() in ("1", "true", "yes", "y")
    try:
        if browser_name == "firefox":
            _browser = await _pw.firefox.launch(headless=headless)
        elif browser_name == "webkit":
            _browser = await _pw.webkit.launch(headless=headless)
        elif browser_name == "chrome":
            # Use system Chrome via channel; requires Chrome installed.
            # Force new headless mode if headless is requested.
            launch_args = ["--headless=new"] if headless else None
            _browser = await _pw.chromium.launch(channel="chrome", headless=headless, args=launch_args)
        else:
            _browser = await _pw.chromium.launch(headless=headless)
    except Exception as e:
        print(f"Failed to launch '{browser_name}', falling back to chromium: {e}")
        _browser = await _pw.chromium.launch(headless=headless)

    # GPT-OSS-20B (opt-in via env; avoid blocking startup)
    if os.getenv("ENABLE_GPT_OSS", "0") == "1":
        model_id = os.getenv("GPT_OSS_MODEL", "openchat/gpt-oss-20b")  # update to your repo name
        print(f"Loading {model_id} ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            # Try GPU FP16; fallback to CPU FP32
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            model_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False
            )
            print("GPT-OSS-20B ready.")
        except Exception as e:
            print(f"Model load skipped due to error: {e}")

@app.on_event("shutdown")
async def shutdown():
    global _pw, _browser
    # Close Playwright
    if _browser:
        await _browser.close()
    if _pw:
        await _pw.stop()

# ---------- Operator Endpoints ----------
@app.get("/")
async def root():
    return {"ok": True, "api": "v1"}

@app.post("/browser/session")
async def new_session():
    ctx = await _browser.new_context()
    page = await ctx.new_page()
    sid = str(id(page))
    _sessions[sid] = page
    return {"session_id": sid}

class OpenBody(BaseModel):
    url: str

@app.post("/browser/open")
async def open_url(body: OpenBody, session_id: Optional[str] = Query(None)):
    if not session_id:
        ctx = await _browser.new_context()
        page = await ctx.new_page()
        sid = str(id(page))
        _sessions[sid] = page
    else:
        sid = session_id
        page = _sessions.get(sid)
        if not page:
            raise HTTPException(404, "session_id not found")

    await page.goto(body.url, wait_until="domcontentloaded")
    title = await page.title()
    return {"session_id": sid, "title": title, "url": body.url}

@app.get("/browser/screenshot")
async def screenshot(session_id: str):
    page = _sessions.get(session_id)
    if not page:
        raise HTTPException(404, "session_id not found")
    png = await page.screenshot(full_page=True)
    b64 = base64.b64encode(png).decode("utf-8")
    return {"png_base64": b64}

# ---------- Browser Controls ----------
def _get_page_or_404(session_id: str) -> Page:
    page = _sessions.get(session_id)
    if not page:
        raise HTTPException(404, "session_id not found")
    return page

class EvalJsBody(BaseModel):
    session_id: str
    expression: str
    arg: Optional[Any] = None
    timeout_ms: Optional[int] = None

@app.post("/browser/eval_js")
async def browser_eval_js(body: EvalJsBody):
    page = _get_page_or_404(body.session_id)
    timeout = (body.timeout_ms or DEFAULT_TIMEOUT_MS) / 1000.0
    try:
        result = await asyncio.wait_for(page.evaluate(body.expression, body.arg), timeout=timeout)
        return {"ok": True, "result": result}
    except asyncio.TimeoutError:
        raise HTTPException(408, "evaluate timed out")
    except Exception as e:
        raise HTTPException(400, f"evaluate failed: {e}")

class ClickBody(BaseModel):
    session_id: str
    selector: str
    timeout_ms: Optional[int] = None

@app.post("/browser/click")
async def browser_click(body: ClickBody):
    page = _get_page_or_404(body.session_id)
    try:
        await page.click(body.selector, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, f"click failed: {e}")

class TypeBody(BaseModel):
    session_id: str
    selector: str
    text: str
    delay_ms: Optional[int] = None
    clear: bool = False
    timeout_ms: Optional[int] = None

@app.post("/browser/type")
async def browser_type(body: TypeBody):
    page = _get_page_or_404(body.session_id)
    try:
        await page.wait_for_selector(body.selector, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        if body.clear:
            await page.click(body.selector, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
            # Select all and delete
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Delete")
        await page.type(body.selector, body.text, delay=body.delay_ms or 0, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, f"type failed: {e}")

class PressBody(BaseModel):
    session_id: str
    key: str
    selector: Optional[str] = None
    timeout_ms: Optional[int] = None

@app.post("/browser/press")
async def browser_press(body: PressBody):
    page = _get_page_or_404(body.session_id)
    try:
        if body.selector:
            await page.wait_for_selector(body.selector, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
            await page.press(body.selector, body.key, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        else:
            await page.keyboard.press(body.key)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, f"press failed: {e}")

class UploadBody(BaseModel):
    session_id: str
    selector: str
    files: List[str]
    timeout_ms: Optional[int] = None

def _validate_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        rp = (Path(p) if not os.path.isabs(p) else Path(p)).resolve()
        if not str(rp).startswith(str(WORKSPACE_ROOT)):
            raise HTTPException(400, f"file outside workspace: {rp}")
        if not rp.exists():
            raise HTTPException(400, f"file not found: {rp}")
        out.append(str(rp))
    return out

@app.post("/browser/upload")
async def browser_upload(body: UploadBody):
    page = _get_page_or_404(body.session_id)
    try:
        files = _validate_paths(body.files)
        await page.set_input_files(body.selector, files, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        return {"ok": True, "files": files}
    except Exception as e:
        raise HTTPException(400, f"upload failed: {e}")

# ---------- Code Fix Endpoint (uses GPT-OSS-20B) ----------
class CodeFixRequest(BaseModel):
    code: str
    issue: str

@app.post("/codefix/analyze")
async def analyze_code(req: CodeFixRequest):
    global model_pipe
    if model_pipe is None:
        raise HTTPException(503, "Model not loaded. Set ENABLE_GPT_OSS=1 and restart.")
    prompt = f"""You are a coding assistant. Fix the issue below with minimal changes.
Return ONLY corrected code with short inline comments if needed.

Issue:
{req.issue}

Code:
{req.code}
"""
    out = model_pipe(prompt)[0]["generated_text"]
    return {"fix": out}

