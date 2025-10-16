#main.py
# ---------- Core Web Framework ----------
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from pathlib import Path
import asyncio, os, json, base64, io, shutil, logging
from urllib.parse import quote_plus

log = logging.getLogger("uvicorn.error")

# ---------- (MOVE THIS BLOCK TO THE VERY TOP, BEFORE TRANSFORMERS) ----------
# hf_transfer guard (prevents startup crash if env enables it but pkg isn't installed)
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").lower() in ("1", "true", "yes"):
    try:
        import hf_transfer  # type: ignore
        log.info("[hf_transfer] fast download enabled.")
    except Exception:
        # Hard-disable so huggingface_hub doesn't error during import
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        log.warning("[hf_transfer] requested but not installed; disabling fast download.")

# Optional hard kill-switch if you ever need it in Render env:
if os.getenv("FORCE_DISABLE_HF_TRANSFER", "").lower() in ("1", "true", "yes"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    log.info("[hf_transfer] force-disabled via FORCE_DISABLE_HF_TRANSFER.")

# ---------- Operator Tools ----------
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, TimeoutError as PWTimeout

# ---------- AI Model ----------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from transformers import BitsAndBytesConfig  # <- uncomment if you want 4-bit


# ---------- App ----------
app = FastAPI(title="Operator + CodeFix API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("uvicorn.error")

# ---------- Globals ----------
_pw = None
_browser: Optional[Browser] = None
_sessions: Dict[str, Page] = {}
model_pipe = None

DEFAULT_TIMEOUT_MS = int(os.getenv("BROWSER_TIMEOUT_MS", "12000"))
HEADLESS = os.getenv("HEADLESS", "true").lower() in ("1","true","yes","on")
BROWSER_CHANNEL = os.getenv("BROWSER_CHANNEL", "chrome")
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", os.getcwd())).resolve()
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "6"))

# === LLM integration ===
MODEL_ID = os.getenv("MODEL_ID", "xenon111/neur-0.0")
HF_TOKEN = os.getenv("HF_TOKEN")
USE_CHAT_TEMPLATE = True
GEN_MAX_CONCURRENCY = int(os.getenv("GEN_MAX_CONCURRENCY", "2"))
_gen_sema = asyncio.Semaphore(GEN_MAX_CONCURRENCY)
_last_load_error: Optional[str] = None

# Prefer persistent caches
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "/data/.cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.getenv("TRANSFORMERS_CACHE", "/data/.cache/huggingface/transformers"))

def _dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32

def _load_model_once():
    """Idempotent model loader; sets global model_pipe or _last_load_error."""
    global model_pipe, _last_load_error
    if model_pipe is not None:
        return
    try:
        log.info(f"[startup] Loading model: {MODEL_ID}")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN, use_fast=True)
        # A) Full precision / auto device map
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=_dtype(),
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        # B) 4-bit option (uncomment if OOM):
        # quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        # model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", trust_remote_code=True, token=HF_TOKEN, quantization_config=quant)

        model_pipe_local = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            device_map="auto",
            torch_dtype=_dtype(),
            trust_remote_code=True,
        )
        _last_load_error = None
        model_pipe = model_pipe_local
        log.info("[startup] Model loaded successfully.")
    except Exception as e:
        _last_load_error = f"{type(e).__name__}: {e}"
        model_pipe = None
        log.exception("[startup] Model load failed.")

async def _ensure_model_loaded():
    global model_pipe
    if model_pipe is None:
        await asyncio.to_thread(_load_model_once)
        if model_pipe is None:
            raise HTTPException(503, f"Model not loaded. Check startup logs. last_error={_last_load_error}")

# ---------- Schemas ----------
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
    path: str  # path inside WORKSPACE_ROOT
    timeout_ms: Optional[int] = None

class WaitForSelectorBody(BaseModel):
    selector: str
    state: str = "visible"  # attached|detached|visible|hidden
    timeout_ms: Optional[int] = None

class ViewportBody(BaseModel):
    width: int
    height: int

class ScreenshotBody(BaseModel):
    full_page: bool = True
    quality: Optional[int] = None  # only for jpeg
    type: str = "png"  # png|jpeg
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

    # Prefer Google Chrome
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

    # Try model load (non-fatal)
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

# ---------- Agent (LLM + Browser Search) ----------
AGENT_SYSTEM_PROMPT = """You are neur, a friendly research assistant that can browse the web.
You must respond using compact JSON on a single line. The JSON schema is:
  {"action": "search", "query": "..."}
  {"action": "final", "answer": "...", "sources": ["..."]}
If you need more information from the internet, use the search action first.
After you receive tool feedback, decide whether another search is required or return a final answer with sources."""

def _format_agent_prompt(question: str, history: List[Dict[str, str]]) -> str:
    conversation = [AGENT_SYSTEM_PROMPT, f"User: {question}"]
    for entry in history:
        prefix = "Assistant" if entry["role"] == "assistant" else "Tool"
        conversation.append(f"{prefix}: {entry['content']}")
    conversation.append("Assistant:")
    return "\n".join(conversation)

def _extract_json_blob(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON found in: {text!r}")
    blob = text[start:end+1]
    json.loads(blob)  # validate
    return blob

async def _agent_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    if _browser is None:
        raise HTTPException(503, "Browser not ready")
    ctx = await _browser.new_context()
    page = await ctx.new_page()
    try:
        url = f"https://duckduckgo.com/?q={quote_plus(query)}&t=h_&ia=web"
        await page.goto(url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)
        await page.wait_for_selector("#links", timeout=DEFAULT_TIMEOUT_MS)
        results = await page.evaluate(
            """
            (limit) => {
                const nodes = Array.from(document.querySelectorAll('#links article, #links .result'));
                return nodes.slice(0, limit).map(item => {
                    const a = item.querySelector('a[href^="http"]');
                    const title = (a?.textContent || '').trim();
                    const url = a?.href || '';
                    const snippetNode = item.querySelector('[data-testid="result-snippet"], .result__snippet');
                    const snippet = (snippetNode?.textContent || '').trim();
                    return url ? { title, url, snippet } : null;
                }).filter(Boolean);
            }
            """,
            limit,
        )
        return results or []
    finally:
        await ctx.close()

@app.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(req: AgentAskRequest):
    await _ensure_model_loaded()
    history: List[Dict[str, str]] = []
    searches: List[Dict[str, Any]] = []
    max_turns = req.max_turns or MAX_AGENT_TURNS

    for _ in range(max_turns):
        prompt = _format_agent_prompt(req.question, history)
        try:
            async with _gen_sema:
                completion = (await asyncio.to_thread(model_pipe, prompt, return_full_text=False))[0]["generated_text"]
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {e}")

        try:
            action_blob = _extract_json_blob(completion)
            action = json.loads(action_blob)
        except Exception as e:
            raise HTTPException(500, f"Model output parsing failed: {e}")

        history.append({"role": "assistant", "content": action_blob})
        act = action.get("action", "").lower()

        if act == "search":
            query = action.get("query", "").strip()
            if not query:
                raise HTTPException(400, "Empty search query.")
            results = await _agent_search(query)
            searches.append({"query": query, "results": results})
            observation = json.dumps({"results": results})
            history.append({"role": "tool", "content": observation})
            continue

        if act == "final":
            answer = action.get("answer", "").strip()
            sources = action.get("sources") or []
            return {"answer": answer, "sources": sources, "searches": searches}

        raise HTTPException(400, f"Unknown action: {act}")

    raise HTTPException(500, "Model did not return a final answer in allotted turns.")

# ---------- Browser Control Endpoints ----------
@app.post("/browser/session")
async def browser_session():
    sid = await _ensure_session(None)
    return {"session_id": sid}

@app.post("/browser/open")
async def browser_open(body: OpenBody, session_id: Optional[str] = Query(None)):
    sid = await _ensure_session(session_id)
    page = await _get_page(sid)
    try:
        await page.goto(body.url, wait_until=body.wait_until, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        title = await page.title()
        return {"session_id": sid, "title": title, "url": page.url}
    except PWTimeout:
        raise HTTPException(408, "Navigation timed out")

@app.post("/browser/evaljs")
async def browser_evaljs(body: EvalJSBody, session_id: str = Query(...)):
    page = await _get_page(session_id)
    try:
        result = await page.evaluate(body.expression, body.arg) if body.arg is not None else await page.evaluate(body.expression)
        return {"session_id": session_id, "result": result}
    except Exception as e:
        raise HTTPException(400, f"evaljs error: {e}")

@app.post("/browser/click")
async def browser_click(body: ClickBody, session_id: str = Query(...)):
    page = await _get_page(session_id)
    try:
        await page.click(
            body.selector,
            button=body.button,
            click_count=body.click_count,
            timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS,
            delay=body.delay_ms
        )
        return {"ok": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(400, f"click error: {e}")

@app.post("/browser/type")
async def browser_type(body: TypeBody, session_id: str = Query(...)):
    page = await _get_page(session_id)
    try:
        await page.type(
            body.selector,
            body.text,
            delay=body.delay_ms or 0,
            timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS,
        )
        return {"ok": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(400, f"type error: {e}")

@app.post("/browser/fill")
async def browser_fill(body: FillBody, session_id: str = Query(...)):
    page = await _get_page(session_id)
    try:
        await page.fill(body.selector, body.value, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        return {"ok": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(400, f"fill error: {e}")

@app.post("/browser/upload")
async def browser_upload(body: UploadBody, session_id: str = Query(...)):
    page = await _get_page(session_id)
    fpath = _safe_path(body.path)
    if not fpath.exists():
        raise HTTPException(404, f"File not found: {fpath}")
    try:
        input_ = await page.wait_for_selector(body.selector, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        await input_.set_input_files(str(fpath))
        return {"ok": True, "session_id": session_id, "path": str(fpath)}
    except Exception as e:
        raise HTTPException(400, f"upload error: {e}")

@app.post("/browser/wait_for_selector")
async def browser_wait_for_selector(body: WaitForSelectorBody, session_id: str = Query(...)):
    page = await _get_page(session_id)
    try:
        await page.wait_for_selector(body.selector, state=body.state, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        return {"ok": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(408, f"wait_for_selector error: {e}")

@app.get("/browser/html")
async def browser_html(session_id: str = Query(...)):
    page = await _get_page(session_id)
    html = await page.content()
    return {"session_id": session_id, "html": html}

@app.post("/browser/query_texts")
async def browser_query_texts(body: QueryBody, session_id: str = Query(...)):
    page = await _get_page(session_id)
    try:
        texts = await page.eval_on_selector_all(body.selector, "els => els.map(e => e.textContent?.trim() ?? '')")
        return {"session_id": session_id, "texts": texts}
    except Exception as e:
        raise HTTPException(400, f"query_texts error: {e}")

@app.post("/browser/query_attrs")
async def browser_query_attrs(body: QueryBody, attr: str = Query("href"), session_id: str = Query(...)):
    page = await _get_page(session_id)
    try:
        vals = await page.eval_on_selector_all(body.selector, f"(els) => els.map(e => e.getAttribute('{attr}'))")
        return {"session_id": session_id, "attrs": vals, "attr": attr}
    except Exception as e:
        raise HTTPException(400, f"query_attrs error: {e}")

@app.post("/browser/viewport")
async def browser_viewport(body: ViewportBody, session_id: str = Query(...)):
    page = await _get_page(session_id)
    await page.set_viewport_size({"width": body.width, "height": body.height})
    return {"ok": True, "session_id": session_id}

@app.post("/browser/screenshot")
async def browser_screenshot(body: ScreenshotBody, session_id: str = Query(...)):
    page = await _get_page(session_id)
    try:
        if body.selector:
            elem = await page.wait_for_selector(body.selector, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
            buf = await elem.screenshot(type=body.type, quality=body.quality, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        else:
            buf = await page.screenshot(full_page=body.full_page, type=body.type, quality=body.quality, timeout=body.timeout_ms or DEFAULT_TIMEOUT_MS)
        b64 = base64.b64encode(buf).decode("ascii")
        return {"session_id": session_id, "image_base64": b64, "mime": f"image/{body.type}"}
    except Exception as e:
        raise HTTPException(400, f"screenshot error: {e}")

@app.delete("/browser/session")
async def browser_close(session_id: str = Query(...)):
    page = _sessions.pop(session_id, None)
    if not page:
        raise HTTPException(404, "session_id not found")
    try:
        await page.context.close()
    except Exception:
        pass
    return {"ok": True}

# ---------- Filesystem (Code Writing) ----------
@app.get("/files/list")
async def files_list(path: str = ""):
    target = _safe_path(path)
    if not target.exists():
        raise HTTPException(404, "Path not found")
    if not target.is_dir():
        raise HTTPException(400, "Not a directory")
    items = []
    for p in sorted(target.iterdir()):
        items.append({
            "name": p.name,
            "path": str(p.relative_to(WORKSPACE_ROOT)),
            "is_dir": p.is_dir(),
            "size": p.stat().st_size if p.is_file() else None,
        })
    return {"root": str(WORKSPACE_ROOT), "items": items}

@app.get("/files/read")
async def files_read(path: str, encoding: str = "utf-8"):
    target = _safe_path(path)
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found")
    try:
        text = target.read_text(encoding=encoding)
        return {"path": path, "content": text}
    except UnicodeDecodeError:
        data = target.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return {"path": path, "base64": b64, "binary": True}

@app.post("/files/write")
async def files_write(body: WriteFileBody):
    target = _safe_path(body.path)
    if body.mkdirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    if body.mode not in ("w","a"):
        raise HTTPException(400, "mode must be 'w' or 'a'")
    with target.open(body.mode, encoding=body.encoding) as f:
        f.write(body.content)
    return {"ok": True, "path": body.path, "bytes": len(body.content.encode(body.encoding))}

@app.post("/files/mkdir")
async def files_mkdir(body: MkdirBody):
    target = _safe_path(body.path)
    target.mkdir(parents=body.parents, exist_ok=body.exist_ok)
    return {"ok": True, "path": body.path}

@app.delete("/files/delete")
async def files_delete(path: str):
    target = _safe_path(path)
    if not target.exists():
        return {"ok": True, "deleted": False, "path": path}
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"ok": True, "deleted": True, "path": path}

# ---------- Local Dev Entry ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
