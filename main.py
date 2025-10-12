# main.py
# ---------- Core Web Framework ----------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from pathlib import Path
import asyncio, os, json
from urllib.parse import quote_plus

# ---------- Operator Tools ----------
from playwright.async_api import async_playwright, Page, Browser

# ---------- AI Model: GPT-OSS-20B (optional) ----------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ---------- App ----------
app = FastAPI(title="Operator + CodeFix API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
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
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "6"))

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

# ---------- Health ----------
@app.get("/")
async def root():
    return {"ok": True, "api": "v1"}

@app.get("/healthz")
async def health():
    return {
        "playwright": bool(_browser is not None),
        "model_loaded": bool(model_pipe is not None),
    }

# ---------- Startup / Shutdown ----------
@app.on_event("startup")
async def startup():
    global _pw, _browser, model_pipe

    # Playwright
    _pw = await async_playwright().start()
    browser_name = os.getenv("BROWSER", "chrome").lower()
    headless = os.getenv("HEADLESS", "1").lower() in ("1", "true", "yes", "y")

    try:
        if browser_name == "firefox":
            _browser = await _pw.firefox.launch(headless=headless)
        elif browser_name == "webkit":
            _browser = await _pw.webkit.launch(headless=headless)
        elif browser_name == "chrome":
            # Prefer system Chrome if present; otherwise fall back to Chromium.
            launch_args = ["--headless=new"] if headless else None
            _browser = await _pw.chromium.launch(
                channel="chrome", headless=headless, args=launch_args
            )
        else:
            _browser = await _pw.chromium.launch(headless=headless)
    except Exception as e:
        print(f"[startup] Failed '{browser_name}', falling back to chromium: {e}")
        _browser = await _pw.chromium.launch(headless=headless)

    # Optional: load LLM only if requested (heavy!)
    if os.getenv("ENABLE_GPT_OSS", "0").lower() in ("1", "true", "yes", "y"):
        model_id = os.getenv("MODEL_ID", "teknium/OpenHermes-2.5-Mistral-7B")  # override to your 20B
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            global model_pipe
            model_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
                do_sample=False,
                temperature=0.0,
            )
        except Exception as e:
            print(f"[startup] Model load failed ({model_id}): {e}")
            model_pipe = None

@app.on_event("shutdown")
async def shutdown():
    global _browser, _pw
    try:
        if _browser is not None:
            await _browser.close()
    finally:
        _browser = None
    try:
        if _pw is not None:
            await _pw.stop()
    finally:
        _pw = None

# ---------- Code Fix Endpoint (uses optional GPT model) ----------
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
    out = model_pipe(prompt, return_full_text=False)[0]["generated_text"]
    return {"fix": out}

# ---------- Operator Agent (LLM + Browser Search) ----------
AGENT_SYSTEM_PROMPT = """You are OperatorGPT, a research assistant that can browse the web.
You must respond using compact JSON on a single line. The JSON schema is:
  {"action": "search", "query": "..."}
  {"action": "final", "answer": "...", "sources": ["..."]}
If you need more information from the internet, use the search action first.
After you receive tool feedback, decide whether another search is required or return a final answer with sources."""

def _format_agent_prompt(question: str, history: List[Dict[str, str]]) -> str:
    conversation: List[str] = [AGENT_SYSTEM_PROMPT, f"User: {question}"]
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
        raise ValueError(f"No JSON object found in: {text!r}")
    candidate = text[start : end + 1]
    # Validate JSON
    json.loads(candidate)
    return candidate

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
                const nodes = Array.from(
                    document.querySelectorAll('#links article, #links .result')
                );
                return nodes.slice(0, limit).map((item) => {
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
    global model_pipe
    if model_pipe is None:
        raise HTTPException(503, "Model not loaded. Set ENABLE_GPT_OSS=1 and restart.")

    history: List[Dict[str, str]] = []
    searches: List[Dict[str, Any]] = []
    max_turns = req.max_turns or MAX_AGENT_TURNS

    for _ in range(max_turns):
        prompt = _format_agent_prompt(req.question, history)
        completion = model_pipe(prompt, return_full_text=False)[0]["generated_text"]

        try:
            action_blob = _extract_json_blob(completion)
            action = json.loads(action_blob)
        except Exception as exc:
            raise HTTPException(500, f"Model output parsing failed: {exc}")

        history.append({"role": "assistant", "content": action_blob})
        action_type = str(action.get("action", "")).lower()

        if action_type == "search":
            query = str(action.get("query", "")).strip()
            if not query:
                raise HTTPException(400, "Model requested search without query")
            results = await _agent_search(query)
            log_entry = {"query": query, "results": results}
            searches.append(log_entry)
            observation = json.dumps({"results": results})
            history.append({"role": "tool", "content": observation})
            continue

        if action_type == "final":
            answer = str(action.get("answer", "")).strip()
            sources = action.get("sources") or []
            if not isinstance(sources, list):
                raise HTTPException(400, "Model returned invalid sources format")
            # Let FastAPI/Pydantic coerce nested dicts to AgentSearchLog schema
            return {
                "answer": answer,
                "sources": [str(s) for s in sources],
                "searches": searches,
            }

        raise HTTPException(400, f"Unknown model action: {action_type}")

    raise HTTPException(500, "Model did not produce a final answer within turn limit")

# ---------- Local dev entry ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=True)
