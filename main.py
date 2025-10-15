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

# ---------- AI Model: GPT-OSS-20B (hardcoded fine-tune) ----------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------- App ----------
app = FastAPI(title="Operator + CodeFix API", version="1.0.0")
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

    # --- Playwright setup ---
    _pw = await async_playwright().start()
    headless = True
    try:
        _browser = await _pw.chromium.launch(headless=headless)
    except Exception as e:
        print(f"[startup] Failed to launch browser: {e}")
        _browser = None

    # --- Hardcoded model load ---
    model_id = "xenon111/neur-0.0"  # your fine-tuned model on Hugging Face
    print(f"[startup] Loading fine-tuned model: {model_id}")

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
        )
        print("[startup] Model loaded successfully.")
    except Exception as e:
        print(f"[startup] Model load failed: {e}")
        model_pipe = None

@app.on_event("shutdown")
async def shutdown():
    global _browser, _pw
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
    global model_pipe
    if model_pipe is None:
        raise HTTPException(503, "Model not loaded. Check startup logs.")

    prompt = f"""You are a coding assistant. Fix the issue below with minimal changes.
Return ONLY corrected code with short inline comments if needed.

Issue:
{req.issue}

Code:
{req.code}
"""
    try:
        out = model_pipe(prompt, return_full_text=False)[0]["generated_text"]
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    return {"fix": out}

# ---------- Operator Agent (LLM + Browser Search) ----------
AGENT_SYSTEM_PROMPT = """You are n0.0, a research assistant that can browse the web.
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
    global model_pipe
    if model_pipe is None:
        raise HTTPException(503, "Model not loaded.")

    history: List[Dict[str, str]] = []
    searches: List[Dict[str, Any]] = []
    max_turns = req.max_turns or MAX_AGENT_TURNS

    for _ in range(max_turns):
        prompt = _format_agent_prompt(req.question, history)
        completion = model_pipe(prompt, return_full_text=False)[0]["generated_text"]

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

# ---------- Local Dev Entry ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
