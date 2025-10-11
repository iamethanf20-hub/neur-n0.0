# ---------- Core Web Framework ----------
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- Async & System Tools ----------
import asyncio, os, base64
from typing import Optional, Dict, List, Any
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import json
import urllib.parse

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
AGENT_SYSTEM_PROMPT = """You are OperatorAgent, an autonomous assistant that can browse the web to answer user questions.\nYou have a single tool available: `search(query)` which executes a web search and returns the top results.\nWhen you want to use the tool respond with a compact one line JSON object: {\"action\": \"search\", \"query\": "<text>"}.\nWhen you are ready to answer the user respond with JSON: {\"action\": \"final\", \"answer\": "<final answer>", \"sources\": [<urls used>]}\nNever respond with anything other than JSON.\n"""

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
@@ -218,47 +221,192 @@ class UploadBody(BaseModel):
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

# ---------- Agent Utilities ----------
class AgentAskRequest(BaseModel):
    question: str
    max_steps: int = 4
    max_results: int = 3


class AgentAskResponse(BaseModel):
    answer: str
    sources: List[str]
    searches: List[Dict[str, Any]]


def _render_conversation(messages: List[Tuple[str, str]]) -> str:
    parts: List[str] = []
    for role, content in messages:
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "tool":
            parts.append(f"Tool: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


async def _call_model(prompt: str) -> str:
    if model_pipe is None:
        raise HTTPException(503, "Model not loaded. Set ENABLE_GPT_OSS=1 and restart.")

    loop = asyncio.get_running_loop()

    def _generate() -> str:
        outputs = model_pipe(prompt, return_full_text=False)
        if not outputs:
            return ""
        return outputs[0].get("generated_text", "").strip()

    return await loop.run_in_executor(None, _generate)


async def _perform_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    if _browser is None:
        raise HTTPException(503, "Browser not ready. Try again later.")

    ctx = await _browser.new_context()
    page = await ctx.new_page()
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://duckduckgo.com/?q={encoded_query}&ia=web"
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_selector("article", timeout=DEFAULT_TIMEOUT_MS)
        results: List[Dict[str, Any]] = await page.evaluate(
            """
            (maxItems) => {
                const articles = Array.from(document.querySelectorAll('article'))
                    .filter(a => a.querySelector('h2') || a.querySelector('a[href]'))
                    .slice(0, maxItems);
                return articles.map(article => {
                    const titleEl = article.querySelector('h2');
                    const linkEl = article.querySelector('a[href]');
                    const snippetEl = article.querySelector('[data-testid="result-snippet"]') || article.querySelector('p');
                    return {
                        title: titleEl ? titleEl.innerText.trim() : '',
                        link: linkEl ? linkEl.href : '',
                        snippet: snippetEl ? snippetEl.innerText.trim() : ''
                    };
                });
            }
            """,
            max_results,
        )
        return results[:max_results]
    finally:
        await ctx.close()


def _format_search_observation(query: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return f"search results for '{query}' were empty."
    lines = [f"search results for '{query}':"]
    for idx, item in enumerate(results, start=1):
        title = item.get("title") or "(no title)"
        snippet = item.get("snippet") or ""
        link = item.get("link") or ""
        lines.append(f"{idx}. {title}\nURL: {link}\nSnippet: {snippet}")
    return "\n".join(lines)


# ---------- Agent Endpoint ----------
@app.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(req: AgentAskRequest) -> AgentAskResponse:
    """Drive the GPT-OSS agent through iterative DuckDuckGo searches.

    Args:
        req: Incoming payload containing the natural-language ``question`` along with
            optional bounds on the number of tool calls (``max_steps``) and how many
            DuckDuckGo results to parse per search (``max_results``).

    Returns:
        A structured :class:`AgentAskResponse` populated with the model's final answer,
        cited sources, and a full search log showing every query plus the scraped
        metadata that was fed back to the LLM as tool observations.

    Raises:
        HTTPException: Surfaces any agent failure such as invalid JSON actions,
        exceeding the step limit, or issues performing the external web search.
    """
    messages: List[Tuple[str, str]] = [("system", AGENT_SYSTEM_PROMPT), ("user", req.question)]
    search_history: List[Dict[str, Any]] = []

    for _ in range(max(1, req.max_steps)):
        prompt = _render_conversation(messages)
        raw_response = await _call_model(prompt)
        if not raw_response:
            raise HTTPException(500, "Model returned empty response.")

        messages.append(("assistant", raw_response))
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise HTTPException(500, f"Model returned invalid JSON: {exc}")

        action = payload.get("action")
        if action == "search":
            query = payload.get("query")
            if not query:
                raise HTTPException(500, "Model search action missing 'query'.")
            results = await _perform_search(query, req.max_results)
            search_history.append({"query": query, "results": results})
            observation = _format_search_observation(query, results)
            messages.append(("tool", observation))
            continue

        if action == "final":
            answer = payload.get("answer", "")
            sources = payload.get("sources", [])
            return AgentAskResponse(answer=answer, sources=sources, searches=search_history)

        raise HTTPException(500, f"Unknown agent action: {action}")

    raise HTTPException(500, "Agent reached step limit without providing an answer.")

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
