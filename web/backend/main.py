"""
Ecolab AI Agent — FastAPI backend with ag-ui protocol + observability

Observability features:
  - Structured JSON logging (every request, tool call, LLM call, error)
  - Request middleware: logs method, path, status, duration for every HTTP call
  - Run-level tracing: run_id ties all events (tool calls, LLM calls) to one user turn
  - Tool call tracing: logs tool name, args, duration, result length per invocation
  - LLM call tracing: logs model, prompt tokens, completion tokens, latency
  - In-memory metrics: counters for requests, tool calls, errors, token usage
  - GET /metrics  — exposes aggregated metrics as JSON
  - GET /health   — liveness check with uptime and model info

Run locally:
  cd web/backend
  uvicorn main:app --reload --port 8086
"""

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
from pydantic import BaseModel

# ── Structured JSON logger ────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        # Attach any extra fields passed via extra={...}
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # Silence noisy libraries
    for lib in ["weaviate", "sentence_transformers", "httpx", "httpcore",
                "urllib3", "uvicorn.access"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger("ecolab.backend")

# ── Import shared tools ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tools.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are EcoAgent, an intelligent water and environmental AI assistant for Ecolab's domain. "
    "Help users with water treatment, water quality, environmental compliance, and sustainability.\n\n"
    "Tool usage rules:\n"
    "- search_environmental_docs: use for standards, regulations, treatment processes, Ecolab tech.\n"
    "- get_water_quality_data: use when user asks about measured water quality in a US location.\n"
    "- get_epa_facility_info: use for industrial facilities or permit questions.\n"
    "- Combine RAG + API results when relevant. Cite sources. Be specific with values and thresholds."
)

_START_TIME = time.time()

# ── In-memory metrics store ───────────────────────────────────────────────────
_metrics: dict = {
    "requests_total":        0,
    "requests_errors":       0,
    "tool_calls_total":      0,
    "tool_calls_by_name":    defaultdict(int),
    "tool_errors":           0,
    "llm_calls_total":       0,
    "llm_errors":            0,
    "tokens_prompt":         0,
    "tokens_completion":     0,
    "latency_llm_ms":        [],   # last 100 values
    "latency_tool_ms":       [],   # last 100 values
    "latency_request_ms":    [],   # last 100 values
}

def _record_latency(key: str, ms: float):
    _metrics[key].append(round(ms, 1))
    if len(_metrics[key]) > 100:
        _metrics[key].pop(0)

def _avg(lst: list) -> float:
    return round(sum(lst) / len(lst), 1) if lst else 0.0

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="EcoAgent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def _request_logger(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    t0     = time.perf_counter()

    logger.info("request_started", extra={
        "req_id": req_id,
        "method": request.method,
        "path":   request.url.path,
    })

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _metrics["requests_total"] += 1
    _record_latency("latency_request_ms", elapsed_ms)

    if response.status_code >= 400:
        _metrics["requests_errors"] += 1

    logger.info("request_finished", extra={
        "req_id":     req_id,
        "method":     request.method,
        "path":       request.url.path,
        "status":     response.status_code,
        "elapsed_ms": round(elapsed_ms, 1),
    })
    return response

# ── Request model ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages:  list[dict]
    thread_id: str = ""

# ── fd silencer ───────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _silence_fds():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull_fd)

# ── ag-ui event helper ────────────────────────────────────────────────────────
def _event(type_: str, **kwargs) -> str:
    return json.dumps({"type": type_, **kwargs}) + "\n"

# ── Groq call with tracing ────────────────────────────────────────────────────
async def groq_chat(messages: list[dict], run_id: str) -> dict:
    t0 = time.perf_counter()
    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "tools":       TOOL_SCHEMAS,
        "tool_choice": "auto",
        "temperature": 0.3,
        "max_tokens":  2048,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(GROQ_URL, json=payload, headers=headers)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _metrics["llm_calls_total"] += 1
        _record_latency("latency_llm_ms", elapsed_ms)

        if r.status_code != 200:
            _metrics["llm_errors"] += 1
            raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:400]}")

        data  = r.json()
        usage = data.get("usage", {})
        prompt_tokens     = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        _metrics["tokens_prompt"]     += prompt_tokens
        _metrics["tokens_completion"] += completion_tokens

        msg        = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []

        logger.info("llm_call", extra={
            "run_id":            run_id,
            "model":             GROQ_MODEL,
            "elapsed_ms":        round(elapsed_ms, 1),
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_calls":        [tc["function"]["name"] for tc in tool_calls],
            "has_content":       bool(msg.get("content")),
        })
        return msg

    except Exception as e:
        _metrics["llm_errors"] += 1
        logger.error("llm_error", extra={"run_id": run_id, "error": str(e)})
        raise

# ── Agentic loop — yields ag-ui events ───────────────────────────────────────
async def run_agent(
    messages: list[dict],
    thread_id: str,
) -> AsyncGenerator[str, None]:

    run_id    = str(uuid.uuid4())
    MAX_LOOPS = 5

    logger.info("run_started", extra={
        "run_id":    run_id,
        "thread_id": thread_id,
        "n_messages": len(messages),
    })

    yield _event("RUN_STARTED", run_id=run_id)

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    run_tool_calls = 0
    run_start      = time.perf_counter()

    try:
        for loop in range(MAX_LOOPS + 1):
            response_msg = await groq_chat(full_messages, run_id)
            tool_calls   = response_msg.get("tool_calls")

            # ── Final answer ──────────────────────────────────────────────────
            if not tool_calls or loop == MAX_LOOPS:
                content = response_msg.get("content", "").strip()
                msg_id  = str(uuid.uuid4())

                yield _event("TEXT_MESSAGE_START", message_id=msg_id, role="assistant")
                words = content.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    yield _event("TEXT_MESSAGE_CONTENT", message_id=msg_id, delta=chunk)
                    await asyncio.sleep(0.01)
                yield _event("TEXT_MESSAGE_END", message_id=msg_id)
                break

            # ── Tool calls ────────────────────────────────────────────────────
            full_messages.append(response_msg)

            for tc in tool_calls:
                fn       = tc["function"]
                name     = fn["name"]
                args_str = fn.get("arguments", "{}")
                tool_id  = tc.get("id", str(uuid.uuid4()))

                if isinstance(args_str, str):
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_str

                for field in {"top_k", "limit"}:
                    if field in args:
                        try:
                            args[field] = int(args[field])
                        except (ValueError, TypeError):
                            args.pop(field)

                yield _event("TOOL_CALL_START", tool_call_id=tool_id, tool_call_name=name)
                yield _event("TOOL_CALL_ARGS",  tool_call_id=tool_id, delta=json.dumps(args))
                yield _event("TOOL_CALL_END",   tool_call_id=tool_id)

                # Execute tool with timing
                tool_fn  = TOOL_FUNCTIONS.get(name)
                t0_tool  = time.perf_counter()
                _metrics["tool_calls_total"]      += 1
                _metrics["tool_calls_by_name"][name] += 1
                run_tool_calls += 1

                if tool_fn is None:
                    result = f"Error: unknown tool '{name}'"
                    _metrics["tool_errors"] += 1
                else:
                    try:
                        with _silence_fds():
                            result = await tool_fn(**args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                        _metrics["tool_errors"] += 1
                        logger.error("tool_error", extra={
                            "run_id": run_id,
                            "tool":   name,
                            "args":   args,
                            "error":  str(e),
                        })

                tool_elapsed_ms = (time.perf_counter() - t0_tool) * 1000
                _record_latency("latency_tool_ms", tool_elapsed_ms)

                logger.info("tool_call", extra={
                    "run_id":     run_id,
                    "tool":       name,
                    "tool_args":       args,
                    "elapsed_ms": round(tool_elapsed_ms, 1),
                    "result_len": len(result),
                })

                yield _event("TOOL_CALL_RESULT", tool_call_id=tool_id, content=result)

                full_messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_id,
                    "name":         name,
                    "content":      result,
                })

        run_elapsed_ms = (time.perf_counter() - run_start) * 1000
        logger.info("run_finished", extra={
            "run_id":        run_id,
            "thread_id":     thread_id,
            "elapsed_ms":    round(run_elapsed_ms, 1),
            "tool_calls":    run_tool_calls,
            "llm_loops":     loop + 1,
        })
        yield _event("RUN_FINISHED", run_id=run_id)

    except Exception as e:
        logger.error("run_error", extra={"run_id": run_id, "error": str(e)})
        yield _event("RUN_ERROR", message=str(e))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not GROQ_API_KEY:
        return JSONResponse({"error": "GROQ_API_KEY not set on server"}, status_code=500)

    return StreamingResponse(
        run_agent(request.messages, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "model":    GROQ_MODEL,
        "uptime_s": round(time.time() - _START_TIME, 1),
    }


@app.get("/metrics")
async def metrics():
    """Aggregated runtime metrics — requests, tool calls, token usage, latencies."""
    return {
        "requests": {
            "total":          _metrics["requests_total"],
            "errors":         _metrics["requests_errors"],
            "avg_latency_ms": _avg(_metrics["latency_request_ms"]),
        },
        "llm": {
            "calls":              _metrics["llm_calls_total"],
            "errors":             _metrics["llm_errors"],
            "tokens_prompt":      _metrics["tokens_prompt"],
            "tokens_completion":  _metrics["tokens_completion"],
            "tokens_total":       _metrics["tokens_prompt"] + _metrics["tokens_completion"],
            "avg_latency_ms":     _avg(_metrics["latency_llm_ms"]),
        },
        "tools": {
            "calls_total":    _metrics["tool_calls_total"],
            "errors":         _metrics["tool_errors"],
            "calls_by_name":  dict(_metrics["tool_calls_by_name"]),
            "avg_latency_ms": _avg(_metrics["latency_tool_ms"]),
        },
        "uptime_s": round(time.time() - _START_TIME, 1),
    }
