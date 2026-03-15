"""
Ecolab AI Agent — CLI chat interface using Groq API
Free tier: extremely generous limits, ~500 tokens/sec inference speed.
No local model download. No Docker. No billing required.

Setup:
  1. Get a free API key: https://console.groq.com/keys
  2. export GROQ_API_KEY="your-key-here"
  3. pip install -r requirements.txt
  4. python scripts/ingest.py   (one-time)
  5. python agent.py

To use a different model:
  GROQ_MODEL=llama-3.1-8b-instant python agent.py
"""

import asyncio
import contextlib
import json
import logging
import os
import sys
import textwrap

import httpx

# ── Silence library noise ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.CRITICAL)
for lib in ["weaviate", "sentence_transformers", "httpx", "httpcore", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.CRITICAL)
logger = logging.getLogger("ecolab.agent")

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


# ── fd silencer (suppresses Weaviate embedded subprocess JSON logs) ───────────
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


# ── Groq API call (OpenAI-compatible) ────────────────────────────────────────

async def groq_chat(messages: list[dict]) -> dict:
    """
    Call Groq's OpenAI-compatible chat completions endpoint with tool support.
    Returns the raw message dict from choices[0].message.
    """
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
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_URL, json=payload, headers=headers)

    if r.status_code != 200:
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:400]}")

    return r.json()["choices"][0]["message"]


# ── Agentic loop ──────────────────────────────────────────────────────────────

async def agent_turn(conversation: list[dict]) -> str:
    """
    Agentic loop: call Groq → execute tool_calls if present → loop until final answer.
    Groq uses the OpenAI tool-calling format natively — no schema conversion needed.
    """
    MAX_TOOL_CALLS = 5

    for _ in range(MAX_TOOL_CALLS + 1):
        response_msg = await groq_chat(conversation)

        tool_calls = response_msg.get("tool_calls")

        # No tool calls → this is the final answer
        if not tool_calls:
            return response_msg.get("content", "").strip()

        # Append assistant message (with tool_calls) to history
        conversation.append(response_msg)

        # Execute each tool call and append results as role=tool messages
        for tc in tool_calls:
            fn   = tc["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            # Coerce integer fields — LLMs sometimes pass them as strings
            _INT_FIELDS = {"top_k", "limit"}
            for field in _INT_FIELDS:
                if field in args:
                    try:
                        args[field] = int(args[field])
                    except (ValueError, TypeError):
                        args.pop(field)   # drop it; Python default will be used

            tool_fn = TOOL_FUNCTIONS.get(name)
            if tool_fn is None:
                result = f"Error: unknown tool '{name}'"
            else:
                try:
                    with _silence_fds():
                        result = await tool_fn(**args)
                except Exception as e:
                    result = f"Tool error: {e}"

            conversation.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         name,
                "content":      result,
            })

    return "(Agent reached tool call limit without producing a final answer.)"


# ── CLI chat loop ─────────────────────────────────────────────────────────────

async def chat_loop():
    conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 60)
    print("  EcoAgent — Water & Environmental AI Assistant")
    print(f"  Model: {GROQ_MODEL} via Groq")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        conversation.append({"role": "user", "content": user_input})

        try:
            response = await agent_turn(conversation)
        except Exception as e:
            response = f"[Agent error: {e}]"
            logger.error(e, exc_info=True)

        wrapped = "\n".join(
            textwrap.fill(line, width=100) if len(line) > 100 else line
            for line in response.splitlines()
        )
        print(f"\nEcoAgent: {wrapped}\n")

        conversation.append({"role": "assistant", "content": response})


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY is not set.")
        print("Get a free key at: https://console.groq.com/keys")
        print("Then run:  export GROQ_API_KEY='your-key-here'")
        sys.exit(1)
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
