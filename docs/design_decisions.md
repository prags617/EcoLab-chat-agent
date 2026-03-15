# Design Decisions, Trade-offs, and Agent Architecture

## Overview

This document describes the key design decisions made in building the Ecolab AI Agent, the trade-offs considered, an explanation of how the agent combines RAG with tool calling, and a detailed comparison of the four LLM backend approaches evaluated during development.

---

## 1. Chat Interface: Web UI over CLI

**Decision:** Deliver the agent as a web application (React + Vite frontend, FastAPI backend) using the ag-ui streaming protocol, rather than a CLI-only tool.

**Rationale:** A browser-based interface makes the agent accessible to non-technical stakeholders — water treatment engineers, compliance officers, and sustainability managers — without requiring terminal access or Python knowledge. The ag-ui protocol provides structured streaming events (`TOOL_CALL_START`, `TEXT_MESSAGE_CONTENT`, etc.) that the frontend renders in real time, giving users visibility into what the agent is doing (which knowledge sources it searched, which APIs it called) as it works.

The CLI (`agent.py`) is retained as a lightweight fallback for local development and testing.

**Trade-off:** A web UI requires running two processes (backend + frontend) compared to one for the CLI. This is a minor operational overhead for significantly better usability.

---

## 2. Tool Architecture: Shared Python Module

**Decision:** Implement all three tools (`search_environmental_docs`, `get_water_quality_data`, `get_epa_facility_info`) as plain async Python functions in `tools/tools.py`, shared by both the CLI agent and the web backend.

**Rationale:** This eliminates code duplication between the two interfaces. The MCP server (`mcp-server/server.py`) wraps the same functions for Claude CLI use. The FastAPI backend calls them directly. A single change to a tool implementation propagates to all interfaces automatically.

Each tool is independently named, schema-validated, and described — matching the "skill tool architecture" pattern where the LLM selects and invokes tools based on intent rather than hard-coded routing logic.

**Trade-off:** The shared module means both interfaces must be deployed together (or the tools module must be on the Python path). In a microservices architecture, tools would be separate HTTP services. For this use case, co-location is simpler and faster.

---

## 3. Vector Database: Weaviate Embedded (no Docker)

**Decision:** Use **Weaviate Embedded** — Weaviate running as a Go binary subprocess launched directly by Python. Vectors are computed locally using `sentence-transformers` (BYO-vector pattern).

**Rationale:**
- Weaviate was specified in the requirements. The Embedded mode satisfies this without Docker or any container runtime.
- The BYO-vector pattern (embed in Python, store float vectors directly) decouples embedding from Weaviate, making the embedding model swappable without changing the DB schema.
- `multi-qa-MiniLM-L6-cos-v1` (~90MB) is optimized for question-answer retrieval and runs CPU-only, requiring no GPU.
- Data persists to `.weaviate_data/` on disk — ingestion is a one-time step.

**Trade-off:** ~190MB one-time download (embedding model + Weaviate binary). Compared to a managed cloud vector DB (Pinecone, Weaviate Cloud), this adds initial setup time but requires zero external accounts or API keys.

---

## 4. Knowledge Base: Curated Domain Documents

**Decision:** Embed domain knowledge as curated text strings in `scripts/ingest.py` rather than requiring external PDF downloads.

**Rationale:** Seven documents cover the core of Ecolab's domain: EPA drinking water regulations, WHO guidelines, water treatment engineering, Legionella control, wastewater treatment, Ecolab sustainability data, and water scarcity context. Embedding them in code makes the project fully self-contained with no external dependencies, broken URLs, or rate limits.

**Trade-off:** The knowledge base is static. Production systems would ingest live PDFs, crawl regulatory databases, and schedule re-ingestion. The `pypdf` dependency and README guidance on `load_pdf()` provide a clear extension path.

---

## 5. Public APIs: USGS NWIS + EPA ECHO

**Decision:** Integrate the USGS NWIS instantaneous values API (`waterservices.usgs.gov/nwis/iv/`) for real-time sensor data and the EPA ECHO Facility Registry for industrial facility lookups.

**Rationale:**
- Both are public, require no authentication, and directly serve Ecolab's business domain.
- USGS NWIS `/iv/` endpoint returns continuous sensor readings (dissolved oxygen, pH, temperature, in-situ nitrate via parameter code `99133`) — unlike the older Water Quality Portal which stopped serving recent USGS data after March 2024.
- EPA ECHO connects real industrial facilities and their NPDES/RCRA/SDWIS permits to the water quality conversation.

**Trade-off:** NWIS in-situ nitrate sensors (`99133`) are deployed at select stations only — not every state has coverage. Parameters like dissolved oxygen, pH, and streamflow have near-universal coverage. The tool returns a clear fallback message when a parameter has no coverage in a given state.

---

## 6. Observability: Structured JSON Logging + Metrics Endpoint

**Decision:** Implement structured JSON logging for every agent event (LLM calls, tool calls, request lifecycle) and expose a `GET /metrics` endpoint for aggregated runtime statistics.

**Rationale:** Structured logs (one JSON object per line) are parseable by any log aggregation platform (Render logs, Datadog, CloudWatch) without custom parsing rules. Each log line carries a `run_id` that ties all events within one user turn together, enabling end-to-end trace reconstruction. The `/metrics` endpoint surfaces token usage, tool call frequency, and latency distributions without needing a dedicated observability platform.

**Trade-off:** In-memory metrics are lost on server restart. Production systems would persist metrics to a time-series database (Prometheus, InfluxDB). For a local/demo deployment, in-memory is sufficient.

---

## 7. How the Agent Combines RAG and Tool Calling

The power of the architecture lies in the LLM orchestrating multiple tool calls dynamically within a single user turn:

**Pattern 1 — Standards + Live Measurement:**
> "What are the nitrate levels in Texas streams, and are they safe?"

1. `search_environmental_docs("EPA nitrate MCL drinking water")` → retrieves EPA 10 mg/L MCL, WHO 50 mg/L guideline, health effects of nitrate
2. `get_water_quality_data(state_code="TX", characteristic="nitrate")` → retrieves real USGS sensor readings from Texas stream gauges
3. Agent synthesizes: compares measured values against regulatory thresholds, flags any exceedances, recommends treatment options from RAG context

**Pattern 2 — Process Knowledge + Facility Context:**
> "Are there industrial dischargers in Texas that could affect water quality?"

1. `get_epa_facility_info(state_code="TX", program="NPDES")` → lists active permitted dischargers
2. `search_environmental_docs("NPDES industrial discharge water quality treatment")` → retrieves regulatory context and treatment requirements
3. Agent synthesizes: connects specific facilities to the regulatory framework they must comply with

**Pattern 3 — Pure RAG:**
> "How does Ecolab's 3D TRASAR technology work?"

Agent recognises this as a knowledge-only question and calls `search_environmental_docs("Ecolab 3D TRASAR cooling water")` — no API call needed.

The LLM decides which tools to call, in what order, and how to combine their outputs. This is the fundamental advantage of schema-driven tool calling over hard-coded retrieval pipelines.

---

## 8. LLM Backend Comparison

During development, four different LLM backends were evaluated. Below is a detailed comparison across the dimensions that matter for this use case.

### 8.1 Groq (llama-3.3-70b-versatile) ✅ Selected

Groq is a cloud inference platform offering free-tier access to open-source models at very high throughput.

| Attribute | Detail |
|-----------|--------|
| Model | llama-3.3-70b-versatile |
| Hosting | Cloud (api.groq.com) |
| API format | OpenAI-compatible |
| Tool calling | Native, structured JSON |
| Speed | ~500 tokens/sec (fastest available) |
| Free tier limits | 14,400 req/day, 500,000 tokens/min |
| Setup | `export GROQ_API_KEY=...` |
| Local download | None |
| Internet required | Yes |

**Why selected:** Groq's OpenAI-compatible API means our `TOOL_SCHEMAS` (already in OpenAI format) work without any conversion. Tool calls are returned as structured JSON — no regex parsing. The free tier is generous enough that rate limiting is not a practical concern for demo usage. At ~500 tokens/sec, responses feel near-instant compared to other options.

**Limitation:** Requires internet access and a Groq account. Not suitable for air-gapped environments.

---

### 8.2 Google Gemini (gemini-2.0-flash)

Gemini is Google's hosted LLM API with a free tier.

| Attribute | Detail |
|-----------|--------|
| Model | gemini-2.0-flash |
| Hosting | Cloud (generativelanguage.googleapis.com) |
| API format | Gemini-specific (not OpenAI-compatible) |
| Tool calling | Native, but different schema format |
| Speed | Fast (~200–400 tokens/sec) |
| Free tier limits | 15 RPM, 1M tokens/day |
| Setup | `export GEMINI_API_KEY=...` |
| Local download | None |
| Internet required | Yes |

**Why not selected:** Gemini's tool schema format differs from OpenAI's — `functionDeclarations` instead of `tools`, no `default` values allowed in parameter schemas, and tool results use `functionResponse` parts in a `user` turn rather than a `tool` role. This required a non-trivial schema conversion layer (`_to_gemini_tools()`) and a custom message history builder (`_build_request()`). Additionally, the free tier's 15 RPM limit caused `429` errors even for single-message turns, making it unreliable for interactive use.

**When to prefer:** If Google ecosystem integration is required, or if Gemini's multimodal capabilities (image input) are needed for future features like reading water quality reports from photos.

---

### 8.3 Ollama (local server)

Ollama is a local model runner that exposes an OpenAI-compatible API over HTTP.

| Attribute | Detail |
|-----------|--------|
| Model | qwen2.5:14b (recommended) |
| Hosting | Local (localhost:11434) |
| API format | OpenAI-compatible |
| Tool calling | Native, structured JSON |
| Speed | ~15–40 tokens/sec (M2 Mac, CPU) |
| Free tier limits | Unlimited (fully local) |
| Setup | `brew install ollama && ollama pull qwen2.5:14b` |
| Local download | ~9GB (14B model) |
| Internet required | No (after initial pull) |

**Why not selected as default:** Ollama requires a one-time ~9GB model download before use, and inference speed on CPU (~15–40 tok/sec) makes responses noticeably slower than Groq. However, Ollama is the correct choice for environments without internet access or where data privacy prevents sending queries to external APIs. The implementation is clean — since Ollama uses the OpenAI format, the same `TOOL_SCHEMAS` and tool execution loop work without modification.

**When to prefer:** Air-gapped deployments, regulated environments where query data cannot leave the network, or when the Groq free tier limits become a constraint at scale.

---

### 8.4 HuggingFace Transformers In-Memory (Qwen2.5-7B-Instruct)

Running a quantized model directly in Python using `transformers` + `bitsandbytes`, with no separate server process.

| Attribute | Detail |
|-----------|--------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Hosting | In-process (Python) |
| API format | Custom (manual tool schema injection + regex parsing) |
| Tool calling | Manual — injected via system prompt, parsed with regex |
| Speed | ~5–15 tokens/sec (Apple Silicon CPU, fp16) |
| Free tier limits | Unlimited (fully local) |
| Setup | `pip install transformers torch accelerate` |
| Local download | ~15GB (7B fp16) |
| Internet required | No (after initial download) |

**Why not selected:** This approach has the highest friction of any option. The `transformers` library does not expose a standard tool-calling API — tool schemas must be injected into the system prompt as JSON and model outputs must be parsed with regex to extract tool calls. This parsing is fragile; the model sometimes outputs tool calls as strings instead of JSON objects (e.g. `"top_k": "5"` instead of `"top_k": 5`), requiring defensive coercion. On Apple Silicon (MPS), the `bitsandbytes` 4-bit quantization library is unsupported, forcing fp16 loading which requires ~15GB RAM. Generation speed is ~5–15 tokens/sec — noticeably slow for interactive use.

**When to prefer:** Strictly offline environments with no ability to install Ollama as a system service, or research contexts requiring direct access to model internals (logits, attention weights, fine-tuning).

---

### 8.5 Comparison Summary

| Dimension | Groq | Gemini | Ollama | Transformers (Qwen) |
|-----------|------|--------|--------|---------------------|
| Internet required | Yes | Yes | No (after pull) | No (after download) |
| API key required | Yes (free) | Yes (free) | No | No |
| Local disk usage | None | None | ~9GB | ~15GB |
| Tool calling format | OpenAI native | Custom conversion needed | OpenAI native | Manual prompt injection |
| Inference speed | ~500 tok/s | ~300 tok/s | ~15–40 tok/s | ~5–15 tok/s |
| Free tier limits | Very generous | 15 RPM (restrictive) | Unlimited | Unlimited |
| Setup complexity | Low | Medium | Medium | High |
| Privacy (data stays local) | No | No | Yes | Yes |
| Best for | Interactive demos, development | Google ecosystem | Privacy-first, offline | Research, no Ollama |

**Bottom line:** Groq was chosen for its combination of zero local setup, OpenAI-compatible tool calling, and best-in-class inference speed on the free tier. For production deployments in regulated or air-gapped environments, Ollama with `qwen2.5:14b` is the recommended alternative with minimal code changes (swap the API endpoint URL and remove the API key).

---

## 9. Chat Context Preservation and Multi-Session Management

### 9.1 Within-Session Context (Conversation Memory)

**Decision:** Send the full conversation history to the backend on every request rather than maintaining server-side session state.

**Rationale:** LLMs are stateless — they have no memory between API calls. The standard pattern for multi-turn conversations is to accumulate the full message history on the client and send it with each new request. This is what the frontend does: every call to `POST /api/chat` includes all prior `user` and `assistant` messages for that thread, allowing the LLM to reference anything said earlier in the conversation.

This means follow-up questions work naturally:
```
You:      What are the EPA nitrate limits?
EcoAgent: The MCL is 10 mg/L as nitrogen...
You:      How does that compare to Texas streams?   ← references prior answer
EcoAgent: [calls get_water_quality_data, then compares against the 10 mg/L just discussed]
```

The backend prepends the system prompt to the history on each call and passes the full context to Groq. `tool_call` and `tool_result` messages are filtered out before sending to the backend — only `user` and `assistant` roles are included, as these are what the LLM needs for conversational continuity.

**Trade-off:** Sending the full history on every request increases token usage as conversations grow longer. For very long conversations this could hit Groq's context window limit (128k tokens for llama-3.3-70b) or increase latency. In practice, water quality conversations are unlikely to reach this limit in normal usage.

---

### 9.2 Persistence Across Page Refreshes (localStorage)

**Decision:** Persist all conversation threads to `localStorage` in the browser, keyed by `thread_id`.

**Rationale:** Without persistence, refreshing the page loses the entire conversation history, forcing users to re-establish context. `localStorage` is a simple, zero-dependency solution that survives page refreshes and browser restarts. All thread data (messages, title, timestamps) is serialised to JSON and written to `localStorage` on every state change via a `useEffect` hook.

```javascript
// Persists automatically whenever threads state changes
useEffect(() => { saveThreads(threads); }, [threads]);
```

On load, threads are read back from `localStorage` and the previously active thread is restored:
```javascript
const [threads] = useState(() => {
  const stored = loadThreads();
  return Object.keys(stored).length ? stored : { [t.id]: newThread() };
});
```

**Trade-off:** `localStorage` is browser-local — conversations do not sync across devices or browsers. For a shared or multi-user deployment, server-side session storage (Redis, PostgreSQL) would be needed. For a single-user local deployment, `localStorage` is sufficient and requires no additional infrastructure.

---

### 9.3 Multiple Independent Chat Sessions

**Decision:** Support multiple named chat threads in a sidebar, each with its own isolated message history and `thread_id`.

**Rationale:** Different investigations require different contexts. A user asking about Legionella control in one thread should not have that context bleed into a separate thread about nitrate compliance. Each thread maintains:
- A unique `thread_id` (UUID) sent to the backend with every request
- Its own message array stored independently in `localStorage`
- An auto-derived title from the first user message (truncated to 40 characters)

Switching threads in the sidebar immediately replaces the message display and changes which history is sent on the next request — the backend receives only the messages from the active thread.

**Thread lifecycle:**
- **Create:** clicking `＋` creates a new thread with an empty message history and a fresh `thread_id`
- **Auto-title:** the first user message in a thread becomes its sidebar label
- **Switch:** clicking a thread in the sidebar loads its full history instantly from `localStorage`
- **Delete:** hovering a thread reveals a `✕` button; deletion removes it from `localStorage` and switches focus to another thread (or creates a new one if none remain)

**Trade-off:** All threads share the same browser's `localStorage` quota (~5MB). Each message is small (typically <2KB), so hundreds of conversations are supported before approaching limits. If a user needs to export or share conversation history, that would require a server-side persistence layer.

---

## 10. Summary of All Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| Web UI over CLI only | Accessible to non-technical users; shows tool call visibility | Requires running two processes |
| Shared tools module | No code duplication across CLI and web | Tools must be on Python path for both |
| Weaviate Embedded | No Docker; fully local; persists to disk | ~190MB one-time download |
| Curated documents | Self-contained; zero external dependencies | Static knowledge; no live crawling |
| USGS NWIS `/iv/` over WQP | Current data; sensor readings | Nitrate sparse; US-only coverage |
| Groq over Gemini | No schema conversion; generous free limits | Requires internet + API key |
| Groq over Ollama | Zero local download; 10–30x faster | Not suitable for air-gapped environments |
| Structured JSON logs | Parseable by any log aggregator | In-memory metrics lost on restart |
| Full history per request | Stateless backend; simple scaling | Token usage grows with conversation length |
| localStorage persistence | Zero infrastructure; survives refresh | Browser-local only; ~5MB quota |
| Multi-thread sidebar | Isolated contexts per investigation | No cross-device sync |
