# Ecolab AI Agent

Intelligent water and environmental AI combining RAG + API tool calling.
Available as a CLI and a web app with full observability and cloud deployment.

---

## Architecture

```
React Frontend (Vite)           hosted on Vercel
     |  ag-ui SSE stream
     v
FastAPI Backend                 hosted on Render
     |-- search_environmental_docs --> Weaviate Embedded (RAG)
     |-- get_water_quality_data    --> USGS NWIS API
     +-- get_epa_facility_info     --> EPA ECHO API
     |
     v  Groq API (llama-3.3-70b-versatile)
```

---

## Option A — Run locally

### Backend
```bash
cd web/backend
pip install -r requirements.txt
cd ../.. && python scripts/ingest.py && cd web/backend
export GROQ_API_KEY="your-key"
uvicorn main:app --reload --port 8086
```

### Frontend (new terminal)
```bash
cd web/frontend
rm -rf node_modules && npm install
npm start
# Open http://localhost:3000
```

---

## Option B — Deploy to cloud (free)

### 1. Deploy backend to Render

1. Push repo to GitHub
2. Go to https://dashboard.render.com → New → Blueprint
3. Connect your repo — Render auto-detects `render.yaml`
4. Set `GROQ_API_KEY` as a secret env var in the Render dashboard
5. Note your backend URL: `https://ecolab-agent-backend.onrender.com`

### 2. Update vercel.json with your Render URL

Edit `vercel.json` — replace the backend URL in the routes section:
```json
"dest": "https://YOUR-RENDER-APP.onrender.com/api/$1"
```

### 3. Deploy frontend to Vercel

```bash
npm install -g vercel
cd web/frontend
vercel --prod
```

Or connect your GitHub repo at https://vercel.com/new and Vercel auto-deploys on every push.

---

## Observability

### Structured JSON logs

Every event emits a JSON log line to stdout:

```json
{"ts": "2026-03-15T10:23:01", "level": "INFO", "logger": "ecolab.backend", "msg": "run_started",   "run_id": "abc123", "thread_id": "xyz", "n_messages": 3}
{"ts": "2026-03-15T10:23:01", "level": "INFO", "logger": "ecolab.backend", "msg": "llm_call",      "run_id": "abc123", "model": "llama-3.3-70b-versatile", "elapsed_ms": 1240, "prompt_tokens": 412, "completion_tokens": 38, "tool_calls": ["search_environmental_docs"]}
{"ts": "2026-03-15T10:23:02", "level": "INFO", "logger": "ecolab.backend", "msg": "tool_call",     "run_id": "abc123", "tool": "search_environmental_docs", "elapsed_ms": 340, "result_len": 1820}
{"ts": "2026-03-15T10:23:04", "level": "INFO", "logger": "ecolab.backend", "msg": "run_finished",  "run_id": "abc123", "elapsed_ms": 3210, "tool_calls": 2, "llm_loops": 2}
{"ts": "2026-03-15T10:23:04", "level": "INFO", "logger": "ecolab.backend", "msg": "request_finished", "method": "POST", "path": "/api/chat", "status": 200, "elapsed_ms": 3215}
```

### Metrics endpoint

```bash
curl http://localhost:8086/metrics
```

```json
{
  "requests":  { "total": 14, "errors": 0, "avg_latency_ms": 3210 },
  "llm":       { "calls": 22, "errors": 0, "tokens_prompt": 9840,
                 "tokens_completion": 1420, "tokens_total": 11260, "avg_latency_ms": 1180 },
  "tools":     { "calls_total": 31, "errors": 0,
                 "calls_by_name": { "search_environmental_docs": 18,
                                    "get_water_quality_data": 11,
                                    "get_epa_facility_info": 2 },
                 "avg_latency_ms": 420 },
  "uptime_s":  182.4
}
```

### Health endpoint

```bash
curl http://localhost:8086/health
# {"status": "ok", "model": "llama-3.3-70b-versatile", "uptime_s": 182.4}
```

---

## Project Structure

```
ecolab-agent/
├── agent.py                     # CLI chat loop
├── tools/
│   ├── __init__.py
│   └── tools.py                 # Shared tool implementations
├── web/
│   ├── backend/
│   │   ├── main.py              # FastAPI + ag-ui + observability
│   │   └── requirements.txt
│   └── frontend/
│       ├── package.json
│       ├── vite.config.js
│       ├── index.html
│       └── src/
│           ├── main.jsx
│           ├── App.jsx
│           ├── EcoAgentChat.jsx
│           └── index.css
├── scripts/ingest.py            # Weaviate document ingestion
├── mcp-server/server.py         # MCP wrapper
├── render.yaml                  # Render IaC (backend hosting)
├── vercel.json                  # Vercel config (frontend hosting)
├── requirements.txt
└── README.md
```
