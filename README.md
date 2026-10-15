# StockMarket (StockSense)

An investing **assessment agent**: you describe a company (or compare a few), pick a risk profile and holding horizon, and the system fetches live market data, scores **risk fit** and **return potential** with deterministic engines, then writes a report for **human approval**. After approval you can ask follow-up questions about that report.

This is a research and workflow tool, **not financial advice**. Numbers come from public market data (Yahoo Finance / Google News RSS) and can be incomplete or delayed.

## What it does

1. Plans what to look up (tickers, news, extras) with an LLM.
2. Resolves company names to exchange tickers (`SBIN.NS`, `AAPL`, …).
3. Fetches price, fundamentals, and optional news in parallel via an MCP server.
4. Scores the stock with pure Python engines (no LLM):
   - **Risk fit** vs `aggressive` / `balanced` / `conservative` and `short` / `medium` / `long`.
   - **Return potential** (relative indicator, not a forecast).
5. Writes a Markdown + JSON report, pauses for approve/reject, and can replan if rejected.
6. Answers questions about the saved report.

## Architecture

```
CLI or FastAPI (api.py)
        │
        ▼
  LangGraph workflow (chat/)
        │     ├── Groq (planning / research)
        │     └── Gemini (report Q&A)
        ├── MCP tools (mcp-server/)  → yfinance, news RSS
        ├── Risk / return engines (analysis/)
        └── PostgreSQL (assessments + LangGraph checkpointer)
```

Reports are also written under `reports/markdown/` and `reports/json/`.

## Requirements

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/)
- **PostgreSQL** (used for assessment rows and graph checkpoints)
- API keys: **Groq** (`GROQ_API_KEY`) and **Google Gemini** (`GOOGLE_API_KEY`)

## Setup

```bash
git clone <repo-url>
cd StockMarket
uv sync
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env` with your credentials. Do not commit `.env`.

| Variable | Purpose |
|---|---|
| `MCP_SERVER_URL` | Streamable HTTP MCP endpoint, e.g. `http://localhost:8001/mcp` |
| `DATABASE_URL` | Postgres URL (`sslmode=require` if your host needs it) |
| `GROQ_API_KEY` | Agent / graph LLM |
| `GOOGLE_API_KEY` | Report Q&A LLM |
| `AGENT_MODEL` | Groq model (default `openai/gpt-oss-120b`) |
| `QNA_MODEL` | Gemini model (default `gemini-3-flash-preview`) |
| `CORS_ORIGINS` | Frontend origins (default includes `http://localhost:5173`) |
| `HOST` / `PORT` | API bind address (default `127.0.0.1:8000` in `.env.example`) |
| `ALLOW_MEMORY_FALLBACK` | If `true`, in-memory checkpointer is allowed when Postgres is down |
| `LANGSMITH_*` | Optional tracing |

Create tables once:

```bash
uv run python -m database.inti_db
```

## Run locally

Use **two processes**. Keep the MCP server on a different port than the API.

**1. MCP market/news server** (Yahoo Finance + news tools):

```bash
# Windows PowerShell
$env:PORT=8001; uv run python mcp-server/server.py
```

```bash
# macOS / Linux
PORT=8001 uv run python mcp-server/server.py
```

**2. Agent API:**

```bash
uv run python api.py
```

Health check: `GET http://127.0.0.1:8000/health`

**CLI** (interactive query, approval, then Q&A):

```bash
uv run python -m chat.chat
```

You will be asked for profile, horizon, and a free-text query (company name or comparison).

## HTTP API

Base URL: `http://127.0.0.1:8000`

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Runtime + discovered MCP tools |
| `POST` | `/assessments` | Start a run; returns when the report awaits approval |
| `POST` | `/assessments/stream` | Same flow as SSE (`started` → `progress` → `result`) |
| `GET` | `/assessments/{thread_id}` | Status / report (refresh, polling) |
| `POST` | `/assessments/{thread_id}/decision` | `approved` or `rejected` (optional `comment`) |
| `POST` | `/assessments/{thread_id}/qna` | Question about the saved report |

Example start:

```json
{
  "query": "I'm considering Maruti Suzuki. Look up the ticker, price trend, fundamentals, and recent news.",
  "profile": "balanced",
  "horizon": "long"
}
```

`profile` must be `aggressive`, `balanced`, or `conservative`.  
`horizon` must be `short`, `medium`, or `long`.  
A request can cover up to **4 tickers**.

## Project layout

| Path | Role |
|---|---|
| `api.py` | FastAPI app |
| `chat/` | LangGraph workflow, runtime, prompts, report files |
| `mcp-server/server.py` | MCP tools: ticker search, price, fundamentals, news, weather |
| `analysis/` | Deterministic risk and return engines |
| `models/` | Shared Pydantic payloads (`PriceData`, `UserPrefs`, …) |
| `database/` | SQLAlchemy models, connection, repository |
| `report/` | Markdown / JSON report builders |
| `reports/` | Generated artifacts (gitignored) |
| `.env.example` | Environment template |

Chat modules: `chat.py` (runtime + CLI), `workflow.py` (graph), `assessment_logic.py` (guardrails), `llm_utils.py` (providers), `config.py` (timeouts and tool names). See `REFACTORING.md` for module notes.

## Tests

```bash
uv run pytest analysis/test_risk_engine.py analysis/test_return_engine.py test/test_repository.py
```

Risk/return tests are pure (no network). Repository tests need a configured database.

## MCP tools

The agent expects these tools on the MCP server:

| Tool | Required | Use |
|---|---|---|
| `search_ticker` | yes | Company name → listing candidates |
| `getStock_price_info` | yes | OHLCV + volatility / drawdown |
| `getStock_fundamentals` | yes | Valuation, profitability, debt, … |
| `get_stock_news` | no | Company headlines |
| `get_general_news` | no | Sector / topic headlines |
| `get_weather` | no | Placeholder weather (demo) |

Indian listings usually need an exchange suffix (`RELIANCE.NS`, `TCS.BO`). US tickers are typically unsuffixed (`AAPL`).

## License

Private / unspecified. Add a license file if you publish this repository.
