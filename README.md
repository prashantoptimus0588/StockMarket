# StockMarket (StockSense)

An investing **assessment agent**: describe a company (or compare a
few), choose a risk profile and holding horizon, and StockSense fetches
market data, evaluates **risk fit** and **return potential** using
deterministic engines, then produces a report for **human approval**.
After approval, you can ask follow-up questions about that report.

This is a research and workflow tool, **not financial advice**. Public
market data (Yahoo Finance / Google News RSS) can be incomplete or
delayed.

## LangGraph State & Workflow

### Shared State (`AssessmentState`)

``` python
query
profile
horizon
messages
tickers
price_data
fundamentals
news
risk_results
return_results
verdict
report_markdown
report_json
assessment_id
approval_status
approval_comment
errors
```

> Keep these names aligned with the actual `AssessmentState` definition
> in the codebase if it changes.

### Workflow Diagram

The Mermaid diagram below is embedded directly in this README. GitHub
renders it as a diagram, so readers can view the workflow without
opening a separate image.

``` mermaid
flowchart TD
    A([START]) --> B[brain_node<br/>Plan assessment and decide next action]
    B --> C{Tool call needed?}
    C -- Yes --> D[MCP ToolNode<br/>Market data, fundamentals, ticker search, news]
    D --> E[collect_data]
    E --> F[risk_engine<br/>Deterministic risk and return analysis]
    F --> G{More research or tools needed?}
    G -- Yes --> B
    G -- No --> H[report_node<br/>Build Markdown and JSON report]
    C -- No, assessment complete --> H
    H --> I[approval_node<br/>Human review / interrupt]
    I --> J{Decision}
    J -- Approved --> K[Persist approval and final assessment]
    K --> L([END])
    J -- Rejected --> M[Capture feedback]
    M --> N[Replan assessment]
    N --> B

    G -. MAX_STEPS guardrail .-> O[Stop runaway tool loops]
    O --> H
```

### Workflow at a glance

1.  **Plan:** `brain_node` interprets the request, risk profile, and
    holding horizon.
2.  **Gather evidence:** the graph calls MCP tools for ticker
    resolution, prices, fundamentals, and optional news.
3.  **Assess:** `collect_data` organizes tool outputs; the deterministic
    risk/return engines calculate assessment metrics.
4.  **Research and report:** the graph synthesizes the available
    evidence and creates Markdown and JSON report artifacts.
5.  **Human approval:** `approval_node` pauses for an approve/reject
    decision.
6.  **Continue or replan:** an approved report is finalized and can be
    used for follow-up Q&A; a rejection feeds comments back into another
    planning pass.
7.  **Guardrails and persistence:** a step limit helps prevent runaway
    tool loops, while PostgreSQL stores assessment records and LangGraph
    checkpoints.

## What it does

1.  Plans what to look up (tickers, news, and other relevant
    information) with an LLM.
2.  Resolves company names to exchange tickers (`SBIN.NS`, `AAPL`, and
    others).
3.  Fetches prices, fundamentals, and optional news through an MCP
    server.
4.  Scores the stock with pure Python engines (not the LLM):
    -   **Risk fit** against `aggressive`, `balanced`, or `conservative`
        profiles and `short`, `medium`, or `long` horizons.
    -   **Return potential**, a relative indicator rather than a
        forecast.
5.  Produces Markdown and JSON reports, pauses for human approval, and
    can replan if a report is rejected.
6.  Answers follow-up questions about the saved report.

## Architecture

``` text
CLI or FastAPI (api.py)
        |
        v
LangGraph workflow (chat/)
        |-- Groq: planning / research
        |-- MCP tools (mcp-server/)
        |     |-- ticker search
        |     |-- price history / market metrics
        |     |-- fundamentals
        |     |-- company and general news
        |
        |-- Deterministic risk / return engines (analysis/)
        |-- Report builders (report/)
        |-- PostgreSQL: assessment records + graph checkpoints
        |
        v
Markdown / JSON report -> Human approval -> Report Q&A
```

Reports are also written under `reports/markdown/` and `reports/json/`.

## Requirements

-   Python **3.12+**
-   [`uv`](https://docs.astral.sh/uv/)
-   PostgreSQL (assessment records and LangGraph checkpoints)
-   API keys:
    -   Groq: `GROQ_API_KEY`
    -   Google Gemini: `GOOGLE_API_KEY`

## Setup

``` bash
git clone https://github.com/prashantoptimus0588/StockMarket.git
cd StockMarket
uv sync
```

Create your environment file:

``` powershell
# Windows PowerShell
Copy-Item .env.example .env
```

``` bash
# macOS / Linux
cp .env.example .env
```

Edit `.env` with your credentials and configuration. **Do not commit
`.env`.**

  -----------------------------------------------------------------------
  Variable                            Purpose
  ----------------------------------- -----------------------------------
  `MCP_SERVER_URL`                    Streamable HTTP MCP endpoint,
                                      e.g. `http://localhost:8001/mcp`

  `DATABASE_URL`                      PostgreSQL connection URL; use
                                      `sslmode=require` if your host
                                      requires it

  `GROQ_API_KEY`                      Agent / graph LLM

  `GOOGLE_API_KEY`                    Report Q&A LLM

  `AGENT_MODEL`                       Groq model; default
                                      `openai/gpt-oss-120b`

  `QNA_MODEL`                         Gemini model; default
                                      `gemini-3-flash-preview`

  `CORS_ORIGINS`                      Allowed frontend origins; default
                                      includes `http://localhost:5173`

  `HOST` / `PORT`                     API bind address; `.env.example`
                                      defaults to `127.0.0.1:8000`

  `ALLOW_MEMORY_FALLBACK`             If `true`, allows an in-memory
                                      checkpointer when PostgreSQL is
                                      unavailable

  `LANGSMITH_*`                       Optional tracing configuration
  -----------------------------------------------------------------------

Initialize the database tables:

``` bash
uv run python -m database.init_db
```

> If your repository currently names the module `database.inti_db`, use
> the actual filename present in the repository. The conventional
> spelling is `init_db`.

## Run locally

Run the MCP server and API as **two separate processes**. Keep them on
different ports.

### 1. MCP market/news server

``` powershell
# Windows PowerShell
$env:PORT=8001
uv run python mcp-server/server.py
```

``` bash
# macOS / Linux
PORT=8001 uv run python mcp-server/server.py
```

### 2. FastAPI agent API

``` bash
uv run python api.py
```

Health check:

``` text
GET http://127.0.0.1:8000/health
```

### CLI

``` bash
uv run python -m chat.chat
```

The CLI asks for a risk profile, holding horizon, and a free-text query
(company name or comparison).

## HTTP API

Base URL: `http://127.0.0.1:8000`

  -------------------------------------------------------------------------------------
  Method                  Path                                  Description
  ----------------------- ------------------------------------- -----------------------
  `GET`                   `/health`                             Runtime status and
                                                                discovered MCP tools

  `POST`                  `/assessments`                        Start an assessment;
                                                                returns when the report
                                                                awaits approval

  `POST`                  `/assessments/stream`                 Stream progress using
                                                                server-sent events
                                                                (SSE): `started` →
                                                                `progress` → `result`

  `GET`                   `/assessments/{thread_id}`            Retrieve status/report;
                                                                supports polling or
                                                                refresh

  `POST`                  `/assessments/{thread_id}/decision`   Submit `approved` or
                                                                `rejected`, optionally
                                                                with a comment

  `POST`                  `/assessments/{thread_id}/qna`        Ask a question about
                                                                the saved report
  -------------------------------------------------------------------------------------

Example assessment request:

``` json
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

  -----------------------------------------------------------------------
  Path                                Role
  ----------------------------------- -----------------------------------
  `api.py`                            FastAPI application

  `chat/`                             LangGraph workflow, runtime,
                                      prompts, and chat/report
                                      interaction

  `mcp-server/server.py`              MCP tools for ticker search,
                                      prices, fundamentals, and news

  `analysis/`                         Deterministic risk and return
                                      engines

  `models/`                           Shared Pydantic payloads
                                      (`PriceData`, `UserPrefs`, and
                                      related models)

  `database/`                         SQLAlchemy models, database
                                      connection, repository, and
                                      initialization

  `report/`                           Markdown and JSON report builders

  `reports/`                          Generated artifacts (gitignored)

  `.env.example`                      Environment-variable template
  -----------------------------------------------------------------------

Important chat modules include `chat.py` (runtime and CLI),
`workflow.py` (graph definition), `assessment_logic.py` (assessment
rules and guardrails), `llm_utils.py` (LLM providers), and `config.py`
(timeouts and tool names). See `REFACTORING.md` for module-specific
notes.

## Tests

``` bash
uv run pytest analysis/test_risk_engine.py analysis/test_return_engine.py test/test_repository.py
```

Risk and return tests are pure Python and do not require network access.
Repository tests require a configured database.

## MCP tools

The agent expects these tools from the MCP server:

  -------------------------------------------------------------------------
  Tool                      Required                Use
  ------------------------- ----------------------- -----------------------
  `search_ticker`           Yes                     Resolve a company name
                                                    to listing candidates

  `getStock_price_info`     Yes                     Retrieve OHLCV history
                                                    and price-derived
                                                    metrics

  `getStock_fundamentals`   Yes                     Retrieve valuation,
                                                    profitability, debt,
                                                    and related
                                                    fundamentals

  `get_stock_news`          No                      Retrieve company
                                                    headlines

  `get_general_news`        No                      Retrieve sector/topic
                                                    headlines

  `get_weather`             Demo / optional         Placeholder weather
                                                    tool
  -------------------------------------------------------------------------

`backtest_signal` is intentionally not listed as an active MCP tool in
this README.

Indian listings commonly use an exchange suffix such as `RELIANCE.NS` or
`TCS.BO`. US tickers are typically unsuffixed, such as `AAPL`.

## Notes and limitations

-   StockSense is a research and workflow tool, **not financial
    advice**.
-   Market data and news feeds can be delayed, incomplete, or
    unavailable.
-   Return potential is a relative indicator, not a guaranteed or
    predicted return.
-   Human approval is part of the workflow; generated assessments should
    be reviewed before being relied upon.
-   Keep API keys, database credentials, and other secrets in `.env` or
    your deployment's secret manager. Never commit them.
