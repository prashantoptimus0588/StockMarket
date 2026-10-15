# StockSense chat workflow refactor

`chat/chat.py` is now the runtime/service and CLI entry point. The graph's behavior and public service functions remain available from `chat.chat` for `api.py` compatibility.

## Module responsibilities

- `chat/chat.py` — `AgentRuntime` lifecycle, checkpointer setup, streaming/status helpers, report Q&A, CLI.
- `chat/workflow.py` — LangGraph nodes and routing, report assembly/persistence, approval interrupt and decision recording.
- `chat/state.py` — `AssessmentState` and progress labels.
- `chat/schemas.py` — structured LLM output schemas.
- `chat/prompts.py` — planning, ticker resolution, research and report Q&A prompts.
- `chat/assessment_logic.py` — deterministic risk/return formatting and guardrails on LLM suggestions.
- `chat/llm_utils.py` — model configuration, provider fallback and bounded LLM calls.
- `chat/tool_helpers.py` — MCP timeout wrappers and tool-result normalization.
- `chat/config.py` — shared runtime limits and MCP tool names.
- `chat/report_files.py` — report paths and atomic Markdown/JSON writes.
- `chat/common.py` — small pure helpers shared between modules.

## Persistence behavior retained

The graph's `report_node` remains the single place that builds the report and writes report artifacts plus the assessment row. The per-thread existing-assessment guard and `report_saved` state guard are retained. Approval remains a LangGraph interrupt and decisions are still recorded through the repository.

## Local setup

1. Copy `.env.example` to `.env` and add your own credentials. Do not commit `.env`.
2. Install dependencies using the existing `uv.lock` / `pyproject.toml` workflow.
3. Start the MCP server and PostgreSQL service as you normally do.
4. Start FastAPI using your existing command.

## Validation performed for this refactor

- Python syntax compilation passed for the refactored `chat/*.py` modules.
- Existing pure risk/return unit tests passed: 27 tests.
- Standalone imports passed for the extracted config, prompt, report-file, schema, assessment-logic and tool-helper modules.

A full live MCP/PostgreSQL/LLM integration run was not performed in this environment. Run the complete project tests and one stock assessment locally before replacing your working copy.
