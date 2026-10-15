"""Investing agent: graph core + CLI.

Layout
------
* Everything that talks to the outside world (MCP tools, LLMs, Postgres) lives in AgentRuntime,
  which is created ONCE (CLI: in main(); FastAPI: in the lifespan handler).
* The graph keeps no per-request data in closures: query / prefs live in the state,
  thread_id comes from config["configurable"]["thread_id"].

Stock flow (deterministic where possible, LLM only where judgement is needed):

  plan_node (LLM, structured) -> resolve_node (search_ticker + LLM picks listing)
    -> fetch_node (code: price + fundamentals [+ news] in parallel)
    -> risk_engine_node (code) -> explain_node (LLM, structured, no tools, validated)
    -> report_node (code) -> approval_node (interrupt)
    reject (max MAX_REPLANS) loops back to plan_node for alternative companies; approve ends then QnA

Everything else (weather, general news, ...) goes to a small tool-calling agent loop.
"""
import asyncio
import json
import logging
import os
import re
import selectors
import sys
import time
from contextlib import AsyncExitStack
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Callable, Literal, Optional, Sequence
from uuid import uuid4

from dotenv import load_dotenv

# chat.py lives in <project-root>/chat/chat.py. Resolve .env from the project
# root so `uv run ...` works even when launched from another working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)  # before LangChain reads env vars

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from analysis.return_engine import CombinedVerdict, ReturnVerdict, combine, evaluate_return  # noqa: E402
from analysis.risk_engine import HORIZONS, PROFILES, evaluate_risk  # noqa: E402
from database.repository import get_assessment_by_thread_id, record_decision, save_assessment  # noqa: E402
from models.models import FundamentalsData, PriceData, RiskVerdict, UserPrefs  # noqa: E402
from report.report import build_multi_report, build_report, multi_report_to_dict, report_to_dict  # noqa: E402

log = logging.getLogger("agent")

from chat.config import (TOOL_TIMEOUT, MCP_CONNECT_TIMEOUT, DB_TIMEOUT, MAX_TICKERS, MAX_STEPS, RECURSION_LIMIT, FETCH_RETRY_DELAY, TOOL_SEARCH, TOOL_PRICE, TOOL_FUNDAMENTALS, TOOL_NEWS, REQUIRED_TOOLS, TOOL_ARG_NAMES)
from chat.llm_utils import (PROVIDER_GROQ, PROVIDER_GEMINI, AGENT_PROVIDER, AGENT_MODEL, AGENT_FALLBACK_MODELS, AGENT_MODEL_CANDIDATES, CHAT_PROVIDER, CHAT_MODEL, CHAT_FALLBACK_MODELS, CHAT_MODEL_CANDIDATES, build_llm, with_model_fallbacks, structured_output_shape, ask)
from chat.report_files import REPORTS_DIR, REPORT_MD_DIR, REPORT_JSON_DIR, RUN_LOG_DIR, safe_name, write_text_atomic, save_report_files
from chat.tool_helpers import with_timeout, with_connect_timeout, to_text, get_field, primary_arg, call_tool
from chat.schemas import Target, QueryPlan, TickerChoice, TickerResolution, Suggestion, TickerSuggestion, Explanation
from chat.prompts import build_plan_prompt, build_resolve_prompt, EXPLAIN_SYSTEM, build_general_prompt, QNA_MAX_TURNS, build_qna_system_prompt
from chat.assessment_logic import CORE_FUND_FIELDS, format_verdict, format_facts, parse_tool_result, build_assessments, engine_stance, validate_explanation, enforce_stance, render_answer
from chat.state import AssessmentState, NODE_LABELS
from chat.common import same_ticker


def setup_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Runtime: everything expensive is created once
# ---------------------------------------------------------------------------
class AgentRuntime:
    def __init__(self):
        self.tools: list = []
        self.tools_by_name: dict = {}
        self.graph = None
        self._stack = AsyncExitStack()
        self.planner = self.resolver = self.explainer = self.agent_llm = None

    async def start(self) -> "AgentRuntime":
        """Connect to MCP, validate its tools, initialize persistence, then compile the graph."""
        server_url = (os.getenv("MCP_SERVER_URL") or "").strip()
        if not server_url:
            raise RuntimeError(
                "MCP_SERVER_URL is missing. Set it in the project-root .env file to the "
                "MCP server's Streamable HTTP endpoint."
            )
        if not server_url.startswith(("http://", "https://")):
            raise RuntimeError(
                "MCP_SERVER_URL must start with http:// or https:// and point to the "
                "Streamable HTTP MCP endpoint."
            )

        # Do not log the URL itself: deployments may include credentials in it.
        log.info("Connecting to MCP server (transport=streamable_http; timeout=%ss) ...",
                 MCP_CONNECT_TIMEOUT)
        try:
            client = MultiServerMCPClient({
                "server": {
                    "url": server_url,
                    "transport": "streamable_http",
                }
            })
            discovered = await asyncio.wait_for(
                client.get_tools(), timeout=MCP_CONNECT_TIMEOUT
            )
            tools = [with_timeout(t) for t in discovered]
            names = [t.name for t in tools]
            log.info("MCP tool discovery succeeded: %d tool(s): %s", len(names), names)
            if not tools:
                raise RuntimeError(
                    "MCP connected but exposed zero tools. Check the MCP server's tool registration."
                )
            missing = [name for name in REQUIRED_TOOLS if name not in set(names)]
            if missing:
                raise RuntimeError(
                    f"MCP server connected, but required tool(s) are missing: {missing}. "
                    f"Tools discovered: {names}"
                )

            checkpointer = await self._make_checkpointer()
            self.build(tools, checkpointer)
            log.info("Agent runtime ready (MCP tools=%d).", len(tools))
            return self
        except asyncio.TimeoutError as exc:
            await self.stop()
            raise RuntimeError(
                f"Timed out after {MCP_CONNECT_TIMEOUT}s while connecting to MCP or listing tools. "
                "Confirm the MCP server is running and MCP_SERVER_URL points to its Streamable HTTP endpoint."
            ) from exc
        except Exception:
            log.exception("Agent runtime startup failed while connecting MCP/building graph")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Release resources even when startup failed part-way through."""
        await self._stack.aclose()

    def _make_llms(self) -> None:
        names = AGENT_MODEL_CANDIDATES
        self.planner = with_model_fallbacks(structured_output_shape(QueryPlan), AGENT_PROVIDER, names)
        self.resolver = with_model_fallbacks(structured_output_shape(TickerResolution), AGENT_PROVIDER, names)
        self.explainer = with_model_fallbacks(structured_output_shape(Explanation), AGENT_PROVIDER, names)
        self.agent_llm = with_model_fallbacks(lambda m: m.bind_tools(self.tools), AGENT_PROVIDER, names)

    def build(self, tools: list, checkpointer) -> None:
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}
        missing = [n for n in REQUIRED_TOOLS if n not in self.tools_by_name]
        if missing:
            raise RuntimeError(f"MCP server is missing required tools: {missing}")
        self._make_llms()
        from chat.workflow import build_graph
        self.graph = build_graph(self).compile(checkpointer=checkpointer)

    async def _make_checkpointer(self):
        url = os.getenv("DATABASE_URL")
        if not url:
            log.warning("DATABASE_URL not set: using in-memory checkpointer (pending approvals are lost on restart).")
            return MemorySaver()
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            pool = AsyncConnectionPool(
                conninfo=with_connect_timeout(url),
                max_size=int(os.getenv("PG_POOL_SIZE", "10")),
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                open=False,
            )
            await pool.open(wait=True, timeout=DB_TIMEOUT)
            self._stack.push_async_callback(pool.close)
            saver = AsyncPostgresSaver(pool)
            await asyncio.wait_for(saver.setup(), timeout=DB_TIMEOUT * 2)
            log.info("Postgres checkpointer ready.")
            return saver
        except Exception as e:
            if os.getenv("ALLOW_MEMORY_FALLBACK", "0") == "1":
                log.error("Postgres checkpointer unavailable (%s: %s). Falling back to MemorySaver.",
                          type(e).__name__, e)
                return MemorySaver()
            raise RuntimeError(
                f"Postgres checkpointer unavailable ({type(e).__name__}: {e}). "
                "Fix DATABASE_URL or set ALLOW_MEMORY_FALLBACK=1 for local dev."
            ) from e


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Service layer (used by the CLI below AND by api.py)
# ---------------------------------------------------------------------------
def make_config(thread_id: str, prefs: Optional[UserPrefs] = None) -> dict:
    """Config for every graph call. Tags/metadata show up in LangSmith traces."""
    return {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": thread_id},
        "run_name": "stock_assessment",
        "tags": ["investing-agent"],
        "metadata": {
            "thread_id": thread_id,
            "profile": str(prefs.profile) if prefs else None,
            "horizon": str(prefs.horizon) if prefs else None,
        },
    }


async def _stream(rt: AgentRuntime, graph_input, thread_id: str, prefs=None) -> AsyncIterator[dict]:
    async for chunk in rt.graph.astream(graph_input, config=make_config(thread_id, prefs), stream_mode="updates"):
        for node in chunk:
            if node.startswith("__"):      # e.g. "__interrupt__"
                continue
            yield {"node": node, "label": NODE_LABELS.get(node, node)}


def start_assessment_stream(rt: AgentRuntime, query: str, prefs: UserPrefs, thread_id: str) -> AsyncIterator[dict]:
    initial = {"messages": [HumanMessage(content=query)], "query": query, "prefs": prefs}
    return _stream(rt, initial, thread_id, prefs)


def resume_assessment_stream(rt: AgentRuntime, thread_id: str, decision: str, comment: Optional[str]) -> AsyncIterator[dict]:
    return _stream(rt, Command(resume={"decision": decision, "comment": comment}), thread_id)


async def get_status(rt: AgentRuntime, thread_id: str) -> dict:
    """not_found | awaiting_approval | completed | incomplete, plus the data a frontend needs."""
    snap = await rt.graph.aget_state(make_config(thread_id))
    vals = snap.values or {}
    if not vals:
        return {"thread_id": thread_id, "status": "not_found"}
    interrupts = [i for task in snap.tasks for i in task.interrupts]
    if interrupts:
        status = "awaiting_approval"
    elif not snap.next:
        status = "completed"
    else:
        status = "incomplete"
    return {
        "thread_id": thread_id,
        "status": status,
        "route": get_field(vals, "route"),
        "answer": get_field(vals, "answer", ""),
        "report_markdown": get_field(vals, "report_markdown", ""),
        "assessment_id": get_field(vals, "assessment_id"),
        "approval_status": get_field(vals, "approval_status"),
        "approval_comment": get_field(vals, "approval_comment"),
        "tickers": list((get_field(vals, "assessments", {}) or {}).keys()),
        "warnings": get_field(vals, "warnings", []),
        "fetch_failures": get_field(vals, "fetch_failures", {}),
        "report_md_path": get_field(vals, "report_md_path"),
        "report_json_path": get_field(vals, "report_json_path"),
        "final_message": _last_ai_text(get_field(vals, "messages", [])),
    }


def _last_ai_text(messages) -> str:
    for m in reversed(list(messages or [])):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", None) and not getattr(m, "tool_calls", None):
            return to_text(m.content)
    return ""


async def load_report(rt: AgentRuntime, thread_id: str) -> Optional[str]:
    row = await asyncio.to_thread(get_assessment_by_thread_id, thread_id)
    if row and row.get("report_markdown"):
        return row["report_markdown"]
    status = await get_status(rt, thread_id)
    return status.get("report_markdown") or None


# ---------------------------------------------------------------------------
# Report Q&A (stateless: the caller owns the history)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _chat_llm():
    return with_model_fallbacks(lambda m: m, CHAT_PROVIDER, CHAT_MODEL_CANDIDATES)


async def answer_question(report_md: str, history: list, question: str) -> str:
    """`history` is a list of (role, text) tuples WITHOUT the system prompt."""
    messages = [("system", build_qna_system_prompt(report_md))]
    messages += history[-QNA_MAX_TURNS:]
    messages.append(("user", question))
    resp = await ask(_chat_llm(), messages, "qna")
    return to_text(resp.content).strip()


async def report_qna_loop(report_md: str) -> None:
    history: list = []
    print("\n" + "=" * 60)
    print("Ask questions about this report only (type 'exit' to quit).")
    print("=" * 60)
    while True:
        q = (await asyncio.to_thread(input, "\nYou: ")).strip()
        if q.lower() in {"", "exit", "quit"}:
            break
        try:
            ans = await answer_question(report_md, history, q)
        except Exception as e:
            print(f"❌ {e}")
            continue
        print(f"\nAI: {ans}")
        history += [("user", q), ("ai", ans)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _choose(label: str, options) -> str:
    while True:
        raw = input(f"{label} ({'/'.join(options)}): ").strip().lower()
        if raw in options:
            return raw
        print(f"Please type one of: {', '.join(options)}")


def ask_prefs() -> UserPrefs:
    return UserPrefs(profile=_choose("Risk profile", PROFILES), horizon=_choose("Investment horizon", HORIZONS))


def save_run_log(query: str, prefs: Optional[UserPrefs], status: Optional[dict] = None,
                 error: Optional[str] = None) -> Optional[Path]:
    """Always-written log of the attempt, so every run leaves a trace even when it fails."""
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = status or {}
        data = {
            "timestamp": stamp, "query": query,
            "prefs": prefs.model_dump() if prefs else None,
            "model": AGENT_MODEL, "error": error,
            "status": status.get("status"), "route": status.get("route"),
            "tickers": status.get("tickers"), "warnings": status.get("warnings"),
            "assessment_id": status.get("assessment_id"),
            "approval_status": status.get("approval_status"),
            "approval_comment": status.get("approval_comment"),
            "report_md_path": status.get("report_md_path"),
            "report_json_path": status.get("report_json_path"),
            "final_answer": status.get("final_message") or status.get("answer"),
        }
        ticker_part = f"_{safe_name('_'.join(status.get('tickers') or []))}" if status.get("tickers") else ""
        return write_text_atomic(RUN_LOG_DIR / f"run_{stamp}{ticker_part}.json",
                                 json.dumps(data, indent=2, default=str, ensure_ascii=False))
    except Exception as e:
        log.warning("could not save run log: %s", e)
        return None


async def main():
    setup_logging()
    prefs = ask_prefs()
    query = input("Enter the Query: ").strip()
    if not query:
        print("No query entered.")
        save_run_log("", prefs, error="empty query")
        return

    rt = AgentRuntime()
    status, error = None, None
    try:
        await rt.start()
        thread_id = str(uuid4())
        async for ev in start_assessment_stream(rt, query, prefs, thread_id):
            print(f"  -> {ev['label']}", flush=True)
        status = await get_status(rt, thread_id)

        while status["status"] == "awaiting_approval":
            print("\n" + "=" * 60 + "\nASSESSMENT APPROVAL REQUIRED\n" + "=" * 60)
            print(f"Assessment ID: {status['assessment_id']}\nTickers: {', '.join(status['tickers'])}\n")
            print(status["report_markdown"])
            while True:
                choice = (await asyncio.to_thread(input, "\nApprove or reject? [approve/reject]: ")).strip().lower()
                if choice in {"approve", "reject"}:
                    break
                print("Please enter 'approve' or 'reject'.")
            if choice == "reject":
                comment = (await asyncio.to_thread(
                    input,
                    "Why, or another company to try? (Enter = suggest alternatives): ",
                )).strip() or None
            else:
                comment = (await asyncio.to_thread(input, "Optional comment (Enter to skip): ")).strip() or None
            decision = "approved" if choice == "approve" else "rejected"
            async for ev in resume_assessment_stream(rt, thread_id, decision, comment):
                print(f"  -> {ev['label']}", flush=True)
            status = await get_status(rt, thread_id)

        print("\n--- AI Response ---")
        print(status.get("final_message") or "The agent stopped before writing an answer. See the run log.")

        if status.get("report_markdown"):
            await report_qna_loop(status["report_markdown"])
    except Exception as e:
        error = str(e)
        log.exception("the agent run failed")
        print(f"\n❌ The agent run failed: {e}")
    finally:
        log_path = save_run_log(query, prefs, status, error)
        if log_path:
            print(f"🗒️ Run log saved: {log_path}")
        await rt.stop()


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )