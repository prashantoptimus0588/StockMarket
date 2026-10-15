"""Pre-flight check: env vars, MCP tools + argument names, Postgres, graph compile.
Run:  uv run python check_setup.py
"""
import asyncio, os, selectors, sys
import agent_chat as chat

def ok(msg): print(f"  [ OK ] {msg}")
def bad(msg): print(f"  [FAIL] {msg}")

async def main():
    chat.setup_logging()
    print("1. Environment")
    for k in ("MCP_SERVER_URL", "GROQ_API_KEY", "GOOGLE_API_KEY", "DATABASE_URL"):
        (ok if os.getenv(k) else bad)(f"{k} {'set' if os.getenv(k) else 'MISSING'}")
    ls = os.getenv("LANGSMITH_TRACING", "").lower() == "true" and os.getenv("LANGSMITH_API_KEY")
    (ok if ls else print)("LangSmith tracing " + ("enabled" if ls else "  [info] LangSmith tracing off (optional)"))

    print("2. Runtime (MCP tools, Postgres checkpointer, graph compile)")
    rt = chat.AgentRuntime()
    try:
        await rt.start()
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        return
    ok(f"graph compiled, {len(rt.tools)} tools: {sorted(rt.tools_by_name)}")

    print("3. Tool argument detection")
    for name in (chat.TOOL_SEARCH, chat.TOOL_PRICE, chat.TOOL_FUNDAMENTALS, chat.TOOL_NEWS):
        t = rt.tools_by_name.get(name)
        if t is None:
            (bad if name in chat.REQUIRED_TOOLS else print)(f"{name} not found")
            continue
        try:
            ok(f"{name}(arg='{chat.primary_arg(t)}')")
        except Exception as e:
            bad(str(e))

    print("4. Live data fetch for RELIANCE.NS (no LLM)")
    try:
        raw_p = await chat.call_tool(rt.tools_by_name[chat.TOOL_PRICE], "RELIANCE.NS")
        raw_f = await chat.call_tool(rt.tools_by_name[chat.TOOL_FUNDAMENTALS], "RELIANCE.NS")
        chat.PriceData.model_validate_json(raw_p); ok("price JSON parses into PriceData")
        chat.FundamentalsData.model_validate_json(raw_f); ok("fundamentals JSON parses into FundamentalsData")
    except Exception as e:
        bad(f"{type(e).__name__}: {str(e)[:300]}")
    await rt.stop()

if __name__ == "__main__":
    asyncio.run(main(), loop_factory=(lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())) if sys.platform == "win32" else None)
