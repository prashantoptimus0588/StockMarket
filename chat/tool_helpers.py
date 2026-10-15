"""Helpers for safe MCP invocation and result normalization."""
import asyncio
import logging
import time
from chat.config import DB_TIMEOUT, TOOL_TIMEOUT, TOOL_ARG_NAMES
log = logging.getLogger("agent")

def with_timeout(tool, seconds: int = TOOL_TIMEOUT):
    """Wrap an MCP tool so a slow/hung call returns an error string instead of blocking forever."""
    orig = tool.coroutine
    if orig is None:
        return tool

    async def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            out = await asyncio.wait_for(orig(*args, **kwargs), timeout=seconds)
            log.info("tool %s done in %.1fs", tool.name, time.perf_counter() - t0)
            return out
        except asyncio.TimeoutError:
            log.warning("tool %s TIMED OUT after %ss", tool.name, seconds)
            # MCP tools use content_and_artifact -> must return (content, artifact)
            return (f"ERROR: tool '{tool.name}' timed out after {seconds}s.", None)

    tool.coroutine = wrapped
    return tool


def with_connect_timeout(url: str, seconds: int = DB_TIMEOUT) -> str:
    if "connect_timeout" in url:
        return url
    return f"{url}{'&' if '?' in url else '?'}connect_timeout={seconds}"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def to_text(content) -> str:
    """MCP adapters / LLMs may return a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        parts.append(block.get("text", "") if isinstance(block, dict) else str(block))
    return "\n".join(parts)


def get_field(state, key, default=None):
    if state is None:
        return default
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def primary_arg(tool) -> str:
    """Name of the tool's main (first required) argument, read from its JSON schema."""
    if tool.name in TOOL_ARG_NAMES:
        return TOOL_ARG_NAMES[tool.name]
    schema = tool.args_schema
    if schema is not None and not isinstance(schema, dict):
        schema = schema.model_json_schema()
    schema = schema or {}
    props = list((schema.get("properties") or {}).keys())
    required = schema.get("required") or props
    if not required:
        raise RuntimeError(f"Tool {tool.name} has no arguments; set TOOL_ARG_NAMES.")
    if len(required) > 1:
        raise RuntimeError(
            f"Tool {tool.name} needs several arguments {required}; set TOOL_ARG_NAMES['{tool.name}']."
        )
    return required[0]


async def call_tool(tool, value: str) -> str:
    out = await tool.ainvoke({primary_arg(tool): value})
    return to_text(out)


