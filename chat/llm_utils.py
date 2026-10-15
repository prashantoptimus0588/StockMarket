"""LLM provider configuration, fallbacks, and bounded invocation."""
import asyncio
import logging
import os
import time
from typing import Any, Callable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
log = logging.getLogger("agent")
LLM_TIMEOUT = 45
LLM_CALL_BUDGET = 150

# ---------------------------------------------------------------------------
# MODEL CONFIG - the only place model names live. Change them here / in .env.
# ---------------------------------------------------------------------------
PROVIDER_GROQ = "groq"
PROVIDER_GEMINI = "gemini"

# Graph LLM work (planning, ticker resolution, explanation, general agent)
AGENT_PROVIDER = PROVIDER_GROQ
AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
AGENT_FALLBACK_MODELS: list = []          # same provider, tried in order if the active one fails
AGENT_MODEL_CANDIDATES = list(dict.fromkeys([AGENT_MODEL] + AGENT_FALLBACK_MODELS))

# Report Q&A chat
CHAT_PROVIDER = PROVIDER_GEMINI
CHAT_MODEL = os.getenv("QNA_MODEL", "gemini-3-flash-preview")
CHAT_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite-preview"]
CHAT_MODEL_CANDIDATES = list(dict.fromkeys([CHAT_MODEL] + CHAT_FALLBACK_MODELS))


def build_llm(provider: str, model_name: str):
    """Single factory for every chat model. The provider SDK retries 429/5xx with backoff."""
    if provider == PROVIDER_GEMINI:
        return ChatGoogleGenerativeAI(model=model_name, temperature=0, timeout=LLM_TIMEOUT, max_retries=2)
    if provider == PROVIDER_GROQ:
        return ChatGroq(model=model_name, temperature=0, timeout=LLM_TIMEOUT, max_retries=2)
    raise ValueError(f"Unknown provider: {provider}")


def with_model_fallbacks(shape: Callable[[Any], Any], provider: str, names: list):
    """Build one runnable per candidate model (shaped by `shape`) and chain them as fallbacks."""
    runnables = [shape(build_llm(provider, n)) for n in names]
    return runnables[0].with_fallbacks(runnables[1:]) if len(runnables) > 1 else runnables[0]


def structured_output_shape(schema):
    """Groq gpt-oss often emits prose instead of a tool call; prefer json_schema."""
    def shape(m):
        if AGENT_PROVIDER == PROVIDER_GROQ:
            js = m.with_structured_output(schema, method="json_schema")
            fc = m.with_structured_output(schema, method="function_calling")
            return js.with_fallbacks([fc])
        return m.with_structured_output(schema)
    return shape


async def ask(runnable, messages, label: str):
    """One logical LLM call with a hard time budget, timing and error logging."""
    t0 = time.perf_counter()
    try:
        out = await asyncio.wait_for(runnable.ainvoke(messages), timeout=LLM_CALL_BUDGET)
    except Exception as e:
        log.error("LLM call '%s' failed after %.1fs: %s: %s", label, time.perf_counter() - t0,
                  type(e).__name__, str(e)[:200])
        raise
    log.info("LLM call '%s' done in %.1fs", label, time.perf_counter() - t0)
    return out


