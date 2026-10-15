"""Shared runtime limits and MCP tool names."""
import os

# ---------------------------------------------------------------------------
# Timeouts / limits
# ---------------------------------------------------------------------------
LLM_TIMEOUT = 45          # seconds per provider request (SDK level)
LLM_CALL_BUDGET = 150     # hard cap for one logical LLM call incl. retries + model fallbacks
TOOL_TIMEOUT = 45         # seconds per MCP tool call
MCP_CONNECT_TIMEOUT = int(os.getenv("MCP_CONNECT_TIMEOUT", "20"))
DB_TIMEOUT = 15           # seconds to connect to Postgres
MAX_TICKERS = 4           # cap per request: protects cost and latency
MAX_STEPS = 15            # general agent loop: max model calls
MAX_REPLANS = 2           # each reject can send the graph back to plan_node this many times
RECURSION_LIMIT = 60
FETCH_RETRY_DELAY = 4     # seconds to wait before the single retry of a failed price/fundamentals fetch

# ---------------------------------------------------------------------------
# MCP tool names (as exposed by your MCP server)
# ---------------------------------------------------------------------------
TOOL_SEARCH = "search_ticker"
TOOL_PRICE = "getStock_price_info"
TOOL_FUNDAMENTALS = "getStock_fundamentals"
TOOL_NEWS = "get_stock_news"
REQUIRED_TOOLS = (TOOL_SEARCH, TOOL_PRICE, TOOL_FUNDAMENTALS)

# If a tool's main argument cannot be auto-detected from its schema, name it here,
# e.g. {"getStock_price_info": "ticker"}.
TOOL_ARG_NAMES: dict = {}


