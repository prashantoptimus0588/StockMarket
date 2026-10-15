import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import yfinance as yf
from mcp.server.fastmcp import FastMCP

# Import Pydantic models
from models.models import (
    DailyBar,
    PriceData,
    FundamentalsData,
    NewsData,
    NewsItem,
)

mcp = FastMCP(
    "news_server",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
)

# Threshold constants
LOW_VOLUME_THRESHOLD = 50_000
STALE_DAYS = 7
NEWS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


############# Helpers ##############

def _fetch_news_data(query: str, days: int, max_items: int) -> NewsData:
    """Shared Google News RSS fetcher returning a NewsData model."""
    try:
        days = max(1, min(days, 30))
        max_items = max(1, min(max_items, 10))

        search = quote(f"{query} when:{days}d")
        url = f"https://news.google.com/rss/search?q={search}&hl=en-IN&gl=IN&ceid=IN:en"

        req = urllib.request.Request(url, headers={"User-Agent": NEWS_USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()

        feed = feedparser.parse(raw)

        if not feed.entries:
            return NewsData(query=query, items=[])

        items = []
        for e in feed.entries:
            if not getattr(e, "published_parsed", None):
                continue
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            source = e.get("source", {}).get("title", "Unknown source")
            items.append(
                NewsItem(
                    published=published.isoformat(),
                    title=e.title,
                    source=source,
                )
            )

        items.sort(key=lambda x: x.published, reverse=True)
        return NewsData(query=query, items=items[:max_items])

    except Exception:
        return NewsData(query=query, items=[])


def _to_pct(val: float | None) -> float | None:
    """Converts a fractional value (0.15) to a plain percentage number (15.0)."""
    if val is None or not isinstance(val, (int, float)):
        return None
    return round(float(val) * 100.0, 2)


############# News and Weather Tools ##############

@mcp.tool()
def get_stock_news(query: str, days: int = 7, max_items: int = 6) -> str:
    """
    Fetch recent news headlines about a company or stock as structured JSON.
    Pass the COMPANY NAME plus the word 'stock' (e.g. 'State Bank of India stock'),
    not the ticker symbol.
    """
    data = _fetch_news_data(query, days, max_items)
    return data.model_dump_json()


@mcp.tool()
def get_general_news(topic: str, days: int = 3, max_items: int = 6) -> str:
    """
    Fetch recent news headlines on a general, non-company topic as structured JSON.
    """
    data = _fetch_news_data(topic, days, max_items)
    return data.model_dump_json()


@mcp.tool()
def get_weather(location: str) -> str:
    """Get current weather for a location as structured JSON."""
    result = {
        "location": location,
        "weather": "15°C and rainy" if "delhi" in location.lower() else "22°C and sunny"
    }
    return json.dumps(result)


############# Stock Tools ##############

@mcp.tool()
def search_ticker(query: str, max_results: int = 5) -> str:
    """
    Look up correct ticker symbols for a company name as structured JSON.
    Use this BEFORE price/fundamentals tools whenever unsure of exact ticker symbol.
    """
    try:
        max_results = max(1, min(max_results, 10))
        quotes = yf.Search(query, max_results=max_results).quotes or []

        candidates = []
        for q in quotes:
            if q.get("quoteType") in ("EQUITY", "ETF"):
                candidates.append({
                    "symbol": q.get("symbol"),
                    "name": q.get("longname") or q.get("shortname"),
                    "exchange": q.get("exchDisp") or q.get("exchange"),
                    "type": q.get("quoteType"),
                })
        return json.dumps({"query": query, "candidates": candidates})
    except Exception as e:
        return json.dumps({"query": query, "candidates": [], "error": str(e)})


@mcp.tool()
def getStock_price_info(stock: str, days: int = 5) -> str:
    """
    Fetch recent daily OHLCV rows and annualized risk metrics as structured PriceData JSON.
    'days' = recent trading days to list (1-30). Risk metrics use 1-year window.
    """
    clean_stock = stock.strip().upper()
    try:
        days = max(1, min(days, 30))
        ticker = yf.Ticker(clean_stock)
        df = ticker.history(period="1y")

        if df.empty:
            return PriceData(ticker=clean_stock).model_dump_json()

        # Build daily bars
        recent_df = df.tail(days)
        bars = [
            DailyBar(
                date=idx.strftime("%Y-%m-%d"),
                open=round(float(row["Open"]), 2),
                high=round(float(row["High"]), 2),
                low=round(float(row["Low"]), 2),
                close=round(float(row["Close"]), 2),
                volume=int(row["Volume"]),
            )
            for idx, row in recent_df.iterrows()
        ]

        # Calculate risk metrics
        returns = df["Close"].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100
        running_max = df["Close"].cummax()
        drawdown = (df["Close"] / running_max - 1) * 100

        data = PriceData(
            ticker=clean_stock,
            latest_close=round(float(df["Close"].iloc[-1]), 2),
            high_52w=round(float(df["High"].max()), 2),
            low_52w=round(float(df["Low"].min()), 2),
            volatility_pct=round(float(volatility), 2) if not returns.empty else None,
            max_drawdown_pct=round(float(drawdown.min()), 2) if not drawdown.empty else None,
            current_drawdown_pct=round(float(drawdown.iloc[-1]), 2) if not drawdown.empty else None,
            avg_volume=int(df["Volume"].mean()),
            recent_bars=bars,
        )
        return data.model_dump_json()

    except Exception:
        return PriceData(ticker=clean_stock).model_dump_json()


@mcp.tool()
def getStock_fundamentals(stock: str) -> str:
    """
    Fetch valuation, profitability, growth, dividend, and debt metrics as structured FundamentalsData JSON.
    """
    clean_stock = stock.strip().upper()
    try:
        ticker = yf.Ticker(clean_stock)
        info = ticker.info or {}

        de_raw = info.get("debtToEquity")
        debt_to_equity = round(float(de_raw) / 100.0, 2) if isinstance(de_raw, (int, float)) else None

        # Compute dividend yield ourselves (yfinance's dividendYield unit is inconsistent)
        px = info.get("currentPrice") or info.get("regularMarketPrice")
        rate = info.get("dividendRate")
        dividend_yield_pct = round(rate / px * 100, 2) if rate and px else None

        data = FundamentalsData(
            ticker=clean_stock,
            company_name=info.get("longName"),
            sector=info.get("sector"),
            market_cap=info.get("marketCap"),
            beta=round(float(info.get("beta")), 2) if isinstance(info.get("beta"), (int, float)) else None,
            trailing_pe=round(float(info.get("trailingPE")), 2) if isinstance(info.get("trailingPE"), (int, float)) else None,
            forward_pe=round(float(info.get("forwardPE")), 2) if isinstance(info.get("forwardPE"), (int, float)) else None,
            price_to_book=round(float(info.get("priceToBook")), 2) if isinstance(info.get("priceToBook"), (int, float)) else None,
            roe_pct=_to_pct(info.get("returnOnEquity")),
            profit_margin_pct=_to_pct(info.get("profitMargins")),
            revenue_growth_pct=_to_pct(info.get("revenueGrowth")),
            earnings_growth_pct=_to_pct(info.get("earningsGrowth")),
            debt_to_equity=debt_to_equity,
            free_cash_flow=info.get("freeCashflow"),
            dividend_yield_pct=dividend_yield_pct,
        )
        return data.model_dump_json()

    except Exception:
        return FundamentalsData(ticker=clean_stock).model_dump_json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")