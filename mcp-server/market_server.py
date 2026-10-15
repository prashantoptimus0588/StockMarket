import os
import math
import json
import urllib.request
from urllib.parse import quote
from datetime import datetime, timezone
import feedparser
import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Market_Server")

@mcp.tool()
def getStock_price_info(stock: str, days: int = 5) -> str:
    """
    Fetch price data for a stock as JSON: recent daily OHLCV bars plus risk metrics
    (annualized volatility, max drawdown, current drawdown, 52-week range, average volume).
    'days' = how many recent trading days to include (1-30). Risk metrics always use the last 1 year.
    Use the exchange suffix for Indian stocks, e.g. SBIN.NS.
    """
    try:
        clean_stock = stock.strip().upper()
        df = yf.Ticker(clean_stock).history(period="1y")

        if df.empty:
            return f'{{"error": "No trading data available for ticker \'{clean_stock}\'."}}'

        days = max(1, min(days, 30))
        bars = [
            DailyBar(
                date=d.strftime('%Y-%m-%d'),
                open=round(r['Open'], 2),
                high=round(r['High'], 2),
                low=round(r['Low'], 2),
                close=round(r['Close'], 2),
                volume=int(r['Volume']),
            )
            for d, r in df.tail(days).iterrows()
        ]

        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100
        drawdown = (df['Close'] / df['Close'].cummax() - 1) * 100

        data = PriceData(
            ticker=clean_stock,
            latest_close=_clean(df['Close'].iloc[-1]),
            high_52w=_clean(df['High'].max()),
            low_52w=_clean(df['Low'].min()),
            volatility_pct=_clean(volatility),
            max_drawdown_pct=_clean(drawdown.min()),
            current_drawdown_pct=_clean(drawdown.iloc[-1]),
            avg_volume=int(df['Volume'].mean()),
            recent_bars=bars,
        )
        return data.model_dump_json()

    except Exception as e:
        return f'{{"error": "Error executing price tool: {str(e)}"}}'
    
@mcp.tool()
def getStock_fundamentals(stock: str) -> str:
    """
    Fetch fundamental metrics for a stock as JSON: valuation (P/E, P/B), profitability (ROE, margin),
    growth (revenue, earnings), debt-to-equity, free cash flow, beta, market cap and dividend yield.
    Missing values are null. Banks usually have no debt-to-equity or free cash flow.
    Use the exchange suffix for Indian stocks, e.g. SBIN.NS.
    """
    try:
        clean_stock = stock.strip().upper()
        info = yf.Ticker(clean_stock).info

        if not info or len(info) <= 5:
            return f'{{"error": "Could not retrieve fundamental analysis metrics for ticker \'{clean_stock}\'."}}'

        de_raw = _clean(info.get('debtToEquity'))
        debt_to_equity = round(de_raw / 100, 2) if de_raw is not None else None

        data = FundamentalsData(
            ticker=clean_stock,
            company_name=info.get('longName'),
            sector=info.get('sector'),
            market_cap=_clean(info.get('marketCap'), 0),
            beta=_clean(info.get('beta')),
            trailing_pe=_clean(info.get('trailingPE')),
            forward_pe=_clean(info.get('forwardPE')),
            price_to_book=_clean(info.get('priceToBook')),
            roe_pct=_pct(info.get('returnOnEquity')),
            profit_margin_pct=_pct(info.get('profitMargins')),
            revenue_growth_pct=_pct(info.get('revenueGrowth')),
            earnings_growth_pct=_pct(info.get('earningsGrowth')),
            debt_to_equity=debt_to_equity,
            free_cash_flow=_clean(info.get('freeCashflow'), 0),
            dividend_yield_pct=_clean(info.get('dividendYield')),
        )
        return data.model_dump_json()

    except Exception as e:
        return f'{{"error": "Error executing fundamental tool: {str(e)}"}}'

if __name__ == "__main__":
    mcp.run(transport='stdio')
