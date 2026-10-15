"""Deterministic assessment formatting and LLM-output guardrails."""
import logging
from typing import Optional
from analysis.return_engine import CombinedVerdict, ReturnVerdict, combine, evaluate_return
from analysis.risk_engine import evaluate_risk
from models.models import FundamentalsData, PriceData, RiskVerdict, UserPrefs
from chat.schemas import Explanation, TickerSuggestion
from chat.common import same_ticker
log = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------
def format_verdict(ticker: str, prefs: UserPrefs, verdict: RiskVerdict,
                   returns: ReturnVerdict, combined: CombinedVerdict) -> str:
    lines = [
        f"[ENGINE - computed by code] ticker={ticker}, profile={prefs.profile}, horizon={prefs.horizon}",
        f"Result: {verdict.result} | Score: {verdict.score if verdict.score is not None else 'n/a'} "
        f"| Rules evaluated: {verdict.rules_evaluated}/{verdict.rules_total}",
        verdict.summary,
        "Rule details:",
    ]
    for r in verdict.rules:
        lines.append(f"- {r.status.upper()} {r.name}: {r.detail}")
    lines.append("")
    lines.append(
        f"Return potential: {returns.level or 'n/a'} | Score: "
        f"{returns.score if returns.score is not None else 'n/a'} "
        f"| Factors evaluated: {returns.factors_evaluated}/{returns.factors_total}"
    )
    lines.append(returns.summary)
    for fct in returns.factors:
        lines.append(f"- {fct.name}: {fct.detail}")
    lines.append("")
    lines.append(f"Combined: {combined.summary}")
    return "\n".join(lines)


def format_facts(price, fund) -> str:
    """Plain figures from the code-computed data, so the narrative quotes real numbers."""
    def g(obj, name):
        return getattr(obj, name, None)

    def fmt(v, suffix="", digits=2):
        return None if v is None else f"{v:,.{digits}f}{suffix}"

    rows = [
        ("Company", g(fund, "company_name")), ("Sector", g(fund, "sector")),
        ("Latest close", fmt(g(price, "latest_close"))),
        ("52-week range", f"{fmt(g(price, 'low_52w'))} - {fmt(g(price, 'high_52w'))}"
         if g(price, "low_52w") is not None and g(price, "high_52w") is not None else None),
        ("Annualized volatility", fmt(g(price, "volatility_pct"), "%")),
        ("Max drawdown (1y)", fmt(g(price, "max_drawdown_pct"), "%")),
        ("Current drawdown from peak", fmt(g(price, "current_drawdown_pct"), "%")),
        ("Beta", fmt(g(fund, "beta"))), ("Trailing P/E", fmt(g(fund, "trailing_pe"))),
        ("Forward P/E", fmt(g(fund, "forward_pe"))), ("Price-to-book", fmt(g(fund, "price_to_book"))),
        ("Return on equity", fmt(g(fund, "roe_pct"), "%")),
        ("Net profit margin", fmt(g(fund, "profit_margin_pct"), "%")),
        ("Revenue growth", fmt(g(fund, "revenue_growth_pct"), "%")),
        ("Earnings growth", fmt(g(fund, "earnings_growth_pct"), "%")),
        ("Dividend yield", fmt(g(fund, "dividend_yield_pct"), "%")),
    ]
    return "KEY FIGURES:\n" + "\n".join(f"- {k}: {v}" for k, v in rows if v is not None)


CORE_FUND_FIELDS = ("market_cap", "beta", "trailing_pe", "forward_pe", "price_to_book", "roe_pct",
                    "profit_margin_pct", "revenue_growth_pct", "earnings_growth_pct",
                    "debt_to_equity", "free_cash_flow")


def parse_tool_result(raw, model, key: str, label: str):
    """-> (parsed model or None, failure reason or None).

    A reason means the FETCH failed (exception, timeout, unreadable output, server `error` field, or a
    fundamentals payload with every core field empty). Thin-but-real data is NOT a failure.
    """
    if isinstance(raw, Exception):
        return None, f"{type(raw).__name__}: {str(raw)[:120]}"
    if raw.lstrip().startswith("ERROR:"):
        return None, raw.strip()[:160]
    try:
        obj = model.model_validate_json(raw)
    except ValueError as exc:
        log.error("%s %s parse failed: %s | raw[:500]=%r", key, label, exc, raw[:500])
        return None, "the tool returned unreadable data"
    if getattr(obj, "error", None):
        return None, str(obj.error)[:200]
    if model is FundamentalsData and all(getattr(obj, f, None) is None for f in CORE_FUND_FIELDS):
        return None, "Yahoo returned no fundamentals"
    return obj, None


def build_assessments(prices: dict, funds: dict, prefs: UserPrefs, warnings: list,
                      failures: Optional[dict] = None) -> dict:
    """ticker -> {price, fundamentals, verdict, returns, combined}; only tickers with both datasets."""
    failures = failures or {}
    out: dict = {}
    for key, price in prices.items():
        fund = funds.get(key)
        if fund is None:
            if key not in failures:
                warnings.append(f"{key}: fundamentals missing, so no assessment was produced.")
            continue
        if not same_ticker(price.ticker, fund.ticker):
            warnings.append(f"{key}: price and fundamentals refer to different tickers "
                            f"({price.ticker} vs {fund.ticker}); skipped.")
            continue
        try:
            verdict = evaluate_risk(price, fund, prefs)
            returns = evaluate_return(price, fund, prefs)
            combined = combine(verdict, returns, prefs)
        except Exception as exc:
            log.exception("engine failed for %s", key)
            warnings.append(f"{key}: the risk engine failed ({type(exc).__name__}).")
            continue
        out[key] = {"price": price, "fundamentals": fund, "verdict": verdict,
                    "returns": returns, "combined": combined}
    for key in funds:
        if key not in prices and key not in failures:
            warnings.append(f"{key}: price data missing, so no assessment was produced.")
    return out


def engine_stance(item: dict, flagged: bool) -> str:
    """What the engine's verdict allows the suggestion to be.

    TUNE THIS to your engine's real values (verdict.result / combined.summary wording).
    """
    verdict, combined = item["verdict"], item["combined"]
    if flagged or verdict.score is None:
        return "insufficient"
    if verdict.result == "does_not_fit" or combined.label == "weak_fit":
        return "negative"
    if combined.score is None:
        return "insufficient"
    if combined.label == "watch":
        return "watch"
    return "other"


def validate_explanation(exp: Explanation, assessments: dict, flagged: set) -> list:
    by_ticker = {s.ticker.strip().upper(): s for s in exp.suggestions}
    problems = []
    for tk, item in assessments.items():
        s = by_ticker.get(tk)
        if s is None:
            problems.append(f"Missing a suggestion for {tk}.")
            continue
        stance = engine_stance(item, tk in flagged)
        if stance == "negative" and s.suggestion in ("Consider", "Hold off"):
            problems.append(f"{tk}: the engine verdict is negative, so the suggestion must be 'Avoid'.")
        if stance == "watch" and s.suggestion == "Consider":
            problems.append(f"{tk}: the engine label is 'watch', so the suggestion must be 'Hold off', not 'Consider'.")
        if stance == "insufficient" and s.suggestion != "Insufficient data":
            problems.append(f"{tk}: data is insufficient/unreliable, so the suggestion must be 'Insufficient data'.")
    return problems


def fallback_explanation(assessments: dict, flagged: set) -> Explanation:
    """Engine-only narrative when the explainer LLM fails (Groq tool_use_failed, etc.)."""
    suggestions = []
    key_points = []
    summaries = []
    for tk, item in assessments.items():
        stance = engine_stance(item, tk in flagged)
        label = {"negative": "Avoid", "insufficient": "Insufficient data", "watch": "Hold off"}.get(
            stance, "Consider")
        reason = item["combined"].summary
        suggestions.append(TickerSuggestion(ticker=tk, suggestion=label, reason=reason))
        summaries.append(f"{tk}: {reason}")
        key_points.append(f"{tk}: {item['verdict'].summary}")
    takeaway = summaries[0] if len(summaries) == 1 else "Engine ranking: " + "; ".join(summaries)
    return Explanation(
        suggestions=suggestions,
        summary=" ".join(summaries),
        key_points=key_points[:6],
        news_points=[],
        takeaway=takeaway,
    )


def enforce_stance(exp: Explanation, assessments: dict, flagged: set) -> Explanation:
    """Last line of defence: code overrides an LLM suggestion that contradicts the engine."""
    by_ticker = {s.ticker.strip().upper(): s for s in exp.suggestions}
    fixed = []
    for tk, item in assessments.items():
        s = by_ticker.get(tk)
        stance = engine_stance(item, tk in flagged)
        forced = {"negative": "Avoid", "insufficient": "Insufficient data", "watch": "Hold off"}.get(stance)
        if s is None:
            fixed.append(TickerSuggestion(
                ticker=tk, suggestion=forced or "Hold off",
                reason=f"Based on the engine verdict: {item['combined'].summary}"))
        elif (stance == "negative" and s.suggestion in ("Consider", "Hold off")) or \
             (stance == "watch" and s.suggestion == "Consider") or \
             (stance == "insufficient" and s.suggestion != "Insufficient data"):
            log.warning("explain: overriding LLM suggestion for %s (%s -> %s)", tk, s.suggestion, forced)
            fixed.append(TickerSuggestion(ticker=tk, suggestion=forced,
                                          reason=f"Based on the engine verdict: {item['combined'].summary}"))
        else:
            fixed.append(TickerSuggestion(ticker=tk, suggestion=s.suggestion, reason=s.reason))
    return exp.model_copy(update={"suggestions": fixed})


def render_answer(exp: Explanation, warnings: list, wants_news: bool, news_missing: bool,
                  failures: Optional[dict] = None) -> str:
    """Report-style markdown: the call first, a Bottom line section last. All from validated structured fields."""
    lines = ["## Suggestion", ""]
    for s_ in exp.suggestions:
        lines.append(f"**{s_.ticker}: {s_.suggestion}**. {s_.reason}")
        lines.append("")
    for tk, why in (failures or {}).items():
        lines.append(f"**{tk}: Not assessed.** {why[:1].upper() + why[1:]} (retried once). "
                     "This is a data-fetch failure, not a verdict on the stock.")
        lines.append("")
    lines += ["## Summary", "", exp.summary]
    if exp.key_points:
        lines += ["", "## Key points", ""] + [f"- {p}" for p in exp.key_points]
    if exp.news_points:
        lines += ["", "## Recent news", ""] + [f"- {p}" for p in exp.news_points]
    elif wants_news and news_missing:
        lines += ["", "## Recent news", "", "- News could not be retrieved for this run."]
    if warnings:
        lines += ["", "## Data notes", ""] + [f"- {w}" for w in warnings]
    lines += ["", "## Bottom line", "", exp.takeaway]
    return "\n".join(lines)


