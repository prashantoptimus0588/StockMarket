"""
report.py - turns the graph's final state into a readable report.

Everything numeric comes straight from the code-computed models (price, fundamentals,
risk verdict, return verdict, combined verdict). The LLM's written explanation is
included as its own clearly-labelled section and is never used for numbers.

Usage (from chat.py, after the graph finishes):

    from report.report import build_report, save_report, report_to_dict

    md = build_report(
        query=query,
        prefs=prefs,
        price=final_state["price"],
        fundamentals=final_state["fundamentals"],
        verdict=final_state["verdict"],
        returns=final_state["returns"],
        combined=final_state["combined"],
        llm_answer=final_message.content,
    )
    path = save_report(md, final_state["verdict_ticker"])
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # type hints only, no runtime dependency on your packages
    from analysis.return_engine import CombinedVerdict, ReturnVerdict
    from models.models import FundamentalsData, PriceData, RiskVerdict, UserPrefs

REPORT_DIR = "reports"

DISCLAIMER = ""  # intentionally empty: no boilerplate footer in reports

_STATUS_ICON = {"pass": "✅ Pass", "fail": "❌ Fail", "skipped": "⏭️ Skipped"}
_RESULT_LABEL = {"fits": "Fits", "borderline": "Borderline", "does_not_fit": "Does not fit"}

# Currency hint from the ticker suffix (Yahoo does not return currency in our PriceData)
_SUFFIX_CURRENCY = {".NS": "INR", ".BO": "INR", ".T": "JPY", ".L": "GBP (pence)", ".HK": "HKD", ".TO": "CAD",
                    ".DE": "EUR", ".F": "EUR", ".PA": "EUR", ".AS": "EUR", ".MI": "EUR", ".MC": "EUR", ".SW": "CHF", ".KS": "KRW", ".KQ": "KRW", ".SI": "SGD", ".AX": "AUD"}


# ---------------------------------------------------------------- formatting helpers

def _num(value: Optional[float], digits: int = 2, suffix: str = "") -> Optional[str]:
    if value is None:
        return None
    return f"{value:,.{digits}f}{suffix}"


def _compact(value: Optional[float]) -> Optional[str]:
    """1_234_000_000 -> 1.23B"""
    if value is None:
        return None
    for limit, tag in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"{value / limit:.2f}{tag}"
    return f"{value:,.0f}"


def _currency(ticker: str) -> str:
    for suffix, cur in _SUFFIX_CURRENCY.items():
        if ticker.upper().endswith(suffix):
            return f"{cur}, inferred from ticker suffix"
    if "." not in ticker:
        return "USD, inferred (no exchange suffix)"
    return "listing currency (not reported)"


def _table(rows: list[tuple[str, Optional[str]]]) -> list[str]:
    """Two-column Markdown table; rows with no value are dropped instead of showing N/A."""
    kept = [(k, v) for k, v in rows if v is not None]
    if not kept:
        return ["_No data available._"]
    out = ["| Metric | Value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in kept]
    return out


def _dump(obj: Any) -> Any:
    """Best-effort conversion of pydantic models / dataclasses to plain dicts."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)
    return getattr(obj, "__dict__", str(obj))


def _esc(text: str) -> str:
    """Keep table cells on one line."""
    return str(text).replace("|", "\\|").replace("\n", " ")


_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_LABEL_RE = re.compile(r"\b(?:52[- ]?(?:weeks?|wk|w)|1[- ]?(?:year|yr|y)|12[- ]?months?)\b", re.IGNORECASE)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*>]\s+|\d+\.\s+)?")


def _to_float(token: str) -> Optional[float]:
    try:
        return abs(float(token.replace(",", "").lstrip("+-")))
    except ValueError:
        return None


def _collect_numbers(obj: Any, out: list[float]) -> None:
    """Every number that appears anywhere in the computed data (values and text fields)."""
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(abs(float(obj)))
    elif isinstance(obj, str):
        out.extend(v for v in (_to_float(t) for t in _NUM_RE.findall(obj)) if v is not None)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect_numbers(v, out)
    else:
        _collect_numbers(_dump(obj), out)


def _allowed_numbers(*objs: Any) -> set[float]:
    """Numbers the narrative is allowed to quote: anything in the structured data, in any of the
    scales a human would write it (1.36T, 13,600 crore, 1360 B ...) plus today's date parts."""
    raw: list[float] = []
    for o in objs:
        _collect_numbers(o, raw)
    now = datetime.now()
    raw += [now.year, now.month, now.day]
    allowed: set[float] = set()
    for v in raw:
        for scale in (1, 1e3, 1e6, 1e7, 1e9, 1e12):
            allowed.add(v / scale)
    return allowed


def _is_grounded(n: float, allowed: set[float]) -> bool:
    tol = max(0.0051, 0.005 * n)
    return any(abs(n - a) <= tol for a in allowed)


def validate_ai_narrative(
    answer: str, tickers: Optional[list[str]] = None, allowed: Optional[set[float]] = None,
) -> tuple[str, str | None]:
    """Keep the AI narrative, but drop individual sentences that quote a number that is not in the
    computed data, or that carry a link. Returns (clean_markdown, note_or_None)."""
    text = (answer or "").strip()
    if not text:
        return "", None
    allowed = allowed or set()
    masks = sorted((t for t in (tickers or []) if t), key=len, reverse=True)

    kept: list[str] = []
    dropped = 0
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            kept.append(line)
            continue
        prefix = _LIST_PREFIX_RE.match(line).group(0)
        good = []
        for sentence in _SENT_RE.split(line[len(prefix):]):
            probe = _LABEL_RE.sub("", sentence)
            for t in masks:
                probe = re.sub(re.escape(t), "", probe, flags=re.IGNORECASE)
            nums = [v for v in (_to_float(t) for t in _NUM_RE.findall(probe)) if v is not None]
            if _URL_RE.search(probe) or any(not _is_grounded(n, allowed) for n in nums):
                dropped += 1
                continue
            good.append(sentence)
        if good:
            kept.append(prefix + " ".join(good))

    # a heading goes if everything under it was dropped
    cleaned: list[str] = []
    section: list[str] = []
    heading: Optional[str] = None
    orig_has_body = False

    def flush():
        if heading is None:
            cleaned.extend(section)
        elif any(l.strip() for l in section) or not orig_has_body:
            cleaned.append(heading)
            cleaned.extend(section)

    src_sections: dict[int, bool] = {}
    cur = -1
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            cur += 1
            src_sections[cur] = False
        elif line.strip():
            src_sections[cur] = True
    idx = -1
    for line in kept:
        if line.lstrip().startswith("#"):
            flush()
            idx += 1
            heading, section = line, []
            orig_has_body = src_sections.get(idx, False)
        else:
            section.append(line)
    flush()
    out = "\n".join(cleaned).strip()
    if not re.sub(r"[#*_\s]", "", out):
        return "", "nothing in the AI narrative could be tied to the computed data"
    return out, (f"{dropped} unsupported sentence(s) removed" if dropped else None)


def _safe_narrative_block(answer: str, tickers: list[str], allowed: Optional[set[float]] = None) -> list[str]:
    safe_answer, _ = validate_ai_narrative(answer, tickers, allowed)
    return [safe_answer, ""] if safe_answer else []


def _ratio(evaluated: Any, total: Any) -> float | None:
    try:
        total_i = int(total)
        evaluated_i = int(evaluated)
    except (TypeError, ValueError):
        return None
    if total_i <= 0:
        return None
    return max(0.0, min(1.0, evaluated_i / total_i))


def _coverage_label(ratio: float | None) -> str:
    if ratio is None:
        return "unavailable"
    if ratio >= 0.80:
        return "good"
    if ratio >= 0.60:
        return "partial"
    return "limited"


def _not_applicable(verdict) -> list[str]:
    """Rules the engine skipped because they do not apply to the sector (not because data is missing)."""
    return [r.name for r in getattr(verdict, "rules", []) or []
            if r.status == "skipped" and "not applicable" in str(r.detail).lower()]


def _risk_counts(verdict) -> tuple[Any, Any, int]:
    """(evaluated, total, n_not_applicable) with not-applicable rules removed from the total."""
    evaluated = getattr(verdict, "rules_evaluated", None)
    total = getattr(verdict, "rules_total", None)
    na = len(_not_applicable(verdict))
    try:
        total = int(total) - na
    except (TypeError, ValueError):
        pass
    return evaluated, total, na


_FUNDAMENTAL_FIELDS = [
    "market_cap", "beta", "trailing_pe", "forward_pe", "price_to_book",
    "roe_pct", "profit_margin_pct", "revenue_growth_pct", "earnings_growth_pct",
    "debt_to_equity", "free_cash_flow",
]


def _fundamental_fields(verdict) -> list[str]:
    fields = list(_FUNDAMENTAL_FIELDS)
    for name in _not_applicable(verdict):
        low = name.lower()
        if "debt" in low and "debt_to_equity" in fields:
            fields.remove("debt_to_equity")
        if "free cash" in low and "free_cash_flow" in fields:
            fields.remove("free_cash_flow")
    return fields


def _data_quality(price, fundamentals, verdict, returns) -> dict[str, Any]:
    """Summarize coverage without changing the underlying rule-engine scores."""
    risk_eval, risk_total, _na = _risk_counts(verdict)
    risk_ratio = _ratio(risk_eval, risk_total)
    return_ratio = _ratio(
        getattr(returns, "factors_evaluated", None) if returns is not None else None,
        getattr(returns, "factors_total", None) if returns is not None else None,
    )
    fundamental_fields = _fundamental_fields(verdict)
    fundamental_values = [getattr(fundamentals, field, None) for field in fundamental_fields]
    fundamental_available = sum(value is not None for value in fundamental_values)
    fundamental_ratio = fundamental_available / len(fundamental_fields) if fundamental_fields else None

    price_fields = [
        "latest_close", "high_52w", "low_52w", "volatility_pct",
        "max_drawdown_pct", "current_drawdown_pct", "avg_volume",
    ]
    price_values = [getattr(price, field, None) for field in price_fields]
    price_available = sum(value is not None for value in price_values)
    price_ratio = price_available / len(price_fields) if price_fields else None

    ratios = [risk_ratio, return_ratio, fundamental_ratio, price_ratio]
    available_ratios = [ratio for ratio in ratios if ratio is not None]
    minimum = min(available_ratios) if available_ratios else 0.0
    if (risk_ratio is not None and risk_ratio < 0.60) or (return_ratio is not None and return_ratio < 0.50) or fundamental_ratio < 0.50 or price_ratio < 0.50:
        overall = "limited"
        message = "Scores are provisional; do not rely on them as a standalone comparison."
    elif any(ratio is not None and ratio < 1.0 for ratio in ratios):
        overall = "partial"
        message = "Some inputs are missing; read the scores with that in mind."
    else:
        overall = "good"
        message = "All tracked inputs and rule/factor checks are available."

    def pct(value):
        return round(value * 100, 1) if value is not None else None

    return {
        "overall": overall,
        "message": message,
        "risk_rules": {"evaluated": risk_eval, "total": risk_total, "coverage_pct": pct(risk_ratio), "status": _coverage_label(risk_ratio)},
        "return_factors": {"evaluated": getattr(returns, "factors_evaluated", None) if returns is not None else 0, "total": getattr(returns, "factors_total", None) if returns is not None else 0, "coverage_pct": pct(return_ratio), "status": _coverage_label(return_ratio)},
        "fundamentals": {"available": fundamental_available, "total": len(fundamental_fields), "coverage_pct": pct(fundamental_ratio), "status": _coverage_label(fundamental_ratio)},
        "price_metrics": {"available": price_available, "total": len(price_fields), "coverage_pct": pct(price_ratio), "status": _coverage_label(price_ratio)},
    }


def _section_data_quality(price, fundamentals, verdict, returns) -> list[str]:
    quality = _data_quality(price, fundamentals, verdict, returns)
    risk = quality["risk_rules"]
    ret = quality["return_factors"]
    fund = quality["fundamentals"]
    px = quality["price_metrics"]
    return [
        "## Data completeness and score confidence",
        "",
        f"**Data coverage: {quality['overall'].upper()}** — {quality['message']}",
        "",
        f"- Price metrics: {px['available']}/{px['total']} available ({px['coverage_pct']}%; {px['status']}).",
        f"- Risk rules: {risk['evaluated']}/{risk['total']} evaluated ({risk['coverage_pct']}%; {risk['status']}).",
        f"- Return factors: {ret['evaluated']}/{ret['total']} evaluated ({ret['coverage_pct']}%; {ret['status']}).",
        f"- Core fundamentals: {fund['available']}/{fund['total']} available ({fund['coverage_pct']}%; {fund['status']}).",
        "",
    ]


# ---------------------------------------------------------------- sections

def _section_glance(prefs, verdict, returns, combined) -> list[str]:
    risk_label = _RESULT_LABEL.get(verdict.result, verdict.result)
    risk_coverage = _ratio(*_risk_counts(verdict)[:2])
    return_coverage = _ratio(
        getattr(returns, "factors_evaluated", None) if returns is not None else None,
        getattr(returns, "factors_total", None) if returns is not None else None,
    )
    risk_score = f"{verdict.score:.0f}/100" if verdict.score is not None else "n/a (insufficient data)"
    if risk_coverage is not None and risk_coverage < 1.0 and verdict.score is not None:
        risk_score += " (provisional)"
    ret_level = (returns.level or "n/a").capitalize() if returns else "n/a"
    ret_score = f"{returns.score:.0f}/100" if returns and returns.score is not None else "n/a"
    if return_coverage is not None and return_coverage < 1.0 and returns and returns.score is not None:
        ret_score += " (provisional)"

    lines = [
        "## Verdict at a glance",
        "",
        "| | Result | Score |",
        "|---|---|---|",
        f"| **Risk fit** ({prefs.profile}, {prefs.horizon}-term) | {risk_label} | {risk_score} |",
        f"| **Return potential** | {ret_level} | {ret_score} |",
        "",
    ]
    if combined is not None:
        label = getattr(combined, "label", None) or getattr(combined, "result", None)
        if label:
            lines.append(f"**Combined:** {label}")
            lines.append("")
        lines.append(f"> {combined.summary}")
        lines.append("")
    return lines


def _section_price(price) -> list[str]:
    rows = [
        ("Latest close", _num(price.latest_close)),
        ("52-week high", _num(price.high_52w)),
        ("52-week low", _num(price.low_52w)),
        ("Annualized volatility", _num(price.volatility_pct, 2, "%")),
        ("Max drawdown (1y)", _num(price.max_drawdown_pct, 2, "%")),
        ("Current drawdown from peak", _num(price.current_drawdown_pct, 2, "%")),
        ("Average daily volume", f"{price.avg_volume:,}" if price.avg_volume is not None else None),
    ]
    lines = ["## Price & risk metrics", "", f"_Prices in {_currency(price.ticker)}._", ""]
    lines += _table(rows)
    lines.append("")

    bars = getattr(price, "recent_bars", None) or []
    today = datetime.now().date().isoformat()
    # a flat bar dated today (open == high == low) is a live/pre-open tick, not a finished session
    bars = [b for b in bars if not (str(b.date)[:10] == today and b.high == b.low)]
    if bars:
        lines += ["**Recent trading days**", "",
                  "| Date | Open | High | Low | Close | Volume |", "|---|---|---|---|---|---|"]
        for b in bars:
            lines.append(
                f"| {b.date} | {b.open:,.2f} | {b.high:,.2f} | {b.low:,.2f} | {b.close:,.2f} | {b.volume:,} |"
            )
        lines.append("")
    return lines


def _section_fundamentals(f) -> list[str]:
    rows = [
        ("Sector", f.sector),
        ("Market cap", _compact(f.market_cap)),
        ("Beta", _num(f.beta)),
        ("Trailing P/E", _num(f.trailing_pe)),
        ("Forward P/E", _num(f.forward_pe)),
        ("Price-to-book", _num(f.price_to_book)),
        ("Return on equity", _num(f.roe_pct, 2, "%")),
        ("Net profit margin", _num(f.profit_margin_pct, 2, "%")),
        ("Revenue growth", _num(f.revenue_growth_pct, 2, "%")),
        ("Earnings growth", _num(f.earnings_growth_pct, 2, "%")),
        ("Debt-to-equity", _num(f.debt_to_equity)),
        ("Free cash flow", _compact(f.free_cash_flow)),
        ("Dividend yield", _num(f.dividend_yield_pct, 2, "%")),
    ]
    return ["## Fundamentals", ""] + _table(rows) + [""]


def _section_risk_rules(verdict) -> list[str]:
    lines = [
        "## Risk assessment",
        "",
        re.sub(r"from \d+/\d+ rules", f"from {_risk_counts(verdict)[0]}/{_risk_counts(verdict)[1]} rules",
               verdict.summary or ""),
        "",
        f"Rules evaluated: **{_risk_counts(verdict)[0]}/{_risk_counts(verdict)[1]}**"
        + (f" ({_risk_counts(verdict)[2]} not applicable to this sector)" if _risk_counts(verdict)[2] else ""),
        "",
        "| Rule | Status | Detail |",
        "|---|---|---|",
    ]
    for r in verdict.rules:
        lines.append(f"| {_esc(r.name)} | {_STATUS_ICON.get(r.status, r.status)} | {_esc(r.detail)} |")
    lines.append("")
    return lines


def _section_return_factors(returns) -> list[str]:
    lines = [
        "## Return potential",
        "",
        returns.summary or "",
        "",
        f"Factors evaluated: **{returns.factors_evaluated}/{returns.factors_total}**",
        "",
        "| Factor | Score | Detail |",
        "|---|---|---|",
    ]
    for fct in returns.factors:
        score = f"{fct.score:.1f}" if getattr(fct, "score", None) is not None else "n/a"
        lines.append(f"| {_esc(fct.name)} | {score} | {_esc(fct.detail)} |")
    lines.append("")
    return lines


def _data_notes(price, fundamentals, verdict, returns) -> list[str]:
    notes: list[str] = []

    na = _not_applicable(verdict)
    if na:
        notes.append(f"Not applicable to this sector (excluded from scoring): {', '.join(na)}.")
    skipped = [r.name for r in verdict.rules if r.status == "skipped" and r.name not in na]
    if skipped:
        notes.append(f"Risk rules skipped for missing data: {', '.join(skipped)}.")
    if verdict.score is None:
        notes.append("No risk score could be computed because too little data was available.")
    if returns is not None and returns.factors_evaluated < returns.factors_total:
        no_data = [f.name for f in returns.factors if getattr(f, "score", None) is None]
        notes.append(
            f"Return potential used {returns.factors_evaluated} of {returns.factors_total} factors"
            + (f" (no data: {', '.join(no_data)})." if no_data else ".")
        )

    if price.avg_volume is not None and price.avg_volume < 50_000:
        notes.append(
            f"Average daily volume is low ({price.avg_volume:,}). This may be an illiquid listing "
            "or the wrong symbol; verify the ticker before relying on these numbers."
        )

    f_fields = [getattr(fundamentals, name, None) for name in _fundamental_fields(verdict)]
    missing = sum(1 for v in f_fields if v is None)
    if missing >= len(f_fields) // 2:
        notes.append(
            f"{missing} of {len(f_fields)} core fundamentals are missing; treat valuation conclusions with caution."
        )

    if not notes:
        return []
    return ["## Data quality notes", ""] + [f"- {n}" for n in notes] + [""]


# ---------------------------------------------------------------- public API

def build_report(
    *,
    query: str,
    prefs: "UserPrefs",
    price: "PriceData",
    fundamentals: "FundamentalsData",
    verdict: "RiskVerdict",
    returns: Optional["ReturnVerdict"] = None,
    combined: Optional["CombinedVerdict"] = None,
    llm_answer: str = "",
    generated_at: Optional[datetime] = None,
) -> str:
    """Return the full report as Markdown."""
    ts = (generated_at or datetime.now()).strftime("%d %B %Y, %H:%M")
    title = fundamentals.company_name or price.ticker

    lines: list[str] = [
        f"# Investment Analysis: {title} ({price.ticker})",
        "",
        f"**Generated:** {ts}  ",
        f"**Investor profile:** {prefs.profile} | **Horizon:** {prefs.horizon}  ",
        f"**Query:** {query}",
        "",
    ]
    allowed = _allowed_numbers(price, fundamentals, verdict, returns, combined)
    lines += _safe_narrative_block(llm_answer, [price.ticker], allowed)   # Suggestion / Summary first
    lines += _section_glance(prefs, verdict, returns, combined)
    lines += _section_data_quality(price, fundamentals, verdict, returns)
    lines += _section_price(price)
    lines += _section_fundamentals(fundamentals)
    lines += _section_risk_rules(verdict)
    if returns is not None:
        lines += _section_return_factors(returns)
    lines += _data_notes(price, fundamentals, verdict, returns)

    if DISCLAIMER:
        lines += ["---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)



def build_multi_report(
    *,
    query: str,
    prefs: "UserPrefs",
    assessments: list[dict[str, Any]],
    llm_answer: str = "",
    generated_at: Optional[datetime] = None,
) -> str:
    """Build one comparison report while keeping each stock's computed data isolated.

    Each assessment dictionary must contain the typed fields accepted by build_report:
    price, fundamentals, verdict, returns, and combined.
    """
    if not assessments:
        raise ValueError("Cannot build a multi-stock report without assessments.")

    ts = (generated_at or datetime.now()).strftime("%d %B %Y, %H:%M")
    tickers = [item["price"].ticker for item in assessments]
    lines = [
        "# Investment Comparison: " + " vs ".join(tickers),
        "",
        f"**Generated:** {ts}  ",
        f"**Investor profile:** {prefs.profile} | **Horizon:** {prefs.horizon}  ",
        f"**Query:** {query}",
        "",
    ]
    allowed_all = _allowed_numbers(*[
        [item["price"], item["fundamentals"], item["verdict"], item.get("returns"), item.get("combined")]
        for item in assessments])
    lines += _safe_narrative_block(llm_answer, tickers, allowed_all)

    for index, item in enumerate(assessments, start=1):
        stock_md = build_report(
            query=query,
            prefs=prefs,
            price=item["price"],
            fundamentals=item["fundamentals"],
            verdict=item["verdict"],
            returns=item.get("returns"),
            combined=item.get("combined"),
            llm_answer="",
            generated_at=generated_at,
        )
        # Keep each stock's full structured sections but omit its repeated title metadata
        # and repeated disclaimer; the comparison report has a single shared header/footer.
        stock_lines = stock_md.splitlines()
        first_section = next(
            (i for i, line in enumerate(stock_lines) if line.startswith("## Verdict at a glance")),
            None,
        )
        if first_section is None:
            raise ValueError(f"Could not find structured report sections for {item['price'].ticker}.")
        body = ["#" + l if l.startswith("## ") else l for l in stock_lines[first_section:]]
        lines += [f"## Stock {index}: {item['price'].ticker}", ""] + body + [""]

    if DISCLAIMER:
        lines += ["---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)


def multi_report_to_dict(
    *,
    query: str,
    prefs: "UserPrefs",
    assessments: list[dict[str, Any]],
    llm_answer: str = "",
) -> dict:
    """Structured JSON representation of a comparison report, with per-ticker records."""
    allowed = _allowed_numbers(*[
        [i["price"], i["fundamentals"], i["verdict"], i.get("returns"), i.get("combined")] for i in assessments])
    safe_answer, _ = validate_ai_narrative(llm_answer, [item["price"].ticker for item in assessments], allowed)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "multi_stock_comparison",
        "query": query,
        "tickers": [item["price"].ticker for item in assessments],
        "prefs": _dump(prefs),
        "assessments": [
            report_to_dict(
                query=query,
                prefs=prefs,
                price=item["price"],
                fundamentals=item["fundamentals"],
                verdict=item["verdict"],
                returns=item.get("returns"),
                combined=item.get("combined"),
                llm_answer="",
            )
            for item in assessments
        ],
        "llm_answer": safe_answer,
    }


def save_report(markdown: str, ticker: str, out_dir: str = REPORT_DIR, stamp: Optional[str] = None) -> Path:
    """Write the report to reports/<TICKER>_<timestamp>.md and return the path.

    Pass the same `stamp` to save_report and save_report_json so the .md and .json filenames match.
    """
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{safe}_{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def report_to_dict(
    *, query, prefs, price, fundamentals, verdict, returns=None, combined=None, llm_answer: str = ""
) -> dict:
    """JSON-serializable version of the same data: for a database row or the follow-up Q&A step."""
    allowed = _allowed_numbers(price, fundamentals, verdict, returns, combined)
    safe_answer, _ = validate_ai_narrative(llm_answer, [price.ticker], allowed)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "ticker": price.ticker,
        "prefs": _dump(prefs),
        "price": _dump(price),
        "fundamentals": _dump(fundamentals),
        "risk_verdict": _dump(verdict),
        "return_verdict": _dump(returns),
        "combined": _dump(combined),
        "data_quality": _data_quality(price, fundamentals, verdict, returns),
        "llm_answer": safe_answer,
    }


def save_report_json(data: dict, ticker: str, out_dir: str = REPORT_DIR, stamp: Optional[str] = None) -> Path:
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{safe}_{stamp}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path