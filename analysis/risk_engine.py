"""Deterministic risk-fit engine.

Pure function: (PriceData, FundamentalsData, UserPrefs) -> RiskVerdict.
No LLM, no network. Measures how well a stock's RISK profile fits the user's
profile + horizon. It does not predict returns.

To tune behaviour, edit the numbers in RULES (limits per profile, weights per horizon).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.models import (
    FundamentalsData,
    PriceData,
    RiskVerdict,
    RuleResult,
    UserPrefs,
)

PROFILES = ("aggressive", "balanced", "conservative")
HORIZONS = ("short", "medium", "long")

FIT_SCORE = 70.0          # score >= this (and no heavy rule failed) -> "fits"
BORDERLINE_SCORE = 50.0   # score >= this -> "borderline", else "does_not_fit"
MIN_COVERAGE = 0.5        # evaluated weight / applicable weight below this -> no verdict
HEAVY_WEIGHT = 3.0        # one failed rule this heavy caps the result at "borderline"; two or more -> "does_not_fit"
SEVERE_MARGIN = 0.5       # a failed rule beyond its limit by this fraction (50%) counts as heavy, whatever its weight
# Price-risk rules: failing any of them means the stock cannot be rated "fits", even for long horizons
# where their weights are small (a stock that fell 40% and is still 30% under its peak is not a clean fit).
CAP_RULES = frozenset({"Volatility", "Max drawdown", "Current drawdown"})


@dataclass(frozen=True)
class Rule:
    name: str
    source: str                 # "price" or "fundamentals"
    field: str                  # attribute on PriceData / FundamentalsData
    op: str                     # "max": value <= limit | "min": value >= limit | "absmax": |value| <= limit
    unit: str
    limits: tuple               # (aggressive, balanced, conservative); None = rule not used for that profile
    weights: tuple              # (short, medium, long) horizon weights
    exempt_sectors: frozenset = frozenset()


# Banks/NBFCs run high leverage and have meaningless free cash flow, so skip those rules for them.
FINANCIALS = frozenset({"Financial Services"})

RULES: tuple[Rule, ...] = (
    #    name                source          field                  op       unit  (agg, bal, cons)          (short, med, long)
    Rule("Volatility",       "price",        "volatility_pct",      "max",    "%", (55, 38, 28),             (3, 2, 1)),
    Rule("Max drawdown",     "price",        "max_drawdown_pct",    "absmax", "%", (50, 35, 25),             (2, 2, 1.5)),
    Rule("Current drawdown", "price",        "current_drawdown_pct", "absmax", "%", (40, 25, 15),            (3, 1.5, 0.5)),
    Rule("Liquidity (avg volume)", "price",  "avg_volume",          "min",    "",  (100_000, 300_000, 500_000), (3, 2, 1)),
    Rule("Beta",             "fundamentals", "beta",                "max",    "",  (1.6, 1.2, 0.9),           (2, 1.5, 1)),
    Rule("Debt/equity",      "fundamentals", "debt_to_equity",      "max",    "",  (2.0, 1.2, 0.6),           (1, 2, 3), FINANCIALS),
    Rule("Profit margin",    "fundamentals", "profit_margin_pct",   "min",    "%", (-10, 5, 10),              (1, 1.5, 2)),
    Rule("Return on equity", "fundamentals", "roe_pct",             "min",    "%", (0, 8, 12),               (0.5, 1.5, 2.5)),
    Rule("Earnings growth",  "fundamentals", "earnings_growth_pct", "min",    "%", (-25, 0, 5),              (1, 1.5, 2)),
    Rule("Trailing P/E",     "fundamentals", "trailing_pe",         "max",    "",  (80, 50, 30),             (0.5, 1, 1.5)),
    Rule("Free cash flow",   "fundamentals", "free_cash_flow",      "min",    "",  (None, 0, 0),             (0.5, 1.5, 2), FINANCIALS),
)


def _skipped(rule: Rule, why: str) -> RuleResult:
    return RuleResult(name=rule.name, status="skipped", detail=why)


def _check(rule: Rule, value: float, limit: float) -> RuleResult:
    if rule.op == "max":
        ok, shown, cmp_ = value <= limit, value, "<="
    elif rule.op == "min":
        ok, shown, cmp_ = value >= limit, value, ">="
    else:  # absmax
        ok, shown, cmp_ = abs(value) <= limit, abs(value), "<="
    detail = f"{shown:,.2f}{rule.unit} (need {cmp_} {limit:,g}{rule.unit})"
    return RuleResult(name=rule.name, status="pass" if ok else "fail", detail=detail)


def _overshoot(rule: Rule, value: float, limit: float) -> Optional[float]:
    """How far past its limit a failed rule is, as a fraction of the limit (None when the limit is 0)."""
    if not limit:
        return None
    if rule.op == "min":
        return (limit - value) / abs(limit)
    shown = abs(value) if rule.op == "absmax" else value
    return (shown - limit) / abs(limit)


def evaluate_risk(
    price: Optional[PriceData],
    fundamentals: Optional[FundamentalsData],
    prefs: UserPrefs,
) -> RiskVerdict:
    p_idx = PROFILES.index(prefs.profile)
    h_idx = HORIZONS.index(prefs.horizon)
    sector = fundamentals.sector if fundamentals else None

    results: list[RuleResult] = []
    applicable_w = evaluated_w = passed_w = 0.0
    heavy_fails = 0
    severe: list[str] = []
    capped_by: list[str] = []

    for rule in RULES:
        weight = rule.weights[h_idx]
        limit = rule.limits[p_idx]

        if sector in rule.exempt_sectors:
            results.append(_skipped(rule, f"not applicable to sector '{sector}'"))
            continue
        if limit is None:
            results.append(_skipped(rule, f"not used for {prefs.profile} profile"))
            continue

        applicable_w += weight
        source = price if rule.source == "price" else fundamentals
        value = getattr(source, rule.field, None) if source is not None else None
        if value is None:
            results.append(_skipped(rule, "no data"))
            continue

        res = _check(rule, float(value), float(limit))
        evaluated_w += weight
        if res.status == "pass":
            passed_w += weight
        else:
            over = _overshoot(rule, float(value), float(limit))
            is_severe = over is not None and over >= SEVERE_MARGIN
            if over is not None and over > 0:
                res = res.model_copy(update={"detail": f"{res.detail}; {over:.0%} past the limit"})
            if weight >= HEAVY_WEIGHT or is_severe:
                heavy_fails += 1
            if is_severe:
                severe.append(rule.name)
            if rule.name in CAP_RULES:
                capped_by.append(rule.name)
        results.append(res)

    evaluated = sum(1 for r in results if r.status != "skipped")
    total = len(results)
    coverage = evaluated_w / applicable_w if applicable_w else 0.0
    label = f"{prefs.profile}/{prefs.horizon}"

    if coverage < MIN_COVERAGE:
        return RiskVerdict(
            result="borderline",
            score=None,
            rules_evaluated=evaluated,
            rules_total=total,
            rules=results,
            summary=f"Insufficient data for {label}: only {evaluated}/{total} rules could be evaluated. No verdict.",
        )

    score = round(passed_w / evaluated_w * 100, 1)
    if heavy_fails >= 2 or score < BORDERLINE_SCORE:
        result = "does_not_fit"
    elif score >= FIT_SCORE and heavy_fails == 0 and not capped_by:
        result = "fits"
    else:
        result = "borderline"

    failed = [r.name for r in results if r.status == "fail"]
    applicable_n = sum(1 for r in results
                       if not (r.status == "skipped" and ("not applicable" in r.detail or "not used for" in r.detail)))
    summary = f"{result} for {label}: score {score}/100 from {evaluated}/{applicable_n} rules."
    if failed:
        summary += " Failed: " + ", ".join(failed) + "."
    if severe:
        summary += " Far beyond the limit: " + ", ".join(severe) + "."
    if result == "borderline" and capped_by and not heavy_fails:
        summary += " Capped at borderline because " + ", ".join(capped_by) + " failed."
    elif heavy_fails and result != "does_not_fit":
        summary += f" {heavy_fails} high-weight or severe failure(s) cap the result."

    return RiskVerdict(
        result=result,
        score=score,
        rules_evaluated=evaluated,
        rules_total=total,
        rules=results,
        summary=summary,
    )