"""Return-potential engine + combined score. Pure functions, no LLM, no network.

IMPORTANT: this does NOT forecast returns. Nobody can do that from a handful of Yahoo fields.
It scores *return potential*: how favourable growth, profitability, valuation, momentum and
income look today, weighted by horizon. Treat high/medium/low as a relative indicator.

Edit FACTORS (anchors + horizon weights) and RETURN_WEIGHT to tune behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from models.models import FundamentalsData, PriceData, RiskVerdict, UserPrefs

HORIZONS = ("short", "medium", "long")

HIGH_SCORE = 65.0        # return score >= this -> "high"
LOW_SCORE = 40.0         # return score <  this -> "low", otherwise "medium"
MIN_COVERAGE = 0.5       # evaluated weight / total weight below this -> no verdict
NEUTRAL_SCORE = 50.0     # missing factors pull the score toward this, so missing data never raises it

# Share of the combined score that comes from return potential (rest comes from risk fit).
RETURN_WEIGHT = {"aggressive": 0.5, "balanced": 0.4, "conservative": 0.25}
STRONG_SCORE = 70.0
WATCH_SCORE = 50.0


# ---------- Output models ----------

class ReturnFactor(BaseModel):
    name: str
    score: Optional[float] = None       # 0-100, None if no data
    detail: str


class ReturnVerdict(BaseModel):
    level: Optional[Literal["high", "medium", "low"]] = None   # None if too little data
    score: Optional[float] = None                              # 0-100
    factors_evaluated: int
    factors_total: int
    factors: list[ReturnFactor] = Field(default_factory=list)
    summary: str = ""


class CombinedVerdict(BaseModel):
    label: Literal["strong_fit", "watch", "weak_fit"]
    score: Optional[float] = None       # 0-100, None if either side lacks data
    risk_score: Optional[float] = None
    return_score: Optional[float] = None
    return_weight: float
    summary: str = ""


# ---------- Factors ----------

def _pos_52w(p: Optional[PriceData], f: Optional[FundamentalsData]) -> Optional[float]:
    """Where the price sits in its 52-week range: 0 = at the low, 1 = at the high."""
    if p is None or None in (p.latest_close, p.high_52w, p.low_52w) or p.high_52w <= p.low_52w:
        return None
    return (p.latest_close - p.low_52w) / (p.high_52w - p.low_52w)


def _fwd_ratio(p: Optional[PriceData], f: Optional[FundamentalsData]) -> Optional[float]:
    """forward P/E / trailing P/E. Below 1 means analysts expect earnings to grow."""
    if f is None or not f.trailing_pe or not f.forward_pe or f.trailing_pe <= 0 or f.forward_pe <= 0:
        return None
    return f.forward_pe / f.trailing_pe


@dataclass(frozen=True)
class Factor:
    name: str
    get: Callable[[Optional[PriceData], Optional[FundamentalsData]], Optional[float]]
    bad: float       # value that scores 0
    good: float      # value that scores 100 (can be lower than bad when lower is better)
    unit: str
    weights: tuple   # (short, medium, long)
    max_abs: Optional[float] = None   # |value| above this is treated as a data glitch and ignored


FACTORS: tuple[Factor, ...] = (
    Factor("Earnings growth",        lambda p, f: f.earnings_growth_pct if f else None, -10, 25,  "%", (1, 2, 3), 300),
    Factor("Revenue growth",         lambda p, f: f.revenue_growth_pct if f else None,    0, 20,  "%", (0.5, 1, 2), 300),
    Factor("Return on equity",       lambda p, f: f.roe_pct if f else None,               5, 20,  "%", (0.5, 1.5, 2.5)),
    Factor("Profit margin",          lambda p, f: f.profit_margin_pct if f else None,     0, 20,  "%", (0.5, 1, 1.5)),
    Factor("Valuation (trailing P/E)", lambda p, f: f.trailing_pe if f else None,        50, 12,  "",  (1, 1.5, 2)),
    Factor("Earnings outlook (fwd/trailing P/E)", _fwd_ratio,                          1.2, 0.7, "",  (1, 1, 1)),
    Factor("Momentum (52w range position)", _pos_52w,                                  0.2, 0.9, "",  (3, 1.5, 0.5)),
    Factor("Dividend yield",         lambda p, f: f.dividend_yield_pct if f else None,    0, 4,   "%", (0.5, 1, 1.5)),
)


def _scale(value: float, bad: float, good: float) -> float:
    t = (value - bad) / (good - bad)
    return max(0.0, min(1.0, t)) * 100.0


def evaluate_return(
    price: Optional[PriceData],
    fundamentals: Optional[FundamentalsData],
    prefs: UserPrefs,
) -> ReturnVerdict:
    h_idx = HORIZONS.index(prefs.horizon)
    factors: list[ReturnFactor] = []
    total_w = evaluated_w = weighted = 0.0

    for fac in FACTORS:
        w = fac.weights[h_idx]
        total_w += w
        value = fac.get(price, fundamentals)
        if value is None:
            factors.append(ReturnFactor(name=fac.name, detail="no data"))
            continue
        if fac.max_abs is not None and abs(float(value)) > fac.max_abs:
            factors.append(ReturnFactor(
                name=fac.name,
                detail=f"{value:,.2f}{fac.unit} looks like a data glitch (beyond {fac.max_abs:g}{fac.unit}); ignored"))
            continue
        s = _scale(float(value), fac.bad, fac.good)
        evaluated_w += w
        weighted += s * w
        factors.append(ReturnFactor(
            name=fac.name,
            score=round(s, 1),
            detail=f"{value:,.2f}{fac.unit} -> {s:.0f}/100 (0 at {fac.bad:g}{fac.unit}, 100 at {fac.good:g}{fac.unit})",
        ))

    evaluated = sum(1 for f in factors if f.score is not None)
    total = len(factors)

    if total_w == 0 or evaluated_w / total_w < MIN_COVERAGE:
        return ReturnVerdict(
            level=None, score=None, factors_evaluated=evaluated, factors_total=total, factors=factors,
            summary=f"Insufficient data for return potential: {evaluated}/{total} factors available. No verdict.",
        )

    coverage = evaluated_w / total_w
    raw = weighted / evaluated_w
    # shrink toward neutral by the share of factor weight that is missing
    score = round(coverage * raw + (1 - coverage) * NEUTRAL_SCORE, 1)
    level = "high" if score >= HIGH_SCORE else "low" if score < LOW_SCORE else "medium"
    note = (f" Adjusted from {raw:.1f} because only {coverage:.0%} of the factor weight had data."
            if coverage < 0.999 else "")
    return ReturnVerdict(
        level=level, score=score, factors_evaluated=evaluated, factors_total=total, factors=factors,
        summary=(f"Return potential {level} for {prefs.horizon} horizon: score {score}/100 "
                 f"from {evaluated}/{total} factors.{note}"),
    )


def combine(risk: RiskVerdict, ret: ReturnVerdict, prefs: UserPrefs) -> CombinedVerdict:
    """Blend risk fit and return potential. Risk acts as a gate: poor risk fit can't be bought back with returns."""
    w = RETURN_WEIGHT[prefs.profile]

    if risk.score is not None and ret.score is None and risk.result == "does_not_fit":
        return CombinedVerdict(
            label="weak_fit", score=None, risk_score=risk.score, return_score=None, return_weight=w,
            summary=(f"weak_fit for {prefs.profile}/{prefs.horizon}: risk fit is poor and "
                     "return potential has insufficient data."),
        )

    if risk.score is None or ret.score is None:
        return CombinedVerdict(
            label="watch", score=None, risk_score=risk.score, return_score=ret.score, return_weight=w,
            summary="Insufficient data to combine risk fit and return potential. No combined score.",
        )

    score = round(w * ret.score + (1 - w) * risk.score, 1)
    if score >= STRONG_SCORE:
        label = "strong_fit"
    elif score >= WATCH_SCORE:
        label = "watch"
    else:
        label = "weak_fit"

    notes = []
    if risk.result == "does_not_fit":
        label = "weak_fit"
        notes.append("risk fit is poor")
    elif label == "strong_fit" and risk.result == "borderline":
        label = "watch"
        notes.append("capped because risk fit is only borderline")
    if label == "strong_fit" and ret.level == "low":
        label = "watch"
        notes.append("capped because return potential is low")

    summary = (f"{label} for {prefs.profile}/{prefs.horizon}: combined {score}/100 "
               f"(risk fit {risk.score} x {1 - w:.2f} + return potential {ret.score} x {w:.2f})")
    if notes:
        summary += "; " + "; ".join(notes)
    return CombinedVerdict(
        label=label, score=score, risk_score=risk.score, return_score=ret.score,
        return_weight=w, summary=summary + ".",
    )