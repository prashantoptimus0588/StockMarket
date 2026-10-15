import math
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


def _clean_num(v):
    """None for NaN/inf/non-numeric strings; numeric strings become floats."""
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = float(v.replace(",", ""))
        except ValueError:
            return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _to_int(v):
    v = _clean_num(v)
    return None if v is None else int(round(v))

# ---------- User choices ----------

Profile = Literal["aggressive", "balanced", "conservative"]
Horizon = Literal["short", "medium", "long"]


class UserPrefs(BaseModel):
    profile: Profile    # how much risk the user accepts
    horizon: Horizon    # how long they plan to hold


# ---------- Market data (what the tools will return) ----------
# All percentage fields hold plain percent numbers: 15.18 means 15.18%.
# Every metric is Optional, because Yahoo does not report everything for every stock.

class DailyBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    @field_validator("volume", mode="before")
    @classmethod
    def _volume(cls, v):
        n = _to_int(v)
        return 0 if n is None else n          # some exchanges report no volume for a day


class PriceData(BaseModel):
    ticker: str
    latest_close: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    volatility_pct: Optional[float] = None          # annualized
    max_drawdown_pct: Optional[float] = None        # negative number, e.g. -22.5
    current_drawdown_pct: Optional[float] = None    # negative number, 0 = at peak
    avg_volume: Optional[int] = None
    recent_bars: list[DailyBar] = Field(default_factory=list)

    @field_validator("latest_close", "high_52w", "low_52w", "volatility_pct",
                     "max_drawdown_pct", "current_drawdown_pct", mode="before")
    @classmethod
    def _floats(cls, v):
        return _clean_num(v)

    @field_validator("avg_volume", mode="before")
    @classmethod
    def _avg_volume(cls, v):
        return _to_int(v)                      # a mean volume is often fractional

    @field_validator("recent_bars", mode="before")
    @classmethod
    def _bars(cls, v):
        """Drop bars with missing prices (holidays, half-days) instead of failing the whole payload."""
        good = []
        for b in v or []:
            if isinstance(b, dict):
                if all(_clean_num(b.get(k)) is not None for k in ("open", "high", "low", "close")):
                    good.append({**b, **{k: _clean_num(b[k]) for k in ("open", "high", "low", "close")}})
            else:
                good.append(b)
        return good

    @model_validator(mode="after")
    def _latest_from_bars(self):
        """If the source computed latest_close from an unfinished row (null), use the last complete bar."""
        if self.latest_close is None and self.recent_bars:
            self.latest_close = self.recent_bars[-1].close
        return self


class FundamentalsData(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    roe_pct: Optional[float] = None
    profit_margin_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    earnings_growth_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None          # ratio, 0.45 means 45%
    free_cash_flow: Optional[float] = None
    dividend_yield_pct: Optional[float] = None

    @field_validator("market_cap", "beta", "trailing_pe", "forward_pe", "price_to_book", "roe_pct",
                     "profit_margin_pct", "revenue_growth_pct", "earnings_growth_pct",
                     "debt_to_equity", "free_cash_flow", "dividend_yield_pct", mode="before")
    @classmethod
    def _floats(cls, v):
        return _clean_num(v)


class NewsItem(BaseModel):
    published: str
    title: str
    source: str


class NewsData(BaseModel):
    query: str
    items: list[NewsItem] = Field(default_factory=list)


# ---------- Risk engine output ----------

class RuleResult(BaseModel):
    name: str
    status: Literal["pass", "fail", "skipped"]
    detail: str                                      # why it passed, failed or was skipped


class RiskVerdict(BaseModel):
    result: Literal["fits", "borderline", "does_not_fit"]
    score: Optional[float] = None                    # 0 to 100, None if too little data
    rules_evaluated: int
    rules_total: int
    rules: list[RuleResult] = Field(default_factory=list)
    summary: str = ""