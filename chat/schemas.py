"""Pydantic schemas used for structured model responses."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured LLM outputs
# ---------------------------------------------------------------------------
class Target(BaseModel):
    company: str = Field(description="Company name (or ticker) exactly as the user wrote it")
    ticker: Optional[str] = Field(
        default=None,
        description="Exact ticker WITH exchange suffix, only if you are 100% sure; otherwise null",
    )


class QueryPlan(BaseModel):
    route: Literal["stock_analysis", "general"] = Field(
        description="'stock_analysis' if the user wants a stock/company assessed or compared; "
                    "'general' for anything else (weather, general news, chit-chat)"
    )
    targets: list[Target] = Field(default_factory=list)
    wants_news: bool = Field(default=False, description="True only if the user asked for recent news")


class TickerChoice(BaseModel):
    company: str
    ticker: Optional[str] = Field(default=None, description="Chosen ticker with suffix, or null if no good match")


class TickerResolution(BaseModel):
    choices: list[TickerChoice]


Suggestion = Literal["Consider", "Hold off", "Avoid", "Insufficient data"]


class TickerSuggestion(BaseModel):
    ticker: str
    suggestion: Suggestion
    reason: str = Field(description="1-2 sentences tying the suggestion to the engine rules and the user's profile/horizon")


class Explanation(BaseModel):
    suggestions: list[TickerSuggestion]
    summary: str = Field(description="Short plain-language explanation of the verdicts")
    key_points: list[str] = Field(default_factory=list)
    news_points: list[str] = Field(default_factory=list, description="Only if a NEWS block was provided; keep source URLs")
    takeaway: str = Field(description="ONE line restating the combined verdict for this profile and horizon")


