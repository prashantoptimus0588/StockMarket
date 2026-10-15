"""LangGraph state schema and progress labels."""
from typing import Annotated, Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from models.models import FundamentalsData, PriceData, UserPrefs
from chat.schemas import Target

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AssessmentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)
    query: str = ""
    prefs: Optional[UserPrefs] = None
    current_step: int = 0                                   # general agent loop only
    route: str = ""
    targets: list[Target] = Field(default_factory=list)
    wants_news: bool = False
    prices: dict[str, PriceData] = Field(default_factory=dict)          # keyed by upper-case ticker
    fundamentals: dict[str, FundamentalsData] = Field(default_factory=dict)
    news: dict[str, str] = Field(default_factory=dict)
    flagged_tickers: list[str] = Field(default_factory=list)            # tool reported a data-quality warning
    warnings: list[str] = Field(default_factory=list)
    fetch_failures: dict[str, str] = Field(default_factory=dict)       # ticker -> why price/fundamentals could not be fetched
    assessments: dict[str, dict] = Field(default_factory=dict)          # ticker -> engine results
    answer: str = ""
    report_markdown: str = ""
    report_json: Optional[dict] = None
    assessment_id: Optional[int] = None
    approval_status: str = "pending"
    approval_comment: Optional[str] = None
    replan_count: int = 0                                   # how many times a reject sent us back to plan
    replan_pending: bool = False                            # approval_node asked to run plan_node again
    rejected_targets: list[str] = Field(default_factory=list)  # companies already assessed and rejected
    replan_hint: str = ""                                   # why the last alternative could not be assessed
    previous_report_markdown: str = ""                      # last good report, kept if a replan fails
    previous_report_json: Optional[dict] = None
    previous_assessment_id: Optional[int] = None
    previous_report_md_path: Optional[str] = None
    previous_report_json_path: Optional[str] = None
    report_saved: bool = False
    report_md_path: Optional[str] = None
    report_json_path: Optional[str] = None


NODE_LABELS = {
    "plan_node": "Understanding your request",
    "resolve_node": "Resolving ticker symbols",
    "fetch_node": "Fetching market data",
    "risk_engine_node": "Running the risk and return engines",
    "explain_node": "Writing the explanation",
    "report_node": "Building and saving the report",
    "approval_node": "Recording your decision",
    "restore_report_node": "Keeping the previous report",
    "agent_node": "Thinking",
    "tools": "Calling tools",
}


