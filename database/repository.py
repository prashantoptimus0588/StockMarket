from datetime import datetime, timezone

from sqlalchemy import select

from database.connection import SessionLocal
from database.models import Assessment


def save_assessment(
    *,
    thread_id:str,
    ticker: str,
    profile: str,
    horizon: str,
    risk_verdict: str,
    return_potential: str | None,
    combined_verdict: str | None,
    report_markdown: str | None = None,
    report_json: dict | None = None,
    
) -> int:
    """Persist an assessment and return its database ID."""

    with SessionLocal.begin() as session:
        # Retry of report_node: update the pending row. After a reject, insert a
        # new row so the reviewed assessment is not overwritten.
        pending = session.scalar(
            select(Assessment)
            .where(Assessment.thread_id == thread_id, Assessment.approval_status == "pending")
            .order_by(Assessment.id.desc())
            .limit(1)
        )

        if pending is not None:
            pending.ticker = ticker
            pending.profile = profile
            pending.horizon = horizon
            pending.risk_verdict = risk_verdict
            pending.return_potential = return_potential
            pending.combined_verdict = combined_verdict
            pending.report_markdown = report_markdown
            pending.report_json = report_json
            session.flush()
            return pending.id

        assessment = Assessment(
            thread_id=thread_id,
            ticker=ticker,
            profile=profile,
            horizon=horizon,
            risk_verdict=risk_verdict,
            return_potential=return_potential,
            combined_verdict=combined_verdict,
            report_markdown=report_markdown,
            report_json=report_json,
            approval_status="pending",
        )
        session.add(assessment)
        session.flush()
        assessment_id = assessment.id

    return assessment_id


def get_assessment_by_thread_id(thread_id: str) -> dict | None:
    """Return the latest assessment for a graph thread, if one was already saved."""
    with SessionLocal() as session:
        assessment = session.scalar(
            select(Assessment)
            .where(Assessment.thread_id == thread_id)
            .order_by(Assessment.id.desc())
            .limit(1)
        )
        if assessment is None:
            return None
        return {
            "id": assessment.id,
            "thread_id": assessment.thread_id,
            "ticker": assessment.ticker,
            "profile": assessment.profile,
            "horizon": assessment.horizon,
            "risk_verdict": assessment.risk_verdict,
            "return_potential": assessment.return_potential,
            "combined_verdict": assessment.combined_verdict,
            "report_markdown": assessment.report_markdown,
            "report_json": assessment.report_json,
            "approval_status": assessment.approval_status,
            "approval_comment": assessment.approval_comment,
            "created_at": assessment.created_at.isoformat(),
        }


def get_assessment(assessment_id: int) -> dict | None:
    """Retrieve an assessment by ID."""

    with SessionLocal() as session:
        assessment = session.get(Assessment, assessment_id)

        if assessment is None:
            return None

        return {
            "id": assessment.id,
            "thread_id": assessment.thread_id,
            "ticker": assessment.ticker,
            "profile": assessment.profile,
            "horizon": assessment.horizon,
            "risk_verdict": assessment.risk_verdict,
            "return_potential": assessment.return_potential,
            "combined_verdict": assessment.combined_verdict,
            "report_markdown": assessment.report_markdown,
            "report_json": assessment.report_json,
            "approval_status": assessment.approval_status,
            "approval_comment": assessment.approval_comment,
            "created_at": assessment.created_at.isoformat(),
            "reviewed_at": (
                assessment.reviewed_at.isoformat()
                if assessment.reviewed_at else None
            ),
        }


def record_decision(
    assessment_id: int,
    decision: str,
    comment: str | None = None,
) -> None:
    """Record a review decision without deleting the assessment."""

    if decision not in {"approved", "rejected"}:
        raise ValueError("Decision must be approved or rejected.")

    with SessionLocal.begin() as session:
        assessment = session.get(Assessment, assessment_id)

        if assessment is None:
            raise ValueError("Assessment not found.")

        if assessment.approval_status != "pending":
            raise ValueError("Assessment has already been reviewed.")

        assessment.approval_status = decision
        assessment.approval_comment = comment
        assessment.reviewed_at = datetime.now(timezone.utc)
