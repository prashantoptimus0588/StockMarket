from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    ticker: Mapped[str] = mapped_column(String(30), index=True)
    profile: Mapped[str] = mapped_column(String(30))
    horizon: Mapped[str] = mapped_column(String(30))

    risk_verdict: Mapped[str] = mapped_column(String(30))
    return_potential: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )
    combined_verdict: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    report_markdown: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    report_json: Mapped[dict | None] = mapped_column(
        JSON, nullable=True
    )

    approval_status: Mapped[str] = mapped_column(
        String(20), default="pending", index=True
    )
    approval_comment: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    thread_id: Mapped[str] = mapped_column(
        String(100), index=True, nullable=False
    )
