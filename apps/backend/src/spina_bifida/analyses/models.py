from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from spina_bifida.extensions import db

if TYPE_CHECKING:
    from spina_bifida.auth.models import User


def utc_now() -> datetime:
    return datetime.now(UTC)


def enum_values(enum_class: type[StrEnum]) -> list[str]:
    return [item.value for item in enum_class]


class JobState(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    QUALITY_CHECK = "quality_check"
    PLANE_DETECTION = "plane_detection"
    ROI_DETECTION = "roi_detection"
    SCORING = "scoring"
    EXPLANATION = "explanation"
    COMPLETED = "completed"
    REJECTED_QUALITY = "rejected_quality"
    FAILED_RECOVERABLE = "failed_recoverable"
    EXPIRED = "expired"
    DELETED = "deleted"


class ProcessingStage(StrEnum):
    VALIDATING = "validating"
    QUALITY_CHECK = "quality_check"
    PLANE_DETECTION = "plane_detection"
    ROI_DETECTION = "roi_detection"
    SCORING = "scoring"
    EXPLANATION = "explanation"


class AssetKind(StrEnum):
    ORIGINAL = "original"
    ROI = "roi"
    HEATMAP = "heatmap"


class RetentionClass(StrEnum):
    TEMPORARY = "temporary"
    FEEDBACK = "feedback"
    RETAINED = "retained"


class FeedbackValue(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class AnalysisJob(db.Model):  # type: ignore[misc, name-defined]
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        CheckConstraint(
            "model_version IN ('v1', 'v2')",
            name="ck_analysis_jobs_model_version",
        ),
        CheckConstraint(
            "batch_position >= 0",
            name="ck_analysis_jobs_batch_position_nonnegative",
        ),
        UniqueConstraint(
            "batch_id",
            "batch_position",
            name="uq_analysis_jobs_batch_position",
        ),
        Index("ix_analysis_jobs_state_created_at_id", "state", "created_at", "id"),
        Index("ix_analysis_jobs_owner_id_batch_id", "owner_id", "batch_id"),
        Index("ix_analysis_jobs_expires_at", "expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    owner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False)
    batch_position: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[Literal["v1", "v2"]] = mapped_column(
        String(2),
        nullable=False,
    )
    state: Mapped[JobState] = mapped_column(
        Enum(JobState, values_callable=enum_values, native_enum=False, length=32),
        nullable=False,
        default=JobState.QUEUED,
    )
    stage: Mapped[ProcessingStage | None] = mapped_column(
        Enum(ProcessingStage, values_callable=enum_values, native_enum=False, length=32),
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(128))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    owner: Mapped[User] = relationship(back_populates="jobs")
    assets: Mapped[list[AnalysisAsset]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="AnalysisAsset.position",
    )
    feedback: Mapped[list[Feedback]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )


class AnalysisAsset(db.Model):  # type: ignore[misc, name-defined]
    __tablename__ = "analysis_assets"
    __table_args__ = (Index("ix_analysis_assets_job_id", "job_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    frame_id: Mapped[str] = mapped_column(String(128), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[AssetKind] = mapped_column(
        Enum(AssetKind, values_callable=enum_values, native_enum=False, length=16),
        nullable=False,
    )
    local_path: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    retention_class: Mapped[RetentionClass] = mapped_column(
        Enum(RetentionClass, values_callable=enum_values, native_enum=False, length=16),
        nullable=False,
        default=RetentionClass.TEMPORARY,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    job: Mapped[AnalysisJob] = relationship(back_populates="assets")


class Feedback(db.Model):  # type: ignore[misc, name-defined]
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("job_id", "author_id", name="uq_feedback_job_author"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    author_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    value: Mapped[FeedbackValue] = mapped_column(
        Enum(FeedbackValue, values_callable=enum_values, native_enum=False, length=16),
        nullable=False,
    )
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    job: Mapped[AnalysisJob] = relationship(back_populates="feedback")
    author: Mapped[User | None] = relationship(back_populates="feedback")
