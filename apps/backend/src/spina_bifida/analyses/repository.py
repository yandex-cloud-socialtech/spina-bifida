from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from flask_sqlalchemy.session import Session as FlaskSession
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session, scoped_session

from spina_bifida.analyses.models import (
    AnalysisAsset,
    AnalysisJob,
    AssetKind,
    JobState,
    ProcessingStage,
    RetentionClass,
)
from spina_bifida.analyses.schemas import NewAnalysisJob, NewFrame
from spina_bifida.extensions import db

type SessionType = Session | scoped_session[Session] | scoped_session[FlaskSession]
PROCESSING_STATES = tuple(JobState(stage.value) for stage in ProcessingStage)


class JobNoLongerProcessable(RuntimeError):
    pass


class JobRepository:
    def __init__(self, session: SessionType | None = None) -> None:
        self.session: SessionType = session or db.session

    def create(
        self,
        owner_id: str,
        frames: Sequence[NewFrame],
        *,
        job_id: str | None = None,
        expires_at: datetime | None = None,
        retention_class: RetentionClass = RetentionClass.TEMPORARY,
    ) -> AnalysisJob:
        created_at = datetime.now(UTC)
        expires_at = expires_at or created_at + timedelta(hours=24)
        resolved_job_id = job_id or str(uuid4())
        job = AnalysisJob(
            id=resolved_job_id,
            owner_id=owner_id,
            batch_id=resolved_job_id,
            batch_position=0,
            model_version="v1",
            created_at=created_at,
            expires_at=expires_at,
        )
        for item in frames:
            job.assets.append(
                AnalysisAsset(
                    frame_id=item.frame_id,
                    position=item.position,
                    kind=AssetKind.ORIGINAL,
                    local_path=str(item.path),
                    checksum=item.checksum,
                    retention_class=retention_class,
                    expires_at=expires_at,
                ),
            )
        try:
            self.session.add(job)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return job

    def create_batch(
        self,
        owner_id: str,
        jobs: Sequence[NewAnalysisJob],
        *,
        expires_at: datetime,
        retention_class: RetentionClass,
    ) -> tuple[AnalysisJob, ...]:
        batch_created_at = datetime.now(UTC)
        records = tuple(
            self._record(
                owner_id,
                item,
                expires_at,
                retention_class,
                created_at=batch_created_at + timedelta(microseconds=item.batch_position),
            )
            for item in jobs
        )
        try:
            self.session.add_all(records)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return records

    def claim_next(self, worker_id: str, now: datetime) -> AnalysisJob | None:
        self.session.rollback()
        connection = self.session.connection()
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        job = self.session.scalars(
            select(AnalysisJob)
            .where(AnalysisJob.state == JobState.QUEUED)
            .order_by(AnalysisJob.created_at, AnalysisJob.id)
            .limit(1),
        ).first()
        if job is None:
            self.session.commit()
            return None
        job.state = JobState.VALIDATING
        job.stage = ProcessingStage.VALIDATING
        job.worker_id = worker_id
        job.started_at = now
        if job.attempt == 0:
            job.attempt = 1
        self.session.commit()
        return job

    def tombstone_batch(
        self,
        owner_id: str,
        batch_id: str,
        *,
        allow_admin: bool,
        now: datetime,
    ) -> tuple[AnalysisJob, ...]:
        self.session.rollback()
        try:
            connection = self.session.connection()
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            statement = select(AnalysisJob).where(AnalysisJob.batch_id == batch_id)
            if not allow_admin:
                statement = statement.where(AnalysisJob.owner_id == owner_id)
            jobs = tuple(
                self.session.scalars(
                    statement.order_by(AnalysisJob.batch_position),
                ),
            )
            for job in jobs:
                if job.state == JobState.DELETED:
                    continue
                job.state = JobState.DELETED
                job.stage = None
                job.deleted_at = now
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return jobs

    def set_stage(self, job_id: str, stage: ProcessingStage) -> AnalysisJob:
        return self._conditional_transition(
            job_id,
            state=JobState(stage.value),
            stage=stage,
        )

    def complete(self, job_id: str, result: dict[str, object], now: datetime) -> AnalysisJob:
        return self._conditional_transition(
            job_id,
            state=JobState.COMPLETED,
            stage=None,
            result=result,
            error_code=None,
            completed_at=now,
            worker_id=None,
        )

    def fail_recoverable(self, job_id: str, code: str, now: datetime) -> AnalysisJob:
        return self._conditional_transition(
            job_id,
            state=JobState.FAILED_RECOVERABLE,
            stage=None,
            error_code=code,
            completed_at=now,
            worker_id=None,
        )

    def reject_quality(
        self,
        job_id: str,
        result: dict[str, object],
        now: datetime,
    ) -> AnalysisJob:
        return self._conditional_transition(
            job_id,
            state=JobState.REJECTED_QUALITY,
            stage=None,
            result=result,
            error_code="insufficient_quality",
            completed_at=now,
            worker_id=None,
        )

    def recover_interrupted(self, now: datetime, worker_id: str | None = None) -> int:
        statement = update(AnalysisJob).where(AnalysisJob.state.in_(PROCESSING_STATES))
        if worker_id is not None:
            statement = statement.where(AnalysisJob.worker_id == worker_id)
        result = self.session.execute(
            statement.values(
                state=JobState.FAILED_RECOVERABLE,
                stage=None,
                error_code="worker_interrupted",
                completed_at=now,
                worker_id=None,
            ).execution_options(synchronize_session=False),
        )
        self.session.commit()
        self.session.expire_all()
        return int(getattr(result, "rowcount", 0))

    def queue_position(self, job: AnalysisJob) -> int | None:
        if job.state != JobState.QUEUED:
            return None
        position = self.session.scalar(
            select(func.count())
            .select_from(AnalysisJob)
            .where(
                AnalysisJob.state == JobState.QUEUED,
                or_(
                    AnalysisJob.created_at < job.created_at,
                    and_(
                        AnalysisJob.created_at == job.created_at,
                        AnalysisJob.id <= job.id,
                    ),
                ),
            ),
        )
        return int(position or 0)

    def require_processable(self, job_id: str) -> AnalysisJob:
        job = self._required(job_id)
        self.session.refresh(job)
        if job.state not in PROCESSING_STATES:
            raise JobNoLongerProcessable(f"Analysis job is no longer processable: {job_id}")
        return job

    def lock_processable(self, job_id: str) -> None:
        """Acquire the database write lock only while the job is still processable."""
        result = self.session.execute(
            update(AnalysisJob)
            .where(
                AnalysisJob.id == job_id,
                AnalysisJob.state.in_(PROCESSING_STATES),
            )
            .values(state=AnalysisJob.state)
            .execution_options(synchronize_session=False),
        )
        if getattr(result, "rowcount", 0) != 1:
            self.session.rollback()
            raise JobNoLongerProcessable(
                f"Analysis job is no longer processable: {job_id}",
            )

    def _conditional_transition(self, job_id: str, **values: object) -> AnalysisJob:
        result = self.session.execute(
            update(AnalysisJob)
            .where(
                AnalysisJob.id == job_id,
                AnalysisJob.state.in_(PROCESSING_STATES),
            )
            .values(**values)
            .execution_options(synchronize_session=False),
        )
        if getattr(result, "rowcount", 0) != 1:
            self.session.rollback()
            raise JobNoLongerProcessable(
                f"Analysis job is no longer processable: {job_id}",
            )
        self.session.commit()
        job = self._required(job_id)
        self.session.refresh(job)
        return job

    def _required(self, job_id: str) -> AnalysisJob:
        job = self.session.get(AnalysisJob, job_id)
        if job is None:
            raise LookupError(f"Analysis job does not exist: {job_id}")
        return job

    @staticmethod
    def _record(
        owner_id: str,
        item: NewAnalysisJob,
        expires_at: datetime,
        retention_class: RetentionClass,
        *,
        created_at: datetime,
    ) -> AnalysisJob:
        job = AnalysisJob(
            id=item.job_id,
            owner_id=owner_id,
            batch_id=item.batch_id,
            batch_position=item.batch_position,
            model_version=item.model_version,
            created_at=created_at,
            expires_at=expires_at,
        )
        job.assets.append(
            AnalysisAsset(
                frame_id=item.frame.frame_id,
                position=item.frame.position,
                kind=AssetKind.ORIGINAL,
                local_path=str(item.frame.path),
                checksum=item.frame.checksum,
                retention_class=retention_class,
                expires_at=expires_at,
            ),
        )
        return job
