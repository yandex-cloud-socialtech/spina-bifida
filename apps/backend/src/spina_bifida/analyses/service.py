from __future__ import annotations

import shutil
import stat
from collections.abc import Sequence
from contextlib import ExitStack, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import IO, BinaryIO, Literal, cast
from uuid import uuid4

from sqlalchemy import select
from werkzeug.datastructures import FileStorage

from spina_bifida.analyses.models import (
    AnalysisAsset,
    AnalysisJob,
    AssetKind,
    Feedback,
    FeedbackValue,
    JobState,
    RetentionClass,
)
from spina_bifida.analyses.repository import JobRepository
from spina_bifida.analyses.schemas import NewAnalysisJob, NewFrame
from spina_bifida.auth.models import User, UserRole
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db
from spina_bifida.observability.audit import audit
from spina_bifida.observability.uploader import RetryManifest
from spina_bifida.storage.files import (
    AssetNotFound,
    FileStore,
    UploadRejected,
    UploadValidator,
)


class UploadBatchRejected(UploadRejected):
    pass


class AnalysisNotFound(LookupError):
    pass


class RetryNotAllowed(ValueError):
    pass


class InvalidFeedback(ValueError):
    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field


class LocalStorageFull(OSError):
    code = "local_storage_full"


class AnalysisService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.validator = UploadValidator(
            max_request_bytes=settings.storage.images.max_request_bytes,
            max_image_pixels=settings.storage.images.max_image_pixels,
        )
        self.file_store = FileStore(settings.storage.jobs_dir)
        self.repository = JobRepository()

    def create_job(
        self,
        owner: User,
        uploads: Sequence[FileStorage],
    ) -> AnalysisJob:
        return self.create_batch(
            owner,
            uploads,
            "v1",
        )[0]

    def create_batch(
        self,
        owner: User,
        uploads: Sequence[FileStorage],
        model_version: str,
    ) -> tuple[AnalysisJob, ...]:
        if model_version not in {"v1", "v2"}:
            raise UploadBatchRejected(
                "model_version_invalid",
                params={"allowed": ["v1", "v2"]},
            )
        selected_model = cast(Literal["v1", "v2"], model_version)
        if not 1 <= len(uploads) <= 10:
            raise UploadBatchRejected(
                "upload_count_invalid",
                params={"minimum": 1, "maximum": 10},
            )
        self._require_capacity()

        with ExitStack() as spool_stack:
            validated: list[IO[bytes]] = []
            checksums: set[str] = set()
            for index, upload in enumerate(uploads):
                try:
                    current = self.validator.validate(upload)
                except UploadRejected as error:
                    raise UploadBatchRejected(
                        error.code,
                        error.status,
                        {**error.params, "file_index": index},
                    ) from error
                try:
                    if current.checksum in checksums:
                        raise UploadBatchRejected(
                            "duplicate_image",
                            params={"file_index": index},
                        )
                    spool = spool_stack.enter_context(
                        SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b"),
                    )
                    current.image.save(spool, format="PNG")
                    spool.seek(0)
                    validated.append(spool)
                    checksums.add(current.checksum)
                finally:
                    current.image.close()

            batch_id = str(uuid4())
            created_at = datetime.now(UTC)
            images = self.settings.storage.images
            if images.mode == "all":
                expires_at = created_at + timedelta(days=images.retained_ttl_days)
                retention_class = RetentionClass.RETAINED
            else:
                expires_at = created_at + timedelta(hours=images.temporary_ttl_hours)
                retention_class = RetentionClass.TEMPORARY
            jobs: list[NewAnalysisJob] = []
            job_ids = [str(uuid4()) for _image in validated]
            try:
                for position, (job_id, normalized) in enumerate(
                    zip(job_ids, validated, strict=True),
                ):
                    frame_id = str(uuid4())
                    stored = self.file_store.save_normalized_stream(
                        job_id,
                        frame_id,
                        normalized,
                    )
                    jobs.append(
                        NewAnalysisJob(
                            job_id=job_id,
                            batch_id=batch_id,
                            batch_position=position,
                            model_version=selected_model,
                            frame=NewFrame(
                                frame_id=frame_id,
                                path=stored.path,
                                checksum=stored.checksum,
                                position=0,
                            ),
                        ),
                    )
                records = self.repository.create_batch(
                    owner.id,
                    jobs,
                    expires_at=expires_at,
                    retention_class=retention_class,
                )
            except Exception:
                for job_id in job_ids:
                    self.file_store.delete_job(job_id)
                raise
        for job in records:
            audit(
                "analysis_created",
                owner.id,
                job_id=job.id,
                batch_id=job.batch_id,
                batch_position=job.batch_position,
                model_version=job.model_version,
                frame_count=1,
            )
        return records

    def get_job(self, actor: User, job_id: str) -> AnalysisJob:
        statement = select(AnalysisJob).where(
            AnalysisJob.id == job_id,
            AnalysisJob.state != JobState.DELETED,
        )
        if actor.role != UserRole.ADMIN:
            statement = statement.where(AnalysisJob.owner_id == actor.id)
        job = db.session.scalar(statement)
        if job is None:
            raise AnalysisNotFound(job_id)
        return job

    def open_asset(self, actor: User, job_id: str, asset_id: str) -> BinaryIO:
        try:
            return self.file_store.open_asset(
                actor.id,
                job_id,
                asset_id,
                allow_admin=actor.role == UserRole.ADMIN,
            )
        except AssetNotFound as error:
            raise AnalysisNotFound(job_id) from error

    def delete_job(self, actor: User, job_id: str) -> None:
        job = self.get_job(actor, job_id)
        job.state = JobState.DELETED
        job.stage = None
        job.deleted_at = datetime.now(UTC)
        db.session.commit()

        self.file_store.delete_job(job.id)
        self._delete_feedback_snapshot(job.id)
        job.assets.clear()
        db.session.commit()
        audit("analysis_deleted", actor.id, job_id=job.id)

    def delete_batch(self, actor: User, batch_id: str) -> None:
        jobs = self.repository.tombstone_batch(
            actor.id,
            batch_id,
            allow_admin=actor.role == UserRole.ADMIN,
            now=datetime.now(UTC),
        )
        if not jobs:
            raise AnalysisNotFound(batch_id)

        cleanup_errors: list[OSError] = []
        for job in jobs:
            job_directory_deleted = False
            try:
                self.file_store.delete_job(job.id)
                job_directory_deleted = True
            except OSError as error:
                cleanup_errors.append(error)
            try:
                self._delete_feedback_snapshot(job.id)
            except OSError as error:
                cleanup_errors.append(error)
            if job_directory_deleted:
                job.assets.clear()
        try:
            self.repository.session.commit()
        except Exception:
            self.repository.session.rollback()
            raise
        if cleanup_errors:
            raise cleanup_errors[0]
        audit(
            "analysis_batch_deleted",
            actor.id,
            batch_id=batch_id,
            job_count=len(jobs),
        )

    def retry_job(self, actor: User, job_id: str) -> AnalysisJob:
        job = self.get_job(actor, job_id)
        if job.state != JobState.FAILED_RECOVERABLE:
            raise RetryNotAllowed(job_id)
        derived = list(
            db.session.scalars(
                select(AnalysisAsset).where(
                    AnalysisAsset.job_id == job.id,
                    AnalysisAsset.kind != AssetKind.ORIGINAL,
                ),
            ),
        )
        for asset in derived:
            self.file_store.delete_asset(Path(asset.local_path))
            db.session.delete(asset)
        job.state = JobState.QUEUED
        job.stage = None
        job.worker_id = None
        job.started_at = None
        job.completed_at = None
        job.error_code = None
        job.result = None
        job.attempt += 1
        db.session.commit()
        audit("analysis_retried", actor.id, job_id=job.id, attempt=job.attempt)
        return job

    def submit_feedback(
        self,
        actor: User,
        job_id: str,
        value: str,
        comment: object,
    ) -> Feedback:
        job = self.get_job(actor, job_id)
        try:
            feedback_value = FeedbackValue(value)
        except ValueError as error:
            raise InvalidFeedback("value") from error
        if comment is not None and (not isinstance(comment, str) or len(comment) > 2000):
            raise InvalidFeedback("comment")
        normalized_comment = comment.strip() or None if isinstance(comment, str) else None
        feedback = db.session.scalar(
            select(Feedback).where(
                Feedback.job_id == job.id,
                Feedback.author_id == actor.id,
            ),
        )
        if feedback is None:
            feedback = Feedback(job_id=job.id, author_id=actor.id)
            db.session.add(feedback)
        feedback.value = feedback_value
        feedback.comment = normalized_comment

        images = self.settings.storage.images
        feedback_settings = self.settings.storage.feedback
        if images.mode == "feedback_only" and feedback_settings.retain_assets:
            retained_until = datetime.now(UTC) + timedelta(
                days=feedback_settings.retained_ttl_days,
            )
            for asset in job.assets:
                asset.retention_class = RetentionClass.FEEDBACK
                asset.expires_at = retained_until
            job.expires_at = retained_until
        db.session.commit()
        if (
            self.settings.object_storage.enabled
            and self.settings.object_storage.upload_feedback
        ):
            try:
                self._stage_feedback_assets(job)
            except OSError:
                audit("feedback_staging_failed", actor.id, job_id=job.id)
        audit("feedback_submitted", actor.id, job_id=job.id, value=feedback_value.value)
        return feedback

    def polling_payload(self, actor: User, job_id: str) -> dict[str, object]:
        job = self.get_job(actor, job_id)
        originals = [asset for asset in job.assets if asset.kind == AssetKind.ORIGINAL]
        return {
            "id": job.id,
            "batch_id": job.batch_id,
            "batch_position": job.batch_position,
            "model_version": job.model_version,
            "state": job.state.value,
            "stage": job.stage.value if job.stage else None,
            "progress": _progress(job.state),
            "queue_position": self.repository.queue_position(job),
            "attempt": job.attempt,
            "frames": [
                {
                    "id": asset.frame_id,
                    "original_asset_url": (
                        f"/api/v1/analyses/{job.id}/assets/{asset.id}"
                    ),
                }
                for asset in originals
            ],
            "result": job.result,
            "error": {"code": job.error_code} if job.error_code else None,
            "expires_at": job.expires_at.isoformat(),
        }

    def _require_capacity(self) -> None:
        images = self.settings.storage.images
        free = shutil.disk_usage(self.settings.storage.jobs_dir).free
        minimum = int(images.min_free_gb * 1024**3)
        used = sum(
            self._visible_file_size(path)
            for root in (
                self.settings.storage.jobs_dir,
                self.settings.storage.feedback_dir,
            )
            for path in root.rglob("*")
        )
        maximum = int(images.max_total_gb * 1024**3)
        if free < minimum or used >= maximum:
            raise LocalStorageFull("local image storage reserve is exhausted")

    @staticmethod
    def _visible_file_size(path: Path) -> int:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return 0
        return metadata.st_size if stat.S_ISREG(metadata.st_mode) else 0

    def _stage_feedback_assets(self, job: AnalysisJob) -> None:
        directory = self._feedback_directory(job.id)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        for asset in job.assets:
            source = Path(asset.local_path)
            destination = directory / f"{asset.id}{source.suffix.casefold()}"
            marker = Path(f"{destination}.uploaded")
            if not destination.exists():
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copyfile(source, temporary)
                temporary.replace(destination)
            if not marker.exists() and not RetryManifest.path_for(destination).exists():
                RetryManifest.pending(destination).save(destination)

    def _delete_feedback_snapshot(self, job_id: str) -> None:
        directory = self._feedback_directory(job_id)
        with suppress(FileNotFoundError):
            shutil.rmtree(directory)

    def _feedback_directory(self, job_id: str) -> Path:
        root = self.settings.storage.feedback_dir.resolve()
        directory = (root / job_id).resolve()
        if directory == root or not directory.is_relative_to(root):
            raise OSError("invalid feedback storage path")
        return directory


_PROGRESS = {
    JobState.QUEUED: 0,
    JobState.VALIDATING: 10,
    JobState.QUALITY_CHECK: 25,
    JobState.PLANE_DETECTION: 40,
    JobState.ROI_DETECTION: 55,
    JobState.SCORING: 75,
    JobState.EXPLANATION: 90,
    JobState.COMPLETED: 100,
    JobState.REJECTED_QUALITY: 100,
    JobState.FAILED_RECOVERABLE: 100,
    JobState.EXPIRED: 100,
    JobState.DELETED: 100,
}


def _progress(state: JobState) -> int:
    return _PROGRESS[state]
