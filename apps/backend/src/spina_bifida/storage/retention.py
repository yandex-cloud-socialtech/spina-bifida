from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from spina_bifida.analyses.models import (
    AnalysisAsset,
    AnalysisJob,
    JobState,
    ProcessingStage,
)
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db
from spina_bifida.storage.files import FileStore


@dataclass(frozen=True, slots=True)
class RetentionReport:
    assets_deleted: int = 0
    jobs_expired: int = 0
    logs_deleted: int = 0
    bytes_freed: int = 0


class RetentionService:
    def __init__(self, settings: AppSettings, file_store: FileStore) -> None:
        self.settings = settings
        self.file_store = file_store

    def run_once(self, now: datetime) -> RetentionReport:
        processing_states = [JobState(stage.value) for stage in ProcessingStage]
        protected_states = [JobState.QUEUED, *processing_states]
        expired_assets = list(
            db.session.scalars(
                select(AnalysisAsset)
                .join(AnalysisJob)
                .where(
                    AnalysisAsset.expires_at <= now,
                    AnalysisJob.state.not_in(protected_states),
                ),
            ),
        )
        assets_deleted = bytes_freed = 0
        affected_jobs: set[str] = set()
        for asset in expired_assets:
            path = Path(asset.local_path)
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self.file_store.delete_asset(path)
            affected_jobs.add(asset.job_id)
            db.session.delete(asset)
            assets_deleted += 1
            bytes_freed += size
        db.session.flush()

        jobs_expired = 0
        for job_id in affected_jobs:
            job = db.session.get(AnalysisJob, job_id)
            remaining = db.session.scalar(
                select(AnalysisAsset.id).where(AnalysisAsset.job_id == job_id).limit(1),
            )
            if job is not None and remaining is None and job.state != JobState.DELETED:
                job.state = JobState.EXPIRED
                job.stage = None
                jobs_expired += 1
        db.session.commit()

        feedback_bytes = self._clean_feedback_snapshots()
        logs_deleted, log_bytes = self._clean_logs(now)
        return RetentionReport(
            assets_deleted=assets_deleted,
            jobs_expired=jobs_expired,
            logs_deleted=logs_deleted,
            bytes_freed=bytes_freed + feedback_bytes + log_bytes,
        )

    def _clean_feedback_snapshots(self) -> int:
        root = self.settings.storage.feedback_dir.resolve()
        freed = 0
        for directory in root.iterdir():
            if not directory.is_dir():
                continue
            job = db.session.get(AnalysisJob, directory.name)
            if job is not None and job.state not in {JobState.EXPIRED, JobState.DELETED}:
                continue
            if self._has_pending_upload(directory):
                continue
            resolved = directory.resolve()
            if resolved == root or not resolved.is_relative_to(root):
                continue
            freed += sum(path.stat().st_size for path in resolved.rglob("*") if path.is_file())
            shutil.rmtree(resolved)
        return freed

    def _clean_logs(self, now: datetime) -> tuple[int, int]:
        archives = sorted(
            self.settings.logs.directory.glob("*.gz"),
            key=lambda path: path.stat().st_mtime,
        )
        threshold = now - timedelta(days=self.settings.logs.retention_days)
        deleted = freed = 0
        survivors: list[Path] = []
        for archive in archives:
            if self._is_pending_upload(archive):
                survivors.append(archive)
                continue
            modified = datetime.fromtimestamp(archive.stat().st_mtime, tz=now.tzinfo)
            if modified < threshold:
                size = self._delete_log_archive(archive)
                deleted += 1
                freed += size
            else:
                survivors.append(archive)

        maximum = int(self.settings.logs.max_total_gb * 1024**3)
        total = sum(path.stat().st_size for path in survivors)
        for archive in survivors:
            if total <= maximum:
                break
            if self._is_pending_upload(archive):
                continue
            size = self._delete_log_archive(archive)
            total -= size
            deleted += 1
            freed += size
        return deleted, freed

    @staticmethod
    def _is_pending_upload(path: Path) -> bool:
        return Path(f"{path}.upload.json").exists() and not Path(f"{path}.uploaded").exists()

    @classmethod
    def _has_pending_upload(cls, directory: Path) -> bool:
        return any(
            cls._is_pending_upload(Path(str(sidecar).removesuffix(".upload.json")))
            for sidecar in directory.rglob("*.upload.json")
        )

    @staticmethod
    def _delete_log_archive(path: Path) -> int:
        size = path.stat().st_size
        path.unlink()
        for suffix in (".uploaded", ".upload.json"):
            sidecar = Path(f"{path}{suffix}")
            if sidecar.exists():
                sidecar.unlink()
        return size
