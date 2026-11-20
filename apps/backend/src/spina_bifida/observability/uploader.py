from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from spina_bifida.config import AppSettings, ObjectStorageRetrySettings
from spina_bifida.storage.object_storage import S3ObjectStorage


class StorageClient(Protocol):
    def put_file(self, local_path: Path, object_key: str, checksum: str) -> None: ...


@dataclass(frozen=True, slots=True)
class UploadReport:
    uploaded: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass(frozen=True, slots=True)
class RetryManifest:
    attempts: int
    next_attempt_at: datetime | None

    @classmethod
    def pending(cls, _path: Path) -> RetryManifest:
        return cls(attempts=0, next_attempt_at=None)

    @classmethod
    def load(cls, path: Path) -> RetryManifest:
        sidecar = cls.path_for(path)
        if not sidecar.exists():
            return cls.pending(path)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        next_value = payload.get("next_attempt_at")
        return cls(
            attempts=int(payload.get("attempts", 0)),
            next_attempt_at=datetime.fromisoformat(next_value) if next_value else None,
        )

    def save(self, path: Path) -> None:
        sidecar = self.path_for(path)
        temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "attempts": self.attempts,
                    "next_attempt_at": (
                        self.next_attempt_at.isoformat() if self.next_attempt_at else None
                    ),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(sidecar)

    @staticmethod
    def path_for(path: Path) -> Path:
        return Path(f"{path}.upload.json")


class OutboxUploader:
    def __init__(
        self,
        *,
        client: StorageClient | None,
        logs_dir: Path,
        feedback_dir: Path,
        logs_prefix: str,
        feedback_prefix: str,
        upload_logs: bool,
        upload_feedback: bool,
        retry: ObjectStorageRetrySettings,
        status_path: Path | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.client = client
        self.logs_dir = logs_dir
        self.feedback_dir = feedback_dir
        self.logs_prefix = logs_prefix.rstrip("/") + "/"
        self.feedback_prefix = feedback_prefix.rstrip("/") + "/"
        self.upload_logs = upload_logs
        self.upload_feedback = upload_feedback
        self.retry = retry
        self.status_path = status_path
        self.now = now or (lambda: datetime.now(UTC))

    @classmethod
    def from_settings(cls, settings: AppSettings) -> OutboxUploader:
        client = (
            S3ObjectStorage(settings.object_storage)
            if settings.object_storage.enabled
            else None
        )
        return cls(
            client=client,
            logs_dir=settings.logs.directory,
            feedback_dir=settings.storage.feedback_dir,
            logs_prefix=settings.object_storage.logs_prefix,
            feedback_prefix=settings.object_storage.feedback_prefix,
            upload_logs=settings.object_storage.upload_logs,
            upload_feedback=settings.object_storage.upload_feedback,
            retry=settings.object_storage.retry,
            status_path=settings.storage.db_dir / "object-storage-last-success",
        )

    def run_once(self) -> UploadReport:
        candidates: list[tuple[Path, str]] = []
        if self.upload_logs:
            candidates.extend((path, self.logs_prefix) for path in self.logs_dir.glob("*.gz"))
        if self.upload_feedback:
            candidates.extend(
                (path, self.feedback_prefix)
                for path in self.feedback_dir.rglob("*")
                if path.is_file()
                and not path.name.endswith((".upload.json", ".uploaded"))
                and RetryManifest.path_for(path).exists()
            )

        uploaded = failed = skipped = 0
        for path, prefix in sorted(candidates):
            marker = Path(f"{path}.uploaded")
            manifest = RetryManifest.load(path)
            current = self.now()
            if (
                marker.exists()
                or self.client is None
                or manifest.attempts >= self.retry.max_attempts
                or (manifest.next_attempt_at is not None and manifest.next_attempt_at > current)
            ):
                skipped += 1
                continue
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            suffix = ".gz" if path.suffix == ".gz" else path.suffix
            object_key = f"{prefix}{checksum}{suffix}"
            try:
                self.client.put_file(path, object_key, checksum)
            except Exception:
                attempts = manifest.attempts + 1
                delay = min(
                    self.retry.max_seconds,
                    self.retry.base_seconds * (2**manifest.attempts),
                )
                RetryManifest(attempts, current + timedelta(seconds=delay)).save(path)
                failed += 1
                continue
            marker_temporary = marker.with_suffix(marker.suffix + ".tmp")
            marker_temporary.write_text(
                json.dumps(
                    {
                        "checksum": checksum,
                        "object_key": object_key,
                        "uploaded_at": current.isoformat(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            marker_temporary.replace(marker)
            with suppress(FileNotFoundError):
                RetryManifest.path_for(path).unlink()
            self._record_success(current)
            uploaded += 1
        return UploadReport(uploaded=uploaded, failed=failed, skipped=skipped)

    def _record_success(self, uploaded_at: datetime) -> None:
        if self.status_path is None:
            return
        self.status_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(uploaded_at.isoformat() + "\n", encoding="utf-8")
        temporary.replace(self.status_path)
