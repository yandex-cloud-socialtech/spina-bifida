from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Protocol
from unittest.mock import patch

from flask import Flask
from werkzeug.datastructures import FileStorage

from spina_bifida.analyses.service import AnalysisService
from spina_bifida.auth.service import AuthService
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db
from spina_bifida.observability.uploader import OutboxUploader, RetryManifest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "images"


class StorageClient(Protocol):
    def put_file(self, local_path: Path, object_key: str, checksum: str) -> None: ...


class FakeStorage:
    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.calls: list[tuple[Path, str, str]] = []

    def put_file(self, local_path: Path, object_key: str, checksum: str) -> None:
        self.calls.append((local_path, object_key, checksum))
        if self.fails:
            raise OSError("simulated object storage outage")


def make_uploader(
    settings: AppSettings,
    client: StorageClient,
    now: datetime,
) -> OutboxUploader:
    return OutboxUploader(
        client=client,
        logs_dir=settings.logs.directory,
        feedback_dir=settings.storage.feedback_dir,
        logs_prefix="logs/",
        feedback_prefix="feedback/",
        upload_logs=True,
        upload_feedback=settings.object_storage.upload_feedback,
        retry=settings.object_storage.retry,
        now=lambda: now,
    )


def test_upload_failure_keeps_local_file_and_records_retry(
    app: Flask,
    tmp_path: Path,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    adjusted = settings.model_copy(
        update={
            "logs": settings.logs.model_copy(update={"directory": tmp_path / "logs"}),
            "storage": settings.storage.model_copy(update={"feedback_dir": tmp_path / "feedback"}),
        },
    )
    adjusted.ensure_directories()
    rotated = adjusted.logs.directory / "api.jsonl.1.gz"
    rotated.write_bytes(b"closed log")
    uploader = make_uploader(adjusted, FakeStorage(fails=True), datetime.now(UTC))

    report = uploader.run_once()

    assert report.failed == 1
    assert rotated.exists()
    manifest = RetryManifest.load(rotated)
    assert manifest.attempts == 1
    assert manifest.next_attempt_at is not None


def test_storage_disabled_never_constructs_s3_client(app: Flask) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)

    with patch("spina_bifida.observability.uploader.S3ObjectStorage") as constructor:
        uploader = OutboxUploader.from_settings(settings)

    assert uploader.client is None
    constructor.assert_not_called()


def test_success_uses_checksum_key_and_keeps_file_until_retention(
    app: Flask,
    tmp_path: Path,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    logs = tmp_path / "logs"
    feedback = tmp_path / "feedback"
    logs.mkdir(exist_ok=True)
    feedback.mkdir(exist_ok=True)
    rotated = logs / "worker.jsonl.1.gz"
    rotated.write_bytes(b"deterministic content")
    client = FakeStorage()
    uploader = OutboxUploader(
        client=client,
        logs_dir=logs,
        feedback_dir=feedback,
        logs_prefix="logs/",
        feedback_prefix="feedback/",
        upload_logs=True,
        upload_feedback=False,
        retry=settings.object_storage.retry,
        now=lambda: datetime.now(UTC),
    )

    first = uploader.run_once()
    second = uploader.run_once()

    checksum = hashlib.sha256(rotated.read_bytes()).hexdigest()
    assert first.uploaded == 1
    assert second.uploaded == 0 and second.skipped == 1
    assert client.calls[0][1] == f"logs/{checksum}.gz"
    assert rotated.exists()
    marker = json.loads(
        rotated.with_suffix(rotated.suffix + ".uploaded").read_text(encoding="utf-8"),
    )
    assert marker["checksum"] == checksum


def test_success_records_last_confirmed_upload(app: Flask, tmp_path: Path) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    logs = tmp_path / "logs"
    feedback = tmp_path / "feedback"
    status = tmp_path / "db" / "object-storage-last-success"
    logs.mkdir(exist_ok=True)
    feedback.mkdir(exist_ok=True)
    rotated = logs / "api.jsonl.1.gz"
    rotated.write_bytes(b"closed")
    now = datetime.now(UTC)
    uploader = OutboxUploader(
        client=FakeStorage(),
        logs_dir=logs,
        feedback_dir=feedback,
        logs_prefix="logs/",
        feedback_prefix="feedback/",
        upload_logs=True,
        upload_feedback=False,
        retry=settings.object_storage.retry,
        status_path=status,
        now=lambda: now,
    )

    uploader.run_once()

    assert status.read_text(encoding="utf-8").strip() == now.isoformat()


def test_retry_backoff_is_exponential_and_capped(app: Flask, tmp_path: Path) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    logs = tmp_path / "logs"
    feedback = tmp_path / "feedback"
    logs.mkdir(exist_ok=True)
    feedback.mkdir(exist_ok=True)
    rotated = logs / "api.jsonl.1.gz"
    rotated.write_bytes(b"closed")
    now = datetime.now(UTC)
    client = FakeStorage(fails=True)
    uploader = OutboxUploader(
        client=client,
        logs_dir=logs,
        feedback_dir=feedback,
        logs_prefix="logs/",
        feedback_prefix="feedback/",
        upload_logs=True,
        upload_feedback=False,
        retry=settings.object_storage.retry.model_copy(
            update={"base_seconds": 5, "max_seconds": 6, "max_attempts": 3},
        ),
        now=lambda: now,
    )

    uploader.run_once()
    first = RetryManifest.load(rotated)
    uploader.now = lambda: now + timedelta(seconds=5)
    uploader.run_once()
    second = RetryManifest.load(rotated)

    assert first.next_attempt_at == now + timedelta(seconds=5)
    assert second.next_attempt_at == now + timedelta(seconds=11)


def test_feedback_disabled_uploads_only_closed_logs(app: Flask, tmp_path: Path) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    logs = tmp_path / "logs"
    feedback = tmp_path / "feedback"
    logs.mkdir(exist_ok=True)
    feedback.mkdir(exist_ok=True)
    (logs / "api.jsonl").write_text("active", encoding="utf-8")
    (logs / "api.jsonl.1.gz").write_bytes(b"closed")
    feedback_file = feedback / "retained.png"
    feedback_file.write_bytes(b"feedback")
    RetryManifest.pending(feedback_file).save(feedback_file)
    client = FakeStorage()
    uploader = OutboxUploader(
        client=client,
        logs_dir=logs,
        feedback_dir=feedback,
        logs_prefix="logs/",
        feedback_prefix="feedback/",
        upload_logs=True,
        upload_feedback=False,
        retry=settings.object_storage.retry,
        now=lambda: datetime.now(UTC),
    )

    uploader.run_once()

    assert [call[0].name for call in client.calls] == ["api.jsonl.1.gz"]


def test_feedback_submission_stages_internal_assets_for_explicit_upload_scope(
    app: Flask,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    db.create_all()
    owner = AuthService().create_user("outbox@example.test", "StrongPass123", "user", "ru")
    adjusted = settings.model_copy(
        update={
            "object_storage": settings.object_storage.model_copy(
                update={"enabled": True, "upload_feedback": True},
            ),
        },
    )
    upload = FileStorage(
        stream=BytesIO((FIXTURES / "valid.png").read_bytes()),
        filename="patient-name.png",
        content_type="image/png",
    )
    service = AnalysisService(adjusted)
    job = service.create_job(owner, [upload])

    service.submit_feedback(owner, job.id, "positive", None)

    staged = [
        path
        for path in (settings.storage.feedback_dir / job.id).iterdir()
        if path.suffix == ".png"
    ]
    assert len(staged) == 1
    assert "patient-name" not in staged[0].name
    assert RetryManifest.path_for(staged[0]).is_file()
