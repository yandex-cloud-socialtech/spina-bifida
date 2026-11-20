from __future__ import annotations

import os
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

from flask import Flask
from werkzeug.datastructures import FileStorage

from spina_bifida.analyses.models import AnalysisJob, JobState, RetentionClass
from spina_bifida.analyses.service import AnalysisService, LocalStorageFull
from spina_bifida.auth.service import AuthService
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db
from spina_bifida.storage.files import FileStore
from spina_bifida.storage.retention import RetentionService

FIXTURES = Path(__file__).parents[1] / "fixtures" / "images"
DiskUsage = namedtuple("DiskUsage", "total used free")


def upload() -> FileStorage:
    return FileStorage(
        stream=BytesIO((FIXTURES / "valid.png").read_bytes()),
        filename="valid.png",
        content_type="image/png",
    )


def setup_job(app: Flask) -> tuple[AppSettings, AnalysisJob]:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    db.create_all()
    owner = AuthService().create_user("retention@example.test", "StrongPass123", "user", "ru")
    job = AnalysisService(settings).create_job(owner, [upload()])
    job.state = JobState.COMPLETED
    db.session.commit()
    return settings, job


def test_retention_deletes_only_db_authorized_expired_assets(app: Flask) -> None:
    settings, job = setup_job(app)
    now = datetime.now(UTC)
    expired = job.assets[0]
    expired.expires_at = now - timedelta(seconds=1)
    expired_path = Path(expired.local_path)
    retained_path = expired_path.with_name("feedback-retained.png")
    retained_path.write_bytes(expired_path.read_bytes())
    from spina_bifida.analyses.models import AnalysisAsset, AssetKind

    job.assets.append(
        AnalysisAsset(
            frame_id="retained",
            position=1,
            kind=AssetKind.ORIGINAL,
            local_path=str(retained_path),
            checksum="a" * 64,
            retention_class=RetentionClass.FEEDBACK,
            expires_at=now + timedelta(days=1),
        ),
    )
    db.session.commit()

    report = RetentionService(settings, FileStore(settings.storage.jobs_dir)).run_once(now)

    assert report.assets_deleted == 1
    assert not expired_path.exists()
    assert retained_path.exists()


def test_expired_job_removes_its_feedback_outbox_snapshot(app: Flask) -> None:
    settings, job = setup_job(app)
    job.assets[0].expires_at = datetime.now(UTC) - timedelta(seconds=1)
    snapshot = settings.storage.feedback_dir / job.id
    snapshot.mkdir()
    (snapshot / "asset.png").write_bytes(b"retained feedback")
    db.session.commit()

    report = RetentionService(settings, FileStore(settings.storage.jobs_dir)).run_once(
        datetime.now(UTC),
    )

    assert report.jobs_expired == 1
    assert not snapshot.exists()


def test_log_retention_keeps_active_and_recent_uploaded_archive(
    app: Flask,
    tmp_path: Path,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    db.create_all()
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    active = logs / "api.jsonl"
    active.write_text("active", encoding="utf-8")
    recent = logs / "api.jsonl.recent.gz"
    recent.write_bytes(b"recent")
    recent.with_suffix(recent.suffix + ".uploaded").write_text("{}", encoding="utf-8")
    old = logs / "api.jsonl.old.gz"
    old.write_bytes(b"old")
    old_time = (datetime.now(UTC) - timedelta(days=60)).timestamp()
    os.utime(old, (old_time, old_time))
    adjusted = settings.model_copy(
        update={"logs": settings.logs.model_copy(update={"directory": logs, "retention_days": 30})},
    )

    report = RetentionService(adjusted, FileStore(settings.storage.jobs_dir)).run_once(
        datetime.now(UTC),
    )

    assert report.logs_deleted == 1
    assert active.exists()
    assert recent.exists()
    assert recent.with_suffix(recent.suffix + ".uploaded").exists()
    assert not old.exists()


def test_log_size_cap_removes_oldest_closed_archive_first(
    app: Flask,
    tmp_path: Path,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    db.create_all()
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    oldest = logs / "api.jsonl.1.gz"
    newest = logs / "api.jsonl.2.gz"
    oldest.write_bytes(b"x" * 100)
    newest.write_bytes(b"y" * 100)
    timestamp = datetime.now(UTC).timestamp()
    os.utime(oldest, (timestamp - 2, timestamp - 2))
    os.utime(newest, (timestamp - 1, timestamp - 1))
    adjusted = settings.model_copy(
        update={
            "logs": settings.logs.model_copy(
                update={"directory": logs, "retention_days": 3650, "max_total_gb": 150 / 1024**3},
            ),
        },
    )

    RetentionService(adjusted, FileStore(settings.storage.jobs_dir)).run_once(datetime.now(UTC))

    assert not oldest.exists()
    assert newest.exists()


def test_log_retention_preserves_pending_outbox_archive(
    app: Flask,
    tmp_path: Path,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    db.create_all()
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    pending = logs / "worker.jsonl.old.gz"
    pending.write_bytes(b"pending upload")
    pending.with_suffix(pending.suffix + ".upload.json").write_text(
        '{"attempts": 1, "next_attempt_at": null}',
        encoding="utf-8",
    )
    old_time = (datetime.now(UTC) - timedelta(days=60)).timestamp()
    os.utime(pending, (old_time, old_time))
    adjusted = settings.model_copy(
        update={"logs": settings.logs.model_copy(update={"directory": logs})},
    )

    report = RetentionService(adjusted, FileStore(settings.storage.jobs_dir)).run_once(
        datetime.now(UTC),
    )

    assert report.logs_deleted == 0
    assert pending.exists()
    assert pending.with_suffix(pending.suffix + ".upload.json").exists()


def test_expired_job_preserves_pending_feedback_outbox(app: Flask) -> None:
    settings, job = setup_job(app)
    job.assets[0].expires_at = datetime.now(UTC) - timedelta(seconds=1)
    snapshot = settings.storage.feedback_dir / job.id
    snapshot.mkdir()
    pending = snapshot / "asset.png"
    pending.write_bytes(b"pending feedback")
    pending.with_suffix(pending.suffix + ".upload.json").write_text(
        '{"attempts": 1, "next_attempt_at": null}',
        encoding="utf-8",
    )
    db.session.commit()

    RetentionService(settings, FileStore(settings.storage.jobs_dir)).run_once(datetime.now(UTC))

    assert pending.exists()


def test_local_disk_reserve_rejects_new_job_before_writing(
    app: Flask,
    monkeypatch,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    db.create_all()
    owner = AuthService().create_user("capacity@example.test", "StrongPass123", "user", "ru")
    monkeypatch.setattr(
        "spina_bifida.analyses.service.shutil.disk_usage",
        lambda _path: DiskUsage(100, 99, 1),
    )

    try:
        AnalysisService(settings).create_job(owner, [upload()])
    except LocalStorageFull as error:
        assert error.code == "local_storage_full"
    else:
        raise AssertionError("capacity guard did not reject the upload")
    assert not list(settings.storage.jobs_dir.iterdir())
