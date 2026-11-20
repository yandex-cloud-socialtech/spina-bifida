from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from threading import Event, Thread, current_thread

import pytest
from flask import Flask
from PIL import Image
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.datastructures import FileStorage

from spina_bifida.analyses.models import (
    AnalysisAsset,
    AnalysisJob,
    JobState,
    RetentionClass,
)
from spina_bifida.analyses.repository import JobNoLongerProcessable, JobRepository
from spina_bifida.analyses.service import AnalysisService, UploadBatchRejected
from spina_bifida.auth.models import User
from spina_bifida.auth.service import AuthService
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db
from spina_bifida.storage.files import ValidatedImage

FIXTURES = Path(__file__).parents[1] / "fixtures" / "images"


def upload(name: str, *, filename: str | None = None) -> FileStorage:
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    return FileStorage(
        stream=BytesIO((FIXTURES / name).read_bytes()),
        filename=filename or name,
        content_type=mime,
    )


def distinct_uploads(count: int) -> list[FileStorage]:
    uploads = []
    for position in range(count):
        stream = BytesIO()
        Image.new(
            "RGB",
            (8, 8),
            (position * 20, position * 11, position * 7),
        ).save(stream, format="PNG")
        stream.seek(0)
        uploads.append(
            FileStorage(
                stream=stream,
                filename=f"frame-{position}.png",
                content_type="image/png",
            ),
        )
    return uploads


@pytest.fixture
def owner(app: Flask) -> User:
    db.create_all()
    return AuthService().create_user(
        "doctor@example.test",
        "StrongPass123",
        "user",
        "ru",
    )


@pytest.fixture
def analysis_service(app: Flask) -> AnalysisService:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    return AnalysisService(settings)


@pytest.mark.parametrize("count", [0, 11])
def test_requires_one_through_ten_files(
    analysis_service: AnalysisService,
    owner: User,
    count: int,
) -> None:
    uploads = [upload("valid.jpg") for _ in range(count)]

    with pytest.raises(UploadBatchRejected) as caught:
        analysis_service.create_batch(owner, uploads, "v1")

    assert caught.value.code == "upload_count_invalid"
    assert caught.value.status == 422


def test_ten_files_create_ten_jobs_in_upload_order(
    analysis_service: AnalysisService,
    owner: User,
) -> None:
    uploads = distinct_uploads(10)

    jobs = analysis_service.create_batch(owner, uploads, "v2")

    assert len(jobs) == 10
    assert len({job.batch_id for job in jobs}) == 1
    assert [job.batch_position for job in jobs] == list(range(10))
    assert {job.model_version for job in jobs} == {"v2"}
    assert all(len(job.assets) == 1 for job in jobs)
    assert all(Path(job.assets[0].local_path).is_file() for job in jobs)


def test_batch_validation_closes_each_decoded_image_before_opening_the_next(
    analysis_service: AnalysisService,
    owner: User,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_validate = analysis_service.validator.validate
    active_images = 0
    peak_active_images = 0
    tracked_images: list[TrackedImage] = []

    class TrackedImage:
        def __init__(self, image: Image.Image) -> None:
            nonlocal active_images, peak_active_images
            self.image = image
            self.closed = False
            active_images += 1
            peak_active_images = max(peak_active_images, active_images)

        def save(self, *args: object, **kwargs: object) -> None:
            self.image.save(*args, **kwargs)

        def convert(self, *args: object, **kwargs: object) -> Image.Image:
            return self.image.convert(*args, **kwargs)

        def close(self) -> None:
            nonlocal active_images
            if not self.closed:
                self.closed = True
                active_images -= 1
                self.image.close()

    def validate_and_track(candidate: FileStorage) -> ValidatedImage:
        validated = original_validate(candidate)
        tracked = TrackedImage(validated.image)
        tracked_images.append(tracked)
        return ValidatedImage(
            image=tracked,  # type: ignore[arg-type]
            checksum=validated.checksum,
            mime_type=validated.mime_type,
        )

    monkeypatch.setattr(analysis_service.validator, "validate", validate_and_track)

    jobs = analysis_service.create_batch(owner, distinct_uploads(10), "v1")

    assert len(jobs) == 10
    assert peak_active_images == 1
    assert active_images == 0
    assert all(image.closed for image in tracked_images)


def test_capacity_scan_tolerates_a_temporary_file_renamed_by_another_request(
    analysis_service: AnalysisService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transient = analysis_service.settings.storage.jobs_dir / "concurrent.tmp"
    transient.write_bytes(b"pending")
    original_stat = Path.stat
    transient_stat_calls = 0

    def stat_with_concurrent_rename(path: Path, *args: object, **kwargs: object):
        nonlocal transient_stat_calls
        if path == transient:
            transient_stat_calls += 1
            if transient_stat_calls == 2:
                transient.unlink()
                raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_with_concurrent_rename)

    analysis_service._require_capacity()


def test_single_valid_upload_creates_one_queued_job(
    analysis_service: AnalysisService,
    owner: User,
) -> None:
    (job,) = analysis_service.create_batch(owner, [upload("valid.jpg")], "v1")

    assert job.state == JobState.QUEUED
    assert job.batch_position == 0
    assert job.model_version == "v1"
    assert [asset.position for asset in job.assets] == [0]
    assert all(Path(asset.local_path).is_file() for asset in job.assets)


def test_invalid_model_version_is_rejected_before_writing(
    analysis_service: AnalysisService,
    owner: User,
) -> None:
    with pytest.raises(UploadBatchRejected) as caught:
        analysis_service.create_batch(owner, [upload("valid.jpg")], "latest")

    assert caught.value.code == "model_version_invalid"
    assert db.session.scalar(select(func.count()).select_from(AnalysisJob)) == 0


@pytest.mark.parametrize(
    ("mode", "ttl_field", "ttl_value", "retention_class"),
    [
        ("temporary", "temporary_ttl_hours", 3, RetentionClass.TEMPORARY),
        ("feedback_only", "temporary_ttl_hours", 5, RetentionClass.TEMPORARY),
        ("all", "retained_ttl_days", 7, RetentionClass.RETAINED),
    ],
)
def test_storage_mode_controls_initial_retention(
    app: Flask,
    owner: User,
    mode: str,
    ttl_field: str,
    ttl_value: int,
    retention_class: RetentionClass,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    images = settings.storage.images.model_copy(
        update={"mode": mode, ttl_field: ttl_value},
    )
    adjusted = settings.model_copy(
        update={"storage": settings.storage.model_copy(update={"images": images})},
    )

    (job,) = AnalysisService(adjusted).create_batch(owner, [upload("valid.png")], "v1")

    expected_seconds = ttl_value * (86400 if ttl_field.endswith("days") else 3600)
    actual_seconds = (job.expires_at - job.created_at).total_seconds()
    assert actual_seconds == pytest.approx(expected_seconds, abs=1)
    assert {asset.retention_class for asset in job.assets} == {retention_class}
    assert all(asset.expires_at == job.expires_at for asset in job.assets)


def test_invalid_second_file_does_not_remove_existing_job(
    analysis_service: AnalysisService,
    owner: User,
) -> None:
    (existing,) = analysis_service.create_batch(owner, [upload("valid.jpg")], "v1")
    existing_paths = [Path(asset.local_path) for asset in existing.assets]

    with pytest.raises(UploadBatchRejected) as caught:
        analysis_service.create_batch(
            owner,
            [upload("valid.png"), upload("corrupt.jpg")],
            "v1",
        )

    assert caught.value.code == "image_decode_failed"
    assert caught.value.params["file_index"] == 1
    assert db.session.get(AnalysisJob, existing.id) is not None
    assert all(path.is_file() for path in existing_paths)
    assert db.session.scalar(select(func.count()).select_from(AnalysisAsset)) == 1


def test_corrupt_tenth_file_creates_no_rows_or_directories(
    analysis_service: AnalysisService,
    owner: User,
) -> None:
    uploads = distinct_uploads(9)
    uploads.append(upload("corrupt.jpg", filename="frame-9.jpg"))

    with pytest.raises(UploadBatchRejected) as caught:
        analysis_service.create_batch(owner, uploads, "v1")

    assert caught.value.code == "image_decode_failed"
    assert caught.value.params["file_index"] == 9
    assert db.session.scalar(select(func.count()).select_from(AnalysisJob)) == 0
    assert not any(analysis_service.settings.storage.jobs_dir.iterdir())


def test_database_failure_rolls_back_rows_and_files(
    analysis_service: AnalysisService,
    owner: User,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = analysis_service.repository.session
    original_commit = session.commit

    def fail_after_flush() -> None:
        session.flush()
        raise RuntimeError("simulated commit failure")

    with monkeypatch.context() as context:
        context.setattr(session, "commit", fail_after_flush)
        with pytest.raises(RuntimeError, match="simulated commit failure"):
            analysis_service.create_batch(owner, [upload("valid.jpg")], "v1")

    original_commit()
    assert db.session.scalar(select(func.count()).select_from(AnalysisJob)) == 0
    assert not any(analysis_service.settings.storage.jobs_dir.iterdir())


@pytest.mark.parametrize("failure_target", ["job_directory", "feedback_snapshot"])
@pytest.mark.parametrize("retry_role", ["owner", "admin"])
def test_batch_delete_retries_tombstoned_cleanup_after_attempting_every_job(
    analysis_service: AnalysisService,
    owner: User,
    monkeypatch: pytest.MonkeyPatch,
    failure_target: str,
    retry_role: str,
) -> None:
    jobs = analysis_service.create_batch(owner, distinct_uploads(2), "v1")
    batch_id = jobs[0].batch_id
    asset_paths = {job.id: Path(job.assets[0].local_path) for job in jobs}
    snapshot_paths = {}
    for job in jobs:
        snapshot = analysis_service._feedback_directory(job.id)
        snapshot.mkdir(parents=True)
        (snapshot / "snapshot.png").write_bytes(b"feedback-snapshot")
        snapshot_paths[job.id] = snapshot

    directory_attempts: list[str] = []
    snapshot_attempts: list[str] = []
    failed = False
    original_delete_job = analysis_service.file_store.delete_job
    original_delete_snapshot = analysis_service._delete_feedback_snapshot

    def delete_job(job_id: str) -> None:
        nonlocal failed
        directory_attempts.append(job_id)
        if failure_target == "job_directory" and job_id == jobs[0].id and not failed:
            failed = True
            raise OSError("simulated batch cleanup failure")
        original_delete_job(job_id)

    def delete_snapshot(job_id: str) -> None:
        nonlocal failed
        snapshot_attempts.append(job_id)
        if failure_target == "feedback_snapshot" and job_id == jobs[0].id and not failed:
            failed = True
            raise OSError("simulated batch cleanup failure")
        original_delete_snapshot(job_id)

    with monkeypatch.context() as context:
        context.setattr(analysis_service.file_store, "delete_job", delete_job)
        context.setattr(analysis_service, "_delete_feedback_snapshot", delete_snapshot)
        with pytest.raises(OSError, match="simulated batch cleanup failure"):
            analysis_service.delete_batch(owner, batch_id)

    assert directory_attempts == [job.id for job in jobs]
    assert snapshot_attempts == [job.id for job in jobs]
    db.session.expire_all()
    assert {
        db.session.get(AnalysisJob, job.id).state for job in jobs  # type: ignore[union-attr]
    } == {JobState.DELETED}

    if retry_role == "admin":
        retry_actor = AuthService().create_user(
            "admin@example.test",
            "StrongPass123",
            "admin",
            "ru",
        )
    else:
        retry_actor = owner
    analysis_service.delete_batch(retry_actor, batch_id)

    db.session.expire_all()
    assert all(not path.exists() for path in asset_paths.values())
    assert all(not path.exists() for path in snapshot_paths.values())
    assert all(
        not db.session.get(AnalysisJob, job.id).assets for job in jobs  # type: ignore[union-attr]
    )


def test_worker_claim_and_service_batch_delete_serialize_without_busy_snapshot(
    app: Flask,
    analysis_service: AnalysisService,
    owner: User,
) -> None:
    (job,) = analysis_service.create_batch(owner, [upload("valid.jpg")], "v1")
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    factory = sessionmaker(bind=db.engine, expire_on_commit=False)
    worker_session: Session = factory()
    delete_session: Session = factory()
    worker_repository = JobRepository(worker_session)
    deleting_service = AnalysisService(settings)
    deleting_service.repository = JobRepository(delete_session)
    worker_has_write_lock = Event()
    delete_write_attempted = Event()
    release_worker = Event()
    claimed_jobs: list[AnalysisJob | None] = []
    errors: list[BaseException] = []
    delete_statements: list[str] = []

    def hold_worker_lock(*args: object) -> None:
        statement = args[2]
        if (
            isinstance(statement, str)
            and current_thread().name == "worker-claim"
            and statement == "BEGIN IMMEDIATE"
        ):
            worker_has_write_lock.set()
            release_worker.wait(timeout=5)

    def observe_delete_write(*args: object) -> None:
        statement = args[2]
        if not isinstance(statement, str):
            return
        if current_thread().name == "batch-delete":
            delete_statements.append(statement)
            if statement == "BEGIN IMMEDIATE" or statement.startswith(
                "UPDATE analysis_jobs SET",
            ):
                delete_write_attempted.set()

    def claim() -> None:
        try:
            claimed_jobs.append(
                worker_repository.claim_next("worker-a", datetime.now(UTC)),
            )
        except BaseException as error:
            errors.append(error)

    def delete() -> None:
        try:
            with app.app_context():
                deleting_service.delete_batch(owner, job.batch_id)
        except BaseException as error:
            errors.append(error)

    event.listen(db.engine, "after_cursor_execute", hold_worker_lock)
    event.listen(db.engine, "before_cursor_execute", observe_delete_write)
    worker_thread = Thread(target=claim, name="worker-claim")
    delete_thread = Thread(target=delete, name="batch-delete")
    try:
        worker_thread.start()
        assert worker_has_write_lock.wait(timeout=5), errors
        delete_thread.start()
        assert delete_write_attempted.wait(timeout=5)
    finally:
        release_worker.set()
        worker_thread.join(timeout=5)
        if delete_thread.ident is not None:
            delete_thread.join(timeout=5)
        event.remove(db.engine, "after_cursor_execute", hold_worker_lock)
        event.remove(db.engine, "before_cursor_execute", observe_delete_write)

    assert not worker_thread.is_alive()
    assert not delete_thread.is_alive()
    assert errors == []
    assert delete_statements[0] == "BEGIN IMMEDIATE"
    assert claimed_jobs and claimed_jobs[0] is not None
    with pytest.raises(JobNoLongerProcessable):
        worker_repository.complete(job.id, {"frames": []}, datetime.now(UTC))
    db.session.expire_all()
    refreshed = db.session.get(AnalysisJob, job.id)
    assert refreshed is not None and refreshed.state == JobState.DELETED

    worker_session.close()
    delete_session.close()
