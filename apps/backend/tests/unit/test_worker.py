from __future__ import annotations

import json
import logging
import socket
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Protocol
from unittest.mock import Mock

import pytest
import torch
from flask import Flask
from PIL import Image
from werkzeug.datastructures import FileStorage

from spina_bifida.analyses.models import AnalysisJob, AssetKind, JobState, ProcessingStage
from spina_bifida.analyses.repository import JobRepository
from spina_bifida.analyses.service import AnalysisService
from spina_bifida.auth.models import User
from spina_bifida.auth.service import AuthService
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db
from spina_bifida.inference.artifacts import image_asset
from spina_bifida.inference.contracts import (
    AnalysisResult,
    AssessmentCategory,
    AssessmentResult,
    AssetType,
    FrameResult,
    InputFrame,
    ModelMetadata,
    PlaneResult,
    PlaneType,
    QualityResult,
    RoiResult,
)
from spina_bifida.inference.model_loader import ModelBundle, ModelFilesMissing, ModelVersion
from spina_bifida.queue.worker import AnalysisWorker, build_worker
from spina_bifida.storage.files import FileStore

FIXTURES = Path(__file__).parents[1] / "fixtures" / "images"
EXPECTED_STAGES = list(ProcessingStage)


class InferenceProtocol(Protocol):
    def analyze(self, frames: list[InputFrame]) -> AnalysisResult: ...


class RecordingRegistry:
    def __init__(self, services: Mapping[str, InferenceProtocol]) -> None:
        self.services = services
        self.requested: list[str] = []

    def for_version(self, version: str) -> InferenceProtocol:
        self.requested.append(version)
        return self.services[version]


def upload() -> FileStorage:
    return FileStorage(
        stream=BytesIO((FIXTURES / "valid.png").read_bytes()),
        filename="valid.png",
        content_type="image/png",
    )


def successful_result(
    frames: list[InputFrame],
    model_version: str = "v1",
) -> AnalysisResult:
    results = []
    for frame in frames:
        generated = Image.new("RGB", (12, 10), (50, 100, 150))
        results.append(
            FrameResult(
                frame_id=frame.frame_id,
                analyzable=True,
                quality=QualityResult(0.9, True),
                plane=PlaneResult(PlaneType.AXIAL, 0.8),
                roi=RoiResult(
                    box=(1, 1, 11, 9),
                    display_box=(0, 0, 12, 10),
                ),
                assessment=AssessmentResult(AssessmentCategory.NO_EXPRESSED_SIGNAL, 0.1),
                explanations=("assessment_regions",),
                assets=(
                    image_asset(AssetType.ROI, generated),
                    image_asset(AssetType.HEATMAP, generated),
                ),
            ),
        )
    return AnalysisResult(
        frames=tuple(results),
        warnings=(),
        model=ModelMetadata(model_version, "cpu", "test-inference"),
    )


class SuccessfulInference:
    def __init__(self, model_version: str = "v1") -> None:
        self.model_version = model_version
        self.calls = 0

    def analyze(self, frames: list[InputFrame]) -> AnalysisResult:
        self.calls += 1
        return successful_result(frames, self.model_version)


class FailingInference:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def analyze(self, frames: list[InputFrame]) -> AnalysisResult:
        raise self.error


class RejectedInference:
    def analyze(self, frames: list[InputFrame]) -> AnalysisResult:
        frame = frames[0]
        return AnalysisResult(
            frames=(
                FrameResult(
                    frame_id=frame.frame_id,
                    analyzable=False,
                    quality=QualityResult(0.1, False),
                    plane=PlaneResult(PlaneType.UNKNOWN, 0.0),
                    roi=None,
                    assessment=None,
                    explanations=(),
                    warnings=("no_detection",),
                ),
            ),
            warnings=("no_detection",),
            model=ModelMetadata("v1", "cpu", "test-inference"),
        )


class SlowInference:
    def analyze(self, frames: list[InputFrame]) -> AnalysisResult:
        sleep(0.2)
        return successful_result(frames)


class CallbackInference:
    def __init__(self, callback) -> None:
        self.callback = callback

    def analyze(self, frames: list[InputFrame]) -> AnalysisResult:
        self.callback()
        return successful_result(frames)


class RecordingRepository(JobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_stages: list[ProcessingStage] = []

    def set_stage(self, job_id: str, stage: ProcessingStage) -> AnalysisJob:
        self.recorded_stages.append(stage)
        return super().set_stage(job_id, stage)


def audit_events(settings: AppSettings) -> list[dict[str, object]]:
    for handler in logging.getLogger("spina_bifida").handlers:
        handler.flush()
    return [
        json.loads(line)
        for line in (settings.logs.directory / "api.jsonl").read_text(encoding="utf-8").splitlines()
    ]


@pytest.fixture
def settings(app: Flask) -> AppSettings:
    configured = app.extensions["settings"]
    assert isinstance(configured, AppSettings)
    return configured


@pytest.fixture
def queued_job(app: Flask, settings: AppSettings) -> AnalysisJob:
    db.create_all()
    owner: User = AuthService().create_user(
        "worker@example.test",
        "StrongPass123",
        "user",
        "ru",
    )
    return AnalysisService(settings).create_job(owner, [upload()])


def make_worker(
    settings: AppSettings,
    inference: InferenceProtocol | None = None,
    repository: JobRepository | None = None,
    *,
    registry: RecordingRegistry | None = None,
) -> AnalysisWorker:
    inference_registry = registry or RecordingRegistry(
        {"v1": inference or SuccessfulInference(), "v2": inference or SuccessfulInference()},
    )
    return AnalysisWorker(
        repository=repository or JobRepository(),
        inference_registry=inference_registry,
        file_store=FileStore(settings.storage.jobs_dir),
        heartbeat_path=settings.storage.db_dir / "worker-heartbeat.json",
        worker_id="test-worker",
        device="cpu",
        timeout_seconds=5,
    )


@pytest.mark.parametrize("model_version", ["v1", "v2"])
def test_worker_uses_the_model_persisted_on_the_claimed_job(
    settings: AppSettings,
    queued_job: AnalysisJob,
    model_version: str,
) -> None:
    queued_job.model_version = model_version  # type: ignore[assignment]
    db.session.commit()
    services = {
        "v1": SuccessfulInference("v1"),
        "v2": SuccessfulInference("v2"),
    }
    registry = RecordingRegistry(services)
    worker = make_worker(settings, registry=registry)

    assert worker.run_once() is True
    assert registry.requested == [model_version]
    assert services[model_version].calls == 1
    assert services[{"v1": "v2", "v2": "v1"}[model_version]].calls == 0
    refreshed = db.session.get(AnalysisJob, queued_job.id)
    assert refreshed is not None
    assert refreshed.result is not None
    assert refreshed.result["model"]["version"] == model_version


def test_build_worker_removes_fresh_heartbeat_while_second_bundle_load_fails(
    settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostname = "startup-test-host"
    heartbeat = settings.storage.db_dir / f"worker-heartbeat-{hostname}.json"
    heartbeat.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    heartbeat.write_text('{"timestamp":"fresh"}', encoding="utf-8")
    second_bundle_started = Event()
    allow_failure = Event()
    loaded: list[ModelVersion] = []
    errors: list[BaseException] = []
    missing = settings.inference.models_root / "v2" / "axial_quality.pt"

    def fake_load(
        models_root: Path,
        version: ModelVersion,
        device: torch.device,
    ) -> ModelBundle:
        loaded.append(version)
        if version == "v2":
            second_bundle_started.set()
            assert allow_failure.wait(timeout=2)
            raise ModelFilesMissing([missing])
        return Mock(spec=ModelBundle, version=version, device=device)

    def load_worker() -> None:
        try:
            build_worker(settings)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(socket, "gethostname", lambda: hostname)
    monkeypatch.setattr(ModelBundle, "load", staticmethod(fake_load))
    thread = Thread(target=load_worker)
    thread.start()
    try:
        assert second_bundle_started.wait(timeout=2)
        assert not heartbeat.exists()
    finally:
        allow_failure.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert loaded == ["v1", "v2"]
    assert len(errors) == 1 and isinstance(errors[0], ModelFilesMissing)
    assert not heartbeat.exists()


def test_worker_audits_persisted_model_version_on_success(
    settings: AppSettings,
    queued_job: AnalysisJob,
) -> None:
    queued_job.model_version = "v2"
    db.session.commit()

    assert make_worker(settings).run_once() is True

    audit_fields = {
        event["event_code"]: event
        for event in audit_events(settings)
        if event["event_code"] in {"analysis_started", "analysis_finished"}
    }
    assert audit_fields["analysis_started"]["model_version"] == "v2"
    assert audit_fields["analysis_finished"]["model_version"] == "v2"


@pytest.mark.parametrize(
    ("error", "error_code"),
    [
        (TimeoutError("timeout detail"), "inference_timeout"),
        (torch.cuda.OutOfMemoryError("cuda detail"), "inference_out_of_memory"),
        (OSError("storage detail"), "storage_unavailable"),
        (RuntimeError("generic detail"), "inference_failed"),
    ],
)
def test_worker_audits_persisted_model_version_and_error_code_on_failure(
    settings: AppSettings,
    queued_job: AnalysisJob,
    error: Exception,
    error_code: str,
) -> None:
    queued_job.model_version = "v2"
    db.session.commit()

    assert make_worker(settings, FailingInference(error)).run_once() is True

    failed = next(
        event
        for event in audit_events(settings)
        if event["event_code"] == "analysis_failed"
    )
    assert failed["model_version"] == "v2"
    assert failed["error_code"] == error_code


def test_worker_completes_one_claimed_job(
    settings: AppSettings,
    queued_job: AnalysisJob,
) -> None:
    repository = RecordingRepository()
    worker = make_worker(settings, SuccessfulInference(), repository)

    assert worker.run_once() is True
    db.session.expire_all()
    refreshed = db.session.get(AnalysisJob, queued_job.id)

    assert refreshed is not None and refreshed.state == JobState.COMPLETED
    assert repository.recorded_stages == EXPECTED_STAGES
    assert {asset.kind for asset in refreshed.assets} == {
        AssetKind.ORIGINAL,
        AssetKind.ROI,
        AssetKind.HEATMAP,
    }
    assert "content" not in str(refreshed.result)
    assert "local_path" not in str(refreshed.result)
    assert refreshed.result is not None
    assert refreshed.result["frames"][0]["roi"] == {
        "box": [1, 1, 11, 9],
        "display_box": [0, 0, 12, 10],
    }


def test_restart_marks_processing_job_recoverable(
    settings: AppSettings,
    queued_job: AnalysisJob,
) -> None:
    queued_job.state = JobState.SCORING
    queued_job.stage = ProcessingStage.SCORING
    worker = make_worker(settings, SuccessfulInference())
    queued_job.worker_id = worker.worker_id
    db.session.commit()

    assert worker.startup_recovery() == 1

    refreshed = db.session.get(AnalysisJob, queued_job.id)
    assert refreshed is not None and refreshed.state == JobState.FAILED_RECOVERABLE
    assert refreshed.error_code == "worker_interrupted"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (TimeoutError("slow and private detail"), "inference_timeout"),
        (OSError("/private/patient/path"), "storage_unavailable"),
        (RuntimeError("sensitive model details"), "inference_failed"),
    ],
)
def test_worker_redacts_recoverable_failures(
    settings: AppSettings,
    queued_job: AnalysisJob,
    error: Exception,
    code: str,
) -> None:
    worker = make_worker(settings, FailingInference(error))

    assert worker.run_once() is True

    refreshed = db.session.get(AnalysisJob, queued_job.id)
    assert refreshed is not None and refreshed.state == JobState.FAILED_RECOVERABLE
    assert refreshed.error_code == code
    assert str(error) not in str(refreshed.result)


def test_quality_rejection_is_not_a_technical_failure(
    settings: AppSettings,
    queued_job: AnalysisJob,
) -> None:
    worker = make_worker(settings, RejectedInference())

    assert worker.run_once() is True

    refreshed = db.session.get(AnalysisJob, queued_job.id)
    assert refreshed is not None and refreshed.state == JobState.REJECTED_QUALITY
    assert refreshed.error_code == "insufficient_quality"


def test_worker_enforces_inference_timeout(
    settings: AppSettings,
    queued_job: AnalysisJob,
) -> None:
    worker = AnalysisWorker(
        repository=JobRepository(),
        inference_registry=RecordingRegistry(
            {"v1": SlowInference(), "v2": SlowInference()},
        ),
        file_store=FileStore(settings.storage.jobs_dir),
        heartbeat_path=settings.storage.db_dir / "worker-heartbeat.json",
        worker_id="timeout-worker",
        device="cpu",
        timeout_seconds=0.02,
    )

    assert worker.run_once() is True

    refreshed = db.session.get(AnalysisJob, queued_job.id)
    assert refreshed is not None and refreshed.state == JobState.FAILED_RECOVERABLE
    assert refreshed.error_code == "inference_timeout"


def test_no_work_returns_false_and_writes_safe_heartbeat(
    settings: AppSettings,
    app: Flask,
) -> None:
    db.create_all()
    heartbeat = settings.storage.db_dir / "worker-heartbeat.json"
    worker = make_worker(settings, SuccessfulInference())

    assert worker.run_once() is False

    content = heartbeat.read_text(encoding="utf-8")
    assert "test-worker" in content
    assert "worker@example.test" not in content
    assert str(settings.storage.jobs_dir) not in content


def test_stopped_worker_does_not_claim_new_work(
    settings: AppSettings,
    queued_job: AnalysisJob,
) -> None:
    stop = Event()
    stop.set()
    worker = make_worker(settings, SuccessfulInference())

    worker.run_forever(stop, idle_seconds=0.001)

    refreshed = db.session.get(AnalysisJob, queued_job.id)
    assert refreshed is not None and refreshed.state == JobState.QUEUED


def test_worker_does_not_resurrect_job_deleted_during_inference(
    settings: AppSettings,
    queued_job: AnalysisJob,
) -> None:
    owner = queued_job.owner
    worker = make_worker(
        settings,
        CallbackInference(lambda: AnalysisService(settings).delete_job(owner, queued_job.id)),
    )

    assert worker.run_once() is True

    db.session.expire_all()
    refreshed = db.session.get(AnalysisJob, queued_job.id)
    assert refreshed is not None and refreshed.state == JobState.DELETED
    assert refreshed.assets == []


def test_busy_worker_runs_maintenance_between_jobs(
    settings: AppSettings,
    queued_job: AnalysisJob,
) -> None:
    stop = Event()
    calls: list[str] = []

    def maintain() -> None:
        calls.append("ran")
        stop.set()

    worker = AnalysisWorker(
        repository=JobRepository(),
        inference_registry=RecordingRegistry(
            {"v1": SuccessfulInference(), "v2": SuccessfulInference()},
        ),
        file_store=FileStore(settings.storage.jobs_dir),
        heartbeat_path=settings.storage.db_dir / "maintenance-worker.json",
        worker_id="maintenance-worker",
        device="cpu",
        timeout_seconds=5,
        maintenance=(maintain,),
    )

    worker.run_forever(stop, idle_seconds=0.001)

    assert calls == ["ran"]


def test_two_workers_complete_a_single_job_once(
    app: Flask,
    settings: AppSettings,
    queued_job: AnalysisJob,
) -> None:
    def run(worker_id: str) -> bool:
        with app.app_context():
            worker = AnalysisWorker(
                repository=JobRepository(),
                inference_registry=RecordingRegistry(
                    {"v1": SuccessfulInference(), "v2": SuccessfulInference()},
                ),
                file_store=FileStore(settings.storage.jobs_dir),
                heartbeat_path=settings.storage.db_dir / f"{worker_id}.json",
                worker_id=worker_id,
                device="cpu",
                timeout_seconds=5,
            )
            return worker.run_once()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(run, ["worker-a", "worker-b"]))

    assert sorted(outcomes) == [False, True]
    db.session.expire_all()
    refreshed = db.session.get(AnalysisJob, queued_job.id)
    assert refreshed is not None and refreshed.attempt == 1
