from __future__ import annotations

import json
import logging
import signal
import socket
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Protocol

import torch
from sqlalchemy import select

from spina_bifida.analyses.models import (
    AnalysisAsset,
    AnalysisJob,
    AssetKind,
    ProcessingStage,
)
from spina_bifida.analyses.repository import JobNoLongerProcessable, JobRepository
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db
from spina_bifida.inference.contracts import AnalysisResult, AssetType, InputFrame
from spina_bifida.inference.model_loader import resolve_device
from spina_bifida.inference.registry import InferenceRegistry
from spina_bifida.observability.audit import audit
from spina_bifida.observability.logging import configure_logging
from spina_bifida.observability.uploader import OutboxUploader
from spina_bifida.storage.files import FileStore
from spina_bifida.storage.retention import RetentionService

LOGGER = logging.getLogger(__name__)


class InferenceProtocol(Protocol):
    def analyze(self, frames: Sequence[InputFrame]) -> AnalysisResult: ...


class InferenceRegistryProtocol(Protocol):
    def for_version(self, version: str) -> InferenceProtocol: ...


class AnalysisWorker:
    def __init__(
        self,
        *,
        repository: JobRepository,
        inference_registry: InferenceRegistryProtocol,
        file_store: FileStore,
        heartbeat_path: Path,
        worker_id: str,
        device: str,
        timeout_seconds: float,
        maintenance: Sequence[Callable[[], object]] = (),
    ) -> None:
        self.repository = repository
        self.inference_registry = inference_registry
        self.file_store = file_store
        self.heartbeat_path = heartbeat_path
        self.worker_id = worker_id
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.maintenance = tuple(maintenance)

    def startup_recovery(self) -> int:
        return self.repository.recover_interrupted(datetime.now(UTC), self.worker_id)

    def run_once(self) -> bool:
        job = self.repository.claim_next(self.worker_id, datetime.now(UTC))
        if job is None:
            self._write_heartbeat(None)
            return False
        self._write_heartbeat(job.id)
        audit(
            "analysis_started",
            None,
            job_id=job.id,
            worker_id=self.worker_id,
            model_version=job.model_version,
        )
        try:
            for stage in ProcessingStage:
                if stage == ProcessingStage.EXPLANATION:
                    break
                self.repository.set_stage(job.id, stage)
            inputs = [
                InputFrame(asset.frame_id, Path(asset.local_path))
                for asset in job.assets
                if asset.kind == AssetKind.ORIGINAL
            ]
            inference = self.inference_registry.for_version(job.model_version)
            result = self._analyze_with_timeout(inference, inputs)
            self.repository.set_stage(job.id, ProcessingStage.EXPLANATION)
            self._persist_generated_assets(job, result)
            payload = self._result_payload(job, result)
            if any(not frame.analyzable or not frame.quality.accepted for frame in result.frames):
                self.repository.reject_quality(job.id, payload, datetime.now(UTC))
            else:
                self.repository.complete(job.id, payload, datetime.now(UTC))
            audit(
                "analysis_finished",
                None,
                job_id=job.id,
                model_version=job.model_version,
            )
        except JobNoLongerProcessable:
            db.session.rollback()
            LOGGER.info(
                "Analysis stopped because the job is no longer processable",
                extra={"job_id": job.id},
            )
            audit("analysis_cancelled", None, job_id=job.id)
        except TimeoutError:
            self._fail_if_processable(job.id, "inference_timeout")
            LOGGER.exception("Inference timed out", extra={"job_id": job.id})
            audit(
                "analysis_failed",
                None,
                job_id=job.id,
                model_version=job.model_version,
                error_code="inference_timeout",
            )
        except torch.cuda.OutOfMemoryError:
            self._fail_if_processable(job.id, "inference_out_of_memory")
            LOGGER.exception("Inference ran out of device memory", extra={"job_id": job.id})
            audit(
                "analysis_failed",
                None,
                job_id=job.id,
                model_version=job.model_version,
                error_code="inference_out_of_memory",
            )
        except OSError:
            self._fail_if_processable(job.id, "storage_unavailable")
            LOGGER.exception("Inference storage operation failed", extra={"job_id": job.id})
            audit(
                "analysis_failed",
                None,
                job_id=job.id,
                model_version=job.model_version,
                error_code="storage_unavailable",
            )
        except Exception:
            self._fail_if_processable(job.id, "inference_failed")
            LOGGER.exception("Unexpected inference failure", extra={"job_id": job.id})
            audit(
                "analysis_failed",
                None,
                job_id=job.id,
                model_version=job.model_version,
                error_code="inference_failed",
            )
        finally:
            self._write_heartbeat(None)
        return True

    def run_forever(self, stop_event: Event, *, idle_seconds: float = 1.0) -> None:
        while not stop_event.is_set():
            processed = self.run_once()
            self._run_maintenance()
            if not processed:
                stop_event.wait(idle_seconds)

    def _run_maintenance(self) -> None:
        for routine in self.maintenance:
            try:
                routine()
            except Exception:
                LOGGER.exception("Worker maintenance cycle failed")

    def _fail_if_processable(self, job_id: str, code: str) -> None:
        try:
            self.repository.fail_recoverable(job_id, code, datetime.now(UTC))
        except JobNoLongerProcessable:
            db.session.rollback()

    def _analyze_with_timeout(
        self,
        inference: InferenceProtocol,
        inputs: list[InputFrame],
    ) -> AnalysisResult:
        if threading.current_thread() is not threading.main_thread():
            started = time.monotonic()
            result = inference.analyze(inputs)
            if time.monotonic() - started > self.timeout_seconds:
                raise TimeoutError("inference timeout")
            return result

        def timeout_handler(_signum: int, _frame: object) -> None:
            raise TimeoutError("inference timeout")

        previous = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)
        try:
            return inference.analyze(inputs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def _persist_generated_assets(self, job: AnalysisJob, result: AnalysisResult) -> None:
        self.repository.require_processable(job.id)
        originals = {
            asset.frame_id: asset
            for asset in job.assets
            if asset.kind == AssetKind.ORIGINAL
        }
        added: list[AnalysisAsset] = []
        saved_paths: list[Path] = []
        try:
            for frame in result.frames:
                original = originals[frame.frame_id]
                for generated in frame.assets:
                    if generated.kind == AssetType.ORIGINAL:
                        continue
                    stored = self.file_store.save_generated(job.id, frame.frame_id, generated)
                    saved_paths.append(stored.path)
                    added.append(
                        AnalysisAsset(
                            job_id=job.id,
                            frame_id=frame.frame_id,
                            position=original.position,
                            kind=AssetKind(generated.kind.value),
                            local_path=str(stored.path),
                            checksum=stored.checksum,
                            retention_class=original.retention_class,
                            expires_at=job.expires_at,
                        ),
                    )
            self.repository.lock_processable(job.id)
            db.session.add_all(added)
            db.session.commit()
        except Exception:
            db.session.rollback()
            for path in saved_paths:
                self.file_store.delete_asset(path)
            raise

    @staticmethod
    def _result_payload(job: AnalysisJob, result: AnalysisResult) -> dict[str, object]:
        payload = result.to_dict()
        frames = payload["frames"]
        if not isinstance(frames, list):
            raise TypeError("invalid inference result")
        for frame_payload, frame_result in zip(frames, result.frames, strict=True):
            if not isinstance(frame_payload, dict):
                raise TypeError("invalid frame result")
            records = list(
                db.session.scalars(
                    select(AnalysisAsset).where(
                        AnalysisAsset.job_id == job.id,
                        AnalysisAsset.frame_id == frame_result.frame_id,
                    ),
                ),
            )
            descriptors: list[dict[str, object]] = []
            for asset in records:
                descriptor: dict[str, object] = {
                    "id": asset.id,
                    "kind": asset.kind.value,
                    "url": f"/api/v1/analyses/{job.id}/assets/{asset.id}",
                    "checksum": asset.checksum,
                }
                generated = next(
                    (
                        item
                        for item in frame_result.assets
                        if item.kind.value == asset.kind.value
                    ),
                    None,
                )
                if generated is not None:
                    descriptor.update(
                        width=generated.width,
                        height=generated.height,
                        media_type=generated.media_type,
                    )
                descriptors.append(descriptor)
            frame_payload["assets"] = descriptors
        return payload

    def _write_heartbeat(self, active_job_id: str | None) -> None:
        self.heartbeat_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "worker_id": self.worker_id,
            "device": self.device,
            "active_job_id": active_job_id,
        }
        temporary = self.heartbeat_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(self.heartbeat_path)


def build_worker(settings: AppSettings) -> AnalysisWorker:
    hostname = socket.gethostname()
    heartbeat_path = settings.storage.db_dir / f"worker-heartbeat-{hostname}.json"
    heartbeat_path.unlink(missing_ok=True)
    device = resolve_device(settings.inference.device)
    inference_registry = InferenceRegistry.load(
        settings.inference.models_root,
        device,
        settings.inference.thresholds,
    )
    file_store = FileStore(settings.storage.jobs_dir)
    retention = RetentionService(settings, file_store)
    uploader = OutboxUploader.from_settings(settings)
    return AnalysisWorker(
        repository=JobRepository(),
        inference_registry=inference_registry,
        file_store=file_store,
        heartbeat_path=heartbeat_path,
        worker_id=f"worker-{hostname}",
        device=device.type,
        timeout_seconds=settings.inference.timeout_seconds,
        maintenance=(
            uploader.run_once,
            lambda: retention.run_once(datetime.now(UTC)),
        ),
    )


def main() -> None:
    from spina_bifida import create_app

    app = create_app()
    stop_event = Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with app.app_context():
        settings = app.extensions["settings"]
        if not isinstance(settings, AppSettings):
            raise RuntimeError("Application settings are unavailable")
        configure_logging(settings, "worker")
        worker = build_worker(settings)
        worker.startup_recovery()
        worker.run_forever(stop_event)


if __name__ == "__main__":
    main()
