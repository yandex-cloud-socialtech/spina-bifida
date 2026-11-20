from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

from flask import Flask

from spina_bifida.config import AppSettings
from spina_bifida.observability.audit import audit
from spina_bifida.observability.logging import configure_logging


def logging_settings(settings: AppSettings, directory: Path) -> AppSettings:
    return settings.model_copy(
        update={
            "logs": settings.logs.model_copy(
                update={
                    "directory": directory,
                    "rotate_max_bytes": 1024,
                    "max_archives": 5,
                    "compress": True,
                },
            ),
        },
    )


def flush(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def test_jsonl_contains_required_context_and_redacts_forbidden_fields(
    app: Flask,
    tmp_path: Path,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    logger = configure_logging(logging_settings(settings, tmp_path), "api")

    with app.test_request_context(headers={"X-Request-ID": "ignored-client-id"}):
        from flask import g

        g.request_id = "request-123"
        audit(
            "security_test",
            "actor-1",
            job_id="job-1",
            outcome="accepted",
            password="secret",
            cookie="session",
            authorization="bearer",
            image_bytes=b"private",
            patient_name="private name",
        )
    flush(logger)

    payload = json.loads((tmp_path / "api.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert {
        "timestamp",
        "severity",
        "component",
        "event_code",
        "app_version",
        "request_id",
        "session_id",
        "job_id",
    } <= set(payload)
    assert payload["event_code"] == "security_test"
    assert payload["request_id"] == "request-123"
    serialized = json.dumps(payload)
    for forbidden in (
        "password",
        "secret",
        "cookie",
        "authorization",
        "image_bytes",
        "patient_name",
        "private name",
    ):
        assert forbidden not in serialized


def test_handler_rotates_by_size_and_compresses_closed_logs(
    app: Flask,
    tmp_path: Path,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    logger = configure_logging(logging_settings(settings, tmp_path), "worker")

    for index in range(20):
        audit("rotation_test", None, sequence=index, detail="x" * 300)
    flush(logger)

    archives = list(tmp_path.glob("worker.jsonl.*.gz"))
    assert archives
    with gzip.open(archives[0], "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["event_code"] == "rotation_test"
    assert (tmp_path / "worker.jsonl").is_file()


def test_manual_age_rollover_closes_and_compresses_archive(
    app: Flask,
    tmp_path: Path,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    logger = configure_logging(logging_settings(settings, tmp_path), "api")
    audit("before_age_rollover", None)
    flush(logger)

    file_handler = next(handler for handler in logger.handlers if hasattr(handler, "doRollover"))
    file_handler.doRollover()  # type: ignore[attr-defined]
    audit("after_age_rollover", None)
    flush(logger)

    assert list(tmp_path.glob("api.jsonl.*.gz"))
    assert (tmp_path / "api.jsonl").exists()
