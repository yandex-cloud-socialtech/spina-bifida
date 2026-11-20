from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Literal

from concurrent_log_handler import ConcurrentTimedRotatingFileHandler

from spina_bifida.config import AppSettings

APP_VERSION = "0.1.0"


class JsonFormatter(logging.Formatter):
    def __init__(self, component: str) -> None:
        super().__init__()
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        record_fields = getattr(record, "audit_fields", {})
        fields = dict(record_fields) if isinstance(record_fields, dict) else {}
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,
            "component": self.component,
            "event_code": getattr(record, "event_code", "log"),
            "app_version": APP_VERSION,
            "request_id": getattr(record, "request_id", None),
            "session_id": getattr(record, "session_id", None),
            "job_id": fields.pop("job_id", getattr(record, "job_id", None)),
        }
        actor_id = getattr(record, "actor_id", None)
        if actor_id is not None:
            payload["actor_id"] = actor_id
        payload.update(fields)
        if payload["event_code"] == "log":
            payload["message"] = record.getMessage()
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))


def configure_logging(
    settings: AppSettings,
    component: Literal["api", "worker"],
) -> logging.Logger:
    logger = logging.getLogger("spina_bifida")
    logger.setLevel(settings.logs.level)
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, "_spina_bifida_managed", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = JsonFormatter(component)
    log_path = settings.logs.directory / f"{component}.jsonl"
    file_handler = ConcurrentTimedRotatingFileHandler(
        log_path,
        when="h",
        interval=settings.logs.rotate_interval_hours,
        backupCount=settings.logs.max_archives,
        encoding="utf-8",
        utc=True,
        maxBytes=settings.logs.rotate_max_bytes,
        use_gzip=settings.logs.compress,
    )
    file_handler.setFormatter(formatter)
    file_handler._spina_bifida_managed = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler._spina_bifida_managed = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)
    return logger
