from __future__ import annotations

import logging
from hashlib import sha256
from typing import Any

from flask import g, has_request_context, session

FORBIDDEN_FIELDS = {
    "password",
    "cookie",
    "authorization",
    "image_bytes",
    "patient_name",
}


def audit(event_code: str, actor_id: str | None, **safe_fields: Any) -> None:
    filtered = {
        key: value
        for key, value in safe_fields.items()
        if key.casefold() not in FORBIDDEN_FIELDS
    }
    request_id = getattr(g, "request_id", None) if has_request_context() else None
    csrf_token = session.get("csrf_token") if has_request_context() else None
    session_id = (
        sha256(csrf_token.encode("utf-8")).hexdigest()[:24]
        if isinstance(csrf_token, str)
        else None
    )
    logging.getLogger("spina_bifida.audit").info(
        event_code,
        extra={
            "event_code": event_code,
            "actor_id": actor_id,
            "request_id": request_id,
            "session_id": session_id,
            "audit_fields": filtered,
        },
    )
