from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path

from flask import Flask
from flask.testing import FlaskClient

from spina_bifida.auth.service import AuthService
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db

FIXTURES = Path(__file__).parents[1] / "fixtures" / "images"


def csrf(client: FlaskClient) -> str:
    return client.get("/api/v1/auth/me").get_json()["csrf_token"]


def flush_logs() -> None:
    logger = logging.getLogger("spina_bifida")
    for handler in logger.handlers:
        handler.flush()


def test_auth_and_analysis_actions_emit_correlated_safe_audit_events(
    app: Flask,
    client: FlaskClient,
) -> None:
    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    db.create_all()
    AuthService().create_user("doctor@example.test", "StrongPass123", "user", "ru")
    failed = client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@example.test", "password": "wrong-secret"},
        headers={"X-CSRF-Token": csrf(client)},
    )
    logged_in = client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@example.test", "password": "StrongPass123"},
        headers={"X-CSRF-Token": csrf(client)},
    )
    token = logged_in.get_json()["csrf_token"]
    accepted = client.post(
        "/api/v1/auth/terms/accept",
        headers={"X-CSRF-Token": token},
    )
    assert accepted.status_code == 200
    created = client.post(
        "/api/v1/analyses",
        data={
            "model_version": "v2",
            "frames": [
                (
                    BytesIO((FIXTURES / "valid.jpg").read_bytes()),
                    "patient-name-must-not-log.jpg",
                    "image/jpeg",
                ),
            ],
        },
        headers={"X-CSRF-Token": token},
    )
    flush_logs()

    events = [
        json.loads(line)
        for line in (settings.logs.directory / "api.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    codes = {event["event_code"] for event in events}
    assert failed.status_code == 401
    assert created.status_code == 202
    assert {"user_created", "login_failed", "login_succeeded", "analysis_created"} <= codes
    audited = [event for event in events if event["event_code"] != "log"]
    assert all(event["request_id"] for event in audited if event["event_code"] != "user_created")
    serialized = json.dumps(events)
    assert "wrong-secret" not in serialized
    assert "patient-name-must-not-log" not in serialized
    creation_event = next(event for event in events if event["event_code"] == "analysis_created")
    assert creation_event["model_version"] == "v2"
