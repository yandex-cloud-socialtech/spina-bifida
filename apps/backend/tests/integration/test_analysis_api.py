from __future__ import annotations

from io import BytesIO
from pathlib import Path

from flask import Flask
from flask.testing import FlaskClient

from spina_bifida.analyses.models import (
    AnalysisAsset,
    AnalysisJob,
    AssetKind,
    JobState,
    ProcessingStage,
    RetentionClass,
)
from spina_bifida.auth.service import AuthService
from spina_bifida.extensions import db

FIXTURES = Path(__file__).parents[1] / "fixtures" / "images"


def login_without_terms(client: FlaskClient, email: str) -> str:
    token = client.get("/api/v1/auth/me").get_json()["csrf_token"]
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "StrongPass123"},
        headers={"X-CSRF-Token": token},
    )
    return response.get_json()["csrf_token"]


def login(client: FlaskClient, email: str) -> str:
    csrf_token = login_without_terms(client, email)
    accepted = client.post(
        "/api/v1/auth/terms/accept",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert accepted.status_code == 200
    return csrf_token


def create_job(client: FlaskClient, token: str) -> AnalysisJob:
    response = client.post(
        "/api/v1/analyses",
        data={
            "model_version": "v1",
            "frames": [
                (
                    BytesIO((FIXTURES / "valid.png").read_bytes()),
                    "valid.png",
                    "image/png",
                ),
            ],
        },
        headers={"X-CSRF-Token": token},
    )
    job = db.session.get(AnalysisJob, response.get_json()["jobs"][0]["id"])
    assert job is not None
    return job


def setup_user(app: Flask, client: FlaskClient) -> tuple[str, AnalysisJob]:
    db.create_all()
    AuthService().create_user("doctor@example.test", "StrongPass123", "user", "ru")
    token = login(client, "doctor@example.test")
    return token, create_job(client, token)


def test_polling_dto_has_ordered_progress_and_terminal_shape(
    app: Flask,
    client: FlaskClient,
) -> None:
    _token, job = setup_user(app, client)
    progress = []
    for stage in ProcessingStage:
        job.state = JobState(stage.value)
        job.stage = stage
        db.session.commit()
        response = client.get(f"/api/v1/analyses/{job.id}")
        progress.append(response.get_json()["progress"])

    job.state = JobState.COMPLETED
    job.stage = None
    job.result = {"frames": [{"assessment": {"score": 0.2}}], "model": {"version": "v1"}}
    db.session.commit()
    completed = client.get(f"/api/v1/analyses/{job.id}").get_json()

    assert progress == sorted(progress)
    assert completed["state"] == "completed"
    assert completed["progress"] == 100
    assert completed["frames"][0]["id"] == job.assets[0].frame_id
    assert completed["result"]["frames"][0]["assessment"]["score"] == 0.2
    assert completed["error"] is None
    assert "local_path" not in str(completed)


def test_polling_reports_global_queue_position(app: Flask, client: FlaskClient) -> None:
    token, first = setup_user(app, client)
    second = create_job(client, token)

    first_payload = client.get(f"/api/v1/analyses/{first.id}").get_json()
    second_payload = client.get(f"/api/v1/analyses/{second.id}").get_json()

    assert first_payload["queue_position"] == 1
    assert second_payload["queue_position"] == 2


def test_retry_requires_recoverable_state_and_preserves_inputs(
    app: Flask,
    client: FlaskClient,
) -> None:
    token, job = setup_user(app, client)
    original = job.assets[0]
    original_path = Path(original.local_path)
    conflict = client.post(
        f"/api/v1/analyses/{job.id}/retry",
        headers={"X-CSRF-Token": token},
    )

    derived_path = original_path.with_name("derived.png")
    derived_path.write_bytes((FIXTURES / "valid.png").read_bytes())
    derived = AnalysisAsset(
        job_id=job.id,
        frame_id=original.frame_id,
        position=original.position,
        kind=AssetKind.HEATMAP,
        local_path=str(derived_path),
        checksum="f" * 64,
        retention_class=RetentionClass.TEMPORARY,
        expires_at=job.expires_at,
    )
    db.session.add(derived)
    job.state = JobState.FAILED_RECOVERABLE
    job.error_code = "inference_timeout"
    job.result = {"private": "must be cleared"}
    job.attempt = 1
    db.session.commit()
    retried = client.post(
        f"/api/v1/analyses/{job.id}/retry",
        headers={"X-CSRF-Token": token},
    )

    assert conflict.status_code == 409
    assert conflict.get_json()["error"]["code"] == "retry_not_allowed"
    assert retried.status_code == 202
    assert retried.get_json()["attempt"] == 2
    db.session.expire_all()
    refreshed = db.session.get(AnalysisJob, job.id)
    assert refreshed is not None and refreshed.state == JobState.QUEUED
    assert refreshed.result is None and refreshed.error_code is None
    assert [asset.kind for asset in refreshed.assets] == [AssetKind.ORIGINAL]
    assert original_path.is_file()
    assert not derived_path.exists()


def test_cross_user_job_is_hidden_and_system_info_has_no_secrets(
    app: Flask,
    client: FlaskClient,
) -> None:
    _token, job = setup_user(app, client)
    other_client = app.test_client()
    AuthService().create_user("other@example.test", "StrongPass123", "user", "ru")
    login(other_client, "other@example.test")

    hidden = other_client.get(f"/api/v1/analyses/{job.id}")
    info = other_client.get("/api/v1/system/info")

    assert hidden.status_code == 404
    assert hidden.get_json()["error"]["code"] == "analysis_not_found"
    assert info.status_code == 200
    info_payload = info.get_json()
    assert {
        "application_version",
        "model_versions",
        "default_model_version",
        "schema_version",
        "device",
    } <= set(info_payload)
    assert info_payload["model_versions"] == ["v1", "v2"]
    assert info_payload["default_model_version"] == "v2"
    assert "model_version" not in info_payload
    serialized = str(info_payload)
    assert "session_secret" not in serialized
    assert "/private/" not in serialized
    assert "/data/" not in serialized


def test_analysis_read_retry_and_delete_require_terms(
    app: Flask,
    client: FlaskClient,
) -> None:
    token, job = setup_user(app, client)
    logged_out = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": token},
    )
    assert logged_out.status_code == 204
    unaccepted_token = login_without_terms(client, "doctor@example.test")

    responses = [
        client.get(f"/api/v1/analyses/{job.id}"),
        client.post(
            f"/api/v1/analyses/{job.id}/retry",
            headers={"X-CSRF-Token": unaccepted_token},
        ),
        client.delete(
            f"/api/v1/analyses/{job.id}",
            headers={"X-CSRF-Token": unaccepted_token},
        ),
    ]

    assert [response.status_code for response in responses] == [403, 403, 403]
    assert {
        response.get_json()["error"]["code"] for response in responses
    } == {"terms_acceptance_required"}
