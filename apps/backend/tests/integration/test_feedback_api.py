from __future__ import annotations

from io import BytesIO
from pathlib import Path

from flask import Flask
from flask.testing import FlaskClient
from sqlalchemy import func, select

from spina_bifida.analyses.models import AnalysisJob, Feedback, JobState, RetentionClass
from spina_bifida.auth.service import AuthService
from spina_bifida.extensions import db

FIXTURES = Path(__file__).parents[1] / "fixtures" / "images"


def login_without_terms(client: FlaskClient, email: str) -> str:
    token = client.get("/api/v1/auth/me").get_json()["csrf_token"]
    return client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "StrongPass123"},
        headers={"X-CSRF-Token": token},
    ).get_json()["csrf_token"]


def login(client: FlaskClient, email: str) -> str:
    csrf_token = login_without_terms(client, email)
    accepted = client.post(
        "/api/v1/auth/terms/accept",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert accepted.status_code == 200
    return csrf_token


def completed_job(client: FlaskClient, token: str) -> AnalysisJob:
    created = client.post(
        "/api/v1/analyses",
        data={
            "model_version": "v1",
            "frames": [
                (
                    BytesIO((FIXTURES / "valid.jpg").read_bytes()),
                    "valid.jpg",
                    "image/jpeg",
                ),
            ],
        },
        headers={"X-CSRF-Token": token},
    ).get_json()
    job = db.session.get(AnalysisJob, created["jobs"][0]["id"])
    assert job is not None
    job.state = JobState.COMPLETED
    db.session.commit()
    return job


def setup_feedback(app: Flask, client: FlaskClient) -> tuple[str, AnalysisJob]:
    db.create_all()
    AuthService().create_user("doctor@example.test", "StrongPass123", "user", "ru")
    token = login(client, "doctor@example.test")
    return token, completed_job(client, token)


def test_feedback_upserts_and_promotes_retention(
    app: Flask,
    client: FlaskClient,
) -> None:
    token, job = setup_feedback(app, client)
    first = client.post(
        f"/api/v1/analyses/{job.id}/feedback",
        json={"value": "positive"},
        headers={"X-CSRF-Token": token},
    )
    second = client.post(
        f"/api/v1/analyses/{job.id}/feedback",
        json={"value": "negative", "comment": "Needs review"},
        headers={"X-CSRF-Token": token},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.get_json()["feedback"]["value"] == "negative"
    assert db.session.scalar(select(func.count()).select_from(Feedback)) == 1
    db.session.expire_all()
    refreshed = db.session.get(AnalysisJob, job.id)
    assert refreshed is not None
    assert {asset.retention_class for asset in refreshed.assets} == {RetentionClass.FEEDBACK}


def test_feedback_validates_value_and_comment_length(
    app: Flask,
    client: FlaskClient,
) -> None:
    token, job = setup_feedback(app, client)

    invalid_value = client.post(
        f"/api/v1/analyses/{job.id}/feedback",
        json={"value": "maybe"},
        headers={"X-CSRF-Token": token},
    )
    long_comment = client.post(
        f"/api/v1/analyses/{job.id}/feedback",
        json={"value": "negative", "comment": "x" * 2001},
        headers={"X-CSRF-Token": token},
    )

    assert invalid_value.status_code == 422
    assert invalid_value.get_json()["error"]["code"] == "invalid_feedback"
    assert long_comment.status_code == 422
    assert long_comment.get_json()["error"]["params"]["field"] == "comment"


def test_feedback_cannot_cross_user_boundary(app: Flask, client: FlaskClient) -> None:
    _token, job = setup_feedback(app, client)
    other_client = app.test_client()
    AuthService().create_user("other@example.test", "StrongPass123", "user", "ru")
    other_token = login(other_client, "other@example.test")

    response = other_client.post(
        f"/api/v1/analyses/{job.id}/feedback",
        json={"value": "positive"},
        headers={"X-CSRF-Token": other_token},
    )

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "analysis_not_found"


def test_feedback_requires_terms_acceptance(app: Flask, client: FlaskClient) -> None:
    token, job = setup_feedback(app, client)
    logged_out = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": token},
    )
    assert logged_out.status_code == 204
    unaccepted_token = login_without_terms(client, "doctor@example.test")

    response = client.post(
        f"/api/v1/analyses/{job.id}/feedback",
        json={"value": "positive"},
        headers={"X-CSRF-Token": unaccepted_token},
    )

    assert response.status_code == 403
    assert response.get_json()["error"]["code"] == "terms_acceptance_required"
