from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from spina_bifida import create_app
from spina_bifida.auth.service import AuthService
from spina_bifida.extensions import db


def csrf_token(client: FlaskClient) -> str:
    response = client.get("/api/v1/auth/me")
    return response.get_json()["csrf_token"]


def login(client: FlaskClient, email: str, password: str = "StrongPass123") -> dict[str, object]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
        headers={"X-CSRF-Token": csrf_token(client)},
    )
    assert response.status_code == 200
    return response.get_json()


@pytest.fixture
def doctor(app: Flask) -> str:
    db.create_all()
    return (
        AuthService()
        .create_user(
            "doctor@example.test",
            "StrongPass123",
            "user",
            "ru",
        )
        .id
    )


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("missing@example.test", "StrongPass123"),
        ("doctor@example.test", "WrongPass123"),
    ],
)
def test_login_does_not_enumerate_accounts(
    client: FlaskClient,
    doctor: str,
    email: str,
    password: str,
) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
        headers={"X-CSRF-Token": csrf_token(client)},
    )

    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "invalid_credentials"


def test_anonymous_me_returns_401_and_bootstrap_csrf(client: FlaskClient) -> None:
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "authentication_required"
    assert response.get_json()["csrf_token"]


def test_mutating_request_without_csrf_is_rejected(client: FlaskClient, doctor: str) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@example.test", "password": "StrongPass123"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "csrf_failed"


def test_login_sets_minimal_protected_session_cookie(client: FlaskClient, doctor: str) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@example.test", "password": "StrongPass123"},
        headers={"X-CSRF-Token": csrf_token(client)},
    )

    cookie = response.headers["Set-Cookie"]
    assert response.status_code == 200
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie
    assert "Secure" not in cookie
    with client.session_transaction() as session:
        assert set(session) == {"csrf_token", "password_version", "user_id"}


def test_blocked_user_loses_access_on_next_request(
    app: Flask,
    client: FlaskClient,
    doctor: str,
) -> None:
    login(client, "doctor@example.test")
    AuthService().block_user(doctor, blocked=True)

    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "authentication_required"


def test_logout_clears_the_session(client: FlaskClient, doctor: str) -> None:
    token = login(client, "doctor@example.test")["csrf_token"]

    response = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": token})

    assert response.status_code == 204
    with client.session_transaction() as current_session:
        assert not current_session


def test_terms_acceptance_is_required_once_per_login(client: FlaskClient, doctor: str) -> None:
    logged_in = login(client, "doctor@example.test")

    assert logged_in["terms_accepted"] is False
    assert logged_in["terms_version"] == "2026-09-22"
    assert client.get("/api/v1/auth/me").get_json()["terms_accepted"] is False

    missing_csrf = client.post("/api/v1/auth/terms/accept")
    assert missing_csrf.status_code == 400
    assert missing_csrf.get_json()["error"]["code"] == "csrf_failed"

    accepted = client.post(
        "/api/v1/auth/terms/accept",
        headers={"X-CSRF-Token": logged_in["csrf_token"]},
    )
    assert accepted.get_json() == {
        "terms_accepted": True,
        "terms_version": "2026-09-22",
    }
    assert client.get("/api/v1/auth/me").get_json()["terms_accepted"] is True
    with client.session_transaction() as current_session:
        assert set(current_session) == {
            "csrf_token",
            "terms_version",
            "password_version",
            "user_id",
        }
        assert current_session["terms_version"] == "2026-09-22"

    client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": logged_in["csrf_token"]})
    next_login = login(client, "doctor@example.test")
    assert next_login["terms_accepted"] is False
    with client.session_transaction() as current_session:
        assert set(current_session) == {"csrf_token", "password_version", "user_id"}


def test_user_can_update_own_locale(client: FlaskClient, doctor: str) -> None:
    token = login(client, "doctor@example.test")["csrf_token"]

    response = client.patch(
        "/api/v1/auth/preferences",
        json={"locale": "en"},
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 200
    assert response.get_json()["user"]["locale"] == "en"
    assert AuthService().get_active_user(doctor).locale == "en"


def test_login_rate_limit_uses_stable_error(client: FlaskClient, doctor: str) -> None:
    for _ in range(5):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "doctor@example.test", "password": "wrong"},
            headers={"X-CSRF-Token": csrf_token(client)},
        )
        assert response.status_code == 401

    limited = client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@example.test", "password": "wrong"},
        headers={"X-CSRF-Token": csrf_token(client)},
    )

    assert limited.status_code == 429
    assert limited.get_json()["error"]["code"] == "rate_limited"
    assert limited.get_json()["error"]["params"]["retry_after"] >= 1


def test_domain_mode_sets_secure_cookie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        "\n".join(
            [
                "access:",
                "  mode: domain",
                "  domain: research.example.test",
                "  acme_email: admin@example.test",
                "storage:",
                f"  db_dir: {tmp_path / 'db'}",
                f"  jobs_dir: {tmp_path / 'jobs'}",
                f"  feedback_dir: {tmp_path / 'feedback'}",
                "logs:",
                f"  directory: {tmp_path / 'logs'}",
                "backup:",
                f"  destination: {tmp_path / 'backups'}",
            ],
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("APP_SESSION_SECRET", "x" * 32)
    secure_app = create_app({"TESTING": True})
    with secure_app.app_context():
        db.create_all()
        AuthService().create_user("doctor@example.test", "StrongPass123", "user", "ru")
        secure_client = secure_app.test_client()
        response = secure_client.post(
            "/api/v1/auth/login",
            json={"email": "doctor@example.test", "password": "StrongPass123"},
            headers={"X-CSRF-Token": csrf_token(secure_client)},
        )

    assert "Secure" in response.headers["Set-Cookie"]


def test_mail_rate_limit_uses_one_normalized_email_across_client_ips(
    client: FlaskClient,
) -> None:
    # Invalid requests still count, without a dependency on configured mail.
    for ip in ["192.0.2.1", "192.0.2.2", "192.0.2.3"]:
        csrf = csrf_token(client)
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "Doctor@Example.Test"},
            headers={"X-CSRF-Token": csrf, "X-Forwarded-For": ip},
        )
        assert response.status_code == 422
    response = client.post(
        "/api/v1/auth/register",
        json={"email": " doctor@example.test "},
        headers={"X-CSRF-Token": csrf_token(client), "X-Forwarded-For": "192.0.2.4"},
    )
    assert response.status_code == 429
