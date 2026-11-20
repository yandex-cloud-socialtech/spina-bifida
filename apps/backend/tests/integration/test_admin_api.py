from unittest.mock import patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from spina_bifida.auth.service import AuthService
from spina_bifida.extensions import db


def csrf_token(client: FlaskClient) -> str:
    return client.get("/api/v1/auth/me").get_json()["csrf_token"]


def login(client: FlaskClient, email: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "StrongPass123"},
        headers={"X-CSRF-Token": csrf_token(client)},
    )
    assert response.status_code == 200
    return response.get_json()["csrf_token"]


def create_users() -> tuple[str, str]:
    db.create_all()
    service = AuthService()
    admin = service.create_user("admin@example.test", "StrongPass123", "admin", "ru")
    doctor = service.create_user("doctor@example.test", "StrongPass123", "user", "ru")
    return admin.id, doctor.id


def test_anonymous_and_ordinary_user_cannot_list_users(
    app: Flask,
    client: FlaskClient,
) -> None:
    _, _doctor = create_users()
    anonymous = client.get("/api/v1/admin/users")
    login(client, "doctor@example.test")
    ordinary = client.get("/api/v1/admin/users")

    assert anonymous.status_code == 401
    assert anonymous.get_json()["error"]["code"] == "authentication_required"
    assert ordinary.status_code == 403
    assert ordinary.get_json()["error"]["code"] == "admin_required"


def test_admin_can_create_block_and_delete_user(app: Flask, client: FlaskClient) -> None:
    admin_id, _doctor = create_users()
    token = login(client, "admin@example.test")
    with patch("spina_bifida.auth.service.audit") as audit_mock:
        created = client.post(
            "/api/v1/admin/users",
            json={
                "email": "new@example.test",
                "password": "StrongPass123",
                "role": "user",
                "locale": "kk",
            },
            headers={"X-CSRF-Token": token},
        )
    user_id = created.get_json()["user"]["id"]
    audit_mock.assert_called_once_with(
        "user_created",
        admin_id,
        target_user_id=user_id,
        role="user",
        locale="kk",
    )
    blocked = client.patch(
        f"/api/v1/admin/users/{user_id}",
        json={"blocked": True},
        headers={"X-CSRF-Token": token},
    )
    deleted = client.delete(
        f"/api/v1/admin/users/{user_id}",
        headers={"X-CSRF-Token": token},
    )

    assert created.status_code == 201
    assert blocked.status_code == 200
    assert blocked.get_json()["user"]["status"] == "blocked"
    assert deleted.status_code == 200
    assert deleted.get_json()["user"]["status"] == "deleted"
    assert deleted.get_json()["user"]["email"].endswith("@invalid.local")


def test_duplicate_email_is_conflict(app: Flask, client: FlaskClient) -> None:
    _admin, _doctor = create_users()
    token = login(client, "admin@example.test")

    response = client.post(
        "/api/v1/admin/users",
        json={
            "email": "DOCTOR@example.test",
            "password": "StrongPass123",
            "role": "user",
            "locale": "ru",
        },
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "email_exists"


def test_last_active_admin_cannot_be_deleted(app: Flask, client: FlaskClient) -> None:
    admin_id, _doctor = create_users()
    token = login(client, "admin@example.test")

    response = client.delete(
        f"/api/v1/admin/users/{admin_id}",
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "last_active_admin"


def test_admin_mutation_requires_csrf(app: Flask, client: FlaskClient) -> None:
    _admin, doctor_id = create_users()
    login(client, "admin@example.test")

    response = client.patch(
        f"/api/v1/admin/users/{doctor_id}",
        json={"blocked": True},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "csrf_failed"


def test_create_admin_cli_prompts_for_hidden_confirmed_password(app: Flask) -> None:
    db.create_all()
    runner = app.test_cli_runner()

    result = runner.invoke(
        args=["create-admin"],
        input="admin@example.test\nStrongPass123\nStrongPass123\nru\n",
    )

    assert result.exit_code == 0
    assert "Admin created: admin@example.test" in result.output
    assert "StrongPass123" not in result.output
    assert AuthService().authenticate("admin@example.test", "StrongPass123") is not None


def test_create_admin_cli_accepts_email_without_echoing_password(app: Flask) -> None:
    db.create_all()
    runner = app.test_cli_runner()

    result = runner.invoke(
        args=["create-admin", "--email", "owner@example.test"],
        input="StrongPass123\nStrongPass123\nru\n",
    )

    assert result.exit_code == 0
    assert "Admin email" not in result.output
    assert "Admin created: owner@example.test" in result.output
    assert "StrongPass123" not in result.output
    assert AuthService().authenticate("owner@example.test", "StrongPass123") is not None


@pytest.mark.parametrize("has_second_admin", [False, True])
def test_admin_cannot_block_own_account(
    app: Flask, client: FlaskClient, has_second_admin: bool,
) -> None:
    admin_id, _doctor = create_users()
    if has_second_admin:
        AuthService().create_user("other@example.test", "StrongPass123", "admin", "ru")
    token = login(client, "admin@example.test")

    with patch("spina_bifida.auth.service.audit") as audit_mock:
        response = client.patch(
            f"/api/v1/admin/users/{admin_id}",
            json={"blocked": True, "role": "user", "locale": "en"},
            headers={"X-CSRF-Token": token},
        )

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "self_block_forbidden"
    audit_mock.assert_not_called()
    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.get_json()["user"]["status"] == "active"
    assert me.get_json()["user"]["role"] == "admin"
    assert me.get_json()["user"]["locale"] == "ru"


def test_admin_can_block_another_admin(app: Flask, client: FlaskClient) -> None:
    create_users()
    other = AuthService().create_user("other@example.test", "StrongPass123", "admin", "ru")
    token = login(client, "admin@example.test")

    response = client.patch(
        f"/api/v1/admin/users/{other.id}",
        json={"blocked": True},
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 200
    assert response.get_json()["user"]["status"] == "blocked"
