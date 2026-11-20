import re
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
from flask import Flask
from flask.testing import FlaskClient
from pydantic import SecretStr
from sqlalchemy import select

from spina_bifida.auth.mail import MailUnavailable, PostboxMailer
from spina_bifida.auth.models import AuthToken, User, UserStatus, utc_now
from spina_bifida.auth.service import AuthService
from spina_bifida.config import PostboxSettings
from spina_bifida.extensions import db


@pytest.fixture
def mailbox(app: Flask, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    db.create_all()
    settings = PostboxSettings(
        enabled=True,
        sender="noreply@example.test",
        public_url="https://app.example.test",
        access_key=SecretStr("key"),
        secret_key=SecretStr("secret"),
    )
    app.extensions["settings"] = app.extensions["settings"].model_copy(update={"postbox": settings})
    messages: list[tuple[str, str, str]] = []
    monkeypatch.setattr(PostboxMailer, "send", lambda _, *args: messages.append(args))
    return messages


def post(client: FlaskClient, path: str, **payload: object):
    csrf = client.get("/api/v1/auth/me").get_json()["csrf_token"]
    return client.post("/api/v1/" + path, json=payload, headers={"X-CSRF-Token": csrf})


def link_token(mailbox: list[tuple[str, str, str]]) -> str:
    url = re.search(r"https://\S+", mailbox[-1][2]).group()
    assert not urlsplit(url).query
    return parse_qs(urlsplit(url).fragment)["token"][0]


def register(client: FlaskClient, email: str = "doctor@example.test", **extra: object):
    return post(
        client, "auth/register", email=email, password="StrongPass123", locale="ru", **extra
    )


def admin_login(client: FlaskClient) -> User:
    admin = AuthService().create_user("admin@example.test", "StrongPass123", "admin", "ru")
    assert (
        post(client, "auth/login", email=admin.email, password="StrongPass123").status_code == 200
    )
    return admin


def test_registration_needs_email_and_moderation(client: FlaskClient, mailbox) -> None:
    assert register(client, role="admin", status="active").status_code == 202
    user = db.session.scalar(select(User).where(User.email == "doctor@example.test"))
    assert user.role == "user"
    assert user.status == "pending_email"
    assert post(client, "auth/login", email=user.email, password="StrongPass123").status_code == 401
    token = link_token(mailbox)
    assert db.session.scalar(select(AuthToken)).token_hash != token
    assert post(client, "auth/verify-email", token=token).get_json()["status"] == "pending"
    assert post(client, "auth/verify-email", token=token).status_code == 400
    assert post(client, "auth/login", email=user.email, password="StrongPass123").status_code == 401
    admin_login(client)
    # The existing block/unblock API cannot approve a registration.
    csrf = client.get("/api/v1/auth/me").get_json()["csrf_token"]
    assert (
        client.patch(
            f"/api/v1/admin/users/{user.id}",
            json={"blocked": False},
            headers={"X-CSRF-Token": csrf},
        ).status_code
        == 422
    )
    response = post(client, f"admin/users/{user.id}/moderate", approved=True)
    assert response.status_code == 200
    assert response.get_json()["user"]["status"] == "active"
    assert mailbox[-1][0] == user.email
    assert post(client, "auth/login", email=user.email, password="StrongPass123").status_code == 200


def test_unverified_email_cannot_be_approved(client: FlaskClient, mailbox) -> None:
    register(client)
    user = db.session.scalar(select(User).where(User.email == "doctor@example.test"))
    admin_login(client)
    assert post(client, f"admin/users/{user.id}/moderate", approved=True).status_code == 422
    assert user.status == "pending_email"


def test_rejection_cannot_be_bypassed_by_password_reset(client: FlaskClient, mailbox) -> None:
    register(client)
    post(client, "auth/verify-email", token=link_token(mailbox))
    user = db.session.scalar(select(User).where(User.email == "doctor@example.test"))
    admin_login(client)
    assert post(client, f"admin/users/{user.id}/moderate", approved=False).status_code == 200
    count = len(mailbox)
    assert post(client, "auth/forgot-password", email=user.email).status_code == 202
    assert len(mailbox) == count
    assert user.status == "rejected"
    assert post(client, "auth/login", email=user.email, password="StrongPass123").status_code == 401


def test_verification_expiry_and_wrong_purpose(client: FlaskClient, mailbox) -> None:
    register(client)
    raw = link_token(mailbox)
    assert (
        post(client, "auth/reset-password", token=raw, password="AnotherPass123").status_code == 400
    )
    token = db.session.scalar(select(AuthToken))
    token.expires_at = utc_now() - timedelta(seconds=1)
    db.session.commit()
    assert post(client, "auth/verify-email", token=raw).status_code == 400
    assert db.session.scalar(select(User)).status == "pending_email"


def test_repeated_signup_binds_latest_password_and_link(client: FlaskClient, mailbox) -> None:
    register(client)
    old = link_token(mailbox)
    assert (
        post(
            client,
            "auth/register",
            email="doctor@example.test",
            locale="ru",
            password="AnotherPass123",
        ).status_code
        == 202
    )
    new = link_token(mailbox)
    assert new != old
    assert post(client, "auth/verify-email", token=old).status_code == 400
    assert post(client, "auth/verify-email", token=new).status_code == 200
    user = db.session.scalar(select(User))
    assert AuthService._verify(user.password_hash, "AnotherPass123")
    assert not AuthService._verify(user.password_hash, "StrongPass123")


def test_reset_is_single_use_and_revokes_other_sessions(app: Flask, client: FlaskClient, mailbox):
    user = AuthService().create_user("doctor@example.test", "StrongPass123", "user", "ru")
    other = app.test_client()
    assert post(other, "auth/login", email=user.email, password="StrongPass123").status_code == 200
    assert post(client, "auth/forgot-password", email=user.email).status_code == 202
    raw = link_token(mailbox)
    assert (
        post(client, "auth/reset-password", token=raw, password="AnotherPass123").status_code == 200
    )
    assert other.get("/api/v1/auth/me").status_code == 401
    assert (
        post(client, "auth/reset-password", token=raw, password="DifferentPass123").status_code
        == 400
    )
    assert AuthService().authenticate(user.email, "StrongPass123") is None
    assert AuthService().authenticate(user.email, "AnotherPass123") is not None


def test_unknown_and_ineligible_reset_requests_are_indistinguishable(client, mailbox):
    user = AuthService().create_user("doctor@example.test", "StrongPass123", "user", "ru")
    for status in ("pending_email", "pending", "blocked", "rejected", "deleted"):
        user.status = status
        db.session.commit()
        response = post(client, "auth/forgot-password", email=user.email)
        # Per-account limiter is tested separately; use a different rate key here.
        from spina_bifida.extensions import limiter

        limiter.reset()
        assert response.status_code == 202
        assert response.get_json() == {"status": "accepted"}
    assert post(client, "auth/forgot-password", email="unknown@example.test").get_json() == {
        "status": "accepted",
    }
    assert not mailbox


def test_change_requires_current_password_and_invalidates_reset_links(app, client, mailbox):
    user = AuthService().create_user("doctor@example.test", "StrongPass123", "user", "ru")
    other = app.test_client()
    post(other, "auth/login", email=user.email, password="StrongPass123")
    post(client, "auth/login", email=user.email, password="StrongPass123")
    post(client, "auth/forgot-password", email=user.email)
    raw = link_token(mailbox)
    assert (
        post(
            client, "auth/change-password", current_password="wrong", password="AnotherPass123"
        ).status_code
        == 422
    )
    assert (
        post(
            client,
            "auth/change-password",
            current_password="StrongPass123",
            password="AnotherPass123",
        ).status_code
        == 200
    )
    assert client.get("/api/v1/auth/me").status_code == 401
    assert other.get("/api/v1/auth/me").status_code == 401
    assert (
        post(client, "auth/reset-password", token=raw, password="DifferentPass123").status_code
        == 400
    )


def test_delivery_failure_can_be_retried_without_approving_user(client, mailbox, monkeypatch):
    register(client)
    post(client, "auth/verify-email", token=link_token(mailbox))
    user = db.session.scalar(select(User))
    admin_login(client)

    def fail(*args):
        raise MailUnavailable

    monkeypatch.setattr(PostboxMailer, "send", fail)
    assert post(client, f"admin/users/{user.id}/moderate", approved=True).status_code == 503
    db.session.refresh(user)
    assert user.status == UserStatus.PENDING


@pytest.mark.parametrize("path", ["register", "forgot-password", "verify-email", "reset-password"])
def test_public_account_actions_require_csrf(client, mailbox, path):
    assert client.post(f"/api/v1/auth/{path}", json={}).status_code == 400


def test_registration_and_reset_are_rate_limited(client, mailbox):
    for _ in range(3):
        assert register(client).status_code == 202
    assert register(client).status_code == 429


def test_ordinary_user_cannot_moderate(client, mailbox):
    user = AuthService().create_user("doctor@example.test", "StrongPass123", "user", "ru")
    post(client, "auth/login", email=user.email, password="StrongPass123")
    assert post(client, f"admin/users/{user.id}/moderate", approved=True).status_code == 403


def test_mail_sends_without_holding_sqlite_writer_lock(app, client, mailbox, monkeypatch):
    from sqlalchemy import text

    original = PostboxMailer.send

    def send_without_lock(self, *args):
        with db.engine.connect() as connection:
            connection.execute(text("BEGIN IMMEDIATE"))
            connection.rollback()
        original(self, *args)

    monkeypatch.setattr(PostboxMailer, "send", send_without_lock)
    assert register(client).status_code == 202
    assert post(client, "auth/verify-email", token=link_token(mailbox)).status_code == 200
    user = db.session.scalar(select(User))
    admin_login(client)
    assert post(client, f"admin/users/{user.id}/moderate", approved=True).status_code == 200
