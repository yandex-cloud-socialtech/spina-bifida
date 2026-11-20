import pytest
from flask import Flask

from spina_bifida.auth.models import UserStatus
from spina_bifida.auth.service import (
    AuthService,
    DuplicateEmail,
    InvalidUserField,
    LastActiveAdmin,
    SelfBlockForbidden,
)
from spina_bifida.extensions import db


@pytest.fixture
def auth_service(app: Flask) -> AuthService:
    db.create_all()
    return AuthService()


def test_passwords_are_argon2id(auth_service: AuthService) -> None:
    user = auth_service.create_user("doctor@example.test", "StrongPass123", "user", "ru")

    assert user.password_hash.startswith("$argon2id$")
    assert auth_service.authenticate(user.email, "StrongPass123") == user


def test_email_is_normalized_and_duplicate_is_rejected(auth_service: AuthService) -> None:
    user = auth_service.create_user(" Doctor@Example.Test ", "StrongPass123", "user", "ru")

    assert user.email == "doctor@example.test"
    with pytest.raises(DuplicateEmail):
        auth_service.create_user("DOCTOR@example.test", "AnotherPass123", "user", "ru")


@pytest.mark.parametrize(
    ("role", "locale"),
    [("owner", "ru"), ("user", "de")],
)
def test_role_and_locale_are_closed_enums(
    auth_service: AuthService,
    role: str,
    locale: str,
) -> None:
    with pytest.raises(InvalidUserField):
        auth_service.create_user("doctor@example.test", "StrongPass123", role, locale)


def test_blocked_user_cannot_authenticate_on_next_check(auth_service: AuthService) -> None:
    user = auth_service.create_user("doctor@example.test", "StrongPass123", "user", "ru")

    auth_service.block_user(user.id, blocked=True)

    assert auth_service.authenticate(user.email, "StrongPass123") is None
    assert db.session.get(type(user), user.id).status == UserStatus.BLOCKED  # type: ignore[union-attr]


def test_delete_anonymizes_user_and_clears_password(auth_service: AuthService) -> None:
    user = auth_service.create_user("doctor@example.test", "StrongPass123", "user", "ru")

    deleted = auth_service.delete_user(user.id)

    assert deleted.status == UserStatus.DELETED
    assert deleted.email.startswith("deleted-")
    assert deleted.email.endswith("@invalid.local")
    assert deleted.password_hash == ""


def test_last_active_admin_cannot_be_deleted(auth_service: AuthService) -> None:
    admin = auth_service.create_user("admin@example.test", "StrongPass123", "admin", "ru")

    with pytest.raises(LastActiveAdmin):
        auth_service.delete_user(admin.id)


def test_deleted_users_do_not_appear_in_admin_list(auth_service: AuthService) -> None:
    admin = auth_service.create_user("admin@example.test", "StrongPass123", "admin", "ru")
    doctor = auth_service.create_user("doctor@example.test", "StrongPass123", "user", "ru")
    auth_service.delete_user(doctor.id)
    assert [user.id for user in auth_service.list_users()] == [admin.id]


@pytest.mark.parametrize("method", ["block_user", "update_user"])
@pytest.mark.parametrize("has_second_admin", [False, True])
def test_service_rejects_self_blocking(
    auth_service: AuthService, method: str, has_second_admin: bool,
) -> None:
    admin = auth_service.create_user("admin@example.test", "StrongPass123", "admin", "ru")
    if has_second_admin:
        auth_service.create_user("other@example.test", "StrongPass123", "admin", "ru")

    with pytest.raises(SelfBlockForbidden):
        getattr(auth_service, method)(admin.id, blocked=True, actor_id=admin.id)

    assert admin.status == UserStatus.ACTIVE


def test_admin_can_update_own_locale_without_blocking(auth_service: AuthService) -> None:
    admin = auth_service.create_user("admin@example.test", "StrongPass123", "admin", "ru")

    updated = auth_service.update_user(admin.id, locale="en", blocked=False, actor_id=admin.id)

    assert updated.locale == "en"
    assert updated.status == UserStatus.ACTIVE


@pytest.mark.parametrize("method", ["block_user", "update_user"])
def test_last_active_admin_blocking_stays_protected(
    auth_service: AuthService, method: str,
) -> None:
    admin = auth_service.create_user("admin@example.test", "StrongPass123", "admin", "ru")

    with pytest.raises(LastActiveAdmin):
        getattr(auth_service, method)(admin.id, blocked=True)

    assert admin.status == UserStatus.ACTIVE
