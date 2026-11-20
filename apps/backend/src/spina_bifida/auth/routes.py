from __future__ import annotations

import secrets
from collections.abc import Callable
from functools import wraps
from hashlib import sha256
from typing import Any, cast

from flask import Blueprint, Response, g, jsonify, request, session
from flask_limiter.util import get_remote_address

from spina_bifida.api.errors import ApiError, error_response
from spina_bifida.auth.accounts import AccountService, InvalidCurrentPassword, InvalidToken
from spina_bifida.auth.mail import MailUnavailable
from spina_bifida.auth.models import User, UserRole
from spina_bifida.auth.service import AuthService, InvalidUserField
from spina_bifida.extensions import limiter
from spina_bifida.observability.audit import audit

auth_blueprint = Blueprint("auth", __name__, url_prefix="/api/v1/auth")
TERMS_VERSION = "2026-09-22"


def _new_csrf_token() -> str:
    token = secrets.token_urlsafe(32)
    session["csrf_token"] = token
    return token


def current_csrf_token() -> str:
    token = session.get("csrf_token")
    if not isinstance(token, str):
        return _new_csrf_token()
    return token


def user_payload(user: User) -> dict[str, str]:
    return {
        "id": user.id,
        "email": user.email,
        "role": str(user.role),
        "status": str(user.status),
        "locale": str(user.locale),
    }


def csrf_required[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        expected = session.get("csrf_token")
        supplied = request.headers.get("X-CSRF-Token")
        if (
            not isinstance(expected, str)
            or not supplied
            or not secrets.compare_digest(expected, supplied)
        ):
            raise ApiError("csrf_failed", 400)
        return function(*args, **kwargs)

    return wrapped


def login_required[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        if getattr(g, "current_user", None) is None:
            raise ApiError("authentication_required", 401)
        return function(*args, **kwargs)

    return wrapped


def terms_accepted() -> bool:
    return session.get("terms_version") == TERMS_VERSION


def terms_required[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        if not terms_accepted():
            raise ApiError("terms_acceptance_required", 403)
        return function(*args, **kwargs)

    return wrapped


def admin_required[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        user = cast(User | None, getattr(g, "current_user", None))
        if user is None:
            raise ApiError("authentication_required", 401)
        if user.role != UserRole.ADMIN:
            raise ApiError("admin_required", 403)
        return function(*args, **kwargs)

    return wrapped


def load_current_user() -> None:
    user_id = session.get("user_id")
    if not isinstance(user_id, str):
        g.current_user = None
        return
    user = AuthService().get_active_user(user_id)
    if user is None or session.get("password_version", 1) != user.password_version:
        session.clear()
        user = None
    g.current_user = user


def _login_rate_key() -> str:
    payload = request.get_json(silent=True)
    email = payload.get("email", "") if isinstance(payload, dict) else ""
    normalized_email = str(email).strip().casefold()
    return f"{get_remote_address()}:{normalized_email}"


@auth_blueprint.post("/login")
@limiter.limit("5 per minute", key_func=_login_rate_key)
@limiter.limit("20 per hour", key_func=_login_rate_key)
@csrf_required
def login() -> Response:
    payload: Any = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError("invalid_request", 422)
    email = payload.get("email")
    password = payload.get("password")
    if not isinstance(email, str) or not isinstance(password, str):
        raise ApiError("invalid_request", 422)
    user = AuthService().authenticate(email, password)
    if user is None:
        raise ApiError("invalid_credentials", 401)
    session.clear()
    session["user_id"] = user.id
    session["password_version"] = user.password_version
    token = _new_csrf_token()
    return jsonify(
        {
            "user": user_payload(user),
            "csrf_token": token,
            "terms_accepted": False,
            "terms_version": TERMS_VERSION,
        },
    )


@auth_blueprint.post("/logout")
@login_required
@csrf_required
def logout() -> tuple[str, int]:
    session.clear()
    return "", 204


@auth_blueprint.post("/terms/accept")
@login_required
@csrf_required
def accept_terms() -> Response:
    current_user = cast(User, g.current_user)
    session["terms_version"] = TERMS_VERSION
    audit("terms_accepted", current_user.id, terms_version=TERMS_VERSION)
    return jsonify({"terms_accepted": True, "terms_version": TERMS_VERSION})


@auth_blueprint.patch("/preferences")
@login_required
@csrf_required
def update_preferences() -> Response:
    payload: Any = request.get_json(silent=True)
    locale = payload.get("locale") if isinstance(payload, dict) else None
    if not isinstance(locale, str):
        raise ApiError("invalid_user", 422, {"field": "locale"})
    current_user = cast(User, g.current_user)
    try:
        user = AuthService().update_user(
            current_user.id,
            locale=locale,
            actor_id=current_user.id,
        )
    except InvalidUserField as error:
        raise ApiError("invalid_user", 422, {"field": "locale"}) from error
    return jsonify({"user": user_payload(user)})


@auth_blueprint.get("/me")
def me() -> Response:
    token = current_csrf_token()
    user = cast(User | None, getattr(g, "current_user", None))
    if user is None:
        raise ApiError(
            "authentication_required",
            401,
            extra={"csrf_token": token},
        )
    return jsonify(
        {
            "user": user_payload(user),
            "csrf_token": token,
            "terms_accepted": terms_accepted(),
            "terms_version": TERMS_VERSION,
        },
    )


def _string_fields(*names: str) -> list[str]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or any(not isinstance(payload.get(n), str) for n in names):
        raise ApiError("invalid_request", 422)
    return [payload[n] for n in names]


@auth_blueprint.errorhandler(InvalidToken)
def invalid_token(error: InvalidToken) -> tuple[Response, int]:
    return error_response("invalid_token", 400)


@auth_blueprint.errorhandler(InvalidCurrentPassword)
def invalid_password(error: InvalidCurrentPassword) -> tuple[Response, int]:
    return error_response("current_password_invalid", 422)


@auth_blueprint.errorhandler(InvalidUserField)
def invalid_field(error: InvalidUserField) -> tuple[Response, int]:
    return error_response("invalid_user", 422, {"field": str(error)})


@auth_blueprint.errorhandler(MailUnavailable)
def mail_unavailable(error: MailUnavailable) -> tuple[Response, int]:
    return error_response("mail_unavailable", 503)


def _mail_rate_key() -> str:
    payload = request.get_json(silent=True)
    email = payload.get("email", "") if isinstance(payload, dict) else ""
    return sha256(str(email).strip().casefold().encode()).hexdigest()


@auth_blueprint.post("/register")
@limiter.limit("10 per hour")
@limiter.limit("3 per hour", key_func=_mail_rate_key)
@csrf_required
def register() -> tuple[Response, int]:
    email, password, locale = _string_fields("email", "password", "locale")
    AccountService().register(email, password, locale)
    return jsonify({"status": "accepted"}), 202


@auth_blueprint.post("/verify-email")
@limiter.limit("20 per hour")
@csrf_required
def verify_email() -> Response:
    (token,) = _string_fields("token")
    AccountService().verify_email(token)
    return jsonify({"status": "pending"})


@auth_blueprint.post("/forgot-password")
@limiter.limit("10 per hour")
@limiter.limit("3 per hour", key_func=_mail_rate_key)
@csrf_required
def forgot_password() -> tuple[Response, int]:
    (email,) = _string_fields("email")
    AccountService().request_reset(email)
    return jsonify({"status": "accepted"}), 202


@auth_blueprint.post("/reset-password")
@limiter.limit("10 per hour")
@csrf_required
def reset_password() -> Response:
    token, password = _string_fields("token", "password")
    AccountService().reset_password(token, password)
    session.clear()
    return jsonify({"status": "password_changed", "csrf_token": _new_csrf_token()})


@auth_blueprint.post("/change-password")
@limiter.limit("5 per minute")
@login_required
@csrf_required
def change_password() -> Response:
    current_password, password = _string_fields("current_password", "password")
    AccountService().change_password(cast(User, g.current_user), current_password, password)
    session.clear()
    return jsonify({"status": "password_changed", "csrf_token": _new_csrf_token()})
