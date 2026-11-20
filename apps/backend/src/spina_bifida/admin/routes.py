from __future__ import annotations

from typing import Any, cast

from flask import Blueprint, Response, g, jsonify, request

from spina_bifida.api.errors import ApiError
from spina_bifida.auth.accounts import AccountService
from spina_bifida.auth.mail import MailUnavailable
from spina_bifida.auth.models import User
from spina_bifida.auth.routes import admin_required, csrf_required, user_payload
from spina_bifida.auth.service import (
    AuthService,
    DuplicateEmail,
    InvalidUserField,
    LastActiveAdmin,
    SelfBlockForbidden,
    UserNotFound,
)

admin_blueprint = Blueprint("admin", __name__, url_prefix="/api/v1/admin")


def _actor_id() -> str:
    return cast(User, g.current_user).id


def _translate_service_error(error: Exception) -> ApiError:
    if isinstance(error, DuplicateEmail):
        return ApiError("email_exists", 409)
    if isinstance(error, SelfBlockForbidden):
        return ApiError("self_block_forbidden", 409)
    if isinstance(error, LastActiveAdmin):
        return ApiError("last_active_admin", 409)
    if isinstance(error, UserNotFound):
        return ApiError("user_not_found", 404)
    if isinstance(error, InvalidUserField):
        return ApiError("invalid_user", 422, {"field": str(error)})
    return ApiError("invalid_request", 422)


@admin_blueprint.get("/users")
@admin_required
def list_users() -> Response:
    return jsonify({"users": [user_payload(user) for user in AuthService().list_users()]})


@admin_blueprint.post("/users")
@admin_required
@csrf_required
def create_user() -> tuple[Response, int]:
    payload: Any = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError("invalid_request", 422)
    try:
        user = AuthService().create_user(
            str(payload.get("email", "")),
            str(payload.get("password", "")),
            str(payload.get("role", "")),
            str(payload.get("locale", "")),
            actor_id=_actor_id(),
        )
    except (DuplicateEmail, InvalidUserField) as error:
        raise _translate_service_error(error) from error
    return jsonify({"user": user_payload(user)}), 201


@admin_blueprint.patch("/users/<user_id>")
@admin_required
@csrf_required
def update_user(user_id: str) -> Response:
    payload: Any = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError("invalid_request", 422)
    blocked = payload.get("blocked")
    if blocked is not None and not isinstance(blocked, bool):
        raise ApiError("invalid_user", 422, {"field": "blocked"})
    role = payload.get("role")
    locale = payload.get("locale")
    if role is not None and not isinstance(role, str):
        raise ApiError("invalid_user", 422, {"field": "role"})
    if locale is not None and not isinstance(locale, str):
        raise ApiError("invalid_user", 422, {"field": "locale"})
    try:
        user = AuthService().update_user(
            user_id,
            role=role,
            locale=locale,
            blocked=blocked,
            actor_id=_actor_id(),
        )
    except (InvalidUserField, LastActiveAdmin, SelfBlockForbidden, UserNotFound) as error:
        raise _translate_service_error(error) from error
    return jsonify({"user": user_payload(user)})


@admin_blueprint.delete("/users/<user_id>")
@admin_required
@csrf_required
def delete_user(user_id: str) -> Response:
    try:
        user = AuthService().delete_user(user_id, actor_id=_actor_id())
    except (LastActiveAdmin, UserNotFound) as error:
        raise _translate_service_error(error) from error
    return jsonify({"user": user_payload(user)})


@admin_blueprint.post("/users/<user_id>/moderate")
@admin_required
@csrf_required
def moderate_user(user_id: str) -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("approved"), bool):
        raise ApiError("invalid_request", 422)
    try:
        user = AccountService().moderate(user_id, payload["approved"], _actor_id())
    except (InvalidUserField, UserNotFound) as error:
        raise _translate_service_error(error) from error
    except MailUnavailable as error:
        raise ApiError("mail_unavailable", 503) from error
    return jsonify({"user": user_payload(user)})
