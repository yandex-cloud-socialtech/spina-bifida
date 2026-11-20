from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from flask import Flask, Response, g, jsonify
from flask_limiter.errors import RateLimitExceeded
from werkzeug.exceptions import BadRequest, MethodNotAllowed, NotFound, RequestEntityTooLarge


@dataclass(slots=True)
class ApiError(Exception):
    code: str
    status: int
    params: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)


def error_response(
    code: str,
    status: int,
    params: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> tuple[Response, int]:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "params": params or {},
            "request_id": getattr(g, "request_id", "unknown"),
        },
    }
    if extra:
        payload.update(extra)
    return jsonify(payload), status


def register_error_handlers(app: Flask) -> None:
    @app.before_request
    def assign_request_id() -> None:
        g.request_id = str(uuid4())

    @app.after_request
    def attach_request_id(response: Response) -> Response:
        response.headers["X-Request-ID"] = g.request_id
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError) -> tuple[Response, int]:
        return error_response(error.code, error.status, error.params, error.extra)

    @app.errorhandler(RateLimitExceeded)
    def handle_rate_limit(_error: RateLimitExceeded) -> tuple[Response, int]:
        return error_response("rate_limited", 429, {"retry_after": 60})

    @app.errorhandler(BadRequest)
    def handle_bad_request(_error: BadRequest) -> tuple[Response, int]:
        return error_response("bad_request", 400)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_entity_too_large(
        _error: RequestEntityTooLarge,
    ) -> tuple[Response, int]:
        return error_response("request_too_large", 413)

    @app.errorhandler(NotFound)
    def handle_not_found(_error: NotFound) -> tuple[Response, int]:
        return error_response("not_found", 404)

    @app.errorhandler(MethodNotAllowed)
    def handle_method_not_allowed(_error: MethodNotAllowed) -> tuple[Response, int]:
        return error_response("method_not_allowed", 405)
