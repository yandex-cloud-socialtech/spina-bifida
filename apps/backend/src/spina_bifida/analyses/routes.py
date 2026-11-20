from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import cast

from flask import Blueprint, Response, current_app, g, jsonify, request, send_file
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from spina_bifida.analyses.service import (
    AnalysisNotFound,
    AnalysisService,
    InvalidFeedback,
    LocalStorageFull,
    RetryNotAllowed,
    UploadBatchRejected,
)
from spina_bifida.api.errors import ApiError
from spina_bifida.auth.models import User
from spina_bifida.auth.routes import csrf_required, login_required, terms_required
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db

analyses_blueprint = Blueprint("analyses", __name__, url_prefix="/api/v1/analyses")
system_blueprint = Blueprint("system", __name__, url_prefix="/api/v1/system")


def _service() -> AnalysisService:
    settings = current_app.extensions["settings"]
    if not isinstance(settings, AppSettings):
        raise RuntimeError("Application settings are unavailable")
    return AnalysisService(settings)


def _current_user() -> User:
    return cast(User, g.current_user)


@analyses_blueprint.post("")
@login_required
@terms_required
@csrf_required
def create_analysis() -> tuple[Response, int]:
    uploads = request.files.getlist("frames") or request.files.getlist("files")
    try:
        jobs = _service().create_batch(
            _current_user(),
            uploads,
            request.form.get("model_version", ""),
        )
    except UploadBatchRejected as error:
        raise ApiError(error.code, error.status, error.params) from error
    except LocalStorageFull as error:
        raise ApiError(
            error.code,
            507,
            {"remediation": "free_local_storage_or_reduce_retention"},
        ) from error
    return (
        jsonify(
            {
                "batch_id": jobs[0].batch_id,
                "jobs": [
                    {
                        "id": job.id,
                        "state": job.state.value,
                        "frame_id": job.assets[0].frame_id,
                        "batch_position": job.batch_position,
                        "polling_url": f"/api/v1/analyses/{job.id}",
                    }
                    for job in jobs
                ],
            },
        ),
        202,
    )


@analyses_blueprint.get("/<job_id>")
@login_required
@terms_required
def get_analysis(job_id: str) -> Response:
    try:
        payload = _service().polling_payload(_current_user(), job_id)
    except AnalysisNotFound as error:
        raise ApiError("analysis_not_found", 404) from error
    return jsonify(payload)


@analyses_blueprint.post("/<job_id>/retry")
@login_required
@terms_required
@csrf_required
def retry_analysis(job_id: str) -> tuple[Response, int]:
    try:
        job = _service().retry_job(_current_user(), job_id)
    except AnalysisNotFound as error:
        raise ApiError("analysis_not_found", 404) from error
    except RetryNotAllowed as error:
        raise ApiError("retry_not_allowed", 409) from error
    return jsonify({"id": job.id, "state": job.state.value, "attempt": job.attempt}), 202


@analyses_blueprint.post("/<job_id>/feedback")
@login_required
@terms_required
@csrf_required
def submit_feedback(job_id: str) -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError("invalid_feedback", 422)
    try:
        feedback = _service().submit_feedback(
            _current_user(),
            job_id,
            str(payload.get("value", "")),
            payload.get("comment"),
        )
    except AnalysisNotFound as error:
        raise ApiError("analysis_not_found", 404) from error
    except InvalidFeedback as error:
        raise ApiError("invalid_feedback", 422, {"field": error.field}) from error
    return jsonify(
        {
            "feedback": {
                "value": feedback.value.value,
                "comment": feedback.comment,
            },
        },
    )


@analyses_blueprint.get("/<job_id>/assets/<asset_id>")
@login_required
@terms_required
def get_asset(job_id: str, asset_id: str) -> Response:
    try:
        stream = _service().open_asset(_current_user(), job_id, asset_id)
    except AnalysisNotFound as error:
        raise ApiError("analysis_not_found", 404) from error
    response = send_file(stream, mimetype="image/png", download_name=f"{asset_id}.png")
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@analyses_blueprint.delete("/<job_id>")
@login_required
@terms_required
@csrf_required
def delete_analysis(job_id: str) -> tuple[str, int]:
    try:
        _service().delete_job(_current_user(), job_id)
    except AnalysisNotFound as error:
        raise ApiError("analysis_not_found", 404) from error
    return "", 204


@analyses_blueprint.delete("/batches/<batch_id>")
@login_required
@terms_required
@csrf_required
def delete_analysis_batch(batch_id: str) -> tuple[str, int]:
    try:
        _service().delete_batch(_current_user(), batch_id)
    except AnalysisNotFound as error:
        raise ApiError("analysis_not_found", 404) from error
    return "", 204


@system_blueprint.get("/info")
@login_required
def system_info() -> Response:
    settings = current_app.extensions["settings"]
    if not isinstance(settings, AppSettings):
        raise RuntimeError("Application settings are unavailable")
    try:
        schema_version = db.session.execute(
            text("SELECT version_num FROM alembic_version"),
        ).scalar_one()
    except SQLAlchemyError:
        db.session.rollback()
        schema_version = "uninitialized"
    try:
        application_version = version("spina-bifida-backend")
    except PackageNotFoundError:
        application_version = "0.1.0"
    return jsonify(
        {
            "application_version": application_version,
            "model_versions": ["v1", "v2"],
            "default_model_version": settings.inference.default_model_version,
            "schema_version": schema_version,
            "device": settings.inference.device,
        },
    )
