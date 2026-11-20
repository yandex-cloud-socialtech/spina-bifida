from __future__ import annotations

from io import BytesIO
from pathlib import Path

from flask import Flask
from flask.testing import FlaskClient
from PIL import Image
from sqlalchemy import func, select

from spina_bifida.analyses.models import AnalysisJob, JobState
from spina_bifida.auth.service import AuthService
from spina_bifida.extensions import db

FIXTURES = Path(__file__).parents[1] / "fixtures" / "images"


def login(client: FlaskClient, email: str) -> str:
    token = client.get("/api/v1/auth/me").get_json()["csrf_token"]
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "StrongPass123"},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 200
    return response.get_json()["csrf_token"]


def accepted_headers(
    client: FlaskClient,
    email: str = "first@example.test",
) -> dict[str, str]:
    csrf_token = login(client, email)
    response = client.post(
        "/api/v1/auth/terms/accept",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    return {"X-CSRF-Token": csrf_token}


def multipart_file(name: str) -> tuple[BytesIO, str, str]:
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    return BytesIO((FIXTURES / name).read_bytes()), name, mime


def distinct_valid_files(count: int = 10) -> list[tuple[BytesIO, str, str]]:
    files = []
    for position in range(count):
        stream = BytesIO()
        Image.new(
            "RGB",
            (8, 8),
            (position * 20, position * 11, position * 7),
        ).save(stream, format="PNG")
        stream.seek(0)
        files.append((stream, f"frame-{position}.png", "image/png"))
    return files


def create_users() -> tuple[str, str]:
    db.create_all()
    service = AuthService()
    first = service.create_user("first@example.test", "StrongPass123", "user", "ru")
    second = service.create_user("second@example.test", "StrongPass123", "user", "ru")
    return first.id, second.id


def test_upload_returns_one_job_and_stable_partial_error(
    app: Flask,
    client: FlaskClient,
) -> None:
    first_id, _ = create_users()
    headers = accepted_headers(client)
    accepted = client.post(
        "/api/v1/analyses",
        data={"model_version": "v1", "frames": [multipart_file("valid.jpg")]},
        headers=headers,
    )
    accepted_payload = accepted.get_json()
    accepted_job = db.session.get(AnalysisJob, accepted_payload["jobs"][0]["id"])
    assert accepted_job is not None
    accepted_path = Path(accepted_job.assets[0].local_path)

    rejected = client.post(
        "/api/v1/analyses",
        data={
            "model_version": "v1",
            "frames": [multipart_file("valid.png"), multipart_file("corrupt.jpg")],
        },
        headers=headers,
    )

    assert accepted.status_code == 202
    assert accepted_payload["batch_id"] == accepted_job.batch_id
    assert accepted_payload["jobs"] == [
        {
            "id": accepted_job.id,
            "state": "queued",
            "frame_id": accepted_job.assets[0].frame_id,
            "batch_position": 0,
            "polling_url": f"/api/v1/analyses/{accepted_job.id}",
        },
    ]
    assert rejected.status_code == 422
    assert rejected.get_json()["error"]["code"] == "image_decode_failed"
    assert rejected.get_json()["error"]["params"]["file_index"] == 1
    assert db.session.get(AnalysisJob, accepted_job.id).owner_id == first_id  # type: ignore[union-attr]
    assert accepted_path.is_file()


def test_ten_file_upload_creates_one_ordered_batch(app: Flask, client: FlaskClient) -> None:
    create_users()

    created = client.post(
        "/api/v1/analyses",
        data={"model_version": "v2", "frames": distinct_valid_files()},
        headers=accepted_headers(client),
    )

    payload = created.get_json()
    jobs = list(db.session.scalars(select(AnalysisJob).order_by(AnalysisJob.batch_position)))
    assert created.status_code == 202
    assert len(payload["jobs"]) == 10
    assert [item["batch_position"] for item in payload["jobs"]] == list(range(10))
    assert {job.batch_id for job in jobs} == {payload["batch_id"]}
    assert [job.batch_position for job in jobs] == list(range(10))
    assert {job.model_version for job in jobs} == {"v2"}
    assert all(len(job.assets) == 1 for job in jobs)


def test_eleven_files_are_rejected_without_partial_batch(
    app: Flask,
    client: FlaskClient,
) -> None:
    create_users()
    files = distinct_valid_files()
    files.append(multipart_file("valid.png"))

    response = client.post(
        "/api/v1/analyses",
        data={"model_version": "v1", "frames": files},
        headers=accepted_headers(client),
    )

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "upload_count_invalid"
    assert db.session.scalar(select(func.count()).select_from(AnalysisJob)) == 0


def test_oversized_multipart_uses_the_json_error_envelope(
    app: Flask,
    client: FlaskClient,
) -> None:
    create_users()
    headers = accepted_headers(client)
    app.config["MAX_CONTENT_LENGTH"] = 64

    response = client.post(
        "/api/v1/analyses",
        data={"model_version": "v1", "frames": [multipart_file("valid.jpg")]},
        headers=headers,
    )

    payload = response.get_json()
    assert response.status_code == 413
    assert response.content_type == "application/json"
    assert payload["error"]["code"] == "request_too_large"
    assert payload["error"]["request_id"] == response.headers["X-Request-ID"]
    assert db.session.scalar(select(func.count()).select_from(AnalysisJob)) == 0


def test_corrupt_tenth_file_creates_no_rows_or_job_directories(
    app: Flask,
    client: FlaskClient,
) -> None:
    create_users()
    files = distinct_valid_files(9)
    files.append(multipart_file("corrupt.jpg"))

    response = client.post(
        "/api/v1/analyses",
        data={"model_version": "v1", "frames": files},
        headers=accepted_headers(client),
    )

    settings = app.extensions["settings"]
    assert response.status_code == 422
    assert response.get_json()["error"]["params"]["file_index"] == 9
    assert db.session.scalar(select(func.count()).select_from(AnalysisJob)) == 0
    assert not any(settings.storage.jobs_dir.iterdir())


def test_normalized_duplicate_tenth_file_rejects_whole_batch(
    app: Flask,
    client: FlaskClient,
) -> None:
    create_users()
    files = distinct_valid_files(9)
    first_bytes = files[0][0].getvalue()
    duplicate_stream = BytesIO()
    with Image.open(BytesIO(first_bytes)) as duplicate:
        duplicate.save(duplicate_stream, format="PNG", compress_level=0)
    duplicate_stream.seek(0)
    assert duplicate_stream.getvalue() != first_bytes
    files.append((duplicate_stream, "frame-9.png", "image/png"))

    response = client.post(
        "/api/v1/analyses",
        data={"model_version": "v1", "frames": files},
        headers=accepted_headers(client),
    )

    settings = app.extensions["settings"]
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "duplicate_image"
    assert response.get_json()["error"]["params"] == {"file_index": 9}
    assert db.session.scalar(select(func.count()).select_from(AnalysisJob)) == 0
    assert not any(settings.storage.jobs_dir.iterdir())


def test_invalid_model_version_and_unaccepted_terms_are_rejected(
    app: Flask,
    client: FlaskClient,
) -> None:
    create_users()
    csrf_token = login(client, "first@example.test")
    blocked = client.post(
        "/api/v1/analyses",
        data={"model_version": "v1", "frames": [multipart_file("valid.jpg")]},
        headers={"X-CSRF-Token": csrf_token},
    )
    client.post(
        "/api/v1/auth/terms/accept",
        headers={"X-CSRF-Token": csrf_token},
    )
    invalid = client.post(
        "/api/v1/analyses",
        data={"model_version": "v3", "frames": [multipart_file("valid.jpg")]},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert blocked.status_code == 403
    assert blocked.get_json()["error"]["code"] == "terms_acceptance_required"
    assert invalid.status_code == 422
    assert invalid.get_json()["error"]["code"] == "model_version_invalid"
    assert db.session.scalar(select(func.count()).select_from(AnalysisJob)) == 0


def test_assets_are_private_and_job_delete_records_tombstone(
    app: Flask,
) -> None:
    first_id, _second_id = create_users()
    owner_client = app.test_client()
    other_client = app.test_client()
    owner_headers = accepted_headers(owner_client)
    created = owner_client.post(
        "/api/v1/analyses",
        data={"model_version": "v1", "frames": [multipart_file("valid.jpg")]},
        headers=owner_headers,
    ).get_json()
    job = db.session.get(AnalysisJob, created["jobs"][0]["id"])
    assert job is not None and job.owner_id == first_id
    asset = job.assets[0]
    asset_path = Path(asset.local_path)

    own_asset = owner_client.get(f"/api/v1/analyses/{job.id}/assets/{asset.id}")
    accepted_headers(other_client, "second@example.test")
    hidden_asset = other_client.get(f"/api/v1/analyses/{job.id}/assets/{asset.id}")
    deleted = owner_client.delete(
        f"/api/v1/analyses/{job.id}",
        headers=owner_headers,
    )

    assert own_asset.status_code == 200
    assert own_asset.headers["Cache-Control"] == "private, no-store"
    assert own_asset.headers["X-Content-Type-Options"] == "nosniff"
    assert hidden_asset.status_code == 404
    assert hidden_asset.get_json()["error"]["code"] == "analysis_not_found"
    assert deleted.status_code == 204
    refreshed = db.session.get(AnalysisJob, job.id)
    assert refreshed is not None and refreshed.state == JobState.DELETED
    assert refreshed.deleted_at is not None
    assert not asset_path.exists()


def test_batch_delete_is_owner_or_admin_scoped_and_preserves_sibling_batch(
    app: Flask,
) -> None:
    first_id, _second_id = create_users()
    AuthService().create_user("admin@example.test", "StrongPass123", "admin", "ru")
    owner_client = app.test_client()
    other_client = app.test_client()
    admin_client = app.test_client()
    owner_headers = accepted_headers(owner_client)
    other_headers = accepted_headers(other_client, "second@example.test")
    admin_headers = accepted_headers(admin_client, "admin@example.test")
    first_batch = owner_client.post(
        "/api/v1/analyses",
        data={"model_version": "v1", "frames": distinct_valid_files(2)},
        headers=owner_headers,
    ).get_json()
    sibling_batch = owner_client.post(
        "/api/v1/analyses",
        data={"model_version": "v2", "frames": [multipart_file("valid.png")]},
        headers=owner_headers,
    ).get_json()

    hidden = other_client.delete(
        f"/api/v1/analyses/batches/{first_batch['batch_id']}",
        headers=other_headers,
    )
    deleted = admin_client.delete(
        f"/api/v1/analyses/batches/{first_batch['batch_id']}",
        headers=admin_headers,
    )

    assert hidden.status_code == 404
    assert deleted.status_code == 204
    db.session.expire_all()
    first_jobs = list(
        db.session.scalars(
            select(AnalysisJob).where(AnalysisJob.batch_id == first_batch["batch_id"]),
        ),
    )
    sibling = db.session.scalar(
        select(AnalysisJob).where(
            AnalysisJob.batch_id == sibling_batch["batch_id"],
        ),
    )
    assert {job.owner_id for job in first_jobs} == {first_id}
    assert {job.state for job in first_jobs} == {JobState.DELETED}
    assert all(job.deleted_at is not None and not job.assets for job in first_jobs)
    assert sibling is not None and sibling.state == JobState.QUEUED

    owner_deleted = owner_client.delete(
        f"/api/v1/analyses/batches/{sibling_batch['batch_id']}",
        headers=owner_headers,
    )
    assert owner_deleted.status_code == 204
    db.session.expire_all()
    refreshed_sibling = db.session.get(AnalysisJob, sibling.id)
    assert refreshed_sibling is not None and refreshed_sibling.state == JobState.DELETED


def test_asset_and_batch_delete_require_terms(app: Flask) -> None:
    create_users()
    owner_client = app.test_client()
    headers = accepted_headers(owner_client)
    created = owner_client.post(
        "/api/v1/analyses",
        data={"model_version": "v1", "frames": [multipart_file("valid.jpg")]},
        headers=headers,
    ).get_json()
    job = db.session.get(AnalysisJob, created["jobs"][0]["id"])
    assert job is not None
    asset_id = job.assets[0].id
    logged_out = owner_client.post("/api/v1/auth/logout", headers=headers)
    assert logged_out.status_code == 204
    unaccepted_token = login(owner_client, "first@example.test")

    asset = owner_client.get(f"/api/v1/analyses/{job.id}/assets/{asset_id}")
    batch_delete = owner_client.delete(
        f"/api/v1/analyses/batches/{created['batch_id']}",
        headers={"X-CSRF-Token": unaccepted_token},
    )

    assert asset.status_code == 403
    assert batch_delete.status_code == 403
    assert asset.get_json()["error"]["code"] == "terms_acceptance_required"
    assert batch_delete.get_json()["error"]["code"] == "terms_acceptance_required"
