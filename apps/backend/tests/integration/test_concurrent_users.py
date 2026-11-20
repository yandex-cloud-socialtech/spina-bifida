from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from io import BytesIO

from flask import Flask
from flask.testing import FlaskClient
from PIL import Image

from spina_bifida.analyses.models import AnalysisJob, JobState
from spina_bifida.analyses.repository import JobRepository
from spina_bifida.auth.service import AuthService
from spina_bifida.config import AppSettings
from spina_bifida.extensions import db

PASSWORD = "StrongPass123"


def distinct_frames(email: str, count: int = 2) -> list[tuple[BytesIO, str, str]]:
    frames = []
    for position in range(count):
        stream = BytesIO()
        Image.new("RGB", (8, 8), (position * 70, position * 40, 120)).save(
            stream,
            format="PNG",
        )
        stream.seek(0)
        frames.append((stream, f"{email}-{position}.png", "image/png"))
    return frames


def submit_jobs(app: Flask, email: str) -> list[str]:
    with app.test_client() as client:
        token = client.get("/api/v1/auth/me").get_json()["csrf_token"]
        login = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": PASSWORD},
            headers={"X-CSRF-Token": token},
        )
        csrf = login.get_json()["csrf_token"]
        accepted = client.post(
            "/api/v1/auth/terms/accept",
            headers={"X-CSRF-Token": csrf},
        )
        assert accepted.status_code == 200
        created = client.post(
            "/api/v1/analyses",
            data={
                "model_version": "v1",
                "frames": distinct_frames(email),
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert created.status_code == 202
        payload = created.get_json()
        assert [job["batch_position"] for job in payload["jobs"]] == [0, 1]
        return [str(job["id"]) for job in payload["jobs"]]


def login(app: Flask, email: str) -> FlaskClient:
    client = app.test_client()
    token = client.get("/api/v1/auth/me").get_json()["csrf_token"]
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": PASSWORD},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 200
    csrf = response.get_json()["csrf_token"]
    accepted = client.post(
        "/api/v1/auth/terms/accept",
        headers={"X-CSRF-Token": csrf},
    )
    assert accepted.status_code == 200
    return client


def test_three_users_share_one_fifo_worker_for_independent_batch_jobs_without_asset_leakage(
    app: Flask,
) -> None:
    db.create_all()
    emails = [f"doctor-{index}@example.test" for index in range(3)]
    users = [AuthService().create_user(email, PASSWORD, "user", "ru") for email in emails]

    with ThreadPoolExecutor(max_workers=3) as executor:
        job_ids_by_owner = list(executor.map(lambda email: submit_jobs(app, email), emails))
    job_ids = [job_id for owner_jobs in job_ids_by_owner for job_id in owner_jobs]

    assert len(set(job_ids)) == 6
    jobs = list(db.session.query(AnalysisJob).filter(AnalysisJob.id.in_(job_ids)).all())
    assert len(jobs) == 6
    assert {job.owner_id for job in jobs} == {user.id for user in users}
    assert all(job.state == JobState.QUEUED for job in jobs)
    assert all(job.batch_position in {0, 1} for job in jobs)
    assert len({job.batch_id for job in jobs}) == 3

    settings = app.extensions["settings"]
    assert isinstance(settings, AppSettings)
    assert settings.inference.max_parallel_jobs == 1
    expected_order = [job.id for job in sorted(jobs, key=lambda job: (job.created_at, job.id))]

    repository = JobRepository()
    completion_order: list[str] = []
    for index in range(6):
        claimed = repository.claim_next("worker-1", datetime.now(UTC))
        assert claimed is not None
        processing = db.session.query(AnalysisJob).filter(
            AnalysisJob.state.notin_([JobState.QUEUED, JobState.COMPLETED]),
        ).all()
        queued = db.session.query(AnalysisJob).filter(AnalysisJob.state == JobState.QUEUED).all()
        assert len(processing) == settings.inference.max_parallel_jobs
        assert len(queued) == 5 - index
        repository.complete(claimed.id, {"frames": []}, datetime.now(UTC))
        completion_order.append(claimed.id)

    assert completion_order == expected_order

    clients = [login(app, email) for email in emails]
    for owner_index, owner_jobs in enumerate(job_ids_by_owner):
        for job_id in owner_jobs:
            job = db.session.get(AnalysisJob, job_id)
            assert job is not None
            asset_id = job.assets[0].id
            for client_index, client in enumerate(clients):
                response = client.get(f"/api/v1/analyses/{job_id}/assets/{asset_id}")
                assert response.status_code == (200 if client_index == owner_index else 404)
