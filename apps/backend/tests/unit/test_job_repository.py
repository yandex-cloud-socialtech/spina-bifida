from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest
from flask import Flask
from sqlalchemy import event, inspect, select, update
from sqlalchemy.orm import Session, sessionmaker

from spina_bifida.analyses import schemas
from spina_bifida.analyses.models import (
    AnalysisJob,
    JobState,
    ProcessingStage,
    RetentionClass,
)
from spina_bifida.analyses.repository import JobNoLongerProcessable, JobRepository
from spina_bifida.analyses.schemas import NewFrame
from spina_bifida.auth.models import User
from spina_bifida.extensions import db


def make_user(email: str = "doctor@example.test") -> User:
    user = User(
        email=email,
        password_hash="not-used-in-repository-tests",
        role="user",
        status="active",
        locale="ru",
    )
    db.session.add(user)
    db.session.commit()
    return user


def frame(tmp_path: Path, name: str = "frame.png") -> NewFrame:
    path = tmp_path / name
    path.write_bytes(b"normalized-image")
    return NewFrame(frame_id=name.removesuffix(".png"), path=path, checksum="a" * 64, position=0)


def batch_job(
    tmp_path: Path,
    *,
    job_id: str,
    batch_id: str,
    batch_position: int,
    model_version: Literal["v1", "v2"] = "v1",
) -> schemas.NewAnalysisJob:
    return schemas.NewAnalysisJob(
        job_id=job_id,
        batch_id=batch_id,
        batch_position=batch_position,
        model_version=model_version,
        frame=frame(tmp_path, f"{job_id}.png"),
    )


def test_create_batch_persists_one_ordered_job_per_frame(
    app: Flask,
    tmp_path: Path,
) -> None:
    db.create_all()
    owner = make_user()
    jobs = [
        batch_job(
            tmp_path,
            job_id=f"job-{position}",
            batch_id="batch-1",
            batch_position=position,
            model_version="v2",
        )
        for position in range(3)
    ]

    created = JobRepository().create_batch(
        owner.id,
        jobs,
        expires_at=datetime.now(UTC) + timedelta(hours=24),
        retention_class=RetentionClass.TEMPORARY,
    )

    assert [job.id for job in created] == ["job-0", "job-1", "job-2"]
    assert {job.batch_id for job in created} == {"batch-1"}
    assert [job.batch_position for job in created] == [0, 1, 2]
    assert {job.model_version for job in created} == {"v2"}
    assert all(len(job.assets) == 1 for job in created)
    assert all(
        later.created_at - earlier.created_at == timedelta(microseconds=1)
        for earlier, later in zip(created[:-1], created[1:], strict=True)
    )


def test_claim_next_uses_one_fifo_for_same_and_cross_owner_batches(
    app: Flask,
    tmp_path: Path,
) -> None:
    db.create_all()
    first_owner = make_user("first@example.test")
    second_owner = make_user("second@example.test")
    repository = JobRepository()
    expires_at = datetime.now(UTC) + timedelta(hours=24)
    first_batch = repository.create_batch(
        first_owner.id,
        [
            batch_job(
                tmp_path,
                job_id=f"first-{position}",
                batch_id="first-batch",
                batch_position=position,
            )
            for position in range(2)
        ],
        expires_at=expires_at,
        retention_class=RetentionClass.TEMPORARY,
    )
    second_batch = repository.create_batch(
        second_owner.id,
        [
            batch_job(
                tmp_path,
                job_id="second-0",
                batch_id="second-batch",
                batch_position=0,
            ),
        ],
        expires_at=expires_at,
        retention_class=RetentionClass.TEMPORARY,
    )

    claimed_ids = []
    for position in range(3):
        claimed = repository.claim_next("worker", datetime.now(UTC))
        assert claimed is not None
        claimed_ids.append(claimed.id)
        repository.complete(claimed.id, {"position": position}, datetime.now(UTC))

    assert claimed_ids == [job.id for job in (*first_batch, *second_batch)]


def test_deleted_claimed_job_cannot_resurrect_or_change_sibling(
    app: Flask,
    tmp_path: Path,
) -> None:
    db.create_all()
    owner = make_user()
    repository = JobRepository()
    first, sibling = repository.create_batch(
        owner.id,
        [
            batch_job(
                tmp_path,
                job_id="first",
                batch_id="batch",
                batch_position=0,
            ),
            batch_job(
                tmp_path,
                job_id="sibling",
                batch_id="batch",
                batch_position=1,
            ),
        ],
        expires_at=datetime.now(UTC) + timedelta(hours=24),
        retention_class=RetentionClass.TEMPORARY,
    )
    claimed = repository.claim_next("worker-a", datetime.now(UTC))
    assert claimed is not None and claimed.id == first.id
    claimed.state = JobState.DELETED
    claimed.stage = None
    claimed.deleted_at = datetime.now(UTC)
    db.session.commit()

    with pytest.raises(JobNoLongerProcessable):
        repository.complete(first.id, {"frames": []}, datetime.now(UTC))
    with pytest.raises(JobNoLongerProcessable):
        repository.set_stage(first.id, ProcessingStage.SCORING)

    next_job = repository.claim_next("worker-b", datetime.now(UTC))
    assert next_job is not None and next_job.id == sibling.id
    repository.complete(sibling.id, {"frames": []}, datetime.now(UTC))
    assert db.session.get(AnalysisJob, first.id).state == JobState.DELETED  # type: ignore[union-attr]
    assert db.session.get(AnalysisJob, sibling.id).state == JobState.COMPLETED  # type: ignore[union-attr]


def test_two_sessions_claim_one_queued_job_exactly_once(app: Flask, tmp_path: Path) -> None:
    db.create_all()
    owner = make_user()
    job = JobRepository().create(owner.id, [frame(tmp_path)])
    factory = sessionmaker(bind=db.engine, expire_on_commit=False)
    first_session: Session = factory()
    second_session: Session = factory()

    first = JobRepository(first_session).claim_next("worker-a", datetime.now(UTC))
    second = JobRepository(second_session).claim_next("worker-b", datetime.now(UTC))

    assert first is not None
    assert first.id == job.id
    assert second is None
    assert first.state == JobState.VALIDATING
    assert first.stage == ProcessingStage.VALIDATING
    assert first.attempt == 1


def test_recover_interrupted_marks_only_processing_states(app: Flask, tmp_path: Path) -> None:
    db.create_all()
    owner = make_user()
    repository = JobRepository()
    states = [
        JobState.QUEUED,
        JobState.VALIDATING,
        JobState.QUALITY_CHECK,
        JobState.PLANE_DETECTION,
        JobState.ROI_DETECTION,
        JobState.SCORING,
        JobState.EXPLANATION,
        JobState.COMPLETED,
        JobState.REJECTED_QUALITY,
    ]
    initial_states: dict[str, JobState] = {}
    for index, state in enumerate(states):
        job = repository.create(owner.id, [frame(tmp_path, f"frame-{index}.png")])
        job.state = state
        job.stage = ProcessingStage(state.value) if state.value in ProcessingStage else None
        initial_states[job.id] = state
    db.session.commit()

    recovered = repository.recover_interrupted(datetime.now(UTC))

    assert recovered == 6
    for job_id, initial_state in initial_states.items():
        current = db.session.get(AnalysisJob, job_id)
        assert current is not None
        if initial_state.value in ProcessingStage:
            assert current.state == JobState.FAILED_RECOVERABLE
            assert current.error_code == "worker_interrupted"
        else:
            assert current.state == initial_state


def test_recover_interrupted_can_be_scoped_to_one_worker(app: Flask, tmp_path: Path) -> None:
    db.create_all()
    owner = make_user()
    repository = JobRepository()
    first = repository.create(owner.id, [frame(tmp_path, "first.png")])
    second = repository.create(owner.id, [frame(tmp_path, "second.png")])
    first.state = second.state = JobState.SCORING
    first.stage = second.stage = ProcessingStage.SCORING
    first.worker_id = "worker-a"
    second.worker_id = "worker-b"
    db.session.commit()

    recovered = repository.recover_interrupted(datetime.now(UTC), "worker-a")

    assert recovered == 1
    assert db.session.get(AnalysisJob, first.id).state == JobState.FAILED_RECOVERABLE  # type: ignore[union-attr]
    assert db.session.get(AnalysisJob, second.id).state == JobState.SCORING  # type: ignore[union-attr]


def test_recover_interrupted_cannot_overwrite_a_concurrent_delete(
    app: Flask,
    tmp_path: Path,
) -> None:
    db.create_all()
    owner = make_user()
    job = JobRepository().create(owner.id, [frame(tmp_path)])
    job.state = JobState.SCORING
    job.stage = ProcessingStage.SCORING
    job.worker_id = "worker-a"
    db.session.commit()
    factory = sessionmaker(bind=db.engine, expire_on_commit=False)
    recovery_session: Session = factory()
    delete_session: Session = factory()
    deleted = False

    def delete_before_recovery_update(*args: object) -> None:
        nonlocal deleted
        statement = args[2]
        if deleted or not isinstance(statement, str) or "UPDATE analysis_jobs SET" not in statement:
            return
        deleted = True
        delete_session.execute(
            update(AnalysisJob)
            .where(AnalysisJob.id == job.id)
            .values(state=JobState.DELETED, stage=None, deleted_at=datetime.now(UTC)),
        )
        delete_session.commit()

    event.listen(db.engine, "before_cursor_execute", delete_before_recovery_update)
    try:
        recovered = JobRepository(recovery_session).recover_interrupted(
            datetime.now(UTC),
            "worker-a",
        )
    finally:
        event.remove(db.engine, "before_cursor_execute", delete_before_recovery_update)
        recovery_session.close()
        delete_session.close()

    db.session.expire_all()
    refreshed = db.session.get(AnalysisJob, job.id)
    assert deleted is True
    assert recovered == 0
    assert refreshed is not None and refreshed.state == JobState.DELETED


def test_repository_persists_lowercase_enums_and_indexed_columns(
    app: Flask,
    tmp_path: Path,
) -> None:
    db.create_all()
    owner = make_user()
    job = JobRepository().create(owner.id, [frame(tmp_path)])

    stored_state = db.session.execute(
        select(AnalysisJob.state).where(AnalysisJob.id == job.id),
    ).scalar_one()
    indexes = {item["name"] for item in inspect(db.engine).get_indexes("analysis_jobs")}

    assert stored_state == JobState.QUEUED
    assert {
        "ix_analysis_jobs_state_created_at_id",
        "ix_analysis_jobs_owner_id_batch_id",
        "ix_analysis_jobs_expires_at",
    } <= indexes


def test_complete_and_fail_recoverable_close_claimed_jobs(app: Flask, tmp_path: Path) -> None:
    db.create_all()
    owner = make_user()
    repository = JobRepository()
    first = repository.create(owner.id, [frame(tmp_path, "first.png")])
    second = repository.create(owner.id, [frame(tmp_path, "second.png")])
    now = datetime.now(UTC)

    assert repository.claim_next("worker-a", now).id == first.id  # type: ignore[union-attr]
    repository.complete(first.id, {"frames": []}, now)
    assert repository.claim_next("worker-b", now).id == second.id  # type: ignore[union-attr]
    repository.fail_recoverable(second.id, "inference_timeout", now + timedelta(seconds=1))

    assert db.session.get(AnalysisJob, first.id).state == JobState.COMPLETED  # type: ignore[union-attr]
    assert db.session.get(AnalysisJob, first.id).result == {"frames": []}  # type: ignore[union-attr]
    assert db.session.get(AnalysisJob, second.id).state == JobState.FAILED_RECOVERABLE  # type: ignore[union-attr]


def test_conditional_transition_cannot_overwrite_a_deleted_job(
    app: Flask,
    tmp_path: Path,
) -> None:
    db.create_all()
    owner = make_user()
    repository = JobRepository()
    job = repository.create(owner.id, [frame(tmp_path)])
    claimed = repository.claim_next("worker-a", datetime.now(UTC))
    assert claimed is not None and claimed.id == job.id
    claimed.state = JobState.DELETED
    claimed.stage = None
    claimed.deleted_at = datetime.now(UTC)
    db.session.commit()

    with pytest.raises(JobNoLongerProcessable):
        repository.complete(job.id, {"frames": []}, datetime.now(UTC))
    with pytest.raises(JobNoLongerProcessable):
        repository.lock_processable(job.id)

    db.session.expire_all()
    refreshed = db.session.get(AnalysisJob, job.id)
    assert refreshed is not None
    assert refreshed.state == JobState.DELETED
    assert refreshed.result is None
