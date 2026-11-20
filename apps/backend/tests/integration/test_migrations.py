import logging
from pathlib import Path

import pytest
from flask import Flask
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from spina_bifida import extensions
from spina_bifida.analyses.models import AnalysisJob
from spina_bifida.extensions import db, downgrade_database, upgrade_database


def test_upgrade_from_empty_database_creates_expected_tables(app: Flask) -> None:
    upgrade_database(app)

    assert set(inspect(db.engine).get_table_names()) == {
        "alembic_version",
        "auth_tokens",
        "analysis_assets",
        "analysis_jobs",
        "feedback",
        "users",
    }


def test_analysis_batch_migration_adds_required_columns_and_indexes(app: Flask) -> None:
    upgrade_database(app)

    inspector = inspect(db.engine)
    columns = {column["name"]: column for column in inspector.get_columns("analysis_jobs")}
    indexes = {item["name"]: item for item in inspector.get_indexes("analysis_jobs")}

    assert columns["batch_id"]["nullable"] is False
    assert columns["batch_position"]["nullable"] is False
    assert columns["model_version"]["nullable"] is False
    assert indexes["ix_analysis_jobs_owner_id_batch_id"]["column_names"] == [
        "owner_id",
        "batch_id",
    ]
    assert indexes["ix_analysis_jobs_state_created_at_id"]["column_names"] == [
        "state",
        "created_at",
        "id",
    ]


def test_analysis_job_constraints_exist_in_migration_and_model_metadata(app: Flask) -> None:
    upgrade_database(app)

    inspector = inspect(db.engine)
    checks = {item["name"] for item in inspector.get_check_constraints("analysis_jobs")}
    unique_constraints = {
        item["name"]: item["column_names"]
        for item in inspector.get_unique_constraints("analysis_jobs")
    }
    model_constraints = {constraint.name for constraint in AnalysisJob.__table__.constraints}

    assert {
        "ck_analysis_jobs_model_version",
        "ck_analysis_jobs_batch_position_nonnegative",
    } <= checks
    assert unique_constraints["uq_analysis_jobs_batch_position"] == [
        "batch_id",
        "batch_position",
    ]
    assert {
        "ck_analysis_jobs_model_version",
        "ck_analysis_jobs_batch_position_nonnegative",
        "uq_analysis_jobs_batch_position",
    } <= model_constraints


def test_analysis_job_constraints_reject_invalid_or_duplicate_batch_metadata(
    app: Flask,
) -> None:
    upgrade_database(app)
    timestamp = "2026-09-22 10:00:00+00:00"
    db.session.execute(
        text(
            """
            INSERT INTO users (
                id, email, password_hash, role, status, locale, created_at, updated_at
            ) VALUES (
                'owner-id', 'owner@example.test', 'hash', 'user', 'active', 'ru',
                :timestamp, :timestamp
            )
            """,
        ),
        {"timestamp": timestamp},
    )
    db.session.execute(
        text(
            """
            INSERT INTO analysis_jobs (
                id, owner_id, batch_id, batch_position, model_version, state, stage,
                attempt, worker_id, result, error_code, created_at, started_at,
                completed_at, expires_at, deleted_at
            ) VALUES (
                'existing-job', 'owner-id', 'batch-id', 0, 'v1', 'queued', NULL,
                0, NULL, NULL, NULL, :timestamp, NULL, NULL, :timestamp, NULL
            )
            """,
        ),
        {"timestamp": timestamp},
    )
    db.session.commit()

    for statement in [
        "UPDATE analysis_jobs SET model_version = 'v3' WHERE id = 'existing-job'",
        "UPDATE analysis_jobs SET batch_position = -1 WHERE id = 'existing-job'",
        """
        INSERT INTO analysis_jobs (
            id, owner_id, batch_id, batch_position, model_version, state, stage,
            attempt, worker_id, result, error_code, created_at, started_at,
            completed_at, expires_at, deleted_at
        ) VALUES (
            'duplicate-job', 'owner-id', 'batch-id', 0, 'v2', 'queued', NULL,
            0, NULL, NULL, NULL, '2026-09-22 10:00:01+00:00', NULL, NULL,
            '2026-09-22 10:00:01+00:00', NULL
        )
        """,
    ]:
        with pytest.raises(IntegrityError):
            db.session.execute(text(statement))
            db.session.commit()
        db.session.rollback()


def test_analysis_batch_migration_backfills_existing_jobs(app: Flask) -> None:
    upgrade_database(app, "0001")
    timestamp = "2026-09-22 10:00:00+00:00"
    db.session.execute(
        text(
            """
            INSERT INTO users (
                id, email, password_hash, role, status, locale, created_at, updated_at
            ) VALUES (
                'owner-id', 'owner@example.test', 'hash', 'user', 'active', 'ru',
                :timestamp, :timestamp
            )
            """,
        ),
        {"timestamp": timestamp},
    )
    db.session.execute(
        text(
            """
            INSERT INTO analysis_jobs (
                id, owner_id, state, stage, attempt, worker_id, result, error_code,
                created_at, started_at, completed_at, expires_at, deleted_at
            ) VALUES (
                'existing-job', 'owner-id', 'queued', NULL, 0, NULL, NULL, NULL,
                :timestamp, NULL, NULL, :timestamp, NULL
            )
            """,
        ),
        {"timestamp": timestamp},
    )
    db.session.commit()

    upgrade_database(app)

    row = db.session.execute(
        text(
            "SELECT batch_id, batch_position, model_version "
            "FROM analysis_jobs WHERE id = 'existing-job'",
        ),
    ).one()
    assert row == ("existing-job", 0, "v1")


def test_alembic_config_uses_explicit_runtime_root(
    app: Flask,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    (runtime_root / "migrations").mkdir(parents=True)
    (runtime_root / "alembic.ini").write_text("[alembic]\n", encoding="utf-8")
    monkeypatch.setenv("APP_MIGRATIONS_ROOT", str(runtime_root))

    config = extensions._alembic_config(app)

    assert config.config_file_name == str(runtime_root / "alembic.ini")
    assert config.get_main_option("script_location") == str(runtime_root / "migrations")


def test_migration_keeps_application_logging_enabled(app: Flask) -> None:
    logger = logging.getLogger("spina_bifida")
    assert logger.disabled is False

    upgrade_database(app)

    assert logger.disabled is False


def test_upgrade_downgrade_upgrade_is_repeatable(app: Flask) -> None:
    upgrade_database(app)
    downgrade_database(app, "0001")
    assert "batch_id" not in {
        column["name"] for column in inspect(db.engine).get_columns("analysis_jobs")
    }
    upgrade_database(app)

    assert {
        "batch_id",
        "batch_position",
        "model_version",
    } <= {column["name"] for column in inspect(db.engine).get_columns("analysis_jobs")}


def test_constraint_migration_downgrade_to_0002_and_upgrade_is_repeatable(app: Flask) -> None:
    upgrade_database(app)
    downgrade_database(app, "0002")
    assert not inspect(db.engine).get_check_constraints("analysis_jobs")

    upgrade_database(app)

    assert {item["name"] for item in inspect(db.engine).get_check_constraints("analysis_jobs")} == {
        "ck_analysis_jobs_batch_position_nonnegative",
        "ck_analysis_jobs_model_version",
    }


def test_sqlite_pragmas_are_enabled(app: Flask) -> None:
    with db.engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar().lower() == "wal"
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar() >= 5000


def test_readiness_reports_database_failure_without_exception_details(
    app: Flask,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extensions, "database_is_ready", lambda: False)

    response = app.test_client().get("/health/ready")

    assert response.status_code == 503
    assert response.get_json() == {"status": "error", "code": "database_unavailable"}
    assert "sqlite" not in response.text.lower()
    assert str(Path(app.extensions["settings"].storage.db_dir)) not in response.text


def test_readiness_reports_storage_failure_without_host_paths(
    app: Flask,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extensions, "database_is_ready", lambda: True)
    monkeypatch.setattr(extensions, "database_schema_is_current", lambda _app: True)
    monkeypatch.setattr(extensions, "storage_is_ready", lambda _settings: False)

    response = app.test_client().get("/health/ready")

    assert response.status_code == 503
    assert response.get_json() == {"status": "error", "code": "storage_unavailable"}
    assert str(Path(app.extensions["settings"].storage.jobs_dir)) not in response.text


def test_readiness_rejects_outdated_schema(
    app: Flask,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extensions, "database_is_ready", lambda: True)
    monkeypatch.setattr(extensions, "database_schema_is_current", lambda _app: False)

    response = app.test_client().get("/health/ready")

    assert response.status_code == 503
    assert response.get_json() == {"status": "error", "code": "schema_outdated"}


def test_readiness_succeeds_for_database_and_writable_paths(app: Flask) -> None:
    upgrade_database(app)

    response = app.test_client().get("/health/ready")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ready"}


def test_registration_migration_preserves_existing_accounts(app: Flask) -> None:
    upgrade_database(app, "0003")
    db.session.execute(
        text(
            "INSERT INTO users (id,email,password_hash,role,status,locale,created_at,updated_at) "
            "VALUES ('existing', 'admin@example.test', 'hash', 'admin', 'active', 'ru', "
            "'2026-10-27 10:00:00', '2026-10-27 10:00:00')"
        )
    )
    db.session.commit()
    upgrade_database(app)
    row = db.session.execute(
        text(
            "SELECT status, password_hash, password_version, email_verified_at "
            "FROM users WHERE id = 'existing'"
        )
    ).one()
    assert row[:3] == ("active", "hash", 1)
    assert row.email_verified_at is not None


def test_clean_installation_has_only_the_explicitly_created_admin(app: Flask) -> None:
    upgrade_database(app)
    assert db.session.execute(text("SELECT count(*) FROM users")).scalar_one() == 0
    result = app.test_cli_runner().invoke(
        args=["create-admin", "--email", "admin@example.test"],
        input="StrongPass123\nStrongPass123\nru\n",
    )
    assert result.exit_code == 0, result.output
    assert db.session.execute(text("SELECT email, role, status FROM users")).all() == [
        ("admin@example.test", "admin", "active"),
    ]
