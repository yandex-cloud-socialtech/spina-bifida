from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event, text

if TYPE_CHECKING:
    from sqlite3 import Connection as SQLiteConnection

    from spina_bifida.config import AppSettings


db = SQLAlchemy(session_options={"expire_on_commit": False})
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri="memory://",
)


def _set_sqlite_pragmas(
    dbapi_connection: SQLiteConnection,
    _connection_record: Any,
) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def init_database(app: Flask, settings: AppSettings) -> None:
    database_path = settings.storage.db_dir / "app.sqlite3"
    app.config.update(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{database_path}",
        SQLALCHEMY_ENGINE_OPTIONS={"connect_args": {"timeout": 5}},
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)
    with app.app_context():
        event.listen(db.engine, "connect", _set_sqlite_pragmas)


def init_limiter(app: Flask) -> None:
    limiter.init_app(app)
    if app.testing:
        limiter.reset()


def _alembic_config(app: Flask) -> Config:
    source_root = Path(__file__).resolve().parents[2]
    backend_root = Path(os.environ.get("APP_MIGRATIONS_ROOT", source_root)).resolve()
    config = Config(str(backend_root / "alembic.ini"))
    config.set_main_option("script_location", str(backend_root / "migrations"))
    config.set_main_option("sqlalchemy.url", str(app.config["SQLALCHEMY_DATABASE_URI"]))
    return config


def _within_app_context(app: Flask, operation: Callable[[], None]) -> None:
    with app.app_context():
        operation()


def upgrade_database(app: Flask, revision: str = "head") -> None:
    _within_app_context(app, lambda: command.upgrade(_alembic_config(app), revision))


def downgrade_database(app: Flask, revision: str = "base") -> None:
    _within_app_context(app, lambda: command.downgrade(_alembic_config(app), revision))


def database_is_ready() -> bool:
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        db.session.rollback()
        return False
    return True


def database_schema_is_current(app: Flask) -> bool:
    try:
        expected = set(ScriptDirectory.from_config(_alembic_config(app)).get_heads())
        with db.engine.connect() as connection:
            current = set(MigrationContext.configure(connection).get_current_heads())
    except Exception:
        db.session.rollback()
        return False
    return current == expected


def _directory_is_writable(path: Path) -> bool:
    if not path.is_dir() or not os.access(path, os.W_OK):
        return False
    try:
        with tempfile.NamedTemporaryFile(prefix=".ready-", dir=path):
            return True
    except OSError:
        return False


def storage_is_ready(settings: AppSettings) -> bool:
    return all(
        _directory_is_writable(path)
        for path in (
            settings.storage.db_dir,
            settings.storage.jobs_dir,
            settings.storage.feedback_dir,
            settings.logs.directory,
            settings.backup.destination,
        )
    )
