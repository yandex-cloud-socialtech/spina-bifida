from collections.abc import Iterator
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from spina_bifida import create_app


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Flask]:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                f"  db_dir: {tmp_path / 'db'}",
                f"  jobs_dir: {tmp_path / 'jobs'}",
                f"  feedback_dir: {tmp_path / 'feedback'}",
                "logs:",
                f"  directory: {tmp_path / 'logs'}",
                "backup:",
                f"  destination: {tmp_path / 'backups'}",
            ],
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("APP_SESSION_SECRET", "x" * 32)
    application = create_app({"TESTING": True})
    with application.app_context():
        yield application


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()
