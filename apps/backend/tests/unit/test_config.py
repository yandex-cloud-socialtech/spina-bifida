from pathlib import Path

import pytest
from pydantic import ValidationError

from spina_bifida import create_app
from spina_bifida.config import AppSettings, ConfigError

SESSION_SECRET = "x" * 32


def write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "app.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def fixture_yaml(tmp_path: Path) -> Path:
    return write_yaml(
        tmp_path,
        """
storage:
  db_dir: data/db
  jobs_dir: data/jobs
  feedback_dir: data/feedback
logs:
  directory: data/logs
backup:
  destination: data/backups
""".strip(),
    )


def load(path: Path, **environ: str) -> AppSettings:
    return AppSettings.load(path, {"APP_SESSION_SECRET": SESSION_SECRET, **environ})


def test_default_config_is_local_and_privacy_preserving(tmp_path: Path) -> None:
    settings = load(fixture_yaml(tmp_path))

    assert settings.inference.device == "auto"
    assert settings.inference.default_model_version == "v2"
    assert settings.inference.max_parallel_jobs == 1
    assert settings.inference.thresholds.plane_detection == 0.1
    assert settings.inference.thresholds.quality == 0.5
    assert settings.inference.thresholds.assessment == 0.5
    assert settings.inference.thresholds.warning_margin == 0.1
    assert settings.storage.images.mode == "feedback_only"
    assert settings.storage.images.max_batch_bytes == 210 * 1024 * 1024
    assert settings.object_storage.enabled is False
    assert settings.localization.supported == ("ru", "en", "kk", "uz")


def test_explicit_default_model_version_is_available_for_ui_bootstrap(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "inference:\n  default_model_version: v2\n")

    assert load(path).inference.default_model_version == "v2"


def test_inference_thresholds_are_loaded_from_yaml(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
inference:
  thresholds:
    plane_detection: 0.23
    quality: 0.61
    assessment: 0.72
    warning_margin: 0.08
""".strip(),
    )

    thresholds = load(path).inference.thresholds

    assert thresholds.plane_detection == 0.23
    assert thresholds.quality == 0.61
    assert thresholds.assessment == 0.72
    assert thresholds.warning_margin == 0.08


def test_legacy_single_model_configuration_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "inference:\n  model_version: v1\n")

    with pytest.raises(ConfigError, match=r"inference\.model_version"):
        load(path)


def test_parallel_worker_configuration_accepts_one_through_four(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "inference:\n  max_parallel_jobs: 4\n")

    assert load(path).inference.max_parallel_jobs == 4


def test_parallel_worker_configuration_rejects_more_than_four(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "inference:\n  max_parallel_jobs: 5\n")

    with pytest.raises(ConfigError, match=r"inference\.max_parallel_jobs"):
        load(path)


def test_unknown_key_fails_with_exact_path(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "inference:\n  unknown: true\n")

    with pytest.raises(ConfigError, match=r"inference\.unknown"):
        load(path)


@pytest.mark.parametrize("secret", [None, "too-short"])
def test_session_secret_is_required_and_at_least_32_characters(
    tmp_path: Path,
    secret: str | None,
) -> None:
    environ = {} if secret is None else {"APP_SESSION_SECRET": secret}

    with pytest.raises(ConfigError, match="APP_SESSION_SECRET"):
        AppSettings.load(fixture_yaml(tmp_path), environ)


@pytest.mark.parametrize(
    ("yaml", "error_path"),
    [
        ("inference:\n  device: metal\n", r"inference\.device"),
        (
            "storage:\n  images:\n    temporary_ttl_hours: 0\n",
            r"storage\.images\.temporary_ttl_hours",
        ),
        ("storage:\n  images:\n    max_total_gb: -1\n", r"storage\.images\.max_total_gb"),
        (
            "inference:\n  thresholds:\n    quality: 1.01\n",
            r"inference\.thresholds\.quality",
        ),
        (
            "inference:\n  thresholds:\n    warning_margin: -0.01\n",
            r"inference\.thresholds\.warning_margin",
        ),
    ],
)
def test_invalid_values_report_their_yaml_path(
    tmp_path: Path,
    yaml: str,
    error_path: str,
) -> None:
    with pytest.raises(ConfigError, match=error_path):
        load(write_yaml(tmp_path, yaml))


def test_enabled_object_storage_requires_endpoint_bucket_and_credentials(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "object_storage:\n  enabled: true\n")

    with pytest.raises(ConfigError) as error:
        load(path)

    message = str(error.value)
    assert "object_storage.endpoint" in message
    assert "object_storage.bucket" in message
    assert "OBJECT_STORAGE_ACCESS_KEY" in message
    assert "OBJECT_STORAGE_SECRET_KEY" in message


def test_enabled_object_storage_loads_only_named_secret_variables(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
object_storage:
  enabled: true
  endpoint: https://storage.example.test
  bucket: research-assets
""".strip(),
    )

    settings = load(
        path,
        OBJECT_STORAGE_ACCESS_KEY="access-key",
        OBJECT_STORAGE_SECRET_KEY="secret-key",
        UNRELATED_PRIVATE_VALUE="must-not-be-loaded",
    )

    assert settings.object_storage.access_key.get_secret_value() == "access-key"
    assert settings.object_storage.secret_key.get_secret_value() == "secret-key"
    assert "must-not-be-loaded" not in repr(settings)


def test_ensure_directories_creates_only_configured_local_paths(tmp_path: Path) -> None:
    settings = load(fixture_yaml(tmp_path))

    settings.ensure_directories()

    expected = {
        tmp_path / "data/db",
        tmp_path / "data/jobs",
        tmp_path / "data/feedback",
        tmp_path / "data/logs",
        tmp_path / "data/backups",
    }
    assert all(path.is_dir() for path in expected)


def test_settings_are_immutable(tmp_path: Path) -> None:
    settings = load(fixture_yaml(tmp_path))

    with pytest.raises(ValidationError):
        settings.inference.device = "cpu"  # type: ignore[misc]


def test_app_factory_loads_settings_before_registering_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = fixture_yaml(tmp_path)
    monkeypatch.setenv("APP_CONFIG_FILE", str(path))
    monkeypatch.setenv("APP_SESSION_SECRET", SESSION_SECRET)

    app = create_app({"TESTING": True})

    assert app.extensions["settings"].inference.device == "auto"
    assert app.config["MAX_CONTENT_LENGTH"] == 210 * 1024 * 1024
    assert app.test_client().get("/health/live").status_code == 200


def test_batch_request_limit_rejects_values_below_one_kibibyte(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "storage:\n  images:\n    max_batch_bytes: 1023\n")

    with pytest.raises(ConfigError, match=r"storage\.images\.max_batch_bytes"):
        load(path)
