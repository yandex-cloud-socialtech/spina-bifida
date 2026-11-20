from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

Locale = Literal["ru", "en", "kk", "uz"]


class ConfigError(ValueError):
    """Raised when the application configuration cannot be safely loaded."""


class StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AccessSettings(StrictSettings):
    mode: Literal["ip", "domain"] = "ip"
    domain: str | None = None
    acme_email: str | None = None

    @model_validator(mode="after")
    def require_domain_configuration(self) -> Self:
        if self.mode == "domain" and (not self.domain or not self.acme_email):
            raise ValueError(
                "access.domain and access.acme_email are required when access.mode is domain",
            )
        return self


class InferenceThresholdSettings(StrictSettings):
    plane_detection: float = Field(default=0.1, ge=0.0, le=1.0)
    quality: float = Field(default=0.5, ge=0.0, le=1.0)
    assessment: float = Field(default=0.5, ge=0.0, le=1.0)
    warning_margin: float = Field(default=0.1, ge=0.0, le=1.0)


class InferenceSettings(StrictSettings):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    default_model_version: Literal["v1", "v2"] = "v2"
    max_parallel_jobs: int = Field(default=1, ge=1, le=4)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)
    models_root: Path = Path("/models")
    thresholds: InferenceThresholdSettings = Field(default_factory=InferenceThresholdSettings)


class ImageStorageSettings(StrictSettings):
    mode: Literal["temporary", "feedback_only", "all"] = "feedback_only"
    temporary_ttl_hours: int = Field(default=24, ge=1, le=24 * 30)
    retained_ttl_days: int = Field(default=90, ge=1, le=3650)
    max_request_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    max_batch_bytes: int = Field(default=210 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=25_000_000, ge=1_000_000)
    max_total_gb: float = Field(default=20.0, gt=0)
    min_free_gb: float = Field(default=2.0, ge=0)


class FeedbackStorageSettings(StrictSettings):
    retained_ttl_days: int = Field(default=365, ge=1, le=3650)
    retain_assets: bool = True


class StorageSettings(StrictSettings):
    db_dir: Path = Path("/data/db")
    jobs_dir: Path = Path("/data/jobs")
    feedback_dir: Path = Path("/data/feedback")
    images: ImageStorageSettings = Field(default_factory=ImageStorageSettings)
    feedback: FeedbackStorageSettings = Field(default_factory=FeedbackStorageSettings)


class LogSettings(StrictSettings):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    directory: Path = Path("/data/logs")
    rotate_max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    rotate_interval_hours: int = Field(default=24, ge=1, le=24 * 30)
    compress: bool = True
    retention_days: int = Field(default=30, ge=1, le=3650)
    max_archives: int = Field(default=30, ge=1, le=10_000)
    max_total_gb: float = Field(default=5.0, gt=0)


class ObjectStorageRetrySettings(StrictSettings):
    max_attempts: int = Field(default=8, ge=1, le=100)
    base_seconds: int = Field(default=5, ge=1, le=3600)
    max_seconds: int = Field(default=3600, ge=1, le=86_400)


class ObjectStorageSettings(StrictSettings):
    enabled: bool = False
    endpoint: str | None = None
    bucket: str | None = None
    logs_prefix: str = "logs/"
    feedback_prefix: str = "feedback/"
    upload_logs: bool = True
    upload_feedback: bool = False
    retry: ObjectStorageRetrySettings = Field(default_factory=ObjectStorageRetrySettings)
    access_key: SecretStr | None = Field(default=None, exclude=True)
    secret_key: SecretStr | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def require_enabled_connection(self) -> Self:
        missing: list[str] = []
        if self.enabled:
            if not self.endpoint:
                missing.append("object_storage.endpoint")
            if not self.bucket:
                missing.append("object_storage.bucket")
            if self.access_key is None or not self.access_key.get_secret_value():
                missing.append("OBJECT_STORAGE_ACCESS_KEY")
            if self.secret_key is None or not self.secret_key.get_secret_value():
                missing.append("OBJECT_STORAGE_SECRET_KEY")
        if missing:
            raise ValueError("required values are missing: " + ", ".join(missing))
        return self


class PostboxSettings(StrictSettings):
    enabled: bool = False
    endpoint: str = "https://postbox.cloud.yandex.net"
    region: str = "ru-central1"
    sender: str = ""
    public_url: str = ""
    access_key: SecretStr | None = Field(default=None, exclude=True)
    secret_key: SecretStr | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_connection(self) -> Self:
        if self.enabled:
            from urllib.parse import urlsplit

            url = urlsplit(self.public_url)
            if (
                url.scheme != "https"
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or url.path not in ("", "/")
            ):
                raise ValueError("postbox.public_url must be an HTTPS origin")
            if not self.endpoint.startswith("https://"):
                raise ValueError("postbox.endpoint must use HTTPS")
            if (
                "@" not in self.sender
                or any(c in self.sender for c in "\r\n")
                or not self.access_key
                or not self.secret_key
            ):
                raise ValueError(
                    "Postbox requires sender, POSTBOX_ACCESS_KEY and POSTBOX_SECRET_KEY",
                )
        return self


class BackupSettings(StrictSettings):
    destination: Path = Path("/data/backups")
    retention_count: int = Field(default=7, ge=1, le=365)


class LocalizationSettings(StrictSettings):
    default: Locale = "ru"
    supported: tuple[Locale, ...] = ("ru", "en", "kk", "uz")

    @model_validator(mode="after")
    def validate_supported_locales(self) -> Self:
        if len(self.supported) != len(set(self.supported)):
            raise ValueError("localization.supported must not contain duplicates")
        if self.default not in self.supported:
            raise ValueError("localization.default must be present in localization.supported")
        if "ru" not in self.supported:
            raise ValueError("localization.supported must include ru as the fallback locale")
        return self


class AppSettings(StrictSettings):
    access: AccessSettings = Field(default_factory=AccessSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logs: LogSettings = Field(default_factory=LogSettings)
    object_storage: ObjectStorageSettings = Field(default_factory=ObjectStorageSettings)
    postbox: PostboxSettings = Field(default_factory=PostboxSettings)
    backup: BackupSettings = Field(default_factory=BackupSettings)
    localization: LocalizationSettings = Field(default_factory=LocalizationSettings)
    session_secret: SecretStr = Field(exclude=True)

    @classmethod
    def load(cls, yaml_path: Path, environ: Mapping[str, str]) -> AppSettings:
        session_secret = environ.get("APP_SESSION_SECRET", "")
        if len(session_secret.encode("utf-8")) < 32:
            raise ConfigError("APP_SESSION_SECRET is required and must contain at least 32 bytes")

        try:
            loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"Configuration file could not be read: {yaml_path}") from error

        if loaded is None:
            raw: dict[str, Any] = {}
        elif isinstance(loaded, dict):
            raw = deepcopy(loaded)
        else:
            raise ConfigError("Configuration root must be a mapping")

        cls._resolve_relative_paths(raw, yaml_path.parent)
        raw["session_secret"] = session_secret
        object_storage = raw.setdefault("object_storage", {})
        if not isinstance(object_storage, dict):
            raise ConfigError("object_storage must be a mapping")
        object_storage["access_key"] = environ.get("OBJECT_STORAGE_ACCESS_KEY") or None
        object_storage["secret_key"] = environ.get("OBJECT_STORAGE_SECRET_KEY") or None

        postbox = raw.setdefault("postbox", {})
        if not isinstance(postbox, dict):
            raise ConfigError("postbox must be a mapping")
        postbox["access_key"] = environ.get("POSTBOX_ACCESS_KEY") or None
        postbox["secret_key"] = environ.get("POSTBOX_SECRET_KEY") or None

        try:
            return cls.model_validate(raw)
        except ValidationError as error:
            messages = []
            for item in error.errors(include_input=False, include_url=False):
                location = ".".join(str(part) for part in item["loc"])
                messages.append(f"{location}: {item['msg']}")
            raise ConfigError("; ".join(messages)) from error

    @staticmethod
    def _resolve_relative_paths(raw: dict[str, Any], base: Path) -> None:
        path_fields = (
            ("inference", "models_root"),
            ("storage", "db_dir"),
            ("storage", "jobs_dir"),
            ("storage", "feedback_dir"),
            ("logs", "directory"),
            ("backup", "destination"),
        )
        for section_name, field_name in path_fields:
            section = raw.get(section_name)
            if not isinstance(section, dict) or field_name not in section:
                continue
            value = Path(section[field_name])
            if not value.is_absolute():
                section[field_name] = base / value

    def ensure_directories(self) -> None:
        for path in (
            self.storage.db_dir,
            self.storage.jobs_dir,
            self.storage.feedback_dir,
            self.logs.directory,
            self.backup.destination,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
