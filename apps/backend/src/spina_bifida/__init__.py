import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from spina_bifida import extensions
from spina_bifida.config import AppSettings
from spina_bifida.observability.logging import configure_logging


def create_app(config_override: Mapping[str, Any] | None = None) -> Flask:
    """Create the Flask application without enabling cross-origin access."""
    override = dict(config_override or {})
    configured_path = override.get("APP_CONFIG_FILE") or os.environ.get(
        "APP_CONFIG_FILE",
        "/config/app.yaml",
    )
    config_path = Path(str(configured_path))
    settings = AppSettings.load(config_path, os.environ)
    settings.ensure_directories()

    app = Flask(__name__)
    # Only Traefik can reach the API on the private Compose network.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)  # type: ignore[method-assign]
    app.config.update(override)
    app.config.update(
        SECRET_KEY=settings.session_secret.get_secret_value(),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.access.mode == "domain",
        MAX_CONTENT_LENGTH=settings.storage.images.max_batch_bytes,
    )
    app.extensions["settings"] = settings
    configure_logging(settings, "api")
    extensions.init_database(app, settings)
    extensions.init_limiter(app)

    from spina_bifida.admin.routes import admin_blueprint
    from spina_bifida.analyses import models as analysis_models  # noqa: F401
    from spina_bifida.analyses.routes import analyses_blueprint, system_blueprint
    from spina_bifida.api.errors import register_error_handlers
    from spina_bifida.auth import models as auth_models  # noqa: F401
    from spina_bifida.auth.routes import auth_blueprint, load_current_user
    from spina_bifida.cli import register_cli

    register_error_handlers(app)
    app.before_request(load_current_user)
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(admin_blueprint)
    app.register_blueprint(analyses_blueprint)
    app.register_blueprint(system_blueprint)
    register_cli(app)

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def readiness() -> tuple[dict[str, str], int] | dict[str, str]:
        if not extensions.database_is_ready():
            return {"status": "error", "code": "database_unavailable"}, 503
        if not extensions.database_schema_is_current(app):
            return {"status": "error", "code": "schema_outdated"}, 503
        if not extensions.storage_is_ready(settings):
            return {"status": "error", "code": "storage_unavailable"}, 503
        return {"status": "ready"}

    return app
