from __future__ import annotations

import click
from flask import Flask

from spina_bifida.analyses.repository import JobRepository
from spina_bifida.auth.service import AuthService, DuplicateEmail, InvalidUserField


def register_cli(app: Flask) -> None:
    @app.cli.command("create-admin")
    @click.option("--email", type=str, help="Administrator email; prompted when omitted.")
    def create_admin(email: str | None) -> None:
        """Create the first or an additional local administrator."""
        email = email or click.prompt("Admin email")
        password = click.prompt("Password", hide_input=True, confirmation_prompt=True)
        locale = click.prompt("Locale", default="ru", show_default=True)
        try:
            user = AuthService().create_user(email, password, "admin", locale)
        except DuplicateEmail as error:
            raise click.ClickException("An account with this email already exists") from error
        except InvalidUserField as error:
            raise click.ClickException(f"Invalid field: {error}") from error
        click.echo(f"Admin created: {user.email}")

    @app.cli.command("recover-jobs")
    def recover_jobs() -> None:
        """Mark jobs interrupted by a worker restart as recoverable."""
        from datetime import UTC, datetime

        count = JobRepository().recover_interrupted(datetime.now(UTC))
        click.echo(f"Recovered jobs: {count}")
