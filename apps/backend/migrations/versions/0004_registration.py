"""Moderated registration, one-use email tokens and session revocation."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("email_verified_at", sa.DateTime(timezone=True)))
    op.add_column(
        "users", sa.Column("password_version", sa.Integer(), nullable=False, server_default="1")
    )
    # Existing accounts were provisioned by an administrator; retain their access.
    op.execute("UPDATE users SET email_verified_at = created_at WHERE status != 'deleted'")
    op.create_table(
        "auth_tokens",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("purpose", sa.String(24), nullable=False),
        sa.Column("password_version", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_auth_tokens_user_id", "auth_tokens", ["user_id"])


def downgrade() -> None:
    op.drop_table("auth_tokens")
    op.drop_column("users", "password_version")
    op.drop_column("users", "email_verified_at")
