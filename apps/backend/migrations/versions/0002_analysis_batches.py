"""Add atomic analysis batch metadata."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column("batch_id", sa.String(36), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("batch_position", sa.Integer(), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("model_version", sa.String(2), nullable=True),
    )
    op.execute(
        sa.text(
            "UPDATE analysis_jobs "
            "SET batch_id = id, batch_position = 0, model_version = 'v1'",
        ),
    )
    op.drop_index("ix_analysis_jobs_owner_id", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_state_created_at", table_name="analysis_jobs")
    with op.batch_alter_table("analysis_jobs") as batch_op:
        batch_op.alter_column(
            "batch_id",
            existing_type=sa.String(36),
            nullable=False,
        )
        batch_op.alter_column(
            "batch_position",
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch_op.alter_column(
            "model_version",
            existing_type=sa.String(2),
            nullable=False,
        )
    op.create_index(
        "ix_analysis_jobs_owner_id_batch_id",
        "analysis_jobs",
        ["owner_id", "batch_id"],
    )
    op.create_index(
        "ix_analysis_jobs_state_created_at_id",
        "analysis_jobs",
        ["state", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_analysis_jobs_state_created_at_id",
        table_name="analysis_jobs",
    )
    op.drop_index(
        "ix_analysis_jobs_owner_id_batch_id",
        table_name="analysis_jobs",
    )
    with op.batch_alter_table("analysis_jobs") as batch_op:
        batch_op.drop_column("model_version")
        batch_op.drop_column("batch_position")
        batch_op.drop_column("batch_id")
    op.create_index(
        "ix_analysis_jobs_owner_id",
        "analysis_jobs",
        ["owner_id"],
    )
    op.create_index(
        "ix_analysis_jobs_state_created_at",
        "analysis_jobs",
        ["state", "created_at"],
    )
