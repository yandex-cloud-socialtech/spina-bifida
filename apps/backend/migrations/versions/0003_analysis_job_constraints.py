"""Constrain analysis job batch metadata."""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analysis_jobs", recreate="always") as batch_op:
        batch_op.create_check_constraint(
            "ck_analysis_jobs_model_version",
            "model_version IN ('v1', 'v2')",
        )
        batch_op.create_check_constraint(
            "ck_analysis_jobs_batch_position_nonnegative",
            "batch_position >= 0",
        )
        batch_op.create_unique_constraint(
            "uq_analysis_jobs_batch_position",
            ["batch_id", "batch_position"],
        )


def downgrade() -> None:
    with op.batch_alter_table("analysis_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "uq_analysis_jobs_batch_position",
            type_="unique",
        )
        batch_op.drop_constraint(
            "ck_analysis_jobs_batch_position_nonnegative",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_analysis_jobs_model_version",
            type_="check",
        )
