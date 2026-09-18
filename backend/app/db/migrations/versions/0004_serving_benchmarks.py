"""Persist reproducible serving benchmark measurements."""
import sqlalchemy as sa
from alembic import op

revision = "0004_serving_benchmarks"
down_revision = "0003_dataset_versions"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "servingbenchmark",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("artifact_id", sa.Integer, sa.ForeignKey("modelartifact.id")),
        sa.Column("provider", sa.String, nullable=False),
        sa.Column("model_ref", sa.String, nullable=False),
        sa.Column("endpoint", sa.String, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("config", sa.JSON, nullable=False),
        sa.Column("hardware", sa.JSON),
        sa.Column("results", sa.JSON),
        sa.Column("error", sa.String),
        sa.Column("started_at", sa.DateTime, nullable=False),
        sa.Column("completed_at", sa.DateTime),
    )
    op.create_index("ix_servingbenchmark_artifact_id", "servingbenchmark", ["artifact_id"])
    op.create_index("ix_servingbenchmark_created_provider", "servingbenchmark", ["provider", "started_at"])


def downgrade():
    op.drop_index("ix_servingbenchmark_created_provider", table_name="servingbenchmark")
    op.drop_index("ix_servingbenchmark_artifact_id", table_name="servingbenchmark")
    op.drop_table("servingbenchmark")
