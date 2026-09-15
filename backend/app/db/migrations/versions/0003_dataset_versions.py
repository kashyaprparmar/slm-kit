"""Immutable dataset/version/recipe/profile and tokenizer metadata."""
import sqlalchemy as sa
from alembic import op

revision = "0003_dataset_versions"
down_revision = "0002_indexes"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "datasetversion",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset_id", sa.Integer, sa.ForeignKey("dataset.id"), nullable=False),
        sa.Column("parent_version_id", sa.Integer, sa.ForeignKey("datasetversion.id")),
        sa.Column("fingerprint", sa.String, nullable=False),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("fmt", sa.String, nullable=False),
        sa.Column("schema", sa.JSON),
        sa.Column("split", sa.String, nullable=False),
        sa.Column("num_rows", sa.Integer),
        sa.Column("num_tokens_est", sa.Integer),
        sa.Column("size_bytes", sa.Integer),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint("dataset_id", "fingerprint", "split", name="uq_dataset_version_content"),
    )
    op.create_table(
        "datasetrecipe",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source_version_id", sa.Integer, sa.ForeignKey("datasetversion.id"), nullable=False),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("config", sa.JSON),
        sa.Column("fingerprint", sa.String, nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "datasetsplit",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("recipe_id", sa.Integer, sa.ForeignKey("datasetrecipe.id"), nullable=False),
        sa.Column("train_version_id", sa.Integer, sa.ForeignKey("datasetversion.id"), nullable=False),
        sa.Column("validation_version_id", sa.Integer, sa.ForeignKey("datasetversion.id")),
        sa.Column("test_version_id", sa.Integer, sa.ForeignKey("datasetversion.id")),
        sa.Column("seed", sa.Integer, nullable=False),
        sa.Column("fractions", sa.JSON),
        sa.Column("fingerprint", sa.String, nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "tokenizerartifact",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_ref", sa.String, nullable=False),
        sa.Column("revision", sa.String),
        sa.Column("resolved_revision", sa.String),
        sa.Column("fingerprint", sa.String, nullable=False, unique=True),
        sa.Column("path", sa.String),
        sa.Column("config", sa.JSON),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "datasetprofile",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset_version_id", sa.Integer, sa.ForeignKey("datasetversion.id"), nullable=False),
        sa.Column("tokenizer_artifact_id", sa.Integer, sa.ForeignKey("tokenizerartifact.id")),
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("cache_key", sa.String, nullable=False, unique=True),
        sa.Column("config", sa.JSON),
        sa.Column("stats", sa.JSON),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    for table, columns in {
        "datasetversion": (("dataset_id",), ("parent_version_id",)),
        "datasetrecipe": (("source_version_id",),),
        "datasetsplit": (("recipe_id",),),
        "datasetprofile": (("dataset_version_id",), ("tokenizer_artifact_id",)),
    }.items():
        for group in columns:
            op.create_index(f"ix_{table}_{'_'.join(group)}", table, list(group))


def downgrade():
    op.drop_table("datasetprofile")
    op.drop_table("tokenizerartifact")
    op.drop_table("datasetsplit")
    op.drop_table("datasetrecipe")
    op.drop_table("datasetversion")
