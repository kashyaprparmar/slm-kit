"""Adopt the existing SQLModel schema without rewriting or removing records.

Definitions are frozen here: never import live application metadata in a revision.
"""
import sqlalchemy as sa
from alembic import op

revision = "0001_legacy"
down_revision = None
branch_labels = None
depends_on = None


def _column(name, type_=sa.String, nullable=False, **kwargs):
    return sa.Column(name, type_, nullable=nullable, **kwargs)


def _id():
    return _column("id", sa.Integer, primary_key=True)


def _created():
    return _column("created_at", sa.DateTime)


def baseline_tables():
    """Frozen tables, also used to validate adoption of an unversioned database."""
    metadata = sa.MetaData()
    sa.Table("dataset", metadata, _id(), _column("name"), _column("kind"),
             _column("path"), _column("fmt"),
             *[_column(n, sa.Integer, True) for n in ("num_rows", "num_tokens_est", "size_bytes")],
             _column("is_sample", sa.Boolean), _column("validation", sa.JSON, True), _created())
    sa.Table("project", metadata, _id(), _column("name"), _column("description"),
             _column("state", sa.JSON, True), _created(), _column("updated_at", sa.DateTime))
    sa.Table("run", metadata, _id(),
             *[_column(n) for n in ("name", "task", "method", "backend", "base_model")],
             sa.Column("dataset_id", sa.Integer, sa.ForeignKey("dataset.id")),
             _column("status"),
             *[_column(n, sa.JSON, True) for n in ("config", "estimate", "metrics", "hardware")],
             *[_column(n, nullable=True) for n in ("output_dir", "hf_repo", "error")],
             _created(), _column("started_at", sa.DateTime, True), _column("finished_at", sa.DateTime, True))
    sa.Table("projectrun", metadata,
             sa.Column("project_id", sa.Integer, sa.ForeignKey("project.id"), primary_key=True),
             sa.Column("run_id", sa.Integer, sa.ForeignKey("run.id"), primary_key=True), _created())
    sa.Table("checkpoint", metadata, _id(),
             sa.Column("run_id", sa.Integer, sa.ForeignKey("run.id"), nullable=False),
             _column("step", sa.Integer), _column("path"), _column("is_final", sa.Boolean), _created())
    sa.Table("modelartifact", metadata, _id(), _column("name"), _column("kind"),
             sa.Column("run_id", sa.Integer, sa.ForeignKey("run.id")),
             *[_column(n, nullable=True) for n in ("base_model", "local_path", "hf_repo")],
             _column("published", sa.Boolean), _column("status"), _column("error", nullable=True),
             _column("meta", sa.JSON, True), _created())
    sa.Table("evalresult", metadata, _id(), _column("model_ref"),
             sa.Column("dataset_id", sa.Integer, sa.ForeignKey("dataset.id")),
             sa.Column("run_id", sa.Integer, sa.ForeignKey("run.id")),
             _column("scores", sa.JSON, True), _column("detail", sa.JSON, True), _created())
    return metadata


def upgrade():
    connection = op.get_bind()
    metadata = baseline_tables()
    inspector = sa.inspect(connection)
    # Check all existing tables before performing any DDL. Missing legacy tables
    # (e.g. Projects in an older install) can safely be created; missing columns
    # in an existing table require an explicit migration, not an invented default.
    for table in metadata.sorted_tables:
        if inspector.has_table(table.name):
            actual = {column["name"] for column in inspector.get_columns(table.name)}
            missing = set(table.columns.keys()) - actual
            if missing:
                raise RuntimeError(f"Cannot adopt legacy table {table.name}: missing columns {sorted(missing)}. Restore a compatible database or add an explicit migration.")
    metadata.create_all(connection)


def downgrade():
    raise RuntimeError("The legacy baseline cannot be downgraded: restore a database backup instead.")
