"""Indexes for relationship lookups and the FIFO run queue."""
from alembic import op

revision = "0002_indexes"
down_revision = "0001_legacy"
branch_labels = None
depends_on = None

INDEXES = {
    "run": [("status", "created_at"), ("dataset_id",)],
    "checkpoint": [("run_id",)],
    "projectrun": [("run_id",)],
    "modelartifact": [("run_id",)],
    "evalresult": [("run_id",), ("dataset_id",)],
}


def upgrade():
    for table, groups in INDEXES.items():
        for columns in groups:
            op.create_index(f"ix_{table}_{'_'.join(columns)}", table, list(columns))


def downgrade():
    for table, groups in INDEXES.items():
        for columns in groups:
            op.drop_index(f"ix_{table}_{'_'.join(columns)}", table_name=table)
