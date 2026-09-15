# Database migrations and integrity

SLM Kit now runs packaged Alembic migrations at backend startup. The initial
revision adopts the existing Dataset, Project, ProjectRun, Run, Checkpoint,
ModelArtifact, and EvalResult tables. A second revision adds indexes for queue
and relationship lookups. Revision `0003_dataset_versions` adds immutable
dataset lineage, preparation recipes/splits, tokenizer artifacts, and cached
profiles. Existing raw files and records are not rewritten or removed.

The migration scripts live under `backend/app/db/migrations/` so both Docker
images and installed Python packages include them. `init_db()` runs the same
upgrade path on a fresh database and on an existing unversioned database.

## Upgrade behavior

- Existing databases receive a SQLite backup under
  `$SLMKIT_HOME/db-backups/` before an upgrade. SQLite's backup API includes
  committed WAL data; do not substitute a raw copy of a running `.db` file.
- Each upgrade runs in an immediate transaction. Concurrent startup migrations
  serialize. Failed DDL is rolled back, and startup stops with a diagnostic.
- Missing legacy tables are created. Existing tables missing required baseline
  columns or databases stamped with unknown revisions are rejected for explicit
  investigation. They are not silently reset or stamped as current.
- Foreign keys and a 30-second busy timeout apply to every application SQLite
  connection. WAL is enabled during initialization for file databases.
- Legacy orphan references are retained and logged for inspection. Foreign-key
  enforcement prevents new invalid references; it does not repair old history.
- After migration, each legacy dataset whose file is present receives one source
  `DatasetVersion` pointing to that exact path and SHA-256. The backfill is
  idempotent and does not copy or alter the file.

## Inspect integrity

`GET /api/system/database` returns the current/expected revision, foreign-key
enforcement, journal mode, `quick_check` result, and up to 100 foreign-key
violations. This explicit diagnostic runs in a request thread, separate from the
frequently polled health endpoint. It does not return dataset or prompt content.

Deleting a run linked to evaluations or registered artifacts returns `409`.
Deleting an otherwise removable run clears its project membership first.
Deleting a project still retains runs and their artifacts.

## Recovery and future revisions

Keep pre-upgrade backups until the upgraded installation has been verified.
Stop the backend before restoring a backup. Do not replace an active database
while its WAL/SHM files are in use. Downgrading the initial baseline is disabled
because dropping those tables would destroy user data.

Future schema additions belong in a new revision referencing
`0003_dataset_versions`; do not edit the frozen baseline or import live SQLModel
metadata into revisions. See [Implementation plan](implementation-plan.md).
