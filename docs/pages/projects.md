# Projects

Projects are persistent experiment workspaces. They let you group related runs
without changing how datasets, model artifacts, or run files are stored.

## Create and select a project

1. Open **Projects** from the sidebar.
2. Enter a name and optional description.
3. Click **Create and select**.

The selected project is remembered in the browser. New runs launched from a
training studio include its project id and are linked to that workspace in the
same database transaction that creates the run.

Project names are trimmed, cannot be blank, and are limited to 120 characters.
If a selected project was deleted in another browser or through the API, the UI
clears that stale selection and the backend rejects attempts to launch against
the missing project.

## Existing and ungrouped runs

Runs do not need a project. Existing runs remain available in **Run History**,
and an API client can attach one to a project with
`POST /api/projects/{project_id}/runs/{run_id}`.

Deleting a project removes only the workspace and its run links. The runs,
checkpoints, registered artifacts, and files are retained. The UI asks for
confirmation before performing this metadata-only deletion.

See the [API reference](../api-reference.md#projects) for CRUD endpoints.
