# Run History

Every run you've ever launched — searchable, with live and historical charts,
full logs, and one-click re-run. Also the place to watch the currently-active
run in detail.

## The list (left)

- **Search** by name, base model, or method.
- **Filter** by status: all / running / queued / done / failed / cancelled.
- Click any run to open it. The list auto-refreshes, so a running job updates
  live. (The Dashboard and studios deep-link here with a run pre-selected.)

## The monitor (right, top)

The same live monitor used in the studios, but it also works for **finished**
runs:
- **Loss curve** — for a running job it streams live; for a finished job it's
  **replotted from saved history** (`metrics.jsonl`), so you always see the full
  curve.
- **Loss / tokens-per-sec / ETA** tiles.
- **Progress bar** and **streaming logs**.
- **Cancel** (for running/queued jobs).
- **Sample generations** (from-scratch runs).

## The detail card (right, bottom)

- **Task / method / duration / backend**, base model, and — if published — a
  link to the Hugging Face repo.
- **Error** message if the run failed (the real reason, e.g. a specific
  exception, not a generic code).
- **Fit estimate** the run was launched with.
- **Full configuration (config-as-data):** the exact JSON that defined the run.
  Copy it, or:

### Two actions

- **Re-run** — re-queues a brand-new run from this run's stored config. Exact,
  reproducible repeats with one click.
- **Export config** — downloads the run's config as a JSON file (for sharing,
  version control, or re-importing later).

## Why this matters

Because every run stores its **complete config**, nothing is ever lost to "what
settings did I use again?" You can always reproduce, tweak, or share a run.

## Tips

- Filter to **failed** to triage problems quickly — the detail card shows the
  real error.
- Use **Re-run** after a crash/cancel to pick up the same config (training also
  auto-resumes from the latest checkpoint if one exists).
- **Export config** → commit it to git for a reproducible record of your
  experiments.
