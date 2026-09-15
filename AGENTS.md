# Repository engineering rules

- Read docs/implementation/STATUS.md and the relevant task before work; inspect actual code rather than trusting completion reports.
- Honor the user's current scope. An audit does not authorize starting implementation phases.
- Preserve pre-existing working-tree changes; never reset or overwrite unrelated work.
- Keep SLM Kit the orchestration layer. LLaMA-Factory and advanced runtimes remain optional.
- Extend TrainingBackend and ModelServingProvider; reuse shared trainer, event and lifecycle code.
- Keep heavy ML imports inside workers; CPU-only API startup must remain usable.
- Reuse datasets/adapters.py, train_entry/tokenization.py, model_refs.py, models/capabilities.py and existing estimator/runtime owners. Avoid parallel normalization, template, loading, capability or fit logic.
- Validate capabilities consistently in API and UI; distinguish metadata, installed dependencies and runtime verification. Persist effective backend/config and reasons.
- Preserve Unicode and conversation/tool semantics. Bind training to reproducible dataset/tokenizer/template identities.
- Preserve existing project/dataset/run/artifact/eval references and config formats. Use additive frozen Alembic migrations with fresh/legacy tests.
- Supervise and cancel subprocess trees; release resource leases reliably. Current GPU leases require one API control process.
- Reuse React Query, workflow drafts and shared studios. Keep advanced controls capability-gated.
- Redact credentials in logs/errors/artifacts while retaining useful diagnostics and tokenizer metadata.
- Run relevant backend pytest/Ruff and frontend test/lint/build checks; mark optional runtime tests and report unverified paths honestly.
- Update task status, acceptance evidence and product docs after each implementation slice. Separate baseline failures from new regressions.

