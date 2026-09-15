# System & Diagnostics

This page shows the health of the local control plane and the dependencies used
by training and serving.

## Runtime services

The service cards report:

- FastAPI readiness;
- a real SQLite `SELECT 1` database probe;
- whether the training queue worker is ready, busy, or unhealthy;
- current shared GPU ownership; and
- independently probed Transformers, Ollama, and vLLM provider states.

An installed but stopped provider is shown as **ready**, an active provider as
**busy**, and an unreachable optional provider as **unavailable**. Provider
failures are isolated, so an offline Ollama or vLLM service does not hide the
health of the API, database, or other providers.

## Dependency checks and activity

**Refresh Checks** runs local GPU, Python-library, Hugging Face, llmfit, and
llama.cpp diagnostics. Each result includes setup guidance.

**Recent requests & failures** displays structured local activity, HTTP status,
correlation metadata, and request duration. Backend logs are also available at
`GET /api/system/logs`; no external observability service is required.

Use this page first when a studio is disabled or an optional provider does not
appear. For command-line checks, see [Troubleshooting](../troubleshooting.md).
