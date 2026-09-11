# Local model deployment

SLM Kit can keep one completed model loaded behind a local,
OpenAI-compatible HTTP endpoint. The same deployment path supports:

- a completed SLM Kit run through its stable `run:<id>` reference;
- a local or Hugging Face causal-language-model checkpoint;
- a PEFT/LoRA adapter (the base model is resolved automatically); and
- a model trained in the Pretraining Studio with the scratch backend.

## Deploy from the UI

1. Finish a training run.
2. Open **Model Registry**.
3. Click **Deploy** beside the run or local artifact.
4. Wait for the deployment card to show **Serving**.

The endpoint is `http://localhost:8802/v1`. Only one model is deployed at a
time. Click **Stop** to terminate the isolated model-server process and release
its GPU memory.

## Call the endpoint

```bash
curl http://localhost:8802/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ignored-for-single-model-server",
    "messages": [{"role": "user", "content": "Explain gross margin."}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

The server also exposes `GET /health`, `GET /v1/models`, and
`POST /v1/completions`. Chat/completion requests support normal JSON and
OpenAI-style server-sent-event streaming with `"stream": true`.

The dedicated **Model Serving** page also detects Ollama, lists installed and
loaded models, manages one loaded model, tests it, and unloads it. Ready GGUF
artifacts can be imported through the supported `ollama create` Modelfile
workflow when the backend can access the native Ollama CLI.

## GPU ownership

Training, evaluation, playground generation, and deployment share one GPU:

- a deployment blocks new eval/playground work with a clear conflict response;
- queued training waits until the deployment is explicitly stopped;
- stopping or restarting the backend terminates the deployment subprocess;
- a warm vLLM playground server is stopped when deployment begins.

This prevents two large models from being loaded into VRAM at once.

## Configuration and security

Native installs bind deployment to `127.0.0.1` by default. Docker publishes
port 8802 on the host loopback interface only. This endpoint has no API-key
authentication, so do not expose it to a LAN or the internet without a reverse
proxy that adds TLS and authentication.

Use `SLMKIT_DEPLOY_PORT` to change the port (update the Docker port mapping too)
and `SLMKIT_DEPLOY_STARTUP_TIMEOUT_SECONDS` for unusually slow model loads.

Repositories that execute custom Hugging Face model code remain disabled by
default. Set `SLMKIT_TRUST_REMOTE_CODE=true` only for repositories you trust.
