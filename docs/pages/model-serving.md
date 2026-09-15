# Model Serving

The **Model Serving** page provides one normalized view over three inference
providers:

- **Transformers** — managed by SLM Kit as an isolated subprocess. It accepts a
  Hugging Face id, local model directory, stable `run:<id>` reference, PEFT
  adapter, or SLM Kit scratch checkpoint.
- **vLLM** — a dedicated, optional Docker service with an OpenAI-compatible API.
  Start and stop it through Docker; the page discovers its models through
  `/v1/models` and can send test prompts.
- **Ollama** — an optional external service. SLM Kit lists installed/loaded
  models, can keep one selected model loaded, unload the managed model, and
  import a ready GGUF artifact when the Ollama CLI is reachable.

## Typical workflow

1. Select a provider.
2. For Transformers, choose a model and click **Serve Model**. For Ollama,
   choose an installed model. For vLLM, start its Compose profile first.
3. Enter a prompt under **Test served model** and click **Test Model**.
4. Stop a managed Transformers or Ollama server when finished to release VRAM.

Transformers and vLLM expose OpenAI-compatible `/v1/chat/completions` APIs.
Ollama uses its native `/api/generate` and `/api/chat` routes.

## GPU ownership

Provider status probes run concurrently and one unavailable provider does not
break the page. SLM Kit also detects a dedicated vLLM service and Ollama models
loaded outside its own process. Training, evaluation, generation, deployment,
and adapter merging reject or wait on those external owners so two large GPU
workloads are not started together.

See [Local deployment](../deployment.md) for endpoint examples and
[Docker](../docker.md#dedicated-vllm-service-optional) for vLLM startup.
