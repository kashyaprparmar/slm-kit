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

## Optional managed engines and normalized APIs

SGLang is optional. Configure an external OpenAI service with `SLMKIT_SGLANG_URL` (default localhost:8803), or install SGLang separately in a compatible Linux/CUDA environment for managed startup. An absent package reports **Not installed · optional**. See the [official SGLang launch reference](https://docs.sglang.ai/backend/pd_disaggregation.html) for supported launch conventions.

Installed vLLM/SGLang may be started through Serve Model. Advanced controls are shown only for flags advertised by the installed CLI; request validation also checks detected CUDA/GPU count and local model context/quantization metadata. External services keep their own lifecycle. SLM Kit stops only workers it owns. LoRA/quantization profiles, tool/reasoning parsers and speculative draft compatibility remain gated until verified; see the [official vLLM argument reference](https://docs.vllm.ai/en/v0.8.5/serving/engine_args.html) for the upstream controls.

The main backend provides GET `/v1/models`, POST `/v1/chat/completions`, POST `/v1/completions` and POST `/v1/scores`. Discovery returns `provider::model` IDs; use those IDs with the main backend URL. Native text completion uses raw text and supports streaming. Native reward deployment provides scalar scores and rejects generation; choose a registered reward artifact in Reward model scoring, start it, then score text. Scores are model-specific. Unconfigured tools/reasoning are explicitly rejected by the gateway.

POST `/api/serving/{provider}/start` accepts model and optional validated engine options; POST `/api/serving/{provider}/stop` stops a managed worker. GET `/api/serving/{provider}/options` reports installed CLI controls. Real optional CUDA provider compatibility remains unverified by the CPU-only fixture suite.
