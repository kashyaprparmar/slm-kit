# Model compatibility

Metadata is required to establish architecture support: a repository name alone
is never sufficient, including when metadata is unavailable offline or gated.
Unknown causal architectures remain experimental across generic training and
inference. Adapter metadata does not override an unsupported base architecture.
Scratch GPT artifacts use the native runtime and scratch training backend;
PEFT, HF SFT, and safetensors export are not supported for scratch weights.

The capability contract retains existing lowercase states, including the legacy
`not_installed`, and adds `missing_dependency` and `incompatible` for future
runtime preflight results. Metadata inspection is not proof of a successful
runtime load. Seq2seq training remains unsupported by the current worker.

SLM Kit does not assume every repository supports every operation. It inspects
small model and tokenizer metadata first, then uses one central capability
registry for training validation, recommendations, inference, merging,
quantization, and export decisions.

## Capability states

- **supported** — covered by a known causal-language-model family and the
  selected path.
- **experimental** — metadata looks compatible, but the exact architecture or
  optimized backend is not fully tested. The generic Transformers loader is the
  usual fallback.
- **requires conversion** — usable after producing a compatible artifact, such
  as GGUF for Ollama/llama.cpp or a merged model from a PEFT adapter.
- **not installed** — the architecture may be viable, but that worker/runtime
  is not part of the current environment.
- **unsupported** — the current causal-LM pipeline cannot perform that operation.

## Recognized causal-LM families

The metadata registry has explicit rules for Qwen/Qwen MoE, Llama/SmolLM,
Mistral/Mixtral, Gemma, Phi, DeepSeek, Falcon, GPT-NeoX/StableLM, GPT-Neo/J,
OPT, BLOOM, and MPT. Unknown decoder-only causal models are marked experimental
rather than silently promised as supported. Encoder-only, encoder-decoder, and
multimodal training are rejected because the current datasets and workers are
text causal-LM focused.

## What inspection checks

`POST /api/registry/inspect` reports the resolved revision, architecture,
model type, parameter count when available, context length, vocabulary,
tokenizer, quantization metadata, suggested PEFT target modules, warnings, and
the full operation-by-operation capability map. Inspection reads local metadata
or small Hugging Face files; it does not load model weights.

Hugging Face metadata and parameter caches include the requested `revision`, so
two revisions of the same repository cannot reuse stale compatibility results.
For reproducibility, pin a commit hash instead of relying on a moving `main`.

Runtime loading remains the final check: gated repositories need a token,
custom-code repositories require `SLMKIT_TRUST_REMOTE_CODE=true`, and the
installed Transformers/PEFT/Unsloth versions must support the model class.
