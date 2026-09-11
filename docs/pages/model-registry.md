# Model Registry

Your locally-trained artifacts **and** your Hugging Face repos, in one place.
Publish, import, export to GGUF, and deploy a local OpenAI-compatible endpoint from here.

## Capability banner

At the top, two badges tell you what's available:
- **HF token configured / not set** — needed to publish and to list your repos.
- **GGUF export available / needs llama.cpp** — needs `SLMKIT_LLAMACPP_DIR`.

See [Configuration](../configuration.md) to enable either.

## Left side — "Trained here"

### Ready to publish
Finished runs expose a stable `run:<id>` reference that is also available in
the Eval Lab model picker. Each has these actions:

- **Deploy** starts an isolated local server on `http://localhost:8802/v1`.
  The deployment card shows the active model and provides copy/stop controls.

- **Publish** → opens a dialog:
  - **Repo id** (e.g. `yourname/finance-qlora`) — use `username/name` to target
    your account.
  - **Private** toggle.
  - On publish, SLM Kit **auto-generates a model card** from the run's stored
    metadata (base model, method, dataset, hyperparameters, training time, final
    metrics, hardware) and uploads the model. You get the repo URL back.

- **GGUF export** → pick a quant type and click **GGUF**:
  - **q4_k_m** (recommended for 8 GB), **q5_k_m**, **q8_0**, **f16**.
  - Runs in the background (not on the GPU queue). The new artifact appears in
    the **Artifacts** list with a live status: *quantizing → GGUF* (or *failed*
    with guidance if llama.cpp isn't set up).
  - Note: GGUF needs a **full/merged** model. Full fine-tunes and from-scratch
    models convert directly; LoRA/QLoRA adapters must be merged first.

### Artifacts
Everything you've published or exported: published models (with HF links),
imported models, and GGUF files (with their local path). Status badges show
`published`, `GGUF`, `local`, `quantizing`, or `failed`.

## Right side — "Hugging Face"

- **Import a model:** paste any `username/model-name` and click **Import** to
  download it into your local models folder — ready to fine-tune further or test
  in the Eval Lab. (repo ids are validated; junk like `..` is rejected.)
- **Your repos:** lists the models on your HF account (needs a token). Click any
  to open it on huggingface.co.

## Typical flows

**Publish what you trained**
Fine-Tuning Studio → run finishes → Registry → *Ready to publish* → **Publish**
→ pick repo name + private → done, URL saved to the run.

**Round-trip for more training**
Registry → **Import** `yourname/model` → it lands in your models folder → use
its local path (or repo id) as the base in a studio.

**Ship a GGUF for llama.cpp / Ollama / LM Studio**
(Full/merged model) → Registry → **GGUF** with `q4_k_m` → grab the file from the
Artifacts list's path.

## Requirements recap

- **Publish / import / list repos** → `SLMKIT_HF_TOKEN`.
- **GGUF export** → a built llama.cpp at `SLMKIT_LLAMACPP_DIR`.

Neither is required for training — they're optional conveniences.
