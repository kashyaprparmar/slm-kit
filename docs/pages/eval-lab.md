# Testing & Eval Lab

Chat with, and rigorously score, **any model** — one you trained here or any
model straight from Hugging Face. Two tabs: **Playground** and **Evaluate**.

Both load a model on the GPU, so they respect the single-GPU rule: if a training
job is running, the Eval Lab waits and tells you the GPU is busy (you'll get a
clear "GPU is busy" message rather than a crash).

---

## Playground (interactive chat)

Quick, hands-on testing.

1. **Model:** pick a curated model, a completed SLM Kit run, or paste any HF repo id / local path.
2. **Prompt:** type anything.
3. **Generation params** (right panel):
   - **Max new tokens** — response length cap
   - **Temperature** — 0 = deterministic/greedy, higher = more creative
   - **Top-p**, **Top-k** — sampling controls
   - **Repetition penalty** — discourages loops
4. **Generate** → tokens **stream in live**, word by word.

First use of a model loads it (a few seconds to a minute depending on size).

---

## Evaluate (the eval harness)

Score one or more models on a held-out set and compare them side by side.

### Set it up

1. **Models to evaluate:** start with one; click **"Compare another"** to add a
   second (or third) for side-by-side. Each can be a curated model, an HF repo
   id, or a local path — mix in-app and external models freely.
2. **Dataset:** pick an *eval* or *instruction* dataset.
3. **Max samples:** keep small (e.g. 25) for quick iterations.
4. **Metrics** (tick what you want):
   - **Exact match** — normalized string equality
   - **Token F1** — soft word-overlap score
   - **ROUGE-L** — needs the `[eval]` extra installed
   - **BLEU** — needs the `[eval]` extra installed
   - **Perplexity** — model's confidence on the reference answers
   - **LLM-as-judge** — an external model rates answer quality (only enabled if
     you've configured a judge API key; see [Configuration](../configuration.md))
5. **Run evaluation.**

### While it runs

A **progress bar per model** shows how many samples are done. Each model is
evaluated in turn (VRAM is freed between models).

### The results

- **Scorecard:** a table of every metric × every model. In a comparison, the
  **best value per metric is highlighted with a ★** (and perplexity is treated
  as lower-is-better).
- **Sample comparisons:** for each example, the question, the reference answer,
  and each model's actual answer side by side — so you can *see* the difference,
  not just the numbers.
- **Past evaluations:** every run is saved; click one to reopen its scorecard.

## Which metric should I trust?

- **Exact match / BLEU / ROUGE** suit short, factual answers.
- **Token F1** is a softer signal for paraphrased answers.
- **Perplexity** measures fluency/confidence, not correctness.
- **LLM-as-judge** is best for open-ended answers where string metrics are
  unfair — but it costs API calls and depends on the judge model.

Use several together; no single number tells the whole story.

## Notes

- Everything here works on **any** model — you do **not** need to have trained it
  in SLM Kit.
- Completed in-app runs use stable `run:<id>` references. The shared loader
  resolves full checkpoints, PEFT adapters, and scratch-pretrained models.
- Without the `[eval]` extra, ROUGE/BLEU are simply skipped (not errored); the
  other metrics still compute.
- Without a judge API key, the LLM-as-judge option is disabled and everything
  else works normally.
