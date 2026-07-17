# Dataset Manager

Upload, validate, and preview your data **before** it ever feeds a run. Bad data
is the #1 cause of wasted training time, so this page checks it first.

## Dataset types

| Type | Used by | Format |
|---|---|---|
| **Instruction / chat** | Fine-Tuning Studio | JSONL/JSON/CSV with Q&A pairs |
| **Domain corpus (text)** | Domain Adaptation | plain `.txt` (raw prose) |
| **Pretraining corpus (text)** | Pretraining Studio | plain `.txt` (raw prose) |
| **Evaluation set** | Eval Lab | JSONL/JSON with Q&A pairs |

## Supported file formats

`.jsonl` · `.json` · `.csv` · `.txt` · `.parquet`
(Parquet needs `pip install pyarrow`.)

### Instruction/eval data — accepted shapes

Any of these per row works (the app auto-detects the fields):

```jsonl
{"instruction": "What is gross margin?", "output": "Gross margin is..."}
{"prompt": "...", "response": "..."}
{"question": "...", "answer": "...", "input": "optional context"}
{"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Uploading

1. Pick the **dataset type** from the dropdown.
2. **Drag a file** onto the drop zone (or click to browse).
3. It's validated instantly and added to your list.

Filenames are sanitized and de-duplicated automatically — uploading two files
named the same won't overwrite anything.

## The validation report (right panel)

Select any dataset to see:

- **Pass/fail banner.** Green "Validation passed" means it's ready. Red means
  errors you must fix before launching a run against it.
- **Specific issues with line numbers** — e.g. *"L42: invalid JSON"*,
  *"L7: missing an output/response/answer field"*. No vague failures.
- **Stats:** row count, estimated tokens, file size.
- **Token histogram:** distribution of tokens per row, so you can spot rows
  that are way too long for your sequence length.
- **Sample rows:** the first few records, rendered.

### Warnings you might see (not blocking, but worth heeding)

- *"Only N examples…"* — instruction fine-tunes usually want **≥ ~50** examples
  (ideally 300+) for a meaningful result.
- *"~X estimated tokens is below the suggested…"* — domain corpora want
  ~100k+ tokens, from-scratch pretraining wants ~1M+. Below that you'll get a
  toy-quality model.
- *"N duplicate/near-duplicate rows detected."*

## Bundled sample datasets

Click **Install sample datasets** (or they auto-install on first boot) to get
four small, license-clean starters — one per pillar plus an eval set — so you
can run the entire pipeline on day one:

- **Tiny Story Corpus** — from-scratch pretraining
- **Finance Domain Corpus** — domain adaptation
- **Finance QA (instruction)** — QLoRA fine-tuning
- **Finance QA (eval)** — evaluation / comparison

Sample datasets are marked with a "sample" tag and can't be deleted.

## Tips

- Fix **all red errors** before launching — the studios won't let you start a
  run on an invalid dataset.
- Use the token histogram to choose a sensible **max sequence length** in the
  studio (don't pay for 2048 tokens if your rows are 200).
