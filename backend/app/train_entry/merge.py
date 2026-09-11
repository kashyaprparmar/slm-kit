"""Merge a PEFT adapter into its base model in an isolated subprocess."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.train_entry.model_runtime import load_runtime


def main(config_path: str) -> int:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    print(f"Loading adapter {cfg['model_ref']}", flush=True)
    runtime = load_runtime(cfg["model_ref"])
    if runtime.spec.kind != "adapter" or not hasattr(runtime.model, "merge_and_unload"):
        raise ValueError("This model is not a mergeable PEFT adapter.")
    print("Merging adapter weights into the base model", flush=True)
    merged = runtime.model.merge_and_unload(progressbar=True)
    merged.save_pretrained(str(output), safe_serialization=True)
    runtime.tokenizer.save_pretrained(str(output))
    print(f"Merged model saved to {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1]))
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise
