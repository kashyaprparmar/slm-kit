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
    runtime = load_runtime(cfg["model_ref"], for_merge=True, base_revision=cfg.get("base_revision"))
    if runtime.spec.kind != "adapter" or not hasattr(runtime.model, "merge_and_unload"):
        raise ValueError("This model is not a mergeable PEFT adapter.")
    if any(getattr(runtime.model, key, False) for key in ("is_loaded_in_4bit", "is_loaded_in_8bit")) or getattr(runtime.model.config, "quantization_config", None):
        raise ValueError("Merging into low-bit quantized weights is unsafe and is not supported.")
    print("Merging adapter weights into the base model", flush=True)
    merged = runtime.model.merge_and_unload(progressbar=True, safe_merge=True)
    merged.save_pretrained(str(output), safe_serialization=True)
    runtime.tokenizer.save_pretrained(str(output))
    lineage = {**runtime.load_metadata, "base_model": runtime.spec.base_model,
               "base_revision": runtime.load_metadata.get("base_revision") or getattr(runtime.model.config, "_commit_hash", None),
               "architectures": getattr(runtime.model.config, "architectures", None),
               "tokenizer_size": len(runtime.tokenizer), "safe_merge": True, "quantized_base": False}
    (output / "merge-lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    print(f"Merged model saved to {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1]))
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise
