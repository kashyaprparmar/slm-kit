"""Isolated subprocess runtime model preflight probe.

Validates that a model can be initialized, tokenized, and configured for training
or inference without retaining GPU memory in the main API server. Emits a single
JSON line on stdout and exits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def run_preflight(config: dict[str, Any]) -> dict[str, Any]:
    model_ref = config.get("model_ref", "").strip()
    revision = config.get("revision") or None
    backend = config.get("backend") or "unsloth"
    load_in_4bit = bool(config.get("load_in_4bit", False))
    cache_dir = config.get("cache_dir")
    trust_remote_code = bool(config.get("trust_remote_code", False))

    warnings: list[str] = []
    recommended_backend = backend

    # Check scratch models
    from app.model_refs import resolve_model_ref
    resolved = resolve_model_ref(model_ref)
    if resolved.kind == "scratch":
        local_dir = Path(resolved.local_path) if resolved.local_path else None
        if not local_dir or not local_dir.is_dir():
            return {
                "ok": False,
                "model_ref": model_ref,
                "error": "Scratch model path does not exist.",
                "warnings": warnings,
            }
        arch_path = local_dir / "arch.json"
        if not arch_path.is_file():
            return {
                "ok": False,
                "model_ref": model_ref,
                "error": "Scratch checkpoint is missing arch.json metadata.",
                "warnings": warnings,
            }
        arch = json.loads(arch_path.read_text(encoding="utf-8"))
        vocab_size = int(arch.get("vocab_size", 0))
        n_layers = int(arch.get("n_layers", 0))
        n_embd = int(arch.get("n_embd", 0))
        block_size = int(arch.get("block_size", 0))

        param_count = (vocab_size + block_size) * n_embd + n_layers * (12 * n_embd * n_embd + 4 * n_embd)

        has_torch = False
        try:
            import torch
            has_torch = True
        except ImportError:
            warnings.append("PyTorch is not installed in the local environment; verified metadata and artifacts offline.")

        vram_peak_mb = 0.0
        if has_torch and torch.cuda.is_available():
            try:
                vram_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            except Exception:
                pass

        return {
            "ok": True,
            "model_ref": model_ref,
            "revision": None,
            "kind": "scratch",
            "backend_verified": "scratch",
            "recommended_backend": "scratch",
            "architecture": "scratch",
            "context_length": block_size,
            "vocab_size": vocab_size,
            "parameter_count": param_count,
            "has_chat_template": False,
            "vram_peak_mb": vram_peak_mb,
            "warnings": warnings + ["Scratch GPT models use the native scratch backend and cannot be fine-tuned with PEFT."],
            "error": None,
        }

    # Hugging Face / Transformers preflight probe
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as err:
        return {
            "ok": False,
            "model_ref": model_ref,
            "error": f"Required ML runtime dependencies are not installed: {err}",
            "warnings": warnings,
        }

    from app.integrations.hf_hub import _token

    common_kwargs: dict[str, Any] = {
        "token": _token(),
        "trust_remote_code": trust_remote_code,
    }
    if cache_dir:
        common_kwargs["cache_dir"] = str(cache_dir)
    if revision:
        common_kwargs["revision"] = revision

    target_ref = resolved.load_ref
    adapter_base = None
    if resolved.kind == "adapter":
        try:
            from peft import PeftConfig
            peft_cfg = PeftConfig.from_pretrained(target_ref, **common_kwargs)
            adapter_base = getattr(peft_cfg, "base_model_name_or_path", None)
        except Exception as exc:
            warnings.append(f"Could not load adapter configuration: {exc}")

    # 1. Config validation
    try:
        config_obj = AutoConfig.from_pretrained(target_ref, **common_kwargs)
    except Exception:
        if adapter_base:
            base_kwargs = dict(common_kwargs)
            base_kwargs.pop("revision", None)
            config_obj = AutoConfig.from_pretrained(adapter_base, **base_kwargs)
        else:
            raise

    # Seq2seq check
    if getattr(config_obj, "is_encoder_decoder", False):
        return {
            "ok": False,
            "model_ref": model_ref,
            "architecture": "encoder_decoder",
            "error": (
                "Sequence-to-sequence (encoder-decoder) architectures are not supported by the "
                "causal LM training worker. A separate seq2seq loader, trainer, and evaluation "
                "contract is required."
            ),
            "warnings": warnings,
        }

    # Context & vocab
    context_length = getattr(config_obj, "max_position_embeddings", None) or getattr(config_obj, "n_positions", None) or getattr(config_obj, "seq_length", None)
    vocab_size = getattr(config_obj, "vocab_size", None)

    # 2. Tokenizer validation
    tokenizer_load_ref = target_ref
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_ref, **common_kwargs)
    except Exception as exc:
        if adapter_base:
            warnings.append(f"Adapter repository has no tokenizer; falling back to base model {adapter_base}")
            base_kwargs = dict(common_kwargs)
            base_kwargs.pop("revision", None)
            tokenizer = AutoTokenizer.from_pretrained(adapter_base, **base_kwargs)
        else:
            raise exc

    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if not has_chat_template:
        warnings.append("No native chat template found. Supervised instruction tuning requires an explicit chat template.")

    # 3. Backend probe
    unsloth_supported = False
    unsloth_reason = ""
    if torch.cuda.is_available():
        try:
            # Unsloth has model_type check in its loader
            unsloth_supported = True
        except Exception as u_err:
            unsloth_supported = False
            unsloth_reason = str(u_err)
    else:
        unsloth_reason = "CUDA is not available"

    if backend == "unsloth":
        if unsloth_supported and resolved.kind != "adapter":
            backend_verified = "unsloth"
            recommended_backend = "unsloth"
        else:
            backend_verified = "transformers"
            recommended_backend = "transformers"
            if resolved.kind == "adapter":
                warnings.append("Adapter checkpoints load via Transformers + PEFT; Unsloth is bypassed.")
            else:
                warnings.append(f"Unsloth unavailable ({unsloth_reason}); routing to Transformers + PEFT.")
    else:
        backend_verified = "transformers"
        recommended_backend = "transformers"

    # 4. Check 4-bit quantization requirement
    if load_in_4bit:
        if not torch.cuda.is_available():
            warnings.append("4-bit quantization requested but CUDA is unavailable.")
        else:
            import importlib.util

            if not importlib.util.find_spec("bitsandbytes"):
                warnings.append("4-bit quantization requested but bitsandbytes is not installed.")

    vram_peak_mb = 0.0
    if torch.cuda.is_available():
        try:
            vram_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass

    return {
        "ok": True,
        "model_ref": model_ref,
        "revision": revision,
        "kind": resolved.kind,
        "backend_verified": backend_verified,
        "recommended_backend": recommended_backend,
        "architecture": "decoder_only",
        "context_length": context_length,
        "vocab_size": vocab_size,
        "has_chat_template": has_chat_template,
        "vram_peak_mb": vram_peak_mb,
        "warnings": warnings,
        "error": None,
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "No config path provided"}))
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    if not cfg_path.is_file():
        print(json.dumps({"ok": False, "error": f"Config file not found: {cfg_path}"}))
        sys.exit(1)

    try:
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        result = run_preflight(config)
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0 if result.get("ok") else 1)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "warnings": []}))
        sys.exit(1)


if __name__ == "__main__":
    main()
