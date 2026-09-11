"""Deterministic training defaults; estimates are planning aids, not guarantees."""
import math

from app.domain import HardwareProfile, Method, RunConfig
from app.integrations import estimator
from app.integrations.hf_hub import get_model_spec
from app.model_refs import resolve_model_ref


def recommend_settings(cfg: RunConfig, hw: HardwareProfile, preset: str, rows=0, tokens=0):
    result = cfg.model_copy(deep=True)
    if cfg.backend == "scratch":
        from app.backends.scratch_backend import ScratchBackend

        def estimate(candidate):
            return ScratchBackend().estimate_footprint(candidate, hw)
    else:
        resolved = resolve_model_ref(cfg.base_model)
        spec = get_model_spec(resolved.base_model or resolved.load_ref)

        def estimate(candidate):
            return estimator.estimate(candidate, hw, spec)
    reasons = {}
    avg_tokens = max(128, math.ceil(tokens / rows)) if rows and tokens else 1024
    context = min(2048, max(256, 2 ** math.ceil(math.log2(avg_tokens))))
    if preset in {"fast", "lowest_memory"}:
        context = min(context, 512)
    if result.arch:
        context = min(context, result.arch.block_size)
    result.train.max_seq_length = context
    reasons["Context Length"] = f"{context} tokens, using the dataset's approximate average length ({avg_tokens:,}) and preset limits."
    if preset == "lowest_memory" and result.backend != "scratch":
        result.method = Method.QLORA
    result.load_in_4bit = result.method == Method.QLORA
    result.gradient_checkpointing = True
    result.optim.optimizer = "adamw_8bit" if hw.gpu_name and result.method != Method.FULL else "adamw_torch"
    result.train.per_device_batch_size = 1
    for batch in (8, 4, 2, 1) if preset != "lowest_memory" else (1,):
        trial = result.model_copy(deep=True)
        trial.train.per_device_batch_size = batch
        if estimate(trial).fit.value == "fits":
            result = trial
            break
    result.train.gradient_accumulation = max(1, 16 // result.train.per_device_batch_size)
    result.train.epochs = 3 if preset == "best_quality" and rows >= 300 else 1
    result.train.max_steps = min(100, max(10, math.ceil(rows / 16))) if preset == "fast" else None
    if cfg.backend == "scratch":
        result.train.max_steps = 100 if preset == "fast" else 1000
    result.optim.learning_rate = 2e-5 if result.method == Method.FULL else 2e-4
    if result.task.value == "continued_pretrain":
        result.optim.learning_rate /= 2
    result.lora.r = 8 if preset == "lowest_memory" else 32 if preset == "best_quality" else 16
    result.lora.alpha = result.lora.r * 2
    result.lora.dropout = 0.05 if rows and rows < 300 else 0.0
    result.optim.lr_scheduler = "cosine"
    result.optim.warmup_ratio = 0.05
    steps = result.train.max_steps or max(1, math.ceil(rows / 16 * result.train.epochs))
    result.train.save_steps = max(10, min(200, steps // 4))
    result.train.logging_steps = max(1, min(10, steps // 20))
    reasons.update({
        "Batch Size": f"{result.train.per_device_batch_size} examples per device; largest conservative candidate with comfortable estimated headroom.",
        "Batch Accumulation": f"{result.train.gradient_accumulation} batches, targeting an effective batch of 16 examples.",
        "Learning Rate": f"{result.optim.learning_rate:g}; full-weight updates and continued pretraining use smaller updates.",
        "Training Length": "Fast uses a short step limit. Other presets use epochs; small datasets stay at one pass to reduce overfitting.",
        "Adapter Settings": f"Rank {result.lora.r}, alpha {result.lora.alpha}, dropout {result.lora.dropout}; balances capacity against memory and dataset size.",
        "Optimizer": result.optim.optimizer + "; chosen from GPU availability and whether all weights are trained.",
        "Schedule": "Cosine decay, 5% warmup, about four checkpoints and twenty progress reports per run.",
        "Quantization": "4-bit base weights for QLoRA; full precision for the other methods.",
        "Memory": "Gradient checkpointing enabled. Free VRAM and system RAM are measured when available.",
    })
    if hw.ram_free_mb and hw.ram_free_mb < 4096:
        reasons["System RAM"] = "Less than 4 GB RAM is currently free. Close other applications before loading model weights."
    return {"config": result.model_dump(mode="json"), "reasons": reasons, "estimate": estimate(result),
            "preset": preset, "warnings": ["Heuristic recommendations; inspect validation before starting.",
            "Token counts use a character estimate, not the model tokenizer."]}
