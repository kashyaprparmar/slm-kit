"""Worker-only advanced optimization hooks.

All imports in this module happen inside training workers.  The API process
uses ``app.optimizations`` for the corresponding metadata and validation.
"""

from __future__ import annotations

from typing import Any

from app.domain import RunConfig


def lora_config_kwargs(cfg: RunConfig) -> dict[str, Any]:
    """Map the canonical LoRA options to PEFT's native constructor safely."""
    if cfg.lora.init_method == "standard":
        return {}
    if cfg.lora.init_method == "pissa":
        return {"init_lora_weights": "pissa"}
    # Validation prevents execution of unverified initializers.  Keep this
    # branch defensive for direct worker invocations.
    raise RuntimeError(f"PEFT initializer '{cfg.lora.init_method}' is not enabled.")


def trainer_optimizers(model, cfg: RunConfig):
    """Return explicit Trainer optimizers only for supported custom paths."""
    if cfg.lora.lora_plus_lr_ratio is None:
        return None
    import torch
    from peft.optimizers import create_loraplus_optimizer

    optimizer = create_loraplus_optimizer(
        model,
        torch.optim.AdamW,
        lr=cfg.optim.learning_rate,
        loraplus_lr_ratio=cfg.lora.lora_plus_lr_ratio,
        weight_decay=cfg.optim.weight_decay,
    )
    return optimizer, None
