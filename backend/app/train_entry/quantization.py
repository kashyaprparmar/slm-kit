"""Worker-only quantization translation shared by trainable and reference loaders."""
from __future__ import annotations


def bitsandbytes_config(cfg, torch, config_type):
    """Build the one authoritative BitsAndBytesConfig for a validated RunConfig."""
    mode = cfg.quantization.mode
    if mode == "none":
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("Bitsandbytes training quantization requires a CUDA GPU.")
    if mode == "int8":
        return config_type(load_in_8bit=True)
    compute_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(cfg.quantization.compute_dtype)
    compute_dtype = compute_dtype or (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    storage_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "uint8": torch.uint8,
    }.get(cfg.quantization.storage_dtype)
    return config_type(
        load_in_4bit=True,
        bnb_4bit_quant_type=mode,
        bnb_4bit_use_double_quant=cfg.quantization.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
        **({"bnb_4bit_quant_storage": storage_dtype} if storage_dtype is not None else {}),
    )
