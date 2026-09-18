"""Architecture-aware trainable-module selection without importing ML stacks.

The functions operate on the standard ``named_modules``/``named_parameters``
protocol exposed by PyTorch models. Keeping torch imports out of this module lets
the API and unit tests inspect policies without initializing CUDA.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.domain import FreezeParams, LoraParams


class ModuleSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class ParameterSummary:
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_percentage: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
            "trainable_percentage": self.trainable_percentage,
        }


@dataclass(frozen=True)
class ModuleSelection:
    strategy: str
    matched_modules: tuple[str, ...]


_LAYER = re.compile(r"(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)(?:\.|$)")
_ATTENTION = ("attn", "attention", "self_attn", "query", "key", "value")
_MLP = ("mlp", "ffn", "feed_forward", "feedforward", "expert")
_EMBEDDING = ("embed", "embedding", "wte", "tok_embeddings")
_HEAD = ("lm_head", "output_projection", "embed_out")
_NORM = ("norm", "layernorm", "layer_norm", "ln_")


def parameter_summary(model) -> ParameterSummary:
    parameters = list(model.parameters())
    total = sum(int(parameter.numel()) for parameter in parameters)
    trainable = sum(int(parameter.numel()) for parameter in parameters if parameter.requires_grad)
    frozen = total - trainable
    return ParameterSummary(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=frozen,
        trainable_percentage=round(100 * trainable / total, 6) if total else 0.0,
    )


def _is_linear_like(module) -> bool:
    class_name = module.__class__.__name__.lower()
    return (
        class_name in {"linear", "conv1d", "linear4bit", "linear8bitlt"}
        or (hasattr(module, "in_features") and hasattr(module, "out_features"))
    )


def _matches_any(name: str, selectors: Iterable[str]) -> bool:
    return any(name == selector or name.endswith(f".{selector}") for selector in selectors)


def discover_lora_targets(
    model,
    config: LoraParams,
    *,
    suggested_modules: Iterable[str] = (),
    excluded_modules: Iterable[str] = (),
) -> ModuleSelection:
    excluded = tuple(excluded_modules)
    linear = {
        name
        for name, module in model.named_modules()
        if name and _is_linear_like(module) and not _matches_any(name, _HEAD)
        and not any(_matches_any(".".join(name.split(".")[:index]), excluded)
                    for index in range(1, len(name.split(".")) + 1))
    }
    strategy = config.target_strategy
    if strategy == "custom":
        if not config.target_modules:
            raise ModuleSelectionError("Custom LoRA targeting requires at least one module name.")
        requested = config.target_modules
        matched = {name for name in linear if _matches_any(name, requested)}
        missing = [selector for selector in requested if not any(_matches_any(name, [selector]) for name in linear)]
        if missing:
            raise ModuleSelectionError(f"LoRA target modules were not found: {', '.join(missing)}.")
    elif strategy == "attention":
        matched = {name for name in linear if any(token in name.lower() for token in _ATTENTION)}
    elif strategy == "mlp":
        matched = {name for name in linear if any(token in name.lower() for token in _MLP)}
    elif strategy == "all_linear":
        matched = linear
    else:
        suggestions = list(suggested_modules)
        matched = {name for name in linear if _matches_any(name, suggestions)} if suggestions else set()
        if not matched:
            matched = linear
    if not matched:
        raise ModuleSelectionError(
            f"LoRA target strategy '{strategy}' did not match any trainable linear modules."
        )
    return ModuleSelection(strategy=strategy, matched_modules=tuple(sorted(matched)))


def _set_trainable(parameter, enabled: bool) -> None:
    if hasattr(parameter, "requires_grad_"):
        parameter.requires_grad_(enabled)
    else:
        parameter.requires_grad = enabled


def _layer_prefixes(parameter_names: Iterable[str]) -> list[tuple[int, str]]:
    found: dict[tuple[int, str], None] = {}
    for name in parameter_names:
        match = _LAYER.search(name)
        if match:
            prefix = name[: match.end()].rstrip(".")
            found[(int(match.group(1)), prefix)] = None
    return sorted(found)


def apply_freeze_policy(model, config: FreezeParams) -> tuple[ParameterSummary, tuple[str, ...]]:
    named = list(model.named_parameters())
    if not named:
        raise ModuleSelectionError("The model exposes no parameters to freeze.")
    for _, parameter in named:
        _set_trainable(parameter, False)

    prefixes = _layer_prefixes(name for name, _ in named)
    if config.last_n_layers and not prefixes:
        raise ModuleSelectionError(
            "Could not identify transformer layer indices for last-N-layer freeze tuning."
        )
    layer_indices = sorted({index for index, _ in prefixes})
    selected_indices = set(layer_indices[-config.last_n_layers :]) if config.last_n_layers else set()
    selected_prefixes = tuple(prefix for index, prefix in prefixes if index in selected_indices)

    custom_matches = {
        selector: any(_matches_any(name.rsplit(".", 1)[0], [selector]) for name, _ in named)
        for selector in config.selected_modules
    }
    missing = [selector for selector, matched in custom_matches.items() if not matched]
    if missing:
        raise ModuleSelectionError(f"Freeze-tuning modules were not found: {', '.join(missing)}.")

    selected: list[str] = []
    for name, parameter in named:
        module_name = name.rsplit(".", 1)[0]
        lower = module_name.lower()
        enabled = (
            any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in selected_prefixes)
            or (config.train_embeddings and any(token in lower for token in _EMBEDDING))
            or (config.train_lm_head and any(token in lower for token in _HEAD))
            or (config.train_norms and any(token in lower for token in _NORM))
            or _matches_any(module_name, config.selected_modules)
        )
        if enabled:
            _set_trainable(parameter, True)
            selected.append(name)
    summary = parameter_summary(model)
    if not summary.trainable_parameters:
        raise ModuleSelectionError("Freeze tuning selected zero trainable parameters.")
    return summary, tuple(selected)
