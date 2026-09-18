"""Shared preference/alignment trainer orchestration.

Objective adapters choose a TRL trainer and loss while model loading, canonical
dataset rendering, callbacks, checkpoints, events, and lineage stay shared.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from app.backends.base import RunContext
from app.core.events import ArtifactEvent, CheckpointEvent, LogEvent, ProfileEvent
from app.domain import Method, RunConfig

PREFERENCE_OBJECTIVES = frozenset({"dpo", "ipo", "orpo", "simpo"})


def normalize_alignment_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Normalize TRL/LLaMA-Factory metric spellings into stable event keys."""
    aliases = {
        "rewards/chosen": "preference_chosen_reward",
        "eval_rewards/chosen": "preference_chosen_reward",
        "rewards/rejected": "preference_rejected_reward",
        "eval_rewards/rejected": "preference_rejected_reward",
        "rewards/margins": "preference_reward_margin",
        "eval_rewards/margins": "preference_reward_margin",
        "rewards/accuracies": "preference_accuracy",
        "eval_rewards/accuracies": "preference_accuracy",
        "logps/chosen": "preference_chosen_logprob",
        "eval_logps/chosen": "preference_chosen_logprob",
        "logps/rejected": "preference_rejected_logprob",
        "eval_logps/rejected": "preference_rejected_logprob",
        "reward/accuracies": "preference_accuracy",
        "reward/margins": "preference_reward_margin",
        "reward/chosen": "preference_chosen_reward",
        "reward/rejected": "preference_rejected_reward",
        "kl": "preference_kl",
    }
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            result[aliases.get(key, key)] = float(value)
    return result


class PreferenceTrainer:
    """One alignment execution path parameterized by objective."""

    def __init__(
        self,
        cfg: RunConfig,
        ctx: RunContext,
        emit,
        *,
        load_trainable_model: Callable,
        load_reward_model: Callable | None = None,
        callback_factory: Callable,
    ) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.emit = emit
        self.load_trainable_model = load_trainable_model
        self.load_reward_model = load_reward_model
        self.callback_factory = callback_factory

    def run(self) -> None:
        import torch

        cfg = self.cfg
        # Validate the requested trainer before loading model weights.
        trainer_cls, config_cls, objective_kwargs = self._objective_runtime()
        signature = inspect.signature(config_cls.__init__).parameters
        unsupported = [key for key in objective_kwargs if key not in signature]
        if unsupported:
            raise RuntimeError(
                f"Installed TRL does not support {cfg.alignment.objective} options: {', '.join(unsupported)}."
            )
        if cfg.alignment.objective == "reward_model":
            if self.load_reward_model is None:
                raise RuntimeError("This backend does not provide reward-model loading.")
            model, tokenizer = self.load_reward_model(cfg, self.emit)
        else:
            model, tokenizer = self.load_trainable_model(cfg, self.emit)
        from app.train_entry.module_selection import parameter_summary

        self.emit(ProfileEvent(name="trainability", values=parameter_summary(model).as_dict()))
        dataset = self._dataset(tokenizer)
        ref_model, reference_lineage = (
            (None, {"strategy": "none", "model": None, "revision": None})
            if cfg.alignment.objective == "reward_model"
            else self._reference_model(model, tokenizer)
        )
        self.emit(ProfileEvent(name="reference_model", values=reference_lineage))

        config_kwargs = {
            "output_dir": str(self.ctx.checkpoint_dir),
            "per_device_train_batch_size": cfg.train.per_device_batch_size,
            "gradient_accumulation_steps": cfg.train.gradient_accumulation,
            "num_train_epochs": cfg.train.epochs if cfg.train.max_steps is None else 1,
            "max_steps": cfg.train.max_steps or -1,
            "learning_rate": cfg.optim.learning_rate,
            "lr_scheduler_type": cfg.optim.lr_scheduler,
            "warmup_ratio": cfg.optim.warmup_ratio,
            "weight_decay": cfg.optim.weight_decay,
            "max_grad_norm": cfg.optim.max_grad_norm,
            "optim": cfg.optim.optimizer,
            "logging_steps": cfg.train.logging_steps,
            "save_steps": cfg.train.save_steps,
            "save_strategy": "steps",
            "seed": cfg.train.seed,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "gradient_checkpointing_kwargs": (
                {"use_reentrant": False}
                if cfg.runtime.gradient_checkpointing == "non_reentrant" else None
            ),
            "bf16": (
                cfg.runtime.precision == "bf16"
                or (
                    cfg.runtime.precision == "auto"
                    and torch.cuda.is_available()
                    and torch.cuda.is_bf16_supported()
                )
            ),
            "fp16": (
                cfg.runtime.precision == "fp16"
                or (
                    cfg.runtime.precision == "auto"
                    and torch.cuda.is_available()
                    and not torch.cuda.is_bf16_supported()
                )
            ),
            "report_to": "none",
            **objective_kwargs,
        }
        if cfg.runtime.use_liger:
            if "use_liger_kernel" not in signature:
                raise RuntimeError(
                    "The installed TRL alignment configuration does not expose Liger support."
                )
            config_kwargs["use_liger_kernel"] = True
        if cfg.runtime.neftune_noise_alpha is not None:
            if "neftune_noise_alpha" not in signature:
                raise RuntimeError("The installed TRL/Transformers runtime does not expose NEFTune support.")
            config_kwargs["neftune_noise_alpha"] = cfg.runtime.neftune_noise_alpha
        length_key = "max_length" if "max_length" in signature else "max_seq_length"
        config_kwargs[length_key] = cfg.train.max_seq_length
        if "max_prompt_length" in signature:
            default_prompt_length = signature["max_prompt_length"].default
            config_kwargs["max_prompt_length"] = min(
                default_prompt_length if isinstance(default_prompt_length, int) else 512,
                max(1, cfg.train.max_seq_length // 2),
            )
        args = config_cls(**{key: value for key, value in config_kwargs.items() if key in signature})

        trainer_signature = inspect.signature(trainer_cls.__init__).parameters
        trainer_kwargs = {
            "model": model,
            "ref_model": ref_model,
            "args": args,
            "train_dataset": dataset,
            "callbacks": [self.callback_factory(self.ctx, self.emit, normalize_alignment_metrics)],
        }
        tokenizer_key = "processing_class" if "processing_class" in trainer_signature else "tokenizer"
        trainer_kwargs[tokenizer_key] = tokenizer
        from app.train_entry.optimization_runtime import trainer_optimizers

        optimizers = trainer_optimizers(model, cfg)
        if optimizers is not None:
            trainer_kwargs["optimizers"] = optimizers
        if "ref_model" not in trainer_signature:
            trainer_kwargs.pop("ref_model")
            if ref_model is not None:
                raise RuntimeError(
                    f"Installed TRL {trainer_cls.__name__} cannot accept the requested reference model."
                )
        trainer = trainer_cls(**trainer_kwargs)
        self.emit(LogEvent(message=f"Starting shared {cfg.alignment.objective.upper()} preference training."))
        trainer.train(resume_from_checkpoint=str(self.ctx.resume_from) if self.ctx.resume_from else None)
        if self.ctx.should_stop():
            self.emit(LogEvent(message="Alignment stopped; optimizer checkpoint retained for resume."))
            return
        final_dir = self.ctx.workdir / "output"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        step = int(getattr(trainer.state, "global_step", 0))
        self.emit(CheckpointEvent(step=step, path=str(final_dir), is_final=True))
        artifact_kind = (
            "reward_model"
            if cfg.alignment.objective == "reward_model"
            else "adapter" if cfg.method in {Method.LORA, Method.QLORA, Method.DORA}
            else "causal_lm"
        )
        self.emit(ArtifactEvent(
            kind=artifact_kind,
            path=str(final_dir),
            metadata={
                "objective": cfg.alignment.objective,
                "reference": reference_lineage,
                "format": "adapter" if cfg.method in {Method.LORA, Method.QLORA, Method.DORA} else "full_model",
                "evaluation_capabilities": (
                    ["pairwise_accuracy", "chosen_score", "rejected_score", "reward_margin"]
                    if cfg.alignment.objective == "reward_model"
                    else ["generation", "preference_log_probability"]
                ),
            },
        ))

    def _objective_runtime(self):
        objective = self.cfg.alignment.objective
        if objective in {"dpo", "ipo"}:
            try:
                from trl import DPOConfig, DPOTrainer
            except ImportError as exc:
                raise RuntimeError("Installed TRL does not provide DPO/IPO training.") from exc
        if objective == "dpo":
            return DPOTrainer, DPOConfig, {
                "beta": self.cfg.alignment.beta,
                "loss_type": self.cfg.alignment.dpo_loss_variant,
                "label_smoothing": self.cfg.alignment.label_smoothing,
            }
        if objective == "ipo":
            return DPOTrainer, DPOConfig, {
                "beta": self.cfg.alignment.beta,
                "loss_type": "ipo",
            }
        if objective == "orpo":
            try:
                from trl import ORPOConfig, ORPOTrainer
            except ImportError as exc:
                raise RuntimeError("Installed TRL does not provide ORPO training.") from exc
            return ORPOTrainer, ORPOConfig, {"beta": self.cfg.alignment.beta}
        if objective == "simpo":
            try:
                from trl import CPOConfig, CPOTrainer
            except ImportError as exc:
                raise RuntimeError("Installed TRL does not provide the shared CPO/SimPO trainer.") from exc
            return CPOTrainer, CPOConfig, {
                "beta": self.cfg.alignment.beta,
                "loss_type": "simpo",
                "simpo_gamma": self.cfg.alignment.simpo_gamma,
                "cpo_alpha": 0.0,
            }
        if objective == "reward_model":
            try:
                from trl import RewardConfig, RewardTrainer
            except ImportError as exc:
                raise RuntimeError("Installed TRL does not provide reward-model training.") from exc
            return RewardTrainer, RewardConfig, {}
        try:
            from trl import KTOConfig, KTOTrainer
        except ImportError as exc:
            raise RuntimeError("Installed TRL does not provide KTO training.") from exc
        return KTOTrainer, KTOConfig, {
            "beta": self.cfg.alignment.beta,
            "desirable_weight": self.cfg.alignment.desirable_weight,
            "undesirable_weight": self.cfg.alignment.undesirable_weight,
        }

    def _dataset(self, tokenizer):
        from datasets import Dataset as HFDataset
        from sqlmodel import Session

        from app.datasets.lineage import file_fingerprint
        from app.datasets.validate import _iter_records, validate
        from app.db.models import Dataset
        from app.db.session import engine
        from app.train_entry.tokenization import render_kto_row, render_preference_row

        with Session(engine) as db:
            dataset = db.get(Dataset, self.cfg.dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {self.cfg.dataset_id} not found.")
        from app.domain import DatasetKind

        expected = DatasetKind.KTO if self.cfg.alignment.objective == "kto" else DatasetKind.PREFERENCE
        if dataset.kind != expected.value:
            raise ValueError(f"Alignment objective requires {expected.value} data, got {dataset.kind}.")
        report, stats = validate(dataset.path, expected)
        if not report.ok:
            raise ValueError("Alignment dataset validation failed: " + "; ".join(
                issue.message for issue in report.issues if issue.level == "error"
            ))

        def rows(source_fingerprint):
            if file_fingerprint(dataset.path) != source_fingerprint:
                raise ValueError("Alignment dataset changed during preparation; validate it again.")
            for line, row in _iter_records(Path(dataset.path), dataset.fmt):
                if not isinstance(row, dict):
                    raise ValueError(f"Alignment dataset row {line} is not an object.")
                renderer = render_kto_row if self.cfg.alignment.objective == "kto" else render_preference_row
                rendered = renderer(row, tokenizer, chat_template=self.cfg.tokenizer.chat_template)
                if self.cfg.alignment.objective == "reward_model":
                    prompt = rendered.pop("prompt")
                    rendered["chosen"] = prompt + rendered["chosen"]
                    rendered["rejected"] = prompt + rendered["rejected"]
                yield rendered
            if file_fingerprint(dataset.path) != source_fingerprint:
                raise ValueError("Alignment dataset changed during preparation; validate it again.")

        return HFDataset.from_generator(rows, gen_kwargs={"source_fingerprint": stats.fingerprint})

    def _reference_model(self, trainable_model, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from app.config import get_settings
        from app.integrations.hf_hub import _token

        reference = self.cfg.alignment.reference
        if reference.strategy == "none":
            return None, {"strategy": "none", "model": None, "revision": None}
        if reference.strategy == "adapter_disabled":
            if not hasattr(trainable_model, "disable_adapter"):
                raise RuntimeError("The loaded PEFT model cannot disable its adapter for reference scoring.")
            return None, {
                "strategy": "adapter_disabled",
                "model": self.cfg.base_model,
                "revision": self.cfg.revision,
                "resolved_revision": getattr(trainable_model.config, "_commit_hash", None),
            }
        from app.model_refs import resolve_model_ref

        requested_ref = self.cfg.base_model if reference.strategy == "base_model" else reference.model
        assert requested_ref
        resolved = resolve_model_ref(requested_ref)
        if resolved.kind == "scratch":
            raise RuntimeError("Scratch models cannot act as a Transformers alignment reference.")
        if resolved.model_category == "reward_model":
            raise RuntimeError("A scalar reward model cannot act as a causal alignment reference.")
        model_ref = resolved.base_model if resolved.kind == "adapter" else resolved.load_ref
        revision = self.cfg.revision if reference.strategy == "base_model" else reference.revision
        if resolved.kind == "adapter":
            revision = None
        settings = get_settings()
        kwargs: dict[str, Any] = {
            "token": _token(),
            "cache_dir": str(settings.hf_cache_dir),
            "trust_remote_code": settings.trust_remote_code,
            "torch_dtype": {
                "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
            }.get(self.cfg.runtime.precision, "auto"),
        }
        if revision:
            kwargs["revision"] = revision
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
        if self.cfg.quantization.mode != "none":
            from app.train_entry.quantization import bitsandbytes_config

            kwargs["quantization_config"] = bitsandbytes_config(
                self.cfg,
                torch,
                BitsAndBytesConfig,
            )
        model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
        if resolved.kind == "adapter" and reference.strategy == "separate_model":
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, resolved.adapter_path, is_trainable=False,
                token=_token(), revision=reference.revision,
            )
        if tokenizer is not None:
            tokenizer_ref = (
                resolved.adapter_path
                if resolved.kind == "adapter" and reference.strategy == "separate_model"
                else model_ref
            )
            token_kwargs = {key: value for key, value in kwargs.items() if key in {
                "token", "cache_dir", "trust_remote_code", "revision",
            }}
            if resolved.kind == "adapter" and reference.strategy == "separate_model" and reference.revision:
                token_kwargs["revision"] = reference.revision
            try:
                reference_tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, **token_kwargs)
            except (OSError, ValueError):
                if resolved.kind != "adapter":
                    raise
                token_kwargs.pop("revision", None)
                reference_tokenizer = AutoTokenizer.from_pretrained(model_ref, **token_kwargs)
            if reference_tokenizer.get_vocab() != tokenizer.get_vocab():
                raise RuntimeError("Reference and policy tokenizer token IDs differ; use a compatible reference tokenizer.")
        if model.get_input_embeddings().num_embeddings != trainable_model.get_input_embeddings().num_embeddings:
            raise RuntimeError("Reference and policy vocabularies differ; they cannot score the same token IDs.")
        model.requires_grad_(False)
        model.eval()
        return model, {
            "strategy": reference.strategy, "requested_model": requested_ref,
            "model": model_ref, "revision": revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "adapter": resolved.adapter_path if resolved.kind == "adapter" and reference.strategy == "separate_model" else None,
        }
