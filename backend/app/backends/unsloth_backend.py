"""Unsloth backend — LoRA / QLoRA / DoRA / full fine-tuning + continued pretraining.

The metadata methods (validate/estimate/export) are import-light and run in the
API process. ``run`` does all the heavy lifting inside the training subprocess and
imports torch/unsloth/trl lazily. Training happens on a worker thread while ``run``
yields events pulled from a thread-safe queue — that lets a blocking
``trainer.train()`` still stream live metrics/checkpoints through our generator
interface.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any

from app.backends.base import RunContext, TrainingBackendCapabilities
from app.capabilities import (
    Capability,
    ChatTemplateCapabilities,
    PeftMethodCapability,
    QuantizationCapability,
    SupportState,
    TokenizerCapabilities,
    dependency_statuses,
)
from app.core.events import (
    CheckpointEvent,
    LogEvent,
    MetricEvent,
    ProfileEvent,
    TrainingEvent,
)
from app.domain import (
    ExportedConfig,
    FitLevel,
    HardwareProfile,
    MemoryEstimate,
    Method,
    RunConfig,
    TaskType,
    ValidationReport,
)
from app.model_refs import ModelReferenceError, resolve_model_ref
from app.optimizations import optimization_capabilities, validate_optimization_config

_SENTINEL = object()
_ADAPTER_METHODS = {Method.LORA, Method.QLORA, Method.DORA}
_UNSLOTH_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _version_at_least(value: str | None, minimum: tuple[int, ...]) -> bool:
    if not value or value == "unknown":
        return False
    parts = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts[: len(minimum)]) >= minimum if parts else False


def _continued_pretraining_text(row: Mapping[str, Any], line: int, eos: str) -> str:
    """Return a raw causal-LM row without applying conversational formatting."""
    from app.datasets.adapters import canonicalize

    record = canonicalize(row)
    if record.kind != "text":
        raise ValueError(
            f"Continued pretraining requires canonical raw text; row {line} is {record.kind}."
        )
    return record.content_text() + eos


def _unsloth_target_modules(requested: list[str]) -> list[str]:
    """Return the explicit projection list Unsloth expects.

    PEFT accepts the convenient ``all-linear`` string, while Unsloth's loader
    expects a sequence and treats that string as individual characters. Empty
    SLM Kit configuration means automatic all-linear targeting, so map it to
    Unsloth's documented default projections instead.
    """
    return list(requested) if requested else list(_UNSLOTH_DEFAULT_TARGET_MODULES)


class UnslothBackend:
    def __init__(self, name: str = "unsloth") -> None:
        if name not in {"unsloth", "transformers"}:
            raise ValueError(f"Unsupported backend profile: {name}")
        self.name = name

    def capabilities(self) -> TrainingBackendCapabilities:
        dependencies = dependency_statuses()
        optimizations = optimization_capabilities(self.name, dependencies)
        required = ["torch", "transformers", "trl"] + (["unsloth"] if self.name == "unsloth" else [])
        missing = [name for name in required if not dependencies[name].installed]
        availability = Capability(
            state=SupportState.MISSING_DEPENDENCY if missing else SupportState.SUPPORTED,
            reason=(
                f"Missing required packages: {', '.join(missing)}."
                if missing
                else (
                    "Unsloth acceleration is used when compatible, with a Transformers + PEFT fallback."
                    if self.name == "unsloth"
                    else "Native Transformers, TRL, and PEFT training is available."
                )
            ),
            requirements=required,
        )
        supported = Capability(state=SupportState.SUPPORTED, reason="Implemented by the causal-LM training worker.")
        unsupported = Capability(state=SupportState.UNSUPPORTED, reason="This training stage is not implemented by this backend.")
        bitsandbytes_installed = dependencies["bitsandbytes"].installed
        peft_installed = dependencies["peft"].installed
        peft_support = supported if peft_installed else Capability(
            state=SupportState.MISSING_DEPENDENCY,
            reason="Install PEFT to use adapter-based training.",
            requirements=["peft"],
        )
        peft_advanced_support = peft_support if (
            peft_installed and _version_at_least(dependencies["peft"].version, (0, 11))
        ) else Capability(
            state=SupportState.INCOMPATIBLE if peft_installed else SupportState.MISSING_DEPENDENCY,
            reason="DoRA/rsLoRA require a verified PEFT version 0.11 or newer.",
            requirements=["peft>=0.11"],
        )
        qlora = Capability(
            state=(SupportState.SUPPORTED if bitsandbytes_installed and peft_installed else SupportState.MISSING_DEPENDENCY),
            reason=(
                "4-bit adapter training is implemented on CUDA through bitsandbytes."
                if bitsandbytes_installed and peft_installed
                else "Install PEFT and bitsandbytes to use 4-bit adapter training."
            ),
            requirements=["peft", "bitsandbytes", "CUDA"],
        )
        peft = {
            Method.LORA.value: PeftMethodCapability(
                method=Method.LORA.value,
                support=peft_support,
                features={
                    option: optimizations[option].support
                    for option in ("rslora", "lora_plus", "pissa", "loftq", "eva", "oft", "qoft")
                },
            ),
            Method.QLORA.value: PeftMethodCapability(
                method=Method.QLORA.value,
                support=qlora,
                requires_quantized_base=True,
            ),
            Method.DORA.value: PeftMethodCapability(method=Method.DORA.value, support=peft_advanced_support),
            Method.PROMPT_TUNING.value: PeftMethodCapability(
                method=Method.PROMPT_TUNING.value,
                support=Capability(state=SupportState.UNSUPPORTED, reason="Prompt tuning is not implemented by the current worker."),
            ),
        }
        methods = {
            Method.FREEZE.value: (
                supported if self.name == "transformers"
                else Capability(state=SupportState.UNSUPPORTED, reason="Freeze tuning uses the native Transformers worker profile.")
            ),
            Method.LORA.value: peft[Method.LORA.value].support,
            Method.QLORA.value: peft[Method.QLORA.value].support,
            Method.DORA.value: peft[Method.DORA.value].support,
            Method.FULL.value: supported,
            Method.PROMPT_TUNING.value: peft[Method.PROMPT_TUNING.value].support,
        }
        alignment_support = (
            supported
            if self.name == "transformers" and dependencies["trl"].installed
            and _version_at_least(dependencies["trl"].version, (0, 9))
            and not _version_at_least(dependencies["trl"].version, (0, 24))
            else Capability(
                state=SupportState.INCOMPATIBLE if dependencies["trl"].installed else SupportState.MISSING_DEPENDENCY,
                reason="Native alignment requires a versioned TRL >=0.9,<0.24 installation.",
                requirements=["trl>=0.9,<0.24"],
            )
            if self.name == "transformers"
            else Capability(
                state=SupportState.UNSUPPORTED,
                reason="Alignment uses the shared native TRL preference trainer; select Auto or Transformers.",
            )
        )
        extended_dpo_losses = (
            alignment_support
            if self.name == "transformers" and _version_at_least(dependencies["trl"].version, (0, 8))
            else Capability(
                state=SupportState.INCOMPATIBLE if dependencies["trl"].installed else SupportState.MISSING_DEPENDENCY,
                reason="This DPO loss requires an installed TRL 0.8 or newer runtime.",
                requirements=["trl>=0.8"],
            )
        )
        simpo_support = (
            alignment_support
            if self.name == "transformers" and _version_at_least(dependencies["trl"].version, (0, 12))
            else Capability(
                state=SupportState.INCOMPATIBLE if self.name == "transformers" else SupportState.UNSUPPORTED,
                reason="SimPO requires a TRL release exposing CPOTrainer with SimPO loss.",
                requirements=["trl>=0.12"],
            )
        )
        return TrainingBackendCapabilities(
            name=self.name,
            display_name="Unsloth" if self.name == "unsloth" else "Transformers / TRL / PEFT",
            description=(
                "Optimized causal-LM training with a native Transformers fallback."
                if self.name == "unsloth"
                else "Native Hugging Face causal-LM training."
            ),
            availability=availability,
            tasks={
                task.value: (
                    alignment_support if task == TaskType.ALIGNMENT
                    else supported if task in {TaskType.CONTINUED_PRETRAIN, TaskType.FINETUNE}
                    else unsupported
                )
                for task in TaskType
            },
            stages={
                "from_scratch_pretraining": unsupported,
                "continued_pretraining": supported,
                "supervised_fine_tuning": supported,
                "alignment": alignment_support,
            },
            methods=methods,
            tokenizer=TokenizerCapabilities(
                modes={"reuse": supported, "extend": unsupported, "train": unsupported},
                loss_policies={
                    "full_sequence": supported,
                    "completion_only": supported,
                    "assistant_only": supported,
                },
                templates=ChatTemplateCapabilities(
                    native=Capability(
                        state=SupportState.SUPPORTED,
                        reason="The worker uses the tokenizer's native template when one is present.",
                    ),
                    explicit_override=Capability(
                        state=SupportState.SUPPORTED,
                        reason="RunConfig may provide an explicit template.",
                    ),
                    fallback=Capability(
                        state=SupportState.UNSUPPORTED,
                        reason="The worker does not guess a family template.",
                    ),
                ),
            ),
            peft=peft,
            quantization={
                "nf4": QuantizationCapability(format="nf4", operations=["train"], support=qlora,
                                               compute_dtypes=["auto", "bf16", "fp16", "fp32"],
                                               storage_dtypes=["auto", "uint8", "bf16", "fp16", "fp32"],
                                               double_quantization=True),
                "fp4": QuantizationCapability(format="fp4", operations=["train"], support=qlora,
                                               compute_dtypes=["auto", "bf16", "fp16", "fp32"],
                                               storage_dtypes=["auto", "uint8", "bf16", "fp16", "fp32"],
                                               double_quantization=True),
                "int8": QuantizationCapability(format="int8", operations=["train"], support=qlora,
                                                compute_dtypes=["auto", "bf16", "fp16", "fp32"]),
                "fp16": QuantizationCapability(format="fp16", operations=["train"], support=supported),
                "bf16": QuantizationCapability(format="bf16", operations=["train"], support=supported),
            },
            precision={name: supported for name in ("auto", "bf16", "fp16", "fp32")},
            attention={
                "auto": supported,
                "sdpa": (
                    supported if _version_at_least(dependencies["torch"].version, (2, 0))
                    else Capability(state=SupportState.INCOMPATIBLE, reason="SDPA requires PyTorch 2.0 or newer.")
                ),
                "eager": supported,
                "flash_attention_2": Capability(
                    state=(SupportState.SUPPORTED if dependencies["flash-attn"].installed else SupportState.MISSING_DEPENDENCY),
                    reason=("flash-attn is installed." if dependencies["flash-attn"].installed else "Install flash-attn to use FlashAttention 2."),
                    requirements=["flash-attn", "CUDA"],
                ),
            },
            gradient_checkpointing={
                "auto": supported,
                "off": supported,
                "standard": supported,
                "non_reentrant": supported,
                "backend_optimized": (
                    supported if self.name == "unsloth"
                    else Capability(state=SupportState.UNSUPPORTED, reason="Backend-optimized checkpointing requires Unsloth.")
                ),
            },
            rope={name: supported for name in ("none", "linear", "dynamic")},
            optimizations=optimizations,
            optional_features={
                "bitsandbytes_optimizer": Capability(
                    state=(SupportState.SUPPORTED if dependencies["bitsandbytes"].installed else SupportState.MISSING_DEPENDENCY),
                    reason=("bitsandbytes optimizers are installed." if dependencies["bitsandbytes"].installed else "Install bitsandbytes to use 8-bit optimizers."),
                    requirements=["bitsandbytes"],
                ),
                # Legacy consumers retain these aliases.  The optimization
                # registry above is the authoritative source for their data.
                **{option: optimizations[option].support for option in ("liger", "rslora", "lora_plus", "pissa", "loftq", "eva", "oft", "qoft")},
                "objective:dpo": alignment_support,
                "objective:ipo": alignment_support,
                "objective:orpo": alignment_support,
                "objective:simpo": simpo_support,
                "objective:kto": alignment_support,
                "objective:reward_model": alignment_support,
                "dpo_loss:sigmoid": alignment_support,
                **{f"reference:{strategy}": alignment_support for strategy in ("base_model", "separate_model", "adapter_disabled", "none")},
                "dpo_loss:hinge": extended_dpo_losses,
                "dpo_loss:robust": extended_dpo_losses,
                "dpo_loss:exo_pair": extended_dpo_losses,
            },
            platforms=["linux", "windows", "wsl"],
            architectures=["decoder-only causal language models"],
            required_dependencies=required,
            optional_dependencies=(
                ["unsloth", "bitsandbytes", "accelerate", "xformers"]
                if self.name == "unsloth"
                else ["bitsandbytes", "accelerate", "xformers"]
            ),
        )

    @property
    def supported_tasks(self) -> set[TaskType]:
        return self.capabilities().supported_tasks

    @property
    def supported_methods(self) -> set[Method]:
        return self.capabilities().supported_methods

    # ---- metadata (runs in API process) -------------------------------
    def validate_config(self, cfg: RunConfig, hw: HardwareProfile) -> ValidationReport:
        report = ValidationReport()
        self.capabilities().validate_operation(cfg, report)
        if not cfg.base_model:
            report.error("A base model (HF repo id or local path) is required.")
        else:
            try:
                resolved = resolve_model_ref(cfg.base_model)
                if resolved.kind == "scratch":
                    report.error(
                        "Scratch checkpoints can be evaluated and deployed, but cannot be fine-tuned by the "
                        "Hugging Face trainer. Start a new scratch pretraining run instead."
                    )
            except ModelReferenceError as exc:
                report.error(str(exc))
        if cfg.method == Method.QLORA and cfg.quantization.mode not in {"nf4", "fp4"}:
            report.error("QLoRA requires quantization.mode to be 'nf4' or 'fp4'.")
        if cfg.quantization.mode in {"nf4", "fp4", "int8"} and cfg.method not in {
            Method.LORA, Method.QLORA, Method.DORA
        }:
            report.error("Bitsandbytes training quantization is supported only with LoRA-family methods.")
        if cfg.quantization.mode in {"nf4", "fp4", "int8"} and not dependency_statuses()["bitsandbytes"].installed:
            report.error("Install bitsandbytes to use training quantization.")
        if "8bit" in cfg.optim.optimizer and not dependency_statuses()["bitsandbytes"].installed:
            report.error(f"Optimizer '{cfg.optim.optimizer}' requires bitsandbytes.")
        if cfg.lora.target_strategy == "custom" and not cfg.lora.target_modules:
            report.error("Custom LoRA targeting requires at least one target module.")
        validate_optimization_config(cfg, self.capabilities().optimizations, report)

        est = self.estimate_footprint(cfg, hw)
        if est.fit == FitLevel.WONT_FIT:
            report.error(
                f"Predicted peak VRAM ~{est.total_mb:.0f} MB exceeds the ~{est.budget_mb:.0f} MB budget. "
                "Reduce sequence length / batch size, switch to QLoRA, or pick a smaller base model."
            )
        elif est.fit == FitLevel.TIGHT:
            report.warn(f"Tight fit (~{est.total_mb:.0f} MB). Close other GPU apps before launching.")

        if cfg.method == Method.FULL:
            report.warn("Full fine-tuning is only viable for very small models on 8GB VRAM.")
        if cfg.method == Method.QLORA and not hw.gpu_name:
            report.error("QLoRA requires a detected CUDA GPU. Use a GPU-enabled environment.")
        if cfg.tokenizer.mode != "reuse":
            report.error(
                f"Tokenizer mode '{cfg.tokenizer.mode}' is not implemented by this worker. "
                "Reuse the model tokenizer; tokenizer changes must never happen silently."
            )
        if cfg.tokenizer.loss_policy != "full_sequence" and cfg.task != TaskType.FINETUNE:
            report.error("Completion/assistant-only loss is available only for supervised fine-tuning data.")
        if cfg.tokenizer.loss_policy != "full_sequence" and cfg.train.packing:
            report.error("Packing with masked completion/assistant loss is not supported by the current collator. Disable packing.")
        if cfg.runtime.precision == "bf16" and not any(gpu.bf16_supported for gpu in hw.gpus):
            report.error("BF16 precision requires a detected BF16-capable GPU.")
        if cfg.runtime.precision == "fp16" and not any(gpu.fp16_supported for gpu in hw.gpus):
            report.error("FP16 precision requires a detected FP16-capable GPU.")
        if cfg.runtime.attention == "flash_attention_2":
            capability = self.capabilities().attention["flash_attention_2"]
            if not capability.allowed:
                report.error(capability.reason)
            if not hw.cuda_available:
                report.error("FlashAttention 2 requires CUDA.")
            if cfg.runtime.precision == "fp32":
                report.error("FlashAttention 2 does not support the requested FP32 training precision.")
        if cfg.runtime.gradient_checkpointing == "backend_optimized" and self.name != "unsloth":
            report.error("Backend-optimized gradient checkpointing requires the Unsloth backend.")
        if cfg.runtime.rope.enabled:
            rope_capability = self.capabilities().rope.get(cfg.runtime.rope.type)
            if rope_capability is None or not rope_capability.allowed:
                reason = rope_capability.reason if rope_capability else "The backend has no verified implementation."
                report.error(f"RoPE policy '{cfg.runtime.rope.type}' is unavailable: {reason}")
        if cfg.task == TaskType.ALIGNMENT:
            objective = self.capabilities().optional_features.get(f"objective:{cfg.alignment.objective}")
            if objective is None or not objective.allowed:
                report.error(objective.reason if objective else f"Objective '{cfg.alignment.objective}' is unsupported.")
            if cfg.train.packing:
                report.error("Preference alignment does not support sequence packing in the shared trainer.")
            if cfg.alignment.objective == "reward_model" and cfg.runtime.rope.enabled:
                report.error("The reward-model loader does not apply RoPE overrides; disable RoPE extension.")
            if cfg.alignment.objective == "dpo":
                loss = self.capabilities().optional_features.get(
                    f"dpo_loss:{cfg.alignment.dpo_loss_variant}"
                )
                if loss is None or not loss.allowed:
                    report.error(loss.reason if loss else "The requested DPO loss is unsupported.")
        return report

    def estimate_footprint(self, cfg: RunConfig, hw: HardwareProfile) -> MemoryEstimate:
        # Lazy import keeps this file importable without the estimator's deps.
        from app.integrations import llmfit

        return llmfit.estimate_fit(cfg, hw)

    def export_config(self, cfg: RunConfig) -> ExportedConfig:
        return ExportedConfig(
            format="json",
            filename=f"{cfg.output_name}.slmkit.json",
            content=cfg.model_dump_json(indent=2),
        )

    # ---- training (runs in subprocess) --------------------------------
    def run(self, cfg: RunConfig, ctx: RunContext) -> Iterator[TrainingEvent]:
        events: queue.Queue = queue.Queue()
        holder: dict = {}

        worker = threading.Thread(
            target=self._train_worker, args=(cfg, ctx, events, holder), daemon=True
        )
        worker.start()

        while True:
            item = events.get()
            if item is _SENTINEL:
                break
            yield item

        worker.join(timeout=5)
        # Re-raise any training error on the generator's thread so the subprocess
        # entrypoint reports the run as FAILED instead of DONE.
        if "exc" in holder:
            raise holder["exc"]

    # -------------------------------------------------------------------
    def _train_worker(self, cfg: RunConfig, ctx: RunContext, out: queue.Queue, holder: dict) -> None:
        def emit(ev: TrainingEvent) -> None:
            out.put(ev)

        try:
            self._train(cfg, ctx, emit)
        except Exception as exc:  # noqa: BLE001 — propagated to run() via holder
            holder["exc"] = exc
            emit(LogEvent(level="error", message=f"Training error: {exc}"))
        finally:
            out.put(_SENTINEL)

    def _train(self, cfg: RunConfig, ctx: RunContext, emit) -> None:
        if cfg.task == TaskType.ALIGNMENT:
            from app.train_entry.alignment import PreferenceTrainer

            PreferenceTrainer(
                cfg,
                ctx,
                emit,
                load_trainable_model=self._load_trainable_model,
                load_reward_model=self._load_reward_model,
                callback_factory=_StreamCallback,
            ).run()
            return
        model, tokenizer = self._load_trainable_model(cfg, emit)
        import torch

        from app.train_entry.module_selection import parameter_summary

        trainability = parameter_summary(model)
        emit(ProfileEvent(name="trainability", values=trainability.as_dict()))
        emit(LogEvent(
            message=(
                f"Trainability: {trainability.trainable_parameters:,} / "
                f"{trainability.total_parameters:,} parameters "
                f"({trainability.trainable_percentage:.4f}%)."
            )
        ))

        dataset, pretokenized = self._load_dataset(cfg, tokenizer, emit)

        import inspect

        from trl import SFTConfig, SFTTrainer

        config_kwargs = dict(
            output_dir=str(ctx.checkpoint_dir),
            per_device_train_batch_size=cfg.train.per_device_batch_size,
            gradient_accumulation_steps=cfg.train.gradient_accumulation,
            warmup_ratio=cfg.optim.warmup_ratio,
            num_train_epochs=cfg.train.epochs if cfg.train.max_steps is None else 1,
            max_steps=cfg.train.max_steps or -1,
            learning_rate=cfg.optim.learning_rate,
            lr_scheduler_type=cfg.optim.lr_scheduler,
            weight_decay=cfg.optim.weight_decay,
            max_grad_norm=cfg.optim.max_grad_norm,
            optim=cfg.optim.optimizer,
            logging_steps=cfg.train.logging_steps,
            save_steps=cfg.train.save_steps,
            save_strategy="steps",
            seed=cfg.train.seed,
            bf16=(cfg.runtime.precision == "bf16" or (
                cfg.runtime.precision == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            )),
            fp16=(cfg.runtime.precision == "fp16" or (
                cfg.runtime.precision == "auto" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
            )),
            packing=cfg.train.packing,
            report_to="none",
        )
        # trl renamed max_seq_length → max_length (~0.20); support both.
        sft_params = inspect.signature(SFTConfig.__init__).parameters
        if cfg.runtime.use_liger:
            if "use_liger_kernel" not in sft_params:
                raise RuntimeError("The installed TRL/Transformers runtime does not expose Liger support.")
            config_kwargs["use_liger_kernel"] = True
        if cfg.runtime.neftune_noise_alpha is not None:
            if "neftune_noise_alpha" not in sft_params:
                raise RuntimeError("The installed TRL/Transformers runtime does not expose NEFTune support.")
            config_kwargs["neftune_noise_alpha"] = cfg.runtime.neftune_noise_alpha
        if not pretokenized:
            config_kwargs["dataset_text_field"] = "text"
            if "dataset_kwargs" in sft_params:
                config_kwargs["dataset_kwargs"] = {"add_special_tokens": False}
            else:
                raise RuntimeError(
                    "This TRL version cannot disable automatic special tokens for pre-rendered chat data. "
                    "Install a supported TRL release instead of training with duplicated BOS/EOS tokens."
                )
        if "max_length" in sft_params:
            config_kwargs["max_length"] = cfg.train.max_seq_length
        else:
            config_kwargs["max_seq_length"] = cfg.train.max_seq_length
        args = SFTConfig(**config_kwargs)

        callback = _StreamCallback(ctx, emit)
        # trl renamed tokenizer → processing_class (~0.12); support both.
        trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        tok_kw = "processing_class" if "processing_class" in trainer_params else "tokenizer"
        trainer_kwargs = {}
        if pretokenized:
            from transformers import DataCollatorForSeq2Seq

            trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100,
            )
        from app.train_entry.optimization_runtime import trainer_optimizers

        optimizers = trainer_optimizers(model, cfg)
        if optimizers is not None:
            trainer_kwargs["optimizers"] = optimizers
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=args,
            callbacks=[callback],
            **trainer_kwargs,
            **{tok_kw: tokenizer},
        )

        emit(LogEvent(message="Starting training…"))
        trainer.train(resume_from_checkpoint=str(ctx.resume_from) if ctx.resume_from else None)

        final_dir = ctx.workdir / "output"
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        emit(CheckpointEvent(step=callback.last_step, path=str(final_dir), is_final=True))
        emit(LogEvent(message=f"Saved final model to {final_dir}"))

    def _load_trainable_model(self, cfg: RunConfig, emit):
        """Load any causal-LM checkpoint, preferring Unsloth where it fits.

        The fallback matters for practical interoperability: a standard Hugging
        Face decoder-only model should remain trainable even if Unsloth has not
        implemented an optimized wrapper for that architecture.
        """
        import torch

        from app.config import get_settings
        from app.integrations.hf_hub import _token
        settings = get_settings()
        common = {"token": _token(), "cache_dir": str(settings.hf_cache_dir),
                  "trust_remote_code": settings.trust_remote_code}
        if cfg.revision:
            common["revision"] = cfg.revision

        # Unsloth must patch Transformers before either Transformers or PEFT is
        # imported. Remote adapter detection below uses PEFT, so importing here
        # prevents that lightweight probe from silently disabling the optimized
        # Qwen/Llama model loaders.
        fast_language_model = None
        unsloth_import_error = None
        if cfg.backend != "transformers" and cfg.method in _ADAPTER_METHODS:
            try:
                from unsloth import FastLanguageModel

                fast_language_model = FastLanguageModel
            except Exception as exc:  # surfaced through the normal fallback log
                unsloth_import_error = exc

        resolved = resolve_model_ref(cfg.base_model)
        if resolved.kind == "transformers" and resolved.local_path is None:
            try:
                from peft import PeftConfig

                adapter_cfg = PeftConfig.from_pretrained(
                    resolved.load_ref,
                    token=_token(),
                    cache_dir=str(settings.hf_cache_dir),
                    revision=cfg.revision,
                )
                adapter_base = getattr(adapter_cfg, "base_model_name_or_path", None)
                if adapter_base:
                    resolved = replace(
                        resolved,
                        kind="adapter",
                        base_model=str(adapter_base),
                        adapter_path=resolved.load_ref,
                    )
            except Exception:
                pass
        if resolved.kind == "scratch":
            raise ValueError("Scratch checkpoints are not supported by the Hugging Face fine-tuning backend.")
        model_ref = resolved.load_ref
        emit(LogEvent(message=f"Loading base model {cfg.base_model} (4bit={cfg.load_in_4bit})..."))
        try:
            if cfg.backend == "transformers" or cfg.method not in _ADAPTER_METHODS or resolved.kind == "adapter":
                raise NotImplementedError("Using the standard Transformers engine for this configuration.")
            if unsloth_import_error is not None:
                raise unsloth_import_error
            if fast_language_model is None:
                raise RuntimeError("Unsloth did not provide FastLanguageModel.")
            if cfg.quantization.mode not in {"none", "nf4"}:
                raise NotImplementedError(
                    f"Unsloth profile does not expose explicit {cfg.quantization.mode} loading; using native Transformers."
                )
            if cfg.runtime.precision != "auto":
                raise NotImplementedError(
                    "Explicit precision selection is handled by the native Transformers engine."
                )
            if cfg.runtime.attention != "auto":
                raise NotImplementedError(
                    "Explicit attention selection is handled by the native Transformers engine."
                )
            if cfg.runtime.rope.enabled:
                raise NotImplementedError(
                    "Explicit RoPE scaling is handled by the native Transformers engine."
                )
            if cfg.runtime.gradient_checkpointing in {"standard", "non_reentrant"}:
                raise NotImplementedError(
                    "The selected checkpointing mode is handled by the native Transformers engine."
                )

            model, tokenizer = fast_language_model.from_pretrained(
                model_name=model_ref,
                max_seq_length=cfg.train.max_seq_length,
                dtype=None,
                load_in_4bit=cfg.load_in_4bit and cfg.method != Method.FULL,
                **common,
            )
            if cfg.method in _ADAPTER_METHODS:
                from app.train_entry.module_selection import discover_lora_targets

                targets = discover_lora_targets(model, cfg.lora)
                emit(ProfileEvent(name="adapter_targets", values={
                    "strategy": targets.strategy,
                    "modules": list(targets.matched_modules),
                    "count": len(targets.matched_modules),
                }))
                emit(LogEvent(message=f"Attaching {cfg.method.value.upper()} adapters with Unsloth (r={cfg.lora.r})..."))
                model = fast_language_model.get_peft_model(
                    model,
                    r=cfg.lora.r,
                    target_modules=list(targets.matched_modules),
                    lora_alpha=cfg.lora.alpha,
                    lora_dropout=cfg.lora.dropout,
                    bias="none",
                    use_gradient_checkpointing="unsloth" if cfg.gradient_checkpointing else False,
                    random_state=cfg.train.seed,
                    use_rslora=cfg.lora.use_rslora,
                    use_dora=(cfg.method == Method.DORA) or cfg.lora.use_dora,
                )
            return model, tokenizer
        except Exception as unsloth_error:
            if "out of memory" in str(unsloth_error).lower():
                raise
            emit(LogEvent(
                level="warning",
                message=f"Unsloth could not load this model ({unsloth_error}). Falling back to Transformers + PEFT.",
            ))

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from app.config import get_settings

        adapter_common = common
        base_common = dict(common)
        if resolved.kind == "adapter":
            # The selected revision belongs to the adapter repository, not
            # necessarily to its independently-versioned base model.
            base_common.pop("revision", None)
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved.adapter_path or model_ref, **adapter_common)
        except (OSError, ValueError):
            if resolved.kind != "adapter" or not resolved.base_model:
                raise
            emit(LogEvent(level="warning", message="Adapter has no tokenizer files; using its base model tokenizer."))
            tokenizer = AutoTokenizer.from_pretrained(resolved.base_model, **base_common)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = "auto"
        if cfg.runtime.precision == "fp32":
            dtype = torch.float32
        elif cfg.runtime.precision == "fp16":
            dtype = torch.float16
        elif cfg.runtime.precision == "bf16":
            dtype = torch.bfloat16
        load_kwargs = {"torch_dtype": dtype, **common}
        if cfg.runtime.attention != "auto":
            load_kwargs["attn_implementation"] = cfg.runtime.attention
        if torch.cuda.is_available():
            load_kwargs["device_map"] = {"": 0}
        if cfg.runtime.rope.enabled:
            config_ref = resolved.base_model if resolved.kind == "adapter" else model_ref
            if not config_ref:
                raise RuntimeError("Could not resolve a base model configuration for RoPE scaling.")
            model_config = AutoConfig.from_pretrained(config_ref, **base_common)
            if not hasattr(model_config, "rope_scaling"):
                raise RuntimeError("This model configuration does not expose RoPE scaling.")
            model_config.rope_scaling = {
                "rope_type": cfg.runtime.rope.type,
                "factor": cfg.runtime.rope.factor,
            }
            load_kwargs["config"] = model_config
            emit(LogEvent(message=(
                f"Applying explicit {cfg.runtime.rope.type} RoPE scaling "
                f"with factor {cfg.runtime.rope.factor:g}."
            )))
        quant_mode = cfg.quantization.mode
        if quant_mode in {"nf4", "fp4", "int8"}:
            from app.train_entry.quantization import bitsandbytes_config

            load_kwargs["quantization_config"] = bitsandbytes_config(
                cfg,
                torch,
                BitsAndBytesConfig,
            )

        if resolved.kind == "adapter":
            from peft import PeftModel

            assert resolved.base_model and resolved.adapter_path
            base_load_kwargs = {**load_kwargs, **base_common}
            base_load_kwargs.pop("revision", None)
            base = AutoModelForCausalLM.from_pretrained(resolved.base_model, **base_load_kwargs)
            if quant_mode != "none":
                from peft import prepare_model_for_kbit_training
                base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=cfg.gradient_checkpointing)
            model = PeftModel.from_pretrained(
                base,
                resolved.adapter_path,
                is_trainable=True,
                token=_token(),
                revision=cfg.revision,
            )
            if cfg.method == Method.FULL:
                model = model.merge_and_unload()
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
            else:
                emit(LogEvent(message="Continuing existing adapter; its saved rank and target modules are retained."))
        else:
            model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)

        if cfg.method in _ADAPTER_METHODS and resolved.kind != "adapter":
            from peft import LoraConfig, get_peft_model
            from peft import TaskType as PeftTaskType

            from app.train_entry.module_selection import discover_lora_targets
            from app.train_entry.optimization_runtime import lora_config_kwargs

            targets = discover_lora_targets(model, cfg.lora)
            emit(ProfileEvent(name="adapter_targets", values={
                "strategy": targets.strategy,
                "modules": list(targets.matched_modules),
                "count": len(targets.matched_modules),
            }))
            if quant_mode != "none":
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)

            model = get_peft_model(model, LoraConfig(
                task_type=PeftTaskType.CAUSAL_LM,
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=list(targets.matched_modules),
                use_dora=(cfg.method == Method.DORA) or cfg.lora.use_dora,
                use_rslora=cfg.lora.use_rslora,
                **lora_config_kwargs(cfg),
            ))
        elif cfg.method == Method.FREEZE:
            if resolved.kind == "adapter":
                raise ValueError("Freeze tuning requires a base model, not an existing PEFT adapter.")
            from app.train_entry.module_selection import apply_freeze_policy

            summary, selected = apply_freeze_policy(model, cfg.freeze)
            emit(ProfileEvent(name="freeze_targets", values={
                **summary.as_dict(), "parameters": list(selected), "count": len(selected),
            }))
        elif cfg.method == Method.FULL:
            for parameter in model.parameters():
                parameter.requires_grad_(True)

        model.config.use_cache = False
        if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            kwargs = ({"gradient_checkpointing_kwargs": {"use_reentrant": False}}
                      if cfg.runtime.gradient_checkpointing == "non_reentrant" else {})
            model.gradient_checkpointing_enable(**kwargs)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        return model, tokenizer

    def _load_reward_model(self, cfg: RunConfig, emit):
        """Load a scalar sequence classifier for pairwise reward training.

        Reward heads deliberately use the native Transformers/PEFT path even
        when the optional Unsloth package is installed.
        """
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        from app.config import get_settings
        from app.integrations.hf_hub import _token

        settings = get_settings()
        common = {
            "token": _token(),
            "cache_dir": str(settings.hf_cache_dir),
            "trust_remote_code": settings.trust_remote_code,
        }
        if cfg.revision:
            common["revision"] = cfg.revision
        resolved = resolve_model_ref(cfg.base_model)
        if resolved.kind == "scratch":
            raise ValueError("Scratch checkpoints cannot be used as reward-model bases.")
        if resolved.kind == "adapter":
            from peft import PeftConfig
            from peft import TaskType as PeftTaskType

            adapter_config = PeftConfig.from_pretrained(resolved.adapter_path, **common)
            if adapter_config.task_type != PeftTaskType.SEQ_CLS:
                raise ValueError("Reward adapter continuation requires a saved SEQ_CLS adapter; merge a causal-LM adapter before reward training.")
        tokenizer_ref = resolved.adapter_path or resolved.load_ref
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, **common)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_ref = resolved.base_model if resolved.kind == "adapter" else resolved.load_ref
        load_kwargs: dict[str, Any] = {**common, "num_labels": 1}
        if resolved.kind == "adapter":
            load_kwargs.pop("revision", None)
        if cfg.runtime.precision == "fp32":
            load_kwargs["torch_dtype"] = torch.float32
        elif cfg.runtime.precision == "fp16":
            load_kwargs["torch_dtype"] = torch.float16
        elif cfg.runtime.precision == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            load_kwargs["torch_dtype"] = "auto"
        if cfg.runtime.attention != "auto":
            load_kwargs["attn_implementation"] = cfg.runtime.attention
        if torch.cuda.is_available():
            load_kwargs["device_map"] = {"": 0}
        if cfg.quantization.mode != "none":
            from app.train_entry.quantization import bitsandbytes_config

            load_kwargs["quantization_config"] = bitsandbytes_config(cfg, torch, BitsAndBytesConfig)
        model = AutoModelForSequenceClassification.from_pretrained(load_ref, **load_kwargs)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.problem_type = "regression"
        if resolved.kind == "adapter":
            from peft import PeftModel

            assert resolved.adapter_path
            if cfg.quantization.mode != "none":
                from peft import prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
            model = PeftModel.from_pretrained(
                model, resolved.adapter_path, is_trainable=True, token=_token(), revision=cfg.revision,
            )
            if cfg.method == Method.FULL:
                model = model.merge_and_unload()
                model.requires_grad_(True)
        elif cfg.method in _ADAPTER_METHODS:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from peft import TaskType as PeftTaskType

            from app.train_entry.module_selection import discover_lora_targets
            from app.train_entry.optimization_runtime import lora_config_kwargs

            # PEFT SEQ_CLS trains/saves these heads in full; targeting them with
            # LoRA as well creates incompatible nested adapter wrappers.
            targets = discover_lora_targets(model, cfg.lora, excluded_modules=("classifier", "score"))
            emit(ProfileEvent(name="adapter_targets", values={
                "strategy": targets.strategy,
                "modules": list(targets.matched_modules),
                "count": len(targets.matched_modules),
            }))
            if cfg.quantization.mode != "none":
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
            model = get_peft_model(model, LoraConfig(
                task_type=PeftTaskType.SEQ_CLS,
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=list(targets.matched_modules),
                use_dora=(cfg.method == Method.DORA) or cfg.lora.use_dora,
                use_rslora=cfg.lora.use_rslora,
                **lora_config_kwargs(cfg),
            ))
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        model.config.use_cache = False
        if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            kwargs = ({"gradient_checkpointing_kwargs": {"use_reentrant": False}}
                      if cfg.runtime.gradient_checkpointing == "non_reentrant" else {})
            model.gradient_checkpointing_enable(**kwargs)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        emit(LogEvent(message=f"Loaded scalar reward model from {cfg.base_model}."))
        return model, tokenizer

    # -------------------------------------------------------------------
    def _load_dataset(self, cfg: RunConfig, tokenizer, emit):
        """Resolve one immutable dataset file into rendered or labelled rows."""
        from datasets import load_dataset
        from sqlmodel import Session

        from app.db.models import Dataset
        from app.db.session import engine

        if cfg.dataset_id is None:
            raise ValueError("No dataset selected for this run.")
        with Session(engine) as db:
            ds_row = db.get(Dataset, cfg.dataset_id)
        if ds_row is None:
            raise ValueError(f"Dataset {cfg.dataset_id} not found.")

        path = ds_row.path
        emit(LogEvent(message=f"Loading dataset '{ds_row.name}' ({ds_row.fmt}) from {path}"))

        if cfg.task == TaskType.CONTINUED_PRETRAIN or ds_row.kind in ("pretrain_corpus", "domain_corpus"):
            # Raw text corpus: one 'text' field per chunk/line.
            from pathlib import Path

            from datasets import Dataset as HFDataset

            from app.datasets.validate import _iter_records
            eos = tokenizer.eos_token or ""

            def raw_rows():
                for line, row in _iter_records(Path(path), ds_row.fmt):
                    if not isinstance(row, dict):
                        raise ValueError(f"Raw corpus row {line} is not an object.")
                    yield {"text": _continued_pretraining_text(row, line, eos)}

            return HFDataset.from_generator(raw_rows), False

        # Instruction/chat: build a 'text' column via the model's chat template.
        if ds_row.fmt in ("jsonl", "json"):
            data = load_dataset("json", data_files=path, split="train")
        elif ds_row.fmt == "csv":
            data = load_dataset("csv", data_files=path, split="train")
        elif ds_row.fmt == "parquet":
            data = load_dataset("parquet", data_files=path, split="train")
        else:
            data = load_dataset("text", data_files=path, split="train")

        if cfg.tokenizer.loss_policy == "full_sequence":
            def to_text(example: Any) -> dict:
                rec = dict(example) if isinstance(example, Mapping) and not isinstance(example, dict) else example
                return {"text": _format_example(rec, tokenizer, cfg.tokenizer.chat_template)}

            return data.map(to_text, remove_columns=[c for c in data.column_names if c != "text"]), False

        from app.train_entry.tokenization import render_and_tokenize

        def tokenize(example: Any) -> dict:
            rec = dict(example) if isinstance(example, Mapping) and not isinstance(example, dict) else example
            result = render_and_tokenize(
                rec, tokenizer, max_length=cfg.train.max_seq_length,
                loss_policy=cfg.tokenizer.loss_policy,
                chat_template=cfg.tokenizer.chat_template,
            )
            return {key: result[key] for key in ("input_ids", "attention_mask", "labels")}

        return data.map(tokenize, remove_columns=data.column_names), True


def _format_example(ex: Any, tokenizer, chat_template: str | None = None) -> str:
    """Render with an explicit/native chat template; never guess a hidden one."""
    from app.train_entry.tokenization import render_and_tokenize

    rec = dict(ex) if isinstance(ex, Mapping) and not isinstance(ex, dict) else ex
    return render_and_tokenize(
        rec, tokenizer, max_length=10**9, loss_policy="full_sequence",
        chat_template=chat_template,
    )["rendered"]


class _StreamCallback:
    """TRL/Transformers trainer callback that streams metrics + honors cancel.

    Not a ``TrainerCallback`` subclass (that would force a transformers import in
    the API process); instead ``__getattr__`` supplies no-op handlers for every
    other ``on_*`` hook the CallbackHandler invokes.
    """

    def __init__(self, ctx: RunContext, emit, metric_normalizer=None) -> None:
        self.ctx = ctx
        self.emit = emit
        self.last_step = 0
        self._t0 = time.time()
        self.metric_normalizer = metric_normalizer

    def __getattr__(self, name: str):
        if name.startswith("on_"):
            return lambda *args, **kwargs: None  # no-op for unhandled trainer hooks
        raise AttributeError(name)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        self.last_step = int(state.global_step)
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if self.metric_normalizer is not None:
            metrics = self.metric_normalizer(metrics)
        import torch
        metrics["elapsed_seconds"] = round(time.time() - self._t0, 1)
        if torch.cuda.is_available():
            metrics["vram_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
            metrics["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        elapsed = max(1e-6, time.time() - self._t0)
        done = max(1, state.global_step)
        if "num_tokens" in metrics:
            metrics["tokens_per_sec"] = round(metrics["num_tokens"] / elapsed, 1)
        else:
            # Approximate throughput: steps × effective batch × seq length.
            # (trl renamed max_seq_length → max_length; check both.)
            seq_len = getattr(args, "max_length", None) or getattr(args, "max_seq_length", None) or 1024
            eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
            metrics["estimated_tokens_per_sec"] = round(done * eff_batch * seq_len / elapsed, 1)
        if state.max_steps:
            eta = elapsed / done * (state.max_steps - done)
            metrics["eta_seconds"] = round(eta, 1)
        self.emit(MetricEvent(step=self.last_step, total_steps=int(state.max_steps or 0), metrics=metrics))
        return control

    def on_save(self, args, state, control, **kwargs):
        path = f"{args.output_dir}/checkpoint-{state.global_step}"
        self.emit(CheckpointEvent(step=int(state.global_step), path=path))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.ctx.should_stop():
            self.emit(LogEvent(level="warning", message="Stop requested — checkpointing and halting."))
            control.should_training_stop = True
            control.should_save = True
        return control
