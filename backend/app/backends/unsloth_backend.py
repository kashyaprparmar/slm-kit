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
from collections.abc import Iterator

from app.backends.base import RunContext
from app.core.events import (
    CheckpointEvent,
    LogEvent,
    MetricEvent,
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

_SENTINEL = object()


class UnslothBackend:
    name = "unsloth"
    supported_tasks = {TaskType.CONTINUED_PRETRAIN, TaskType.FINETUNE}
    supported_methods = {Method.LORA, Method.QLORA, Method.DORA, Method.FULL}

    # ---- metadata (runs in API process) -------------------------------
    def validate_config(self, cfg: RunConfig, hw: HardwareProfile) -> ValidationReport:
        report = ValidationReport()
        if cfg.task not in self.supported_tasks:
            report.error(f"Unsloth backend does not support task '{cfg.task.value}'.")
        if cfg.method not in self.supported_methods:
            report.error(f"Unsloth backend does not support method '{cfg.method.value}'.")
        if not cfg.base_model:
            report.error("A base model (HF repo id or local path) is required.")
        if cfg.method == Method.QLORA and not cfg.load_in_4bit:
            report.warn("QLoRA selected but load_in_4bit is off — enabling 4-bit is recommended on 8GB.")

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
        events: "queue.Queue" = queue.Queue()
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
    def _train_worker(self, cfg: RunConfig, ctx: RunContext, out: "queue.Queue", holder: dict) -> None:
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
        import torch
        from unsloth import FastLanguageModel

        emit(LogEvent(message=f"Loading base model {cfg.base_model} (4bit={cfg.load_in_4bit})…"))
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.train.max_seq_length,
            dtype=None,  # auto: bf16 on Ampere+, else fp16
            load_in_4bit=cfg.load_in_4bit and cfg.method != Method.FULL,
        )

        if cfg.method != Method.FULL:
            emit(LogEvent(message=f"Attaching {cfg.method.value.upper()} adapters (r={cfg.lora.r})…"))
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora.r,
                target_modules=cfg.lora.target_modules,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=cfg.train.seed,
                use_rslora=cfg.lora.use_rslora,
                use_dora=(cfg.method == Method.DORA) or cfg.lora.use_dora,
            )

        dataset = self._load_dataset(cfg, tokenizer, emit)

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
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            dataset_text_field="text",
            packing=cfg.train.packing,
            report_to="none",
        )
        # trl renamed max_seq_length → max_length (~0.20); support both.
        sft_params = inspect.signature(SFTConfig.__init__).parameters
        if "max_length" in sft_params:
            config_kwargs["max_length"] = cfg.train.max_seq_length
        else:
            config_kwargs["max_seq_length"] = cfg.train.max_seq_length
        args = SFTConfig(**config_kwargs)

        callback = _StreamCallback(ctx, emit)
        # trl renamed tokenizer → processing_class (~0.12); support both.
        trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        tok_kw = "processing_class" if "processing_class" in trainer_params else "tokenizer"
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=args,
            callbacks=[callback],
            **{tok_kw: tokenizer},
        )

        emit(LogEvent(message="Starting training…"))
        trainer.train(resume_from_checkpoint=str(ctx.resume_from) if ctx.resume_from else None)

        final_dir = ctx.workdir / "output"
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        emit(CheckpointEvent(step=callback.last_step, path=str(final_dir), is_final=True))
        emit(LogEvent(message=f"Saved final model to {final_dir}"))

    # -------------------------------------------------------------------
    def _load_dataset(self, cfg: RunConfig, tokenizer, emit):
        """Resolve the dataset row and produce a HF Dataset with a 'text' column."""
        from datasets import load_dataset

        from app.db.models import Dataset
        from app.db.session import engine
        from sqlmodel import Session

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
            data = load_dataset("text", data_files=path, split="train")
            return data

        # Instruction/chat: build a 'text' column via the model's chat template.
        if ds_row.fmt in ("jsonl", "json"):
            data = load_dataset("json", data_files=path, split="train")
        elif ds_row.fmt == "csv":
            data = load_dataset("csv", data_files=path, split="train")
        elif ds_row.fmt == "parquet":
            data = load_dataset("parquet", data_files=path, split="train")
        else:
            data = load_dataset("text", data_files=path, split="train")

        def to_text(example: dict) -> dict:
            return {"text": _format_example(example, tokenizer)}

        return data.map(to_text, remove_columns=[c for c in data.column_names if c != "text"])


def _format_example(ex: dict, tokenizer) -> str:
    """Turn a variety of common instruction schemas into a single training string."""
    if "messages" in ex and isinstance(ex["messages"], list):
        messages = ex["messages"]
    else:
        instruction = ex.get("instruction") or ex.get("prompt") or ex.get("question") or ""
        context = ex.get("input") or ex.get("context") or ""
        answer = ex.get("output") or ex.get("response") or ex.get("answer") or ""
        user = instruction if not context else f"{instruction}\n\n{context}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Model has no chat template — fall back to a plain format.
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n".join(parts) + (tokenizer.eos_token or "")


class _StreamCallback:
    """TRL/Transformers trainer callback that streams metrics + honors cancel.

    Not a ``TrainerCallback`` subclass (that would force a transformers import in
    the API process); instead ``__getattr__`` supplies no-op handlers for every
    other ``on_*`` hook the CallbackHandler invokes.
    """

    def __init__(self, ctx: RunContext, emit) -> None:
        self.ctx = ctx
        self.emit = emit
        self.last_step = 0
        self._t0 = time.time()

    def __getattr__(self, name: str):
        if name.startswith("on_"):
            return lambda *args, **kwargs: None  # no-op for unhandled trainer hooks
        raise AttributeError(name)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        self.last_step = int(state.global_step)
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        elapsed = max(1e-6, time.time() - self._t0)
        done = max(1, state.global_step)
        if "tokens_per_sec" not in metrics:
            # Approximate throughput: steps × effective batch × seq length.
            # (trl renamed max_seq_length → max_length; check both.)
            seq_len = getattr(args, "max_length", None) or getattr(args, "max_seq_length", None) or 1024
            eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
            metrics["tokens_per_sec"] = round(done * eff_batch * seq_len / elapsed, 1)
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
