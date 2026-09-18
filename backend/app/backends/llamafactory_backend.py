"""Optional LLaMA-Factory adapter behind SLM Kit's TrainingBackend contract."""
from __future__ import annotations

import ast
import json
import queue
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

from sqlmodel import Session

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
from app.core.events import ArtifactEvent, CheckpointEvent, LogEvent, MetricEvent, TrainingEvent
from app.db.models import Dataset
from app.db.session import engine
from app.domain import (
    ExportedConfig,
    HardwareProfile,
    MemoryEstimate,
    Method,
    RunConfig,
    TaskType,
    ValidationReport,
)
from app.optimizations import optimization_capabilities


def _installation() -> tuple[bool, str | None, list[str] | None]:
    dependencies = dependency_statuses()
    executable = shutil.which("llamafactory-cli")
    package = dependencies["llamafactory"]
    if executable:
        return True, package.version, [executable]
    if package.installed:
        return True, package.version, [sys.executable, "-m", "llamafactory.cli"]
    return False, None, None


def _supported(reason: str) -> Capability:
    return Capability(state=SupportState.SUPPORTED, reason=reason, requirements=["llamafactory"])


def _version_at_least(value: str | None, minimum: tuple[int, ...]) -> bool:
    if not value:
        return False
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts[: len(minimum)]) >= minimum if parts else False


class LlamaFactoryBackend:
    name = "llamafactory"

    def capabilities(self) -> TrainingBackendCapabilities:
        optimizations = optimization_capabilities(self.name)
        installed, version, _ = _installation()
        if not installed:
            availability_state = SupportState.NOT_INSTALLED
        elif version is None:
            availability_state = SupportState.EXPERIMENTAL
        elif _version_at_least(version, (0, 9)):
            availability_state = SupportState.SUPPORTED
        else:
            availability_state = SupportState.INCOMPATIBLE
        if not installed:
            availability_reason = (
                "LLaMA-Factory backend is not installed. Optional: install the compatible "
                "llamafactory package and expose llamafactory-cli."
            )
        elif version is None:
            availability_reason = (
                "LLaMA-Factory is installed, but its version could not be verified; "
                "runtime use is experimental."
            )
        elif availability_state == SupportState.SUPPORTED:
            availability_reason = (
                f"LLaMA-Factory {version} is installed and satisfies the adapter's "
                ">=0.9 compatibility floor."
            )
        else:
            availability_reason = (
                f"LLaMA-Factory {version} is older than the adapter's supported >=0.9 contract."
            )
        availability = Capability(
            state=availability_state,
            reason=availability_reason,
            requirements=["llamafactory (optional)"],
        )
        supported = _supported("Translated to a versioned LLaMA-Factory training configuration.")
        unsupported = Capability(state=SupportState.UNSUPPORTED, reason="This operation is not mapped by the optional LLaMA-Factory adapter.")
        methods = {
            method.value: (supported if method in {Method.FULL, Method.FREEZE, Method.LORA, Method.QLORA, Method.DORA} else unsupported)
            for method in Method
        }
        peft = {
            method.value: PeftMethodCapability(
                method=method.value,
                support=methods[method.value],
                requires_quantized_base=method == Method.QLORA,
            )
            for method in (Method.LORA, Method.QLORA, Method.DORA)
        }
        return TrainingBackendCapabilities(
            name=self.name,
            display_name="LLaMA-Factory",
            description="Optional external training engine managed by the normal SLM Kit worker lifecycle.",
            availability=availability,
            tasks={
                task.value: (supported if task in {TaskType.CONTINUED_PRETRAIN, TaskType.FINETUNE, TaskType.ALIGNMENT} else unsupported)
                for task in TaskType
            },
            stages={
                "from_scratch_pretraining": unsupported,
                "continued_pretraining": supported,
                "supervised_fine_tuning": supported,
                "alignment": supported,
            },
            methods=methods,
            tokenizer=TokenizerCapabilities(
                modes={"reuse": supported, "extend": unsupported, "train": unsupported, "import": unsupported},
                loss_policies={
                    "full_sequence": supported,
                    "completion_only": Capability(
                        state=SupportState.UNSUPPORTED,
                        reason="Exact final-response-only masking is not mapped by this adapter.",
                    ),
                    "assistant_only": supported,
                },
                templates=ChatTemplateCapabilities(
                    native=supported,
                    explicit_override=Capability(
                        state=SupportState.UNSUPPORTED,
                        reason="Custom Jinja templates cannot be translated safely to a LLaMA-Factory template name.",
                    ),
                    fallback=unsupported,
                ),
            ),
            peft=peft,
            quantization={
                "nf4": QuantizationCapability(format="nf4", operations=["train"], support=supported,
                                                compute_dtypes=["auto", "bf16", "fp16"], double_quantization=True),
                "fp4": QuantizationCapability(format="fp4", operations=["train"], support=supported,
                                                compute_dtypes=["auto", "bf16", "fp16"], double_quantization=True),
                "int8": QuantizationCapability(format="int8", operations=["train"], support=supported),
            },
            precision={name: supported for name in ("auto", "bf16", "fp16", "fp32")},
            attention={
                "auto": supported,
                "sdpa": supported,
                "flash_attention_2": supported,
                "eager": supported,
            },
            gradient_checkpointing={
                name: (supported if name != "backend_optimized" else unsupported)
                for name in ("auto", "off", "standard", "non_reentrant", "backend_optimized")
            },
            rope={"none": supported, "linear": supported, "dynamic": supported},
            optimizations=optimizations,
            optional_features={
                "liger": supported,
                "dpo_loss:sigmoid": supported,
                "dpo_loss:hinge": supported,
                "reference:base_model": supported,
                "reference:adapter_disabled": supported,
                "reference:none": supported,
                "reference:separate_model": Capability(
                    state=SupportState.UNSUPPORTED,
                    reason="Separate reference models/revisions are not mapped by the optional adapter; use Transformers.",
                ),
                "version": Capability(
                    state=SupportState.SUPPORTED if installed else SupportState.NOT_INSTALLED,
                    reason=version or "No installed version detected.",
                ),
                **{
                    f"objective:{objective}": Capability(
                        state=SupportState.EXPERIMENTAL,
                        reason=(
                            f"{objective.upper()} is delegated to the installed LLaMA-Factory >=0.9 runtime; "
                            "the generated configuration preserves the requested objective."
                        ),
                        requirements=["llamafactory>=0.9"],
                    )
                    for objective in ("dpo", "ipo", "orpo", "simpo", "kto")
                },
            },
            platforms=["linux", "windows", "wsl"],
            architectures=["decoder-only causal language models supported by the installed LLaMA-Factory version"],
            required_dependencies=[],
            optional_dependencies=["llamafactory"],
        )

    @property
    def supported_tasks(self) -> set[TaskType]:
        return self.capabilities().supported_tasks

    @property
    def supported_methods(self) -> set[Method]:
        return self.capabilities().supported_methods

    def validate_config(self, cfg: RunConfig, hw: HardwareProfile) -> ValidationReport:
        report = ValidationReport()
        descriptor = self.capabilities()
        descriptor.validate_availability(report, required=True)
        descriptor.validate_operation(cfg, report)
        if not cfg.base_model:
            report.error("LLaMA-Factory requires a base model.")
        if cfg.tokenizer.mode != "reuse":
            report.error("The LLaMA-Factory adapter currently supports base-tokenizer reuse only.")
        if cfg.tokenizer.chat_template:
            report.error(
                "The LLaMA-Factory adapter cannot preserve a custom Jinja chat template; "
                "use the native backend or the model tokenizer's native template."
            )
        if cfg.task == TaskType.FINETUNE and cfg.tokenizer.loss_policy == "completion_only":
            report.error(
                "The LLaMA-Factory adapter cannot guarantee exact final-response-only masking for multi-turn rows."
            )
        if cfg.method == Method.QLORA and not hw.cuda_available:
            report.error("LLaMA-Factory QLoRA requires a detected CUDA GPU.")
        if cfg.runtime.gradient_checkpointing == "backend_optimized":
            report.error("Backend-optimized checkpointing is specific to Unsloth.")
        if cfg.alignment.objective in {"orpo", "simpo"} and cfg.task == TaskType.ALIGNMENT:
            report.warn("Reference-free objective support is delegated to the installed LLaMA-Factory version and verified at runtime.")
        if cfg.task == TaskType.ALIGNMENT and cfg.alignment.objective == "reward_model":
            report.error("Reward-model training currently uses the native Transformers/TRL backend.")
        if cfg.task == TaskType.ALIGNMENT:
            if cfg.train.packing:
                report.error("Preference alignment does not support sequence packing in this adapter.")
            if cfg.alignment.reference.strategy == "separate_model":
                report.error(descriptor.optional_features["reference:separate_model"].reason)
            if cfg.alignment.objective == "dpo":
                loss = descriptor.optional_features.get(f"dpo_loss:{cfg.alignment.dpo_loss_variant}")
                if loss is None or not loss.allowed:
                    report.error("The optional adapter supports sigmoid and hinge DPO losses only; use Transformers.")
        return report

    def estimate_footprint(self, cfg: RunConfig, hw: HardwareProfile) -> MemoryEstimate:
        from app.integrations import llmfit

        return llmfit.estimate_fit(cfg, hw)

    def export_config(self, cfg: RunConfig) -> ExportedConfig:
        content = json.dumps(self._config(cfg, Path("DATASET_DIR"), Path("OUTPUT_DIR"), "slmkit_dataset"), indent=2)
        return ExportedConfig(format="json", filename=f"{cfg.output_name}.llamafactory.json", content=content)

    def run(self, cfg: RunConfig, ctx: RunContext) -> Iterator[TrainingEvent]:
        installed, version, command = _installation()
        if not installed or command is None:
            raise RuntimeError(self.capabilities().availability.reason)
        from app.core.hardware import read_hardware

        validation = self.validate_config(cfg, read_hardware())
        if not validation.ok:
            detail = "; ".join(issue.message for issue in validation.issues if issue.level == "error")
            raise RuntimeError(f"LLaMA-Factory worker validation failed: {detail}")
        data_dir = ctx.workdir / "llamafactory_data"
        output_dir = ctx.workdir / "output"
        dataset_name = "slmkit_dataset"
        self._prepare_dataset(cfg, data_dir, dataset_name)
        native = self._config(cfg, data_dir, output_dir, dataset_name)
        config_path = ctx.workdir / "llamafactory.json"
        config_path.write_text(json.dumps(native, indent=2, ensure_ascii=False), encoding="utf-8")
        yield LogEvent(message=f"Starting optional LLaMA-Factory {version or 'unknown'} with generated config {config_path.name}.")

        proc = subprocess.Popen(
            [*command, "train", str(config_path)],
            cwd=str(ctx.workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.put(line.rstrip())
            lines.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        finished_output = False
        while proc.poll() is None or not finished_output:
            if ctx.should_stop() and proc.poll() is None:
                yield LogEvent(level="warning", message="Stop requested; terminating the LLaMA-Factory process tree.")
                _terminate_process_tree(proc)
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                finished_output = True
                continue
            if line:
                yield LogEvent(message=line)
                metrics = _parse_metrics(line)
                if metrics:
                    step = int(metrics.pop("step", 0))
                    yield MetricEvent(step=step, total_steps=cfg.train.max_steps, metrics=metrics)
        returncode = proc.wait()
        if ctx.should_stop():
            return
        if returncode != 0:
            raise RuntimeError(f"LLaMA-Factory exited with code {returncode}.")

        for checkpoint in sorted(output_dir.glob("checkpoint-*")):
            try:
                step = int(checkpoint.name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            yield CheckpointEvent(step=step, path=str(checkpoint))
        yield CheckpointEvent(step=cfg.train.max_steps or 0, path=str(output_dir), is_final=True)
        yield ArtifactEvent(
            kind="adapter" if cfg.method in {Method.LORA, Method.QLORA, Method.DORA} else "model",
            path=str(output_dir),
            metadata={
                "engine": "llamafactory", "version": version, "configuration": str(config_path),
                **({"objective": cfg.alignment.objective, "reference": cfg.alignment.reference.model_dump()}
                   if cfg.task == TaskType.ALIGNMENT else {}),
            },
        )

    def _config(self, cfg: RunConfig, data_dir: Path, output_dir: Path, dataset_name: str) -> dict:
        stage = "pt" if cfg.task == TaskType.CONTINUED_PRETRAIN else (
            "kto" if cfg.task == TaskType.ALIGNMENT and cfg.alignment.objective == "kto"
            else "dpo" if cfg.task == TaskType.ALIGNMENT
            else "sft"
        )
        finetuning_type = {
            Method.FULL: "full",
            Method.FREEZE: "freeze",
            Method.LORA: "lora",
            Method.QLORA: "lora",
            Method.DORA: "lora",
        }.get(cfg.method, cfg.method.value)
        native = {
            "model_name_or_path": cfg.base_model,
            "stage": stage,
            "do_train": True,
            "finetuning_type": finetuning_type,
            "dataset": dataset_name,
            "dataset_dir": str(data_dir),
            "output_dir": str(output_dir),
            "overwrite_output_dir": False,
            "cutoff_len": cfg.train.max_seq_length,
            "packing": bool(cfg.train.packing),
            "per_device_train_batch_size": cfg.train.per_device_batch_size,
            "gradient_accumulation_steps": cfg.train.gradient_accumulation,
            "learning_rate": cfg.optim.learning_rate,
            "num_train_epochs": cfg.train.epochs,
            "lr_scheduler_type": cfg.optim.lr_scheduler,
            "warmup_ratio": cfg.optim.warmup_ratio,
            "weight_decay": cfg.optim.weight_decay,
            "max_grad_norm": cfg.optim.max_grad_norm,
            "logging_steps": cfg.train.logging_steps,
            "save_steps": cfg.train.save_steps,
            "seed": cfg.train.seed,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "report_to": "none",
        }
        if cfg.task == TaskType.FINETUNE:
            native["train_on_prompt"] = cfg.tokenizer.loss_policy == "full_sequence"
        if cfg.revision:
            native["revision"] = cfg.revision
        if cfg.train.max_steps:
            native["max_steps"] = cfg.train.max_steps
        if cfg.method in {Method.LORA, Method.QLORA, Method.DORA}:
            native.update(lora_rank=cfg.lora.r, lora_alpha=cfg.lora.alpha,
                          lora_dropout=cfg.lora.dropout, lora_target=",".join(cfg.lora.target_modules) or "all")
        if cfg.method == Method.DORA:
            native["use_dora"] = True
        if cfg.method == Method.QLORA:
            native.update(
                quantization_bit=8 if cfg.quantization.mode == "int8" else 4,
                quantization_method="bnb",
            )
            if cfg.quantization.mode in {"nf4", "fp4"}:
                native.update(
                    quantization_type=cfg.quantization.mode,
                    double_quantization=cfg.quantization.double_quant,
                )
        if cfg.method == Method.FREEZE:
            native.update(
                freeze_trainable_layers=cfg.freeze.last_n_layers,
                freeze_trainable_modules=",".join(cfg.freeze.selected_modules),
                freeze_extra_modules=",".join(
                    name for enabled, name in (
                        (cfg.freeze.train_embeddings, "embed_tokens"),
                        (cfg.freeze.train_lm_head, "lm_head"),
                        (cfg.freeze.train_norms, "norm"),
                    ) if enabled
                ),
            )
        if cfg.runtime.precision == "bf16":
            native["bf16"] = True
        elif cfg.runtime.precision == "fp16":
            native["fp16"] = True
        elif cfg.runtime.precision == "fp32":
            native["fp16"] = False
            native["bf16"] = False
        if cfg.runtime.attention != "auto":
            native["flash_attn"] = {
                "flash_attention_2": "fa2", "sdpa": "sdpa", "eager": "disabled",
            }[cfg.runtime.attention]
        if cfg.runtime.use_liger:
            native["enable_liger_kernel"] = True
        if cfg.task == TaskType.ALIGNMENT:
            native["pref_beta"] = cfg.alignment.beta
            if cfg.alignment.objective != "kto":
                native["pref_loss"] = cfg.alignment.dpo_loss_variant if cfg.alignment.objective == "dpo" else cfg.alignment.objective
            if cfg.alignment.objective == "dpo":
                native["dpo_label_smoothing"] = cfg.alignment.label_smoothing
            if cfg.alignment.objective == "kto":
                native["kto_chosen_weight"] = cfg.alignment.desirable_weight
                native["kto_rejected_weight"] = cfg.alignment.undesirable_weight
            if cfg.alignment.objective == "simpo":
                native["simpo_gamma"] = cfg.alignment.simpo_gamma
        return native

    @staticmethod
    def _prepare_dataset(cfg: RunConfig, data_dir: Path, dataset_name: str) -> None:
        from app.datasets.adapters import canonicalize
        from app.datasets.validate import _iter_records

        with Session(engine) as db:
            dataset = db.get(Dataset, cfg.dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {cfg.dataset_id} not found.")
        data_dir.mkdir(parents=True, exist_ok=True)
        output = data_dir / "dataset.jsonl"
        rows = []
        canonical_kind: str | None = None
        for line, row in _iter_records(Path(dataset.path), dataset.fmt):
            if not isinstance(row, dict):
                raise ValueError(f"Dataset row {line} is not an object.")
            record = canonicalize(row)
            if canonical_kind is None:
                canonical_kind = record.kind
            elif canonical_kind != record.kind:
                raise ValueError("LLaMA-Factory datasets must contain one canonical row kind.")
            if record.kind == "text":
                rows.append({"text": record.content_text()})
            elif record.kind == "conversation":
                rows.append({"messages": [message.model_dump() for message in record.messages]})
            elif record.kind == "preference":
                if len(record.chosen) != 1 or len(record.rejected) != 1:
                    raise ValueError(
                        "LLaMA-Factory preference translation requires one chosen and one rejected response; "
                        "use the native backend for multi-message continuations."
                    )
                rows.append({
                    "messages": [message.model_dump() for message in record.prompt],
                    "chosen": record.chosen[0].model_dump(),
                    "rejected": record.rejected[0].model_dump(),
                })
            elif record.kind == "kto":
                rows.append({
                    "messages": [
                        *[message.model_dump() for message in record.prompt],
                        record.response.model_dump(),
                    ],
                    "kto_tag": record.desirable,
                })
            else:
                raise ValueError(f"LLaMA-Factory adapter cannot translate canonical {record.kind} rows.")
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if canonical_kind is None:
            raise ValueError("The selected dataset is empty.")
        dataset_info: dict = {"file_name": output.name}
        if canonical_kind == "text":
            dataset_info["columns"] = {"prompt": "text"}
        else:
            dataset_info.update(
                formatting="sharegpt",
                columns={"messages": "messages"},
                tags={
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                    "observation_tag": "tool",
                },
            )
            if canonical_kind == "preference":
                dataset_info["ranking"] = True
                dataset_info["columns"].update(chosen="chosen", rejected="rejected")
            elif canonical_kind == "kto":
                dataset_info["columns"]["kto_tag"] = "kto_tag"
        (data_dir / "dataset_info.json").write_text(
            json.dumps({dataset_name: dataset_info}, indent=2),
            encoding="utf-8",
        )


def _parse_metrics(line: str) -> dict[str, float]:
    start, end = line.find("{"), line.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = ast.literal_eval(line[start:end + 1])
    except (ValueError, SyntaxError):
        try:
            value = json.loads(line[start:end + 1])
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    allowed = {"loss", "learning_rate", "epoch", "grad_norm", "step", "rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies"}
    return {str(key): float(item) for key, item in value.items() if key in allowed and isinstance(item, (int, float))}


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        import psutil

        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=3)
        for process in alive:
            process.kill()
    except Exception:
        proc.kill()
