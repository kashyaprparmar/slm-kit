"""From-scratch pretraining backend (Pillar 1).

A compact, self-contained GPT-style trainer modeled on the structure of
FareedKhan-dev/train-llm-from-scratch: train a custom tokenizer on the corpus,
build a small decoder-only transformer, and run a plain PyTorch training loop.
Pure torch — no Unsloth — so it doubles as the "does my GPU work at all?" smoke
test. All heavy imports are inside ``_train``.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path

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
    SampleEvent,
    TrainingEvent,
)
from app.domain import (
    ExportedConfig,
    FitLevel,
    HardwareProfile,
    MemoryEstimate,
    Method,
    RunConfig,
    ScratchArch,
    TaskType,
    ValidationReport,
)

_SENTINEL = object()


class ScratchBackend:
    name = "scratch"

    def capabilities(self) -> TrainingBackendCapabilities:
        dependencies = dependency_statuses()
        required = ["torch", "tokenizers"]
        missing = [name for name in required if not dependencies[name].installed]
        availability = Capability(
            state=SupportState.MISSING_DEPENDENCY if missing else SupportState.SUPPORTED,
            reason=(f"Missing required packages: {', '.join(missing)}." if missing else "Built-in from-scratch trainer is available."),
            requirements=required,
        )
        supported = Capability(state=SupportState.SUPPORTED, reason="Implemented by the scratch training loop.")
        unsupported = Capability(state=SupportState.UNSUPPORTED, reason="The scratch backend only initializes a new SLM Kit GPT.")
        return TrainingBackendCapabilities(
            name=self.name,
            display_name="From scratch",
            description="Train a small decoder-only GPT and tokenizer from an uninitialized architecture.",
            availability=availability,
            tasks={task.value: (supported if task == TaskType.PRETRAIN else unsupported) for task in TaskType},
            stages={
                "from_scratch_pretraining": supported,
                "continued_pretraining": unsupported,
                "supervised_fine_tuning": unsupported,
                "alignment": unsupported,
            },
            methods={method.value: (supported if method == Method.FULL else unsupported) for method in Method},
            tokenizer=TokenizerCapabilities(
                modes={"reuse": unsupported, "extend": unsupported, "train": supported, "import": supported},
                loss_policies={
                    "full_sequence": supported,
                    "completion_only": unsupported,
                    "assistant_only": unsupported,
                },
                templates=ChatTemplateCapabilities(
                    native=unsupported,
                    explicit_override=unsupported,
                    fallback=unsupported,
                ),
            ),
            peft={
                method.value: PeftMethodCapability(method=method.value, support=unsupported)
                for method in (Method.LORA, Method.QLORA, Method.DORA, Method.PROMPT_TUNING)
            },
            quantization={
                "fp32": QuantizationCapability(format="fp32", operations=["train"], support=supported),
            },
            platforms=["linux", "windows", "wsl"],
            architectures=["SLM Kit GPT"],
            required_dependencies=required,
        )

    @property
    def supported_tasks(self) -> set[TaskType]:
        return self.capabilities().supported_tasks

    @property
    def supported_methods(self) -> set[Method]:
        return self.capabilities().supported_methods

    def validate_config(self, cfg: RunConfig, hw: HardwareProfile) -> ValidationReport:
        report = ValidationReport()
        self.capabilities().validate_operation(cfg, report)
        if cfg.dataset_id is None:
            report.error("A text corpus dataset is required for pretraining.")
        arch = cfg.arch or ScratchArch()
        if arch.n_embd % arch.n_heads != 0:
            report.error("n_embd must be divisible by n_heads.")
        if cfg.tokenizer.mode == "extend":
            report.error("From-scratch pretraining supports a newly trained or imported tokenizer, not extend mode.")
        if cfg.tokenizer.mode == "reuse":
            report.info("Legacy scratch tokenizer mode 'reuse' is interpreted as training a new tokenizer.")
        if cfg.tokenizer.mode == "import" and not cfg.tokenizer.source:
            report.error("Imported scratch tokenizers require tokenizer.source.")
        est = self.estimate_footprint(cfg, hw)
        if est.fit == FitLevel.WONT_FIT:
            report.error(f"Predicted VRAM ~{est.total_mb:.0f} MB exceeds budget; shrink the architecture.")
        elif est.fit == FitLevel.TIGHT:
            report.warn("Tight fit — reduce n_layers/n_embd/block_size if you hit OOM.")
        report.info("From-scratch models need a lot of tokens; expect a toy model from small corpora.")
        return report

    def estimate_footprint(self, cfg: RunConfig, hw: HardwareProfile) -> MemoryEstimate:
        arch = cfg.arch or ScratchArch()
        # Params ≈ embeddings + 12 * n_layers * n_embd^2 (standard GPT estimate).
        n_params = arch.vocab_size * arch.n_embd + 12 * arch.n_layers * arch.n_embd**2
        mb = 1024 * 1024
        weights = n_params * 2 / mb
        optimizer = n_params * (4 + 4 + 4) / mb   # fp32 AdamW: grad + m + v
        b = cfg.train.per_device_batch_size
        acts = b * arch.block_size * arch.n_embd * arch.n_layers * 4 * 2 / mb
        overhead = 600 + 0.15 * (weights + optimizer + acts)
        total = weights + optimizer + acts + overhead
        budget = float(hw.vram_total_mb or 8192)
        safe = budget * 0.9
        fit = FitLevel.FITS if total <= safe * 0.8 else FitLevel.TIGHT if total <= safe else FitLevel.WONT_FIT
        return MemoryEstimate(
            weights_mb=round(weights, 1), optimizer_mb=round(optimizer, 1),
            activations_mb=round(acts, 1), overhead_mb=round(overhead, 1),
            total_mb=round(total, 1), budget_mb=budget, fit=fit, source="fallback",
            notes=[f"~{n_params/1e6:.1f}M parameters"],
            total_parameters=n_params, trainable_parameters=n_params,
            frozen_parameters=0, trainable_percentage=100.0,
        )

    def export_config(self, cfg: RunConfig) -> ExportedConfig:
        return ExportedConfig(
            format="json",
            filename=f"{cfg.output_name}.scratch.json",
            content=cfg.model_dump_json(indent=2),
        )

    def run(self, cfg: RunConfig, ctx: RunContext) -> Iterator[TrainingEvent]:
        events: queue.Queue = queue.Queue()
        holder: dict = {}
        worker = threading.Thread(target=self._worker, args=(cfg, ctx, events, holder), daemon=True)
        worker.start()
        while True:
            item = events.get()
            if item is _SENTINEL:
                break
            yield item
        worker.join(timeout=5)
        if "exc" in holder:
            raise holder["exc"]

    def _worker(self, cfg: RunConfig, ctx: RunContext, out: queue.Queue, holder: dict) -> None:
        try:
            self._train(cfg, ctx, out.put)
        except Exception as exc:  # noqa: BLE001 — propagated to run() via holder
            holder["exc"] = exc
            out.put(LogEvent(level="error", message=f"Pretraining error: {exc}"))
        finally:
            out.put(_SENTINEL)

    def _train(self, cfg: RunConfig, ctx: RunContext, emit) -> None:
        import torch

        arch = cfg.arch or ScratchArch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(cfg.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.train.seed)
        emit(LogEvent(message=f"Pretraining on {device}; arch={arch.n_layers}L/{arch.n_embd}d/{arch.block_size}ctx"))

        corpus = self._read_corpus(cfg, emit)
        tokenizer = self._load_or_train_tokenizer(corpus, arch, cfg, ctx, emit)
        ids = tokenizer.encode(corpus).ids
        data = torch.tensor(ids, dtype=torch.long)
        emit(LogEvent(message=f"Corpus tokenized: {len(ids):,} tokens, vocab={tokenizer.get_vocab_size()}"))

        n = int(0.9 * len(data))
        train_data, val_data = data[:n], data[n:]
        if len(train_data) <= arch.block_size + 1:
            raise ValueError(
                f"Corpus too small: {len(train_data)} train tokens but block_size={arch.block_size}. "
                "Use a bigger corpus or a smaller context length."
            )

        model = _GPT(arch, tokenizer.get_vocab_size()).to(device)
        total_parameters = sum(p.numel() for p in model.parameters())
        emit(ProfileEvent(name="parameters", values={
            "total": total_parameters,
            "trainable": total_parameters,
            "frozen": 0,
            "trainable_percentage": 100.0,
        }))
        emit(LogEvent(message=f"Model built: {total_parameters/1e6:.1f}M params"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.learning_rate,
                                      weight_decay=cfg.optim.weight_decay)
        start_step = 1
        if ctx.resume_from:
            arch_path = ctx.resume_from / "arch.json"
            if arch_path.is_file():
                checkpoint_arch = ScratchArch.model_validate_json(arch_path.read_text(encoding="utf-8"))
                if checkpoint_arch != arch:
                    raise ValueError(
                        "Scratch resume architecture does not match the checkpoint arch.json. "
                        "Use the checkpoint architecture unchanged when resuming."
                    )
            weights_path = ctx.resume_from / "model.pt"
            if not weights_path.is_file():
                raise ValueError(f"Scratch resume checkpoint is missing {weights_path.name}.")
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            state_path = ctx.resume_from / "training.pt"
            if state_path.is_file():
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(state["optimizer"])
                for optimizer_state in optimizer.state.values():
                    for key, value in optimizer_state.items():
                        if torch.is_tensor(value):
                            optimizer_state[key] = value.to(device)
                torch.set_rng_state(state["cpu_rng"])
                if torch.cuda.is_available() and state.get("cuda_rng"):
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
                start_step = int(state["step"]) + 1
                emit(LogEvent(message=f"Resuming scratch optimizer/RNG state at step {start_step}."))
            else:
                emit(LogEvent(level="warning", message=(
                    "Legacy scratch checkpoint has weights only; optimizer/RNG state cannot be restored."
                )))

        block = arch.block_size
        batch = cfg.train.per_device_batch_size
        max_steps = cfg.train.max_steps or 1000
        t0 = time.time()

        def get_batch(split):
            src = train_data if split == "train" else val_data
            ix = torch.randint(max(1, len(src) - block), (batch,))
            x = torch.stack([src[i:i + block] for i in ix])
            y = torch.stack([src[i + 1:i + 1 + block] for i in ix])
            return x.to(device), y.to(device)

        final_step = start_step - 1
        for step in range(start_step, max_steps + 1):
            final_step = step
            model.train()
            x, y = get_batch("train")
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
            optimizer.step()

            if step % cfg.train.logging_steps == 0 or step == 1:
                tps = step * batch * block / max(1e-6, time.time() - t0)
                eta = (time.time() - t0) / step * (max_steps - step)
                metrics = {
                    "loss": round(float(loss.item()), 4),
                    "tokens_per_sec": round(tps, 1),
                    "eta_seconds": round(eta, 1),
                }
                if len(val_data) > block + 1:
                    cpu_rng = torch.get_rng_state()
                    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    try:
                        model.eval()
                        with torch.no_grad():
                            val_x, val_y = get_batch("val")
                            _, val_loss = model(val_x, val_y)
                        metrics["validation_loss"] = round(float(val_loss.item()), 4)
                    finally:
                        torch.set_rng_state(cpu_rng)
                        if cuda_rng is not None:
                            torch.cuda.set_rng_state_all(cuda_rng)
                emit(MetricEvent(step=step, total_steps=max_steps, metrics=metrics))

            if step % cfg.train.save_steps == 0:
                self._save(model, tokenizer, optimizer, cfg, ctx, step, is_final=False, emit=emit)
                sample = self._sample(model, tokenizer, device, emit_step=step)
                emit(SampleEvent(step=step, text=sample))

            if ctx.should_stop():
                emit(LogEvent(level="warning", message="Stop requested — saving and halting."))
                break

        self._save(model, tokenizer, optimizer, cfg, ctx, final_step, is_final=True, emit=emit)

    # -------------------------------------------------------------------
    def _read_corpus(self, cfg: RunConfig, emit) -> str:
        from sqlmodel import Session

        from app.db.models import Dataset
        from app.db.session import engine

        with Session(engine) as db:
            row = db.get(Dataset, cfg.dataset_id)
        if row is None:
            raise ValueError(f"Dataset {cfg.dataset_id} not found.")
        from pathlib import Path

        from app.datasets.validate import _extract_text, _iter_records
        return "\n".join(_extract_text(record) for _, record in _iter_records(Path(row.path), row.fmt)
                         if isinstance(record, dict))

    def _load_or_train_tokenizer(self, corpus: str, arch: ScratchArch, cfg: RunConfig, ctx: RunContext, emit):
        from tokenizers import ByteLevelBPETokenizer

        source = ctx.resume_from if ctx.resume_from else (
            Path(cfg.tokenizer.source) if cfg.tokenizer.mode == "import" and cfg.tokenizer.source else None
        )
        if source:
            vocab, merges = source / "vocab.json", source / "merges.txt"
            if not vocab.is_file() or not merges.is_file():
                raise ValueError("Imported/resumed scratch tokenizer requires vocab.json and merges.txt.")
            emit(LogEvent(message=f"Loading scratch tokenizer from {source}."))
            return ByteLevelBPETokenizer(str(vocab), str(merges))

        emit(LogEvent(message=f"Training byte-level BPE tokenizer (vocab={arch.vocab_size})…"))
        tok_dir = ctx.workdir / "tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        corpus_file = ctx.workdir / "corpus.txt"
        corpus_file.write_text(corpus, encoding="utf-8")

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[str(corpus_file)],
            vocab_size=arch.vocab_size,
            min_frequency=2,
            special_tokens=["<|endoftext|>", "<|pad|>"],
        )
        tokenizer.save_model(str(tok_dir))
        return tokenizer

    def _save(self, model, tokenizer, optimizer, cfg, ctx, step, is_final, emit):
        import torch

        out_dir = ctx.workdir / ("output" if is_final else f"checkpoints/checkpoint-{step}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "model.pt")
        torch.save({
            "step": step,
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, out_dir / "training.pt")
        (out_dir / "arch.json").write_text((cfg.arch or ScratchArch()).model_dump_json(indent=2))
        tokenizer.save_model(str(out_dir))
        emit(CheckpointEvent(step=step, path=str(out_dir), is_final=is_final))

    def _sample(self, model, tokenizer, device, emit_step, max_new=60) -> str:
        import torch

        model.eval()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                out = model.generate(idx, max_new_tokens=max_new)
            return tokenizer.decode(out[0].tolist())
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


# --------------------------------------------------------------------------- #
# Minimal GPT (defined lazily-friendly: only used after torch import in _train).
# --------------------------------------------------------------------------- #
def _make_gpt_classes():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Block(nn.Module):
        def __init__(self, arch: ScratchArch):
            super().__init__()
            self.ln1 = nn.LayerNorm(arch.n_embd)
            self.attn = nn.MultiheadAttention(arch.n_embd, arch.n_heads,
                                              dropout=arch.dropout, batch_first=True)
            self.ln2 = nn.LayerNorm(arch.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(arch.n_embd, 4 * arch.n_embd), nn.GELU(),
                nn.Linear(4 * arch.n_embd, arch.n_embd), nn.Dropout(arch.dropout),
            )

        def forward(self, x):
            T = x.size(1)
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            a = self.ln1(x)
            attn, _ = self.attn(a, a, a, attn_mask=mask, need_weights=False)
            x = x + attn
            x = x + self.mlp(self.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(self, arch: ScratchArch, vocab_size: int):
            super().__init__()
            self.arch = arch
            self.tok_emb = nn.Embedding(vocab_size, arch.n_embd)
            self.pos_emb = nn.Embedding(arch.block_size, arch.n_embd)
            self.drop = nn.Dropout(arch.dropout)
            self.blocks = nn.ModuleList([Block(arch) for _ in range(arch.n_layers)])
            self.ln_f = nn.LayerNorm(arch.n_embd)
            self.head = nn.Linear(arch.n_embd, vocab_size, bias=False)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
            for blk in self.blocks:
                x = blk(x)
            logits = self.head(self.ln_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.arch.block_size:]
                logits, _ = self(idx_cond)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, nxt], dim=1)
            return idx

    return GPT


class _GPT:
    """Thin factory so the module imports without torch present."""

    def __new__(cls, arch: ScratchArch, vocab_size: int):
        gpt_cls = _make_gpt_classes()
        return gpt_cls(arch, vocab_size)
