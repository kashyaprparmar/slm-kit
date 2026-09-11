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

from app.backends.base import RunContext
from app.core.events import (
    CheckpointEvent,
    LogEvent,
    MetricEvent,
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
    supported_tasks = {TaskType.PRETRAIN}
    supported_methods = {Method.FULL}

    def validate_config(self, cfg: RunConfig, hw: HardwareProfile) -> ValidationReport:
        report = ValidationReport()
        if cfg.task != TaskType.PRETRAIN:
            report.error("The scratch backend only supports from-scratch pretraining.")
        if cfg.dataset_id is None:
            report.error("A text corpus dataset is required for pretraining.")
        arch = cfg.arch or ScratchArch()
        if arch.n_embd % arch.n_heads != 0:
            report.error("n_embd must be divisible by n_heads.")
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
        tokenizer = self._train_tokenizer(corpus, arch, ctx, emit)
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
        emit(LogEvent(message=f"Model built: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.learning_rate,
                                      weight_decay=cfg.optim.weight_decay)

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

        for step in range(1, max_steps + 1):
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
                emit(MetricEvent(step=step, total_steps=max_steps, metrics={
                    "loss": round(float(loss.item()), 4),
                    "tokens_per_sec": round(tps, 1),
                    "eta_seconds": round(eta, 1),
                }))

            if step % cfg.train.save_steps == 0:
                self._save(model, tokenizer, cfg, ctx, step, is_final=False, emit=emit)
                sample = self._sample(model, tokenizer, device, emit_step=step)
                emit(SampleEvent(step=step, text=sample))

            if ctx.should_stop():
                emit(LogEvent(level="warning", message="Stop requested — saving and halting."))
                break

        self._save(model, tokenizer, cfg, ctx, step, is_final=True, emit=emit)

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

    def _train_tokenizer(self, corpus: str, arch: ScratchArch, ctx: RunContext, emit):
        from tokenizers import ByteLevelBPETokenizer

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

    def _save(self, model, tokenizer, cfg, ctx, step, is_final, emit):
        import torch

        out_dir = ctx.workdir / ("output" if is_final else f"checkpoints/checkpoint-{step}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "model.pt")
        (out_dir / "arch.json").write_text((cfg.arch or ScratchArch()).model_dump_json(indent=2))
        tokenizer.save_model(str(out_dir))
        emit(CheckpointEvent(step=step, path=str(out_dir), is_final=is_final))

    def _sample(self, model, tokenizer, device, emit_step, max_new=60) -> str:
        import torch

        model.eval()
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=max_new)
        return tokenizer.decode(out[0].tolist())


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
