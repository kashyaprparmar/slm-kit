"""Runtime loader shared by generation, evaluation, and local deployment.

Heavy imports intentionally stay in this subprocess-only module.  It turns the
lightweight model-reference contract into a single inference interface for:

* normal Hugging Face decoder-only models;
* PEFT/LoRA adapters (base model + adapter loaded explicitly); and
* SLM Kit's from-scratch GPT checkpoints.

Keeping this in one place prevents the Playground and Eval Lab from drifting
into subtly different loading behaviour.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from app.model_refs import ResolvedModel, resolve_model_ref

# PEFT probes TorchAO integrations while loading ordinary LoRA adapters.  Some
# TorchAO wheels include architecture-specific optional extensions (for example
# Hopper-only CUTLASS kernels) that cannot load on otherwise-supported GPUs.
# Those warnings do not affect PEFT inference and only make successful Eval Lab
# runs look broken; real import/runtime failures still propagate normally.
logging.getLogger("torchao").setLevel(logging.ERROR)


def _device_for(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:  # noqa: BLE001 - third-party model implementations vary
        for param in model.parameters():
            if param.device.type != "meta":
                return param.device
        raise RuntimeError("The loaded model has no materialized parameters.")


def _encode(tokenizer, prompt: str, device, params=None):
    params = params or {}
    messages = params.get("messages") or ([{"role": "system", "content": params["system_prompt"]}] if params.get("system_prompt") else []) + [{"role": "user", "content": prompt}]
    encoded = None
    try:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except (TypeError, ValueError):
        try:
            value = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            encoded = {"input_ids": value}
        except Exception:  # Models without a chat template are still valid.
            encoded = tokenizer(prompt, return_tensors="pt")
    if not hasattr(encoded, "items"):
        encoded = {"input_ids": encoded}
    return {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}


@dataclass
class ModelRuntime:
    spec: ResolvedModel
    model: object
    tokenizer: object
    device: object
    scratch: bool = False

    @property
    def display_name(self) -> str:
        return self.spec.label

    def stream(self, prompt: str, params: dict) -> Iterator[str]:
        if self.scratch:
            yield from self._stream_scratch(prompt, params)
            return
        yield from self._stream_transformers(prompt, params)

    def generate(self, prompt: str, params: dict) -> str:
        return "".join(self.stream(prompt, params)).strip()

    def _stream_transformers(self, prompt: str, params: dict) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        tokenizer = self.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = _encode(tokenizer, prompt, self.device, params)
        context = getattr(self.model.config, "max_position_embeddings", 4096)
        if encoded["input_ids"].shape[-1] + int(params.get("max_new_tokens", 256)) > context:
            raise ValueError(f"Prompt plus response exceeds the model's {context}-token context. Shorten the prompt or response limit.")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        temperature = float(params.get("temperature", 0.7))
        kwargs = {
            **encoded,
            "streamer": streamer,
            "max_new_tokens": int(params.get("max_new_tokens", 256)),
            "do_sample": temperature > 0,
            "repetition_penalty": float(params.get("repetition_penalty", 1.1)),
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs.update(
                temperature=temperature,
                top_p=float(params.get("top_p", 0.95)),
                top_k=int(params.get("top_k", 50)),
            )
        else:
            # Override sampling values embedded in some models' generation
            # configs as well as omitting request-level sampling controls.
            kwargs.update(temperature=None, top_p=None, top_k=None)
        errors = []
        def generate_worker():
            try:
                import torch
                if params.get("seed") is not None:
                    torch.manual_seed(int(params["seed"]))
                with torch.inference_mode():
                    self.model.generate(**kwargs)
            except BaseException as exc:
                errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)
        worker = threading.Thread(target=generate_worker, daemon=True)
        worker.start()
        for text in streamer:
            if text:
                yield text
        worker.join()
        if errors:
            raise errors[0]

    def _stream_scratch(self, prompt: str, params: dict) -> Iterator[str]:
        import torch

        tokenizer = self.tokenizer
        if params.get("seed") is not None:
            torch.manual_seed(int(params["seed"]))
        ids = tokenizer.encode(prompt).ids or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        max_new = int(params.get("max_new_tokens", 256))
        temperature = float(params.get("temperature", 0.7))
        top_k = int(params.get("top_k", 50))
        top_p = float(params.get("top_p", 0.95))
        generated: list[int] = []
        emitted = ""
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new):
                logits, _ = self.model(idx[:, -self.model.arch.block_size:])
                logits = logits[:, -1, :]
                if temperature <= 0:
                    nxt = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-5)
                    if 0 < top_k < logits.shape[-1]:
                        cutoff = torch.topk(logits, top_k).values[:, -1, None]
                        logits = logits.masked_fill(logits < cutoff, float("-inf"))
                    if 0 < top_p < 1:
                        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        remove = cumulative > top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        logits.scatter_(1, sorted_idx, sorted_logits.masked_fill(remove, float("-inf")))
                    nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                token_id = int(nxt.item())
                generated.append(token_id)
                idx = torch.cat((idx, nxt), dim=1)
                # Byte-level BPE can split Unicode characters across tokens.
                decoded = tokenizer.decode(generated)
                text = decoded[len(emitted):] if not decoded.endswith("\ufffd") else ""
                if text:
                    emitted = decoded
                    yield text

    def reference_perplexity(self, prompt: str, reference: str) -> float | None:
        """Compute continuation perplexity while excluding prompt tokens from loss."""
        if not reference.strip():
            return None
        import torch

        if self.scratch:
            ids = self.tokenizer.encode(f"{prompt}\n{reference}").ids
            prompt_len = len(self.tokenizer.encode(prompt).ids)
            overflow = max(0, len(ids) - self.model.arch.block_size - 1)
            ids = ids[overflow:]
            prompt_len = max(0, prompt_len - overflow)
            if len(ids) < 2:
                return None
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)
            # Labels predict ids[1:], so positions before the answer boundary
            # must be ignored. At least one continuation token must remain.
            y[:, : max(0, prompt_len - 1)] = -100
            if torch.all(y == -100):
                return None
            with torch.no_grad():
                logits, _ = self.model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100
                )
            return float(math.exp(min(float(loss), 20)))

        tokenizer = self.tokenizer
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        full = tokenizer(f"{prompt}\n{reference}", return_tensors="pt").input_ids.to(self.device)
        labels = full.clone()
        labels[:, : min(labels.shape[1], prompt_ids.shape[1])] = -100
        if (labels != -100).sum().item() == 0:
            return None
        with torch.no_grad():
            loss = self.model(full, labels=labels).loss
        return float(math.exp(min(float(loss), 20)))

    def unload(self) -> None:
        """Release references before the caller moves to another model."""
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_scratch(spec: ResolvedModel) -> ModelRuntime:
    import torch
    from tokenizers import ByteLevelBPETokenizer

    from app.backends.scratch_backend import _GPT
    from app.domain import ScratchArch

    path = spec.local_path
    assert path is not None
    root = Path(path)
    arch = ScratchArch.model_validate_json((root / "arch.json").read_text(encoding="utf-8"))
    tokenizer = ByteLevelBPETokenizer(str(root / "vocab.json"), str(root / "merges.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _GPT(arch, tokenizer.get_vocab_size()).to(device)
    try:
        state = torch.load(root / "model.pt", map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0 compatibility for existing checkpoints
        state = torch.load(root / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return ModelRuntime(spec=spec, model=model, tokenizer=tokenizer, device=device, scratch=True)


def _load_transformers(spec: ResolvedModel) -> ModelRuntime:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from app.config import get_settings

    settings = get_settings()
    trust_remote_code = settings.trust_remote_code
    from app.integrations.hf_hub import _token
    common = {"token": _token(), "cache_dir": str(settings.hf_cache_dir), "trust_remote_code": trust_remote_code}
    load_ref = spec.load_ref
    if spec.kind == "transformers" and spec.local_path is None:
        # Remote PEFT repositories look like ordinary HF ids to the lightweight
        # resolver. Probe their small adapter config before loading gigabytes of
        # weights so published SLM Kit adapters work by repo id too.
        try:
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(load_ref, token=_token(), cache_dir=str(settings.hf_cache_dir))
            base_ref = getattr(peft_config, "base_model_name_or_path", None)
            if base_ref:
                spec = replace(
                    spec,
                    kind="adapter",
                    base_model=str(base_ref),
                    adapter_path=load_ref,
                )
        except Exception:
            pass
    if spec.kind == "adapter":
        from peft import PeftModel

        assert spec.base_model and spec.adapter_path
        tokenizer_ref = spec.adapter_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, **common)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(spec.base_model, **common)
        base = AutoModelForCausalLM.from_pretrained(
            spec.base_model,
            dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            **common,
        )
        model = PeftModel.from_pretrained(base, spec.adapter_path, token=_token())
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_ref, **common)
        model = AutoModelForCausalLM.from_pretrained(
            load_ref,
            dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            **common,
        )
    model.eval()
    return ModelRuntime(spec=spec, model=model, tokenizer=tokenizer, device=_device_for(model))


def load_runtime(model_ref: str) -> ModelRuntime:
    """Load any supported reference and return a common inference runtime."""
    spec = resolve_model_ref(model_ref)
    if spec.kind == "scratch":
        return _load_scratch(spec)
    return _load_transformers(spec)
