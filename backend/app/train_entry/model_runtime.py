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
from dataclasses import dataclass, field, replace
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
    if params.get("raw_prompt"):
        encoded = tokenizer(prompt, return_tensors="pt")
        return {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}
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
    load_metadata: dict = field(default_factory=dict)

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


@dataclass
class RewardModelRuntime:
    """Scalar sequence-scoring runtime; it intentionally has no generate API."""

    spec: ResolvedModel
    model: object
    tokenizer: object
    device: object

    def score(self, text: str) -> float:
        import torch

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        if logits.numel() != 1:
            raise RuntimeError("Reward model must produce exactly one scalar score per sequence.")
        return float(logits.reshape(-1)[0].detach().cpu())

    def unload(self) -> None:
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


def _load_transformers(spec: ResolvedModel, *, for_merge: bool = False, base_revision: str | None = None) -> ModelRuntime:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from app.config import get_settings

    settings = get_settings()
    trust_remote_code = settings.trust_remote_code
    from app.integrations.hf_hub import _token
    common = {"token": _token(), "cache_dir": str(settings.hf_cache_dir), "trust_remote_code": trust_remote_code}
    load_ref = spec.load_ref
    load_metadata = {}
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
        from peft import PeftConfig, PeftModel

        assert spec.base_model and spec.adapter_path
        adapter_config = PeftConfig.from_pretrained(spec.adapter_path, token=_token())
        recorded_revision = getattr(adapter_config, "revision", None)
        if for_merge and base_revision and recorded_revision and recorded_revision != base_revision:
            raise ValueError("Requested base revision differs from the adapter's recorded revision.")
        base_common = {**common, "revision": base_revision or recorded_revision}
        if for_merge:
            if Path(spec.base_model).is_dir():
                from app.train_entry.export import fingerprint
                load_metadata["base_fingerprint"] = fingerprint(Path(spec.base_model))
            if not Path(spec.base_model).exists():
                import re
                if not re.fullmatch(r"[0-9a-fA-F]{40}", base_common["revision"] or ""):
                    raise ValueError("Remote adapter merging requires the immutable base commit used for training; mutable or unknown revisions are unsafe.")
            base_config = AutoConfig.from_pretrained(spec.base_model, **base_common)
            if getattr(base_config, "quantization_config", None):
                raise ValueError("Cannot merge adapters into a quantized base. Use the original unquantized base.")
        tokenizer_ref = spec.adapter_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, **common)
        except Exception:
            if for_merge and Path(tokenizer_ref).is_dir() and any((Path(tokenizer_ref) / name).exists() for name in ("tokenizer.json", "tokenizer_config.json")):
                raise ValueError("Adapter tokenizer metadata exists but cannot be loaded safely.")
            tokenizer = AutoTokenizer.from_pretrained(spec.base_model, **base_common)
        if for_merge:
            base_tokenizer = AutoTokenizer.from_pretrained(spec.base_model, **base_common)
            if tokenizer.get_vocab() != base_tokenizer.get_vocab() or tokenizer.special_tokens_map != base_tokenizer.special_tokens_map:
                raise ValueError("Adapter and base tokenizers differ; merging requires verified vocabulary and special-token compatibility.")
        base = AutoModelForCausalLM.from_pretrained(
            spec.base_model,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            **base_common,
        )
        expected_class = (getattr(adapter_config, "auto_mapping", None) or {}).get("base_model_class")
        if for_merge and expected_class and type(base).__name__ != expected_class:
            raise ValueError(f"Adapter expects {expected_class}; loaded base architecture is {type(base).__name__}.")
        model = PeftModel.from_pretrained(base, spec.adapter_path, token=_token())
        if for_merge:
            load_metadata["base_revision"] = base_common["revision"]
            if "base_fingerprint" in load_metadata and fingerprint(Path(spec.base_model)) != load_metadata["base_fingerprint"]:
                raise ValueError("Local base changed while loading; safe merging requires an immutable base.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_ref, **common)
        model = AutoModelForCausalLM.from_pretrained(
            load_ref,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            **common,
        )
    model.eval()
    return ModelRuntime(spec=spec, model=model, tokenizer=tokenizer, device=_device_for(model), load_metadata=load_metadata)


def load_runtime(model_ref: str, *, for_merge: bool = False, base_revision: str | None = None) -> ModelRuntime:
    """Load any supported reference and return a common inference runtime."""
    spec = resolve_model_ref(model_ref)
    if spec.model_category == "reward_model":
        raise ValueError("Reward models score sequences and cannot be loaded for text generation.")
    if spec.kind == "scratch":
        return _load_scratch(spec)
    return _load_transformers(spec, for_merge=for_merge, base_revision=base_revision)


def load_reward_runtime(model_ref: str) -> RewardModelRuntime:
    """Load a registered scalar reward artifact for pairwise evaluation."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from app.config import get_settings
    from app.integrations.hf_hub import _token

    spec = resolve_model_ref(model_ref)
    if spec.model_category != "reward_model":
        raise ValueError("Pairwise reward evaluation requires a registered reward-model artifact.")
    settings = get_settings()
    common = {
        "token": _token(),
        "cache_dir": str(settings.hf_cache_dir),
        "trust_remote_code": settings.trust_remote_code,
    }
    if spec.kind == "adapter":
        from peft import PeftModel

        assert spec.base_model and spec.adapter_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(spec.adapter_path, **common)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(spec.base_model, **common)
        base = AutoModelForSequenceClassification.from_pretrained(
            spec.base_model,
            num_labels=1,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            **common,
        )
        model = PeftModel.from_pretrained(base, spec.adapter_path, token=_token())
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.load_ref, **common)
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.load_ref,
            num_labels=1,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            **common,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return RewardModelRuntime(spec=spec, model=model, tokenizer=tokenizer, device=_device_for(model))
