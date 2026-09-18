"""Offline trainable adapter merge/save/reload acceptance on tiny CPU weights."""
import json

import pytest


@pytest.mark.export_runtime
def test_tiny_adapter_merge_is_equivalent_and_rejects_unsafe_bases(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        AutoModelForCausalLM,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )

    from app.model_refs import ResolvedModel
    from app.train_entry import merge, model_runtime

    base_path, adapter_path = tmp_path / "base", tmp_path / "adapter"
    torch.manual_seed(8)
    base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=16, vocab_size=8))
    base.save_pretrained(base_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7}, unk_token="[UNK]")), unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.save_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(base_path)
    adapter = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", target_modules=["c_attn"], r=2, lora_alpha=4))
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.03)
    adapter.eval()
    batch = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        expected = adapter(batch).logits
    adapter.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    spec = ResolvedModel(requested_ref=str(adapter_path), load_ref=str(adapter_path), kind="adapter",
                         label="tiny", base_model=str(base_path), adapter_path=str(adapter_path), local_path=str(adapter_path))
    monkeypatch.setattr(model_runtime, "resolve_model_ref", lambda _: spec)
    output = tmp_path / "merged"
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"model_ref": str(adapter_path), "output_dir": str(output)}))
    assert merge.main(str(request)) == 0
    merged = AutoModelForCausalLM.from_pretrained(output).eval()
    with torch.no_grad():
        assert torch.allclose(merged(batch).logits, expected, atol=1e-5)
    assert json.loads((output / "merge-lineage.json").read_text())["safe_merge"]
    tokenizer.add_tokens(["new"])
    tokenizer.save_pretrained(adapter_path)
    with pytest.raises(ValueError, match="tokenizers differ"):
        model_runtime.load_runtime(str(adapter_path), for_merge=True)
    config_path = base_path / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {"quant_method": "bitsandbytes", "load_in_4bit": True}
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="quantized base"):
        model_runtime.load_runtime(str(adapter_path), for_merge=True)
