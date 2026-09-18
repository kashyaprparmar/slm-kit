from __future__ import annotations

import pytest


@pytest.mark.optimization_runtime
def test_pissa_and_loraplus_execute_on_a_tiny_cpu_peft_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    from app.domain import Method, RunConfig, TaskType
    from app.train_entry.optimization_runtime import lora_config_kwargs, trainer_optimizers

    cfg = RunConfig(
        backend="transformers", task=TaskType.FINETUNE, method=Method.LORA,
        base_model="tiny", dataset_id=1, output_name="tiny-pissa",
        quantization={"mode": "none"},
        lora={"r": 2, "alpha": 4, "init_method": "pissa", "lora_plus_lr_ratio": 2},
        optim={"optimizer": "adamw_torch", "learning_rate": 1e-3},
    )
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=16, vocab_size=32))
    model = get_peft_model(model, LoraConfig(target_modules=["c_attn"], r=2, lora_alpha=4, **lora_config_kwargs(cfg)))
    optimizer, _ = trainer_optimizers(model, cfg)
    batch = torch.randint(0, 32, (2, 8))
    output = model(input_ids=batch, labels=batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()
    assert optimizer.param_groups
