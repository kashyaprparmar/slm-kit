import { describe, expect, it } from "vitest";
import { defaultForm, formFromConfig, toPayload, toPretrainPayload, type RunForm } from "./runconfig";
import { readWorkflow, writeWorkflow } from "./workflow";

describe("runconfig and workflow defensive hydration", () => {
  it("handles legacy drafts missing chat_template and tokenizer settings in toPayload", () => {
    // Simulates an older draft stored in localStorage before Phase 2 fields were added
    const legacyForm = {
      backend: "unsloth",
      task: "finetune",
      method: "qlora",
      base_model: "Qwen/Qwen2.5-0.5B-Instruct",
      revision: "",
      seed: 42,
      gradient_checkpointing: true,
      dataset_id: 1,
      output_name: "test-run",
      r: 16,
      alpha: 16,
      dropout: 0,
      learning_rate: 2e-4,
      lr_scheduler: "cosine",
      warmup_ratio: 0.03,
      optimizer: "adamw_8bit",
      epochs: 1,
      max_steps: null,
      per_device_batch_size: 2,
      gradient_accumulation: 4,
      max_seq_length: 1024,
      save_steps: 100,
      logging_steps: 5,
    } as unknown as RunForm;

    expect(() => toPayload(legacyForm)).not.toThrow();
    const payload = toPayload(legacyForm);
    expect(payload.base_model).toBe("Qwen/Qwen2.5-0.5B-Instruct");
    expect((payload.tokenizer as any).chat_template).toBeNull();
    expect((payload.tokenizer as any).mode).toBe("reuse");
    expect((payload.tokenizer as any).loss_policy).toBe("full_sequence");
  });

  it("handles empty or undefined string fields gracefully", () => {
    const emptyForm = {
      ...defaultForm(),
      base_model: undefined as any,
      output_name: undefined as any,
      chat_template: undefined as any,
    };
    expect(() => toPayload(emptyForm)).not.toThrow();
    const payload = toPayload(emptyForm);
    expect(payload.base_model).toBe("");
    expect(payload.output_name).toBe("my-finetune");
    expect((payload.tokenizer as any).chat_template).toBeNull();
  });

  it("readWorkflow merges existing draft with default fallback keys", () => {
    writeWorkflow("test.finetune.legacy", {
      base_model: "unsloth/Llama-3.2-1B-Instruct",
      epochs: 3,
    });
    const result = readWorkflow("test.finetune.legacy", defaultForm());
    expect(result.base_model).toBe("unsloth/Llama-3.2-1B-Instruct");
    expect(result.epochs).toBe(3);
    // Newly added fields should be populated from defaultForm
    expect(result.chat_template).toBe("");
    expect(result.tokenizer_mode).toBe("reuse");
    expect(result.loss_policy).toBe("full_sequence");
  });

  it("toPretrainPayload handles undefined output_name gracefully", () => {
    const payload = toPretrainPayload({
      output_name: undefined as any,
      dataset_id: null,
      vocab_size: 8192,
      n_layers: 6,
      n_heads: 6,
      n_embd: 384,
      block_size: 256,
      dropout: 0.1,
      learning_rate: 3e-4,
      per_device_batch_size: 8,
      max_steps: 1000,
      save_steps: 200,
      logging_steps: 10,
      tokenizer_mode: "train",
      tokenizer_source: "",
    });
    expect(payload.output_name).toBe("my-scratch-llm");
    expect((payload.tokenizer as any).mode).toBe("train");
  });

  it("preserves an imported scratch tokenizer source", () => {
    const payload = toPretrainPayload({
      output_name: "scratch-import",
      dataset_id: 1,
      vocab_size: 8192,
      n_layers: 6,
      n_heads: 6,
      n_embd: 384,
      block_size: 256,
      dropout: 0.1,
      learning_rate: 3e-4,
      per_device_batch_size: 8,
      max_steps: 1000,
      save_steps: 200,
      logging_steps: 10,
      tokenizer_mode: "import",
      tokenizer_source: " C:\\models\\tokenizer ",
    });
    expect(payload.tokenizer).toEqual({ mode: "import", source: "C:\\models\\tokenizer" });
  });

  it("round-trips Phase B method and runtime policy through RunConfig", () => {
    const form = defaultForm({
      backend: "transformers",
      method: "freeze",
      freeze_last_n_layers: 3,
      freeze_embeddings: true,
      precision: "bf16",
      attention: "sdpa",
      checkpointing_mode: "non_reentrant",
      rope_enabled: true,
      rope_factor: 2,
    });
    const payload = toPayload(form);
    expect((payload.freeze as any).last_n_layers).toBe(3);
    expect((payload.quantization as any).mode).toBe("none");
    expect((payload.runtime as any).attention).toBe("sdpa");
    const hydrated = formFromConfig(payload);
    expect(hydrated.method).toBe("freeze");
    expect(hydrated.freeze_last_n_layers).toBe(3);
    expect(hydrated.precision).toBe("bf16");
    expect(hydrated.rope_enabled).toBe(true);
  });

  it("normalizes reference-free alignment objectives in the shared payload", () => {
    const payload = toPayload(defaultForm({
      task: "alignment",
      alignment_objective: "simpo",
      reference_strategy: "base_model",
    }));
    expect((payload.alignment as any).reference.strategy).toBe("none");
    expect((payload.alignment as any).objective).toBe("simpo");
  });

  it("round-trips alignment objective and reference lineage", () => {
    const payload = toPayload(defaultForm({
      task: "alignment",
      alignment_objective: "dpo",
      reference_strategy: "separate_model",
      reference_model: "org/reference",
      reference_revision: "abc123",
    }));
    const hydrated = formFromConfig(payload);
    expect(hydrated.alignment_objective).toBe("dpo");
    expect(hydrated.reference_strategy).toBe("separate_model");
    expect(hydrated.reference_model).toBe("org/reference");
    expect(hydrated.reference_revision).toBe("abc123");
  });

  it("keeps DPO loss settings objective-specific", () => {
    const payload = toPayload(defaultForm({
      task: "alignment",
      alignment_objective: "dpo",
      alignment_dpo_loss_variant: "robust",
      alignment_label_smoothing: 0.1,
    }));
    expect((payload.alignment as any).dpo_loss_variant).toBe("robust");
    expect((payload.alignment as any).label_smoothing).toBe(0.1);
  });

  it("normalizes reward models to reference-free alignment", () => {
    const payload = toPayload(defaultForm({
      task: "alignment",
      alignment_objective: "reward_model",
      reference_strategy: "separate_model",
      reference_model: "org/reference",
    }));
    expect((payload.alignment as any).reference).toEqual({
      strategy: "none",
      model: null,
      revision: null,
    });
  });

  it("drops inactive DPO settings when changing objectives", () => {
    const payload = toPayload(defaultForm({
      task: "alignment", alignment_objective: "ipo",
      alignment_dpo_loss_variant: "robust", alignment_label_smoothing: 0.1,
    }));
    expect((payload.alignment as any).dpo_loss_variant).toBe("sigmoid");
    expect((payload.alignment as any).label_smoothing).toBe(0);
  });
});
