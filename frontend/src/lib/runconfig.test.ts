import { describe, expect, it } from "vitest";
import { defaultForm, toPayload, toPretrainPayload, type RunForm } from "./runconfig";
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
    });
    expect(payload.output_name).toBe("my-scratch-llm");
  });
});
