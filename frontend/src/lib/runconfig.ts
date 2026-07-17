import type { Method, TaskType } from "./types";

/** Flat form state for a training run, mirrored to the backend RunConfig. */
export interface RunForm {
  backend: string;
  task: TaskType;
  method: Method;
  base_model: string;
  dataset_id: number | null;
  output_name: string;
  // LoRA
  r: number;
  alpha: number;
  dropout: number;
  // Optim
  learning_rate: number;
  lr_scheduler: string;
  warmup_ratio: number;
  optimizer: string;
  // Train
  epochs: number;
  max_steps: number | null;
  per_device_batch_size: number;
  gradient_accumulation: number;
  max_seq_length: number;
  save_steps: number;
  logging_steps: number;
}

export function defaultForm(overrides: Partial<RunForm> = {}): RunForm {
  return {
    backend: "unsloth",
    task: "finetune",
    method: "qlora",
    base_model: "unsloth/Qwen2.5-0.5B-Instruct",
    dataset_id: null,
    output_name: "my-finetune",
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
    ...overrides,
  };
}

// --------------------------------------------------------------------------- //
// From-scratch pretraining
// --------------------------------------------------------------------------- //
export interface PretrainForm {
  output_name: string;
  dataset_id: number | null;
  vocab_size: number;
  n_layers: number;
  n_heads: number;
  n_embd: number;
  block_size: number;
  dropout: number;
  learning_rate: number;
  per_device_batch_size: number;
  max_steps: number;
  save_steps: number;
  logging_steps: number;
}

export function defaultPretrainForm(overrides: Partial<PretrainForm> = {}): PretrainForm {
  return {
    output_name: "my-scratch-llm",
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
    ...overrides,
  };
}

export function toPretrainPayload(f: PretrainForm): Record<string, unknown> {
  return {
    backend: "scratch",
    task: "pretrain",
    method: "full",
    base_model: "",
    dataset_id: f.dataset_id,
    output_name: f.output_name.trim() || "my-scratch-llm",
    arch: {
      vocab_size: f.vocab_size,
      n_layers: f.n_layers,
      n_heads: f.n_heads,
      n_embd: f.n_embd,
      block_size: f.block_size,
      dropout: f.dropout,
    },
    optim: { learning_rate: f.learning_rate },
    train: {
      max_steps: f.max_steps,
      per_device_batch_size: f.per_device_batch_size,
      max_seq_length: f.block_size,
      save_steps: f.save_steps,
      logging_steps: f.logging_steps,
    },
  };
}

/** Build the nested RunConfig payload the backend expects. */
export function toPayload(f: RunForm): Record<string, unknown> {
  return {
    backend: f.backend,
    task: f.task,
    method: f.method,
    base_model: f.base_model.trim(),
    dataset_id: f.dataset_id,
    output_name: f.output_name.trim() || "my-finetune",
    load_in_4bit: f.method === "qlora",
    lora: {
      r: f.r,
      alpha: f.alpha,
      dropout: f.dropout,
      use_dora: f.method === "dora",
    },
    optim: {
      learning_rate: f.learning_rate,
      lr_scheduler: f.lr_scheduler,
      warmup_ratio: f.warmup_ratio,
      optimizer: f.optimizer,
    },
    train: {
      epochs: f.epochs,
      max_steps: f.max_steps,
      per_device_batch_size: f.per_device_batch_size,
      gradient_accumulation: f.gradient_accumulation,
      max_seq_length: f.max_seq_length,
      save_steps: f.save_steps,
      logging_steps: f.logging_steps,
    },
  };
}
