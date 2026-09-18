import type { Method, TaskType } from "./types";

/** Flat form state for a training run, mirrored to the backend RunConfig. */
export interface RunForm {
  backend: string;
  task: TaskType;
  method: Method;
  base_model: string;
  revision: string;
  seed: number;
  gradient_checkpointing: boolean;
  dataset_id: number | null;
  output_name: string;
  // LoRA
  r: number;
  alpha: number;
  dropout: number;
  lora_target_strategy: "auto" | "all_linear" | "attention" | "mlp" | "custom";
  lora_target_modules: string;
  use_rslora: boolean;
  lora_init_method: "standard" | "pissa" | "loftq" | "eva";
  lora_plus_lr_ratio: number | null;
  freeze_last_n_layers: number;
  freeze_embeddings: boolean;
  freeze_lm_head: boolean;
  freeze_norms: boolean;
  freeze_modules: string;
  precision: "auto" | "bf16" | "fp16" | "fp32";
  attention: "auto" | "sdpa" | "flash_attention_2" | "eager";
  checkpointing_mode: "auto" | "off" | "standard" | "non_reentrant" | "backend_optimized";
  quantization_mode: "none" | "nf4" | "fp4" | "int8";
  quantization_compute_dtype: "auto" | "bf16" | "fp16" | "fp32";
  quantization_double: boolean;
  quantization_storage_dtype: "auto" | "uint8" | "bf16" | "fp16" | "fp32";
  use_liger: boolean;
  neftune_noise_alpha: number | null;
  rope_enabled: boolean;
  rope_factor: number;
  rope_type: "linear" | "dynamic" | "yarn";
  alignment_objective: "dpo" | "ipo" | "orpo" | "simpo" | "kto" | "reward_model";
  alignment_beta: number;
  alignment_dpo_loss_variant: "sigmoid" | "hinge" | "robust" | "exo_pair";
  alignment_label_smoothing: number;
  alignment_simpo_gamma: number;
  alignment_desirable_weight: number;
  alignment_undesirable_weight: number;
  reference_strategy: "base_model" | "separate_model" | "adapter_disabled" | "none";
  reference_model: string;
  reference_revision: string;
  // Optim
  learning_rate: number;
  lr_scheduler: string;
  warmup_ratio: number;
  optimizer: string;
  optimizer_strategy: "default" | "galore" | "apollo" | "badam" | "adam_mini" | "muon";
  optimizer_target_modules: string;
  optimizer_low_rank_rank: number;
  optimizer_update_interval: number;
  // Train
  epochs: number;
  max_steps: number | null;
  per_device_batch_size: number;
  gradient_accumulation: number;
  max_seq_length: number;
  save_steps: number;
  logging_steps: number;
  tokenizer_mode: "reuse";
  loss_policy: "full_sequence" | "completion_only" | "assistant_only";
  chat_template: string;
}

export function defaultForm(overrides: Partial<RunForm> = {}): RunForm {
  return {
    backend: "auto",
    task: "finetune",
    method: "qlora",
    base_model: "unsloth/Qwen2.5-0.5B-Instruct",
    revision: "",
    seed: 42,
    gradient_checkpointing: true,
    dataset_id: null,
    output_name: "my-finetune",
    r: 16,
    alpha: 16,
    dropout: 0,
    lora_target_strategy: "auto",
    lora_target_modules: "",
    use_rslora: false,
    lora_init_method: "standard",
    lora_plus_lr_ratio: null,
    freeze_last_n_layers: 1,
    freeze_embeddings: false,
    freeze_lm_head: true,
    freeze_norms: false,
    freeze_modules: "",
    precision: "auto",
    attention: "auto",
    checkpointing_mode: "auto",
    quantization_mode: "nf4",
    quantization_compute_dtype: "auto",
    quantization_double: true,
    quantization_storage_dtype: "auto",
    use_liger: false,
    neftune_noise_alpha: null,
    rope_enabled: false,
    rope_factor: 2,
    rope_type: "linear",
    alignment_objective: "dpo",
    alignment_beta: 0.1,
    alignment_dpo_loss_variant: "sigmoid",
    alignment_label_smoothing: 0,
    alignment_simpo_gamma: 0.5,
    alignment_desirable_weight: 1,
    alignment_undesirable_weight: 1,
    reference_strategy: "base_model",
    reference_model: "",
    reference_revision: "",
    learning_rate: 2e-4,
    lr_scheduler: "cosine",
    warmup_ratio: 0.03,
    optimizer: "adamw_8bit",
    optimizer_strategy: "default",
    optimizer_target_modules: "",
    optimizer_low_rank_rank: 128,
    optimizer_update_interval: 200,
    epochs: 1,
    max_steps: null,
    per_device_batch_size: 2,
    gradient_accumulation: 4,
    max_seq_length: 1024,
    save_steps: 100,
    logging_steps: 5,
    tokenizer_mode: "reuse",
    loss_policy: "full_sequence",
    chat_template: "",
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
  tokenizer_mode: "train" | "import";
  tokenizer_source: string;
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
    tokenizer_mode: "train",
    tokenizer_source: "",
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
    output_name: (f.output_name ?? "").trim() || "my-scratch-llm",
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
    tokenizer: {
      mode: f.tokenizer_mode ?? "train",
      source: f.tokenizer_mode === "import" ? (f.tokenizer_source ?? "").trim() : null,
    },
  };
}

/** Build the nested RunConfig payload the backend expects. */
export function formFromConfig(config: Record<string, unknown>): RunForm {
  const train = (config.train ?? {}) as Record<string, unknown>;
  const optim = (config.optim ?? {}) as Record<string, unknown>;
  const lora = (config.lora ?? {}) as Record<string, unknown>;
  const tokenizer = (config.tokenizer ?? {}) as Record<string, unknown>;
  const freeze = (config.freeze ?? {}) as Record<string, unknown>;
  const quantization = (config.quantization ?? {}) as Record<string, unknown>;
  const runtime = (config.runtime ?? {}) as Record<string, unknown>;
  const rope = (runtime.rope ?? {}) as Record<string, unknown>;
  const alignment = (config.alignment ?? {}) as Record<string, unknown>;
  const reference = (alignment.reference ?? {}) as Record<string, unknown>;
  const defaults = defaultForm();
  const result = { ...defaults };
  for (const key of Object.keys(defaults) as (keyof RunForm)[]) {
    const value = config[key] ?? train[key] ?? optim[key] ?? lora[key] ?? tokenizer[key];
    if (value !== undefined) Object.assign(result, { [key]: value });
  }
  result.revision = typeof config.revision === "string" ? config.revision : "";
  result.max_steps = typeof train.max_steps === "number" ? train.max_steps : null;
  result.lora_target_strategy = (lora.target_strategy as RunForm["lora_target_strategy"]) ?? "auto";
  result.lora_target_modules = Array.isArray(lora.target_modules) ? lora.target_modules.join(", ") : "";
  result.use_rslora = Boolean(lora.use_rslora);
  result.lora_init_method = (lora.init_method as RunForm["lora_init_method"]) ?? "standard";
  result.lora_plus_lr_ratio = typeof lora.lora_plus_lr_ratio === "number" ? lora.lora_plus_lr_ratio : null;
  result.freeze_last_n_layers = Number(freeze.last_n_layers ?? 1);
  result.freeze_embeddings = Boolean(freeze.train_embeddings);
  result.freeze_lm_head = freeze.train_lm_head === undefined ? true : Boolean(freeze.train_lm_head);
  result.freeze_norms = Boolean(freeze.train_norms);
  result.freeze_modules = Array.isArray(freeze.selected_modules) ? freeze.selected_modules.join(", ") : "";
  result.precision = (runtime.precision as RunForm["precision"]) ?? "auto";
  result.attention = (runtime.attention as RunForm["attention"]) ?? "auto";
  result.checkpointing_mode = (runtime.gradient_checkpointing as RunForm["checkpointing_mode"]) ?? "auto";
  result.quantization_mode = (quantization.mode as RunForm["quantization_mode"]) ?? (result.method === "qlora" ? "nf4" : "none");
  result.quantization_compute_dtype = (quantization.compute_dtype as RunForm["quantization_compute_dtype"]) ?? "auto";
  result.quantization_double = quantization.double_quant === undefined ? true : Boolean(quantization.double_quant);
  result.quantization_storage_dtype = (quantization.storage_dtype as RunForm["quantization_storage_dtype"]) ?? "auto";
  result.use_liger = Boolean(runtime.use_liger);
  result.neftune_noise_alpha = typeof runtime.neftune_noise_alpha === "number" ? runtime.neftune_noise_alpha : null;
  result.rope_enabled = Boolean(rope.enabled);
  result.rope_factor = Number(rope.factor ?? 2);
  result.rope_type = (rope.type as RunForm["rope_type"]) ?? "linear";
  result.alignment_objective = (alignment.objective as RunForm["alignment_objective"]) ?? "dpo";
  result.alignment_beta = Number(alignment.beta ?? 0.1);
  result.alignment_dpo_loss_variant = (alignment.dpo_loss_variant as RunForm["alignment_dpo_loss_variant"]) ?? "sigmoid";
  result.alignment_label_smoothing = Number(alignment.label_smoothing ?? 0);
  result.alignment_simpo_gamma = Number(alignment.simpo_gamma ?? 0.5);
  result.alignment_desirable_weight = Number(alignment.desirable_weight ?? 1);
  result.alignment_undesirable_weight = Number(alignment.undesirable_weight ?? 1);
  result.reference_strategy = (reference.strategy as RunForm["reference_strategy"]) ?? "base_model";
  result.reference_model = typeof reference.model === "string" ? reference.model : "";
  result.reference_revision = typeof reference.revision === "string" ? reference.revision : "";
  result.optimizer_strategy = (optim.strategy as RunForm["optimizer_strategy"]) ?? "default";
  result.optimizer_target_modules = Array.isArray(optim.target_modules) ? optim.target_modules.join(", ") : "";
  result.optimizer_low_rank_rank = Number(optim.low_rank_rank ?? 128);
  result.optimizer_update_interval = Number(optim.update_interval ?? 200);
  return result;
}

export function toPayload(f: RunForm): Record<string, unknown> {
  const referenceFree = ["orpo", "simpo", "reward_model"].includes(f.alignment_objective ?? "dpo");
  const referenceStrategy = referenceFree ? "none" : (f.reference_strategy ?? "base_model");
  return {
    backend: f.backend,
    task: f.task,
    method: f.method,
    base_model: (f.base_model ?? "").trim(),
    revision: f.revision?.trim() || null,
    gradient_checkpointing: f.gradient_checkpointing ?? true,
    dataset_id: f.dataset_id,
    output_name: (f.output_name ?? "").trim() || "my-finetune",
    load_in_4bit: f.method === "qlora",
    lora: {
      r: f.r,
      alpha: f.alpha,
      dropout: f.dropout,
      target_strategy: f.lora_target_strategy ?? "auto",
      target_modules: (f.lora_target_modules ?? "").split(",").map(v => v.trim()).filter(Boolean),
      use_dora: f.method === "dora",
      use_rslora: f.use_rslora ?? false,
      init_method: f.lora_init_method ?? "standard",
      lora_plus_lr_ratio: f.lora_plus_lr_ratio ?? null,
    },
    freeze: {
      last_n_layers: f.freeze_last_n_layers ?? 1,
      train_embeddings: f.freeze_embeddings ?? false,
      train_lm_head: f.freeze_lm_head ?? true,
      train_norms: f.freeze_norms ?? false,
      selected_modules: (f.freeze_modules ?? "").split(",").map(v => v.trim()).filter(Boolean),
    },
    quantization: {
      mode: ["full", "freeze"].includes(f.method)
        ? "none"
        : f.method === "qlora" && (!f.quantization_mode || f.quantization_mode === "none") ? "nf4" : (f.quantization_mode ?? "none"),
      compute_dtype: f.quantization_mode === "int8" ? "auto" : (f.quantization_compute_dtype ?? "auto"),
      double_quant: f.quantization_mode === "int8" ? false : (f.quantization_double ?? true),
      storage_dtype: f.quantization_mode === "int8" ? "auto" : (f.quantization_storage_dtype ?? "auto"),
    },
    runtime: {
      precision: f.precision ?? "auto",
      attention: f.attention ?? "auto",
      gradient_checkpointing: f.checkpointing_mode ?? "auto",
      use_liger: f.use_liger ?? false,
      neftune_noise_alpha: f.neftune_noise_alpha ?? null,
      rope: { enabled: f.rope_enabled ?? false, factor: f.rope_factor ?? 2, type: f.rope_type ?? "linear" },
    },
    alignment: {
      objective: f.alignment_objective ?? "dpo",
      beta: f.alignment_beta ?? 0.1,
      dpo_loss_variant: f.alignment_objective === "dpo" ? (f.alignment_dpo_loss_variant ?? "sigmoid") : "sigmoid",
      label_smoothing: f.alignment_objective === "dpo" && f.alignment_dpo_loss_variant !== "hinge" ? (f.alignment_label_smoothing ?? 0) : 0,
      simpo_gamma: f.alignment_objective === "simpo" ? (f.alignment_simpo_gamma ?? 0.5) : 0.5,
      desirable_weight: f.alignment_objective === "kto" ? (f.alignment_desirable_weight ?? 1) : 1,
      undesirable_weight: f.alignment_objective === "kto" ? (f.alignment_undesirable_weight ?? 1) : 1,
      reference: {
        strategy: referenceStrategy,
        model: referenceStrategy === "separate_model" ? (f.reference_model ?? "").trim() || null : null,
        revision: referenceStrategy === "separate_model" ? (f.reference_revision ?? "").trim() || null : null,
      },
    },
    optim: {
      learning_rate: f.learning_rate,
      lr_scheduler: f.lr_scheduler,
      warmup_ratio: f.warmup_ratio,
      optimizer: f.optimizer,
      strategy: f.optimizer_strategy ?? "default",
      target_modules: (f.optimizer_target_modules ?? "").split(",").map(v => v.trim()).filter(Boolean),
      low_rank_rank: f.optimizer_low_rank_rank ?? 128,
      update_interval: f.optimizer_update_interval ?? 200,
    },
    train: {
      seed: f.seed ?? 42,
      epochs: f.epochs,
      max_steps: f.max_steps,
      per_device_batch_size: f.per_device_batch_size,
      gradient_accumulation: f.gradient_accumulation,
      max_seq_length: f.max_seq_length,
      save_steps: f.save_steps,
      logging_steps: f.logging_steps,
    },
    tokenizer: {
      mode: f.tokenizer_mode ?? "reuse",
      loss_policy: f.loss_policy ?? "full_sequence",
      chat_template: (f.chat_template ?? "").trim() || null,
    },
  };
}
