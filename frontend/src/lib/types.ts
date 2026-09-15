// Mirrors the backend Pydantic models (app/domain.py, app/db/models.py).

export type TaskType = "pretrain" | "continued_pretrain" | "finetune";
export type Method = "lora" | "qlora" | "dora" | "full" | "prompt_tuning";
export type RunStatus = "queued" | "running" | "done" | "failed" | "cancelled";
export type DatasetKind =
  | "pretrain_corpus"
  | "domain_corpus"
  | "instruction"
  | "eval";
export type FitLevel = "fits" | "tight" | "wont_fit";

export interface HardwareProfile {
  gpu_name?: string | null;
  vram_total_mb?: number | null;
  vram_free_mb?: number | null;
  gpu_util_pct?: number | null;
  ram_total_mb?: number | null;
  ram_free_mb?: number | null;
  cpu_count?: number | null;
  cpu_util_pct?: number | null;
  disk_free_mb?: number | null;
  disk_total_mb?: number | null;
  gpus: GPUDeviceProfile[];
  gpu_count: number;
  cuda_available: boolean;
  cuda_runtime_version?: string | null;
  nvidia_driver_version?: string | null;
  mps_available: boolean;
  platform?: string | null;
  source: string;
}

export interface GPUDeviceProfile {
  id: number;
  uuid?: string | null;
  name: string;
  vram_total_mb: number;
  vram_free_mb: number;
  utilization_pct?: number | null;
  compute_capability?: string | null;
  temperature_c?: number | null;
  power_watts?: number | null;
  bf16_supported: boolean;
  fp16_supported: boolean;
  fp8_supported: boolean;
  flash_attention_feasible: boolean;
}

export type SupportState = "supported" | "experimental" | "unsupported" | "not_installed" | "missing_dependency" | "incompatible" | "requires_conversion";
export type EvidenceLevel = "declared" | "installed" | "metadata" | "runtime";
export interface Capability { state: SupportState; reason: string; requirements: string[]; evidence?: EvidenceLevel }
export interface PeftMethodCapability { method: string; support: Capability; adapter_based: boolean; requires_quantized_base: boolean }
export interface QuantizationCapability { format: string; operations: string[]; support: Capability }
export interface TrainingBackendCapabilities {
  schema_version: 1;
  name: string;
  display_name: string;
  description: string;
  availability: Capability;
  tasks: string[];
  methods: string[];
  quantization: string[];
  requirements: string[];
  task_capabilities: Record<string, Capability>;
  stages: Record<string, Capability>;
  method_capabilities: Record<string, Capability>;
  tokenizer: {
    modes: Record<string, Capability>;
    loss_policies: Record<string, Capability>;
    templates: { native: Capability; explicit_override: Capability; fallback: Capability };
  };
  peft: Record<string, PeftMethodCapability>;
  quantization_capabilities: Record<string, QuantizationCapability>;
  platforms: string[];
  architectures: string[];
  required_dependencies: string[];
  optional_dependencies: string[];
}
export interface ModelCapabilities {
  family: string;
  architecture_kind: string;
  model_type?: string | null;
  architectures: string[];
  is_moe: boolean;
  is_multimodal: boolean;
  trust_remote_code: boolean;
  chat_template: boolean;
  training: Record<string, Capability>;
  backends: Record<string, Capability>;
  inference: Record<string, Capability>;
  quantization: Record<string, Capability>;
  export: Record<string, Capability>;
  precision: Record<string, Capability>;
  distributed: Record<string, Capability>;
  template_capabilities?: {
    native: Capability;
    explicit_override: Capability;
    fallback: Capability;
  };
  peft_methods?: Record<string, PeftMethodCapability>;
  quantization_matrix?: Record<string, QuantizationCapability>;
  dependencies?: Record<string, { installed: boolean; version?: string | null }>;
  suggested_target_modules: string[];
  warnings: string[];
}

export interface PreflightResult {
  ok: boolean;
  model_ref: string;
  revision?: string | null;
  kind?: string;
  backend_verified?: string;
  recommended_backend?: string;
  architecture?: string;
  context_length?: number | null;
  vocab_size?: number | null;
  parameter_count?: number | null;
  has_chat_template?: boolean;
  vram_peak_mb?: number;
  warnings?: string[];
  error?: string | null;
}

export interface MemoryEstimate {
  weights_mb: number;
  optimizer_mb: number;
  gradients_mb?: number;
  adapters_mb?: number;
  safe_budget_mb?: number;
  available_mb?: number | null;
  headroom_mb?: number;
  verdict?: string;
  suggestions?: string[];
  activations_mb: number;
  kv_cache_mb: number;
  overhead_mb: number;
  total_mb: number;
  budget_mb: number;
  fit: FitLevel;
  source: string;
  notes: string[];
}

export interface ValidationIssue {
  level: "error" | "warning" | "info";
  message: string;
  line?: number | null;
}
export interface ValidationReport {
  ok: boolean;
  issues: ValidationIssue[];
}

export interface Dataset {
  id: number;
  name: string;
  kind: DatasetKind;
  path: string;
  fmt: string;
  num_rows?: number | null;
  num_tokens_est?: number | null;
  size_bytes?: number | null;
  is_sample: boolean;
  validation?: ValidationReport | null;
  created_at: string;
}

export interface DatasetVersion {
  id: number;
  dataset_id: number;
  parent_version_id?: number | null;
  fingerprint: string;
  path: string;
  fmt: string;
  schema: Record<string, unknown>;
  split: "source" | "train" | "validation" | "test" | string;
  num_rows?: number | null;
  num_tokens_est?: number | null;
  size_bytes?: number | null;
  created_at: string;
}

export interface DatasetRecipe {
  id: number;
  source_version_id: number;
  name: string;
  config: Record<string, unknown>;
  fingerprint: string;
  created_at: string;
}

export interface DatasetProfile {
  id: number;
  dataset_version_id: number;
  tokenizer_artifact_id?: number | null;
  kind: "tokenizer" | "quality" | string;
  cache_key: string;
  config: Record<string, unknown>;
  stats: Record<string, unknown>;
  created_at: string;
}

export interface TokenizerArtifact {
  id: number;
  model_ref: string;
  revision?: string | null;
  resolved_revision?: string | null;
  fingerprint: string;
  config: Record<string, unknown>;
  status: string;
  created_at: string;
}

export interface TokenizerProfileResult {
  tokenizer: {
    model_ref: string;
    resolved_revision?: string | null;
    fingerprint: string;
    class: string;
    vocab_size: number;
    model_max_length?: number | null;
    chat_template: boolean;
  };
  stats: {
    sampled_rows: number;
    total_tokens: number;
    min: number;
    max: number;
    mean: number;
    median: number;
    p90: number;
    p95: number;
    p99: number;
    truncation_percentage: number;
    tokens_per_example: number;
    tokens_per_character: number;
    tokens_per_word: number;
    unknown_token_rate: number;
    padding_overhead_percentage: number;
    packing_efficiency_percentage: number;
    by_dominant_script: Record<string, Record<string, number>>;
  };
  previews: { line: number; rendered: string; tokens: string[]; input_ids: number[]; loss_mask: number[]; original_length: number; truncated: boolean }[];
  errors: { line: number; message: string }[];
}

export interface QualityProfileResult {
  sampled_rows: number;
  valid_rows: number;
  max_rows: number;
  truncated_scan: boolean;
  script_distribution: Record<string, number>;
  warnings: { code: string; count: number; examples: { line: number; text: string; [key: string]: unknown }[] }[];
  destructive_changes: false;
}

export interface Run {
  id: number;
  name: string;
  task: TaskType;
  method: Method;
  backend: string;
  base_model: string;
  dataset_id?: number | null;
  status: RunStatus;
  config: Record<string, unknown>;
  estimate?: MemoryEstimate | null;
  metrics?: Record<string, number> | null;
  hardware?: HardwareProfile | null;
  output_dir?: string | null;
  hf_repo?: string | null;
  error?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface Checkpoint {
  id: number;
  run_id: number;
  step: number;
  path: string;
  is_final: boolean;
  created_at: string;
}

export interface SystemStatus {
  llmfit_available: boolean;
  hf_token_set: boolean;
  judge_configured: boolean;
  current_run?: number | null;
  queued?: number[];
  resource?: { kind: string; id?: string | number; since?: number };
  home: string;
  vram_budget_mb: number;
}

export interface ModelArtifact {
  id: number;
  name: string;
  kind: string;
  run_id?: number | null;
  base_model?: string | null;
  local_path?: string | null;
  hf_repo?: string | null;
  published: boolean;
  status: string; // ready | quantizing | failed
  error?: string | null;
  meta?: Record<string, unknown> | null;
  created_at: string;
}

export interface UnpublishedRun {
  run_id: number;
  name: string;
  output_dir?: string | null;
  hf_repo?: string | null;
  method: string;
  base_model: string;
  model_ref?: string;
}

export interface ModelOption {
  ref: string;
  requested_ref: string;
  load_ref: string;
  kind: "transformers" | "adapter" | "scratch";
  label: string;
  base_model?: string | null;
  run_id?: number | null;
  local_path?: string | null;
  deployable: boolean;
  source: "run" | "artifact";
  task?: string;
  method?: string;
}

export interface ModelInspection {
  model_ref: string;
  reachable: boolean;
  revision?: string | null;
  kind?: string;
  architecture?: string[];
  model_type?: string | null;
  parameters?: number | null;
  context_length?: number | null;
  vocab_size?: number | null;
  tokenizer_class?: string | null;
  supports_causal_lm?: boolean;
  supports_lora?: boolean;
  supports_full_training?: boolean;
  supports_4bit?: boolean;
  suggested_target_modules?: string[];
  warnings?: string[];
  capabilities?: ModelCapabilities;
  resolved_commit?: string | null;
  config_fingerprint?: string | null;
  tokenizer_fingerprint?: string | null;
  dependencies?: Record<string, { installed: boolean; version?: string | null }>;
}

export interface PreflightResult {
  ok: boolean;
  model_ref: string;
  revision?: string | null;
  kind?: string;
  backend_verified?: string;
  recommended_backend?: string;
  architecture?: string;
  context_length?: number | null;
  vocab_size?: number | null;
  parameter_count?: number | null;
  has_chat_template?: boolean;
  vram_peak_mb?: number;
  warnings?: string[];
  error?: string | null;
  resolved?: Record<string, unknown>;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  state: Record<string, unknown>;
  run_count?: number;
  runs?: Run[];
  created_at: string;
  updated_at: string;
}

export interface ModelLineage {
  run_id: number;
  name: string;
  model_ref: string;
  parent_ref?: string | null;
  task: string;
  method: string;
  status: string;
  dataset_id?: number | null;
  created_at: string;
}

export interface DeploymentStatus {
  available: boolean;
  active: boolean;
  state?: string;
  model_ref?: string | null;
  kind?: string | null;
  endpoint?: string | null;
  health_url?: string | null;
  started_at?: number | null;
  pid?: number | null;
  logs?: string[];
}

export interface HFModel {
  repo_id: string;
  private?: boolean | null;
  downloads?: number | null;
  likes?: number | null;
  updated?: string;
}

export interface AdvisorRec {
  repo: string;
  params_b: number;
  method: Method;
  context_length: number;
  estimate: MemoryEstimate;
  quality_tier: number;
  fit: FitLevel;
  score: number;
  reasoning: string;
}

export interface EvalPerModel {
  scores: Record<string, number>;
  seconds: number;
}
export interface EvalSample {
  prompt: string;
  reference: string;
  preds: Record<string, string>;
}
export interface EvalResult {
  id: number;
  model_ref: string;
  dataset_id?: number | null;
  run_id?: number | null;
  scores: Record<string, Record<string, number>>; // model -> metric -> value
  detail?: {
    status: string;
    models?: string[];
    per_model?: Record<string, EvalPerModel>;
    samples?: EvalSample[];
    error?: string;
  } | null;
  created_at: string;
}

// Persisted log line (backend: app/core/log_capture.py)
export interface LogLine {
  ts?: number | null;
  level: string;
  message: string;
}

// WebSocket training events
export type TrainingEvent =
  | { type: "log"; ts: number; level: string; message: string }
  | { type: "metric"; ts: number; step: number; total_steps?: number; metrics: Record<string, number> }
  | { type: "checkpoint"; ts: number; step: number; path: string; is_final: boolean }
  | { type: "sample"; ts: number; step: number; prompt: string; text: string }
  | { type: "status"; ts: number; status: string; detail?: string }
  | { type: "progress"; ts: number; current: number; total?: number; unit: string; message?: string }
  | { type: "resource"; ts: number; resources: Record<string, number> }
  | { type: "artifact"; ts: number; kind: string; path: string; metadata: Record<string, unknown> }
  | { type: "warning"; ts: number; code: string; message: string; action?: string }
  | { type: "profile"; ts: number; name: string; values: Record<string, unknown> }
  | { type: "error"; ts: number; code: string; message: string; detail?: string; retryable: boolean };
