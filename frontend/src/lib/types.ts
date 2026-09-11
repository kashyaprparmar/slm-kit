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
  source: string;
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
  | { type: "status"; ts: number; status: string; detail?: string };
