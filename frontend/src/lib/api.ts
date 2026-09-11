import type {
  AdvisorRec,
  Checkpoint,
  Dataset,
  EvalResult,
  HardwareProfile,
  HFModel,
  LogLine,
  MemoryEstimate,
  ModelOption,
  ModelInspection,
  ModelLineage,
  DeploymentStatus,
  ModelArtifact,
  Run,
  SystemStatus,
  UnpublishedRun,
  ValidationReport,
} from "./types";

import { createLogger } from "./logger";

const log = createLogger("api");

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET";
  const t0 = performance.now();
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }), ...(init?.headers ?? {}) },
    });
  } catch (e) {
    log.error(`${method} ${path} — network error`, String(e));
    throw new ApiError(0, { message: "SLM Kit backend is offline. Start the backend and try again." });
  }
  const ms = Math.round(performance.now() - t0);
  if (!res.ok) {
    const raw = await res.text();
    let detail: unknown = raw;
    try { detail = JSON.parse(raw); } catch { /* Non-JSON gateway response. */ }
    log.error(`${method} ${path} → ${res.status} (${ms}ms)`, detail);
    throw new ApiError(res.status, detail, res.headers.get("X-Request-ID"));
  }
  // Only log mutations at info; GETs at debug to avoid noise.
  (method === "GET" ? log.debug : log.info)(`${method} ${path} → ${res.status} (${ms}ms)`);
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  suggestions: string[];
  constructor(public status: number, public detail: unknown, public requestId: string | null = null) {
    const d = detail as { message?: string; detail?: unknown; suggestions?: string[] } | null;
    super(typeof detail === "string" ? detail : d?.message || (typeof d?.detail === "string" ? d.detail : `Request failed (${status})`));
    this.suggestions = d?.suggestions ?? [];
  }
}

export const api = {
  prepareDataset: (id: number, body: Record<string, unknown>) => req<{ datasets: Dataset[]; dropped_rows: number }>(`/api/datasets/${id}/prepare`, { method: "POST", body: JSON.stringify(body) }),
  trainingSettings: (config: Record<string, unknown>, preset: string) => req<{ config: Record<string, unknown>; reasons: Record<string, string>; warnings: string[] }>("/api/advisor/settings", { method: "POST", body: JSON.stringify({ config, preset }) }),
  diagnostics: () => req<{ checks: { name: string; status: string; detail: string; guidance: string }[] }>("/api/system/diagnostics"),
  activity: () => req<{ events: { id: string; ts: number; message: string; level: string; duration_ms?: number; correlation_id?: string }[] }>("/api/system/activity"),
  servingProviders: () => req<{ transformers: DeploymentStatus; ollama: { installed: boolean; running: boolean; managed_model?: string; endpoint: string; guidance: string; models: { name: string }[]; loaded: { name: string }[] } }>("/api/serving/providers"),
  startOllama: (model: string) => req<unknown>("/api/serving/ollama/start", { method: "POST", body: JSON.stringify({ model }) }),
  stopOllama: () => req<{ stopped: boolean }>("/api/serving/ollama/stop", { method: "POST" }),
  importOllama: (artifact_id: number, model: string) => req<{ model: string; status: string }>("/api/serving/ollama/import", { method: "POST", body: JSON.stringify({ artifact_id, model }) }),
  testServing: (body: { provider: string; prompt: string; max_tokens: number }) => req<{ output: string }>("/api/serving/test", { method: "POST", body: JSON.stringify(body) }),
  cloneRun: (id: number) => req<{ config: Record<string, unknown> }>(`/api/runs/${id}/clone`, { method: "POST" }),
  // system
  health: () => req<{ status: string; version: string }>("/api/health"),
  hardware: () => req<HardwareProfile>("/api/system/hardware"),
  status: () => req<SystemStatus>("/api/system/status"),

  // datasets
  listDatasets: () => req<Dataset[]>("/api/datasets"),
  getDataset: (id: number) => req<Dataset>(`/api/datasets/${id}`),
  previewDataset: (id: number) =>
    req<{
      validation: ValidationReport;
      stats: {
        fmt: string;
        num_rows: number;
        num_tokens_est: number;
        size_bytes: number;
        sample_rows: unknown[];
        token_histogram: number[];
        columns: string[];
        duplicate_rows: number;
        invalid_rows: number;
        empty_rows: number;
        long_rows: number;
        fingerprint: string;
      };
    }>(`/api/datasets/${id}/preview`),
  installSamples: () => req<{ installed: Dataset[] }>("/api/datasets/install-samples", { method: "POST" }),
  deleteDataset: (id: number) => req<{ deleted: number }>(`/api/datasets/${id}`, { method: "DELETE" }),
  hfDatasetCapabilities: () => req<{ available: boolean }>("/api/datasets/hf-capabilities"),
  importFromHF: (body: { repo_id: string; config?: string; split?: string; kind: string; name?: string; max_rows?: number }) =>
    req<{ dataset: Dataset; validation: ValidationReport; rows_imported: number }>("/api/datasets/import-hf", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  uploadDataset: async (file: File, kind: string, name?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("kind", kind);
    if (name) fd.append("name", name);
    return req<{ dataset: Dataset; validation: ValidationReport }>("/api/datasets/upload", { method: "POST", body: fd });
  },

  // runs
  listRuns: () => req<Run[]>("/api/runs"),
  getRun: (id: number) =>
    req<{ run: Run; checkpoints: Checkpoint[]; queue_position: number | null }>(`/api/runs/${id}`),
  backends: () =>
    req<{ name: string; tasks: string[]; methods: string[] }[]>("/api/runs/backends"),
  estimate: (cfg: Record<string, unknown>) =>
    req<{ estimate: MemoryEstimate; validation: ValidationReport; hardware: HardwareProfile }>(
      "/api/runs/estimate",
      { method: "POST", body: JSON.stringify(cfg) },
    ),
  createRun: (cfg: Record<string, unknown>) =>
    req<{ run_id: number; queue_size: number; estimate: MemoryEstimate }>("/api/runs", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),
  cancelRun: (id: number) => req<{ cancelled: number }>(`/api/runs/${id}/cancel`, { method: "POST" }),
  deleteRun: (id: number) => req<{ deleted: number }>(`/api/runs/${id}`, { method: "DELETE" }),
  runMetrics: (id: number) => req<{ metrics: Record<string, number>[] }>(`/api/runs/${id}/metrics`),
  runLogs: (id: number, lines = 2000) =>
    req<{ lines: LogLine[] }>(`/api/runs/${id}/logs?lines=${lines}`),
  rerun: (id: number) => req<{ run_id: number; queue_size: number }>(`/api/runs/${id}/rerun`, { method: "POST" }),
  exportConfig: (id: number) =>
    req<{ format: string; filename: string; content: string }>(`/api/runs/${id}/export-config`),

  // advisor
  recommend: (body: { task: string; method: string; priority: string; max_seq_length?: number }) =>
    req<{ hardware: HardwareProfile; source: string; recommendations: AdvisorRec[] }>(
      "/api/advisor/recommend",
      { method: "POST", body: JSON.stringify(body) },
    ),

  // registry
  localModels: () =>
    req<{ artifacts: ModelArtifact[]; unpublished_runs: UnpublishedRun[]; lineage: ModelLineage[] }>("/api/registry/local"),
  hfModels: () => req<{ models: HFModel[]; token_set: boolean }>("/api/registry/hf"),
  registryCapabilities: () =>
    req<{ gguf_available: boolean; hf_token_set: boolean; quant_types: string[]; deployment: DeploymentStatus }>(
      "/api/registry/capabilities",
    ),
  modelOptions: () => req<{ models: ModelOption[] }>("/api/registry/model-options"),
  inspectModel: (model_ref: string, revision?: string) =>
    req<ModelInspection>("/api/registry/inspect", {
      method: "POST",
      body: JSON.stringify({ model_ref, revision: revision?.trim() || null }),
    }),
  deploymentStatus: () => req<DeploymentStatus>("/api/registry/deployment"),
  deployModel: (model_ref: string) =>
    req<DeploymentStatus>("/api/registry/deploy", { method: "POST", body: JSON.stringify({ model_ref }) }),
  stopDeployment: () => req<{ stopped: boolean }>("/api/registry/deployment", { method: "DELETE" }),
  publishModel: (body: { run_id: number; repo_id: string; private: boolean }) =>
    req<{ hf_repo: string }>("/api/registry/publish", { method: "POST", body: JSON.stringify(body) }),
  importModel: (repo_id: string) =>
    req<{ artifact: ModelArtifact; path: string }>("/api/registry/import", {
      method: "POST",
      body: JSON.stringify({ repo_id }),
    }),
  deleteArtifact: (id: number) => req<{ deleted: number }>(`/api/registry/artifacts/${id}`, { method: "DELETE" }),
  quantize: (body: { run_id: number; quant_type: string }) =>
    req<{ artifact_id: number; status: string }>("/api/registry/quantize", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  quantizeArtifact: (body: { artifact_id: number; quant_type: string }) =>
    req<{ artifact_id: number; status: string }>("/api/registry/quantize", { method: "POST", body: JSON.stringify(body) }),
  mergeAdapter: (model_ref: string, name?: string) =>
    req<{ artifact_id: number; status: string }>("/api/registry/merge", { method: "POST", body: JSON.stringify({ model_ref, name }) }),
  quantizeLogs: (artifactId: number, lines = 2000) =>
    req<{ lines: LogLine[] }>(`/api/registry/quantize/${artifactId}/logs?lines=${lines}`),

  // system logs
  serverLogs: (lines = 200) => req<{ lines: string[]; path: string }>(`/api/system/logs?lines=${lines}`),

  // eval lab
  evalStatus: () =>
    req<{
      busy: boolean;
      judge_available: boolean;
      current: { kind: string; id: string | number } | null;
      vllm_available: boolean;
      serve_engine_setting: "auto" | "vllm" | "transformers";
      vllm_warm_model: string | null;
    }>("/api/eval/status"),
  cancelEval: () => req<{ cancelled: boolean }>("/api/eval/cancel", { method: "POST" }),
  jobLogs: (kind: "eval" | "gen", id: number | string, lines = 2000) =>
    req<{ lines: LogLine[] }>(`/api/eval/logs/${kind}/${id}?lines=${lines}`),
  generate: (body: Record<string, unknown>) =>
    req<{ gen_id: string }>("/api/eval/generate", { method: "POST", body: JSON.stringify(body) }),
  generationResult: (id: string) => req<{ status: string; output?: string; metrics?: Record<string, number>; error?: string | null }>(`/api/eval/generation/${id}`),
  runEval: (body: Record<string, unknown>) =>
    req<{ eval_id: number }>("/api/eval/run", { method: "POST", body: JSON.stringify(body) }),
  evalResults: () => req<EvalResult[]>("/api/eval/results"),
  evalResult: (id: number) => req<EvalResult>(`/api/eval/results/${id}`),
};
