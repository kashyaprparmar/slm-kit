import type {
  AdvisorRec,
  Dataset,
  EvalResult,
  HardwareProfile,
  HFModel,
  LogLine,
  MemoryEstimate,
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
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      ...init,
    });
  } catch (e) {
    log.error(`${method} ${path} — network error`, String(e));
    throw e;
  }
  const ms = Math.round(performance.now() - t0);
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text();
    }
    log.error(`${method} ${path} → ${res.status} (${ms}ms)`, detail);
    throw new ApiError(res.status, detail);
  }
  // Only log mutations at info; GETs at debug to avoid noise.
  (method === "GET" ? log.debug : log.info)(`${method} ${path} → ${res.status} (${ms}ms)`);
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : `Request failed (${status})`);
  }
}

export const api = {
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
    const res = await fetch("/api/datasets/upload", { method: "POST", body: fd });
    if (!res.ok) throw new ApiError(res.status, await res.text());
    return res.json() as Promise<{ dataset: Dataset; validation: ValidationReport }>;
  },

  // runs
  listRuns: () => req<Run[]>("/api/runs"),
  getRun: (id: number) =>
    req<{ run: Run; checkpoints: unknown[]; queue_position: number | null }>(`/api/runs/${id}`),
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
    req<{ artifacts: ModelArtifact[]; unpublished_runs: UnpublishedRun[] }>("/api/registry/local"),
  hfModels: () => req<{ models: HFModel[]; token_set: boolean }>("/api/registry/hf"),
  registryCapabilities: () =>
    req<{ gguf_available: boolean; hf_token_set: boolean; quant_types: string[] }>(
      "/api/registry/capabilities",
    ),
  publishModel: (body: { run_id: number; repo_id: string; private: boolean }) =>
    req<{ hf_repo: string }>("/api/registry/publish", { method: "POST", body: JSON.stringify(body) }),
  importModel: (repo_id: string) =>
    req<{ artifact: ModelArtifact; path: string }>("/api/registry/import", {
      method: "POST",
      body: JSON.stringify({ repo_id }),
    }),
  quantize: (body: { run_id: number; quant_type: string }) =>
    req<{ artifact_id: number; status: string }>("/api/registry/quantize", {
      method: "POST",
      body: JSON.stringify(body),
    }),
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
  runEval: (body: Record<string, unknown>) =>
    req<{ eval_id: number }>("/api/eval/run", { method: "POST", body: JSON.stringify(body) }),
  evalResults: () => req<EvalResult[]>("/api/eval/results"),
  evalResult: (id: number) => req<EvalResult>(`/api/eval/results/${id}`),
};
