import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, Plus, Trash2, Trophy, History, Square, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { useWebSocket } from "@/lib/ws";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { NumberField, Field } from "@/components/ui/field";
import { BaseModelPicker } from "@/components/studio/BaseModelPicker";
import { DatasetPicker } from "@/components/studio/DatasetPicker";
import { EmptyState } from "@/components/ui/empty-state";
import { LogPanel, type LogLine } from "@/components/LogPanel";
import { relativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { EvalPerModel, EvalSample } from "@/lib/types";

const METRIC_OPTIONS = [
  { key: "exact_match", label: "Exact match" },
  { key: "token_f1", label: "Token F1" },
  { key: "rouge_l", label: "ROUGE-L" },
  { key: "bleu", label: "BLEU" },
  { key: "perplexity", label: "Perplexity" },
];
// For highlighting the better model: perplexity is lower-is-better.
const LOWER_BETTER = new Set(["perplexity"]);

interface ResultView {
  per_model: Record<string, EvalPerModel>;
  samples: EvalSample[];
}

export function EvalHarness({ judgeAvailable }: { judgeAvailable: boolean }) {
  const qc = useQueryClient();
  const [models, setModels] = useState<string[]>(["unsloth/Qwen2.5-0.5B-Instruct"]);
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [maxSamples, setMaxSamples] = useState(25);
  const [metrics, setMetrics] = useState<Set<string>>(new Set(["exact_match", "token_f1", "rouge_l", "bleu"]));
  const [useJudge, setUseJudge] = useState(false);

  const [evalId, setEvalId] = useState<number | null>(null);
  const [lastEvalId, setLastEvalId] = useState<number | null>(null); // kept after finish so logs stay viewable
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Record<string, { done: number; total: number }>>({});
  const [result, setResult] = useState<ResultView | null>(null);
  const [liveLogs, setLiveLogs] = useState<LogLine[]>([]);
  const [phase, setPhase] = useState<{ phase: string; model?: string; elapsed?: number } | null>(null);

  const past = useQuery({ queryKey: ["eval-results"], queryFn: api.evalResults });
  const status = useQuery({ queryKey: ["eval-status"], queryFn: api.evalStatus, refetchInterval: 3000 });
  const busyElsewhere = (status.data?.busy ?? false) && !running;

  // Persisted logs for the most recent eval run — keeps the Logs panel useful
  // after the job finishes, not just while it's actively streaming.
  const logHistory = useQuery({
    queryKey: ["eval-logs", lastEvalId],
    queryFn: () => api.jobLogs("eval", lastEvalId!),
    enabled: !!lastEvalId && !running,
  });
  const persistedLogs = logHistory.data?.lines ?? [];
  const logs: LogLine[] = running ? liveLogs : persistedLogs.length ? persistedLogs : liveLogs;

  // Polling fallback: if the WS missed the terminal event (e.g. an instant
  // failure before the socket attached), the stored row still resolves the run.
  useQuery({
    queryKey: ["eval-poll", evalId],
    queryFn: async () => {
      const r = await api.evalResult(evalId!);
      const st = r.detail?.status;
      if (st === "done" || st === "failed") {
        setRunning(false);
        if (st === "done" && r.detail?.per_model && !result) {
          setResult({ per_model: r.detail.per_model, samples: r.detail.samples ?? [] });
        }
        if (st === "failed") toast.error(r.detail?.error ?? "Evaluation failed");
        qc.invalidateQueries({ queryKey: ["eval-results"] });
      }
      return r;
    },
    enabled: evalId != null && running,
    refetchInterval: 2500,
  });

  useWebSocket<Record<string, unknown>>(evalId != null && running ? `/ws/eval/${evalId}` : null, (ev) => {
    if (ev.type === "progress") {
      setPhase(null);
      setProgress((p) => ({ ...p, [ev.model as string]: { done: ev.done as number, total: ev.total as number } }));
      setLiveLogs((l) => [...l, { ts: Date.now() / 1000, level: "info", message: `progress: ${ev.model} ${ev.done}/${ev.total}` }]);
    } else if (ev.type === "phase") {
      setPhase({ phase: ev.phase as string, model: ev.model as string, elapsed: ev.elapsed as number });
      setLiveLogs((l) => [...l, { ts: Date.now() / 1000, level: "info", message: `[${ev.phase}] model=${ev.model ?? ""} elapsed=${ev.elapsed ?? 0}s` }]);
    } else if (ev.type === "result") {
      setResult({ per_model: ev.per_model as Record<string, EvalPerModel>, samples: (ev.samples as EvalSample[]) ?? [] });
    } else if (ev.type === "status") {
      setRunning(false);
      setPhase(null);
      if (ev.status === "failed") toast.error("Evaluation failed — see logs below.");
      else toast.success("Evaluation complete");
      qc.invalidateQueries({ queryKey: ["eval-results"] });
      qc.invalidateQueries({ queryKey: ["eval-status"] });
      qc.invalidateQueries({ queryKey: ["eval-logs"] });
    } else if (ev.type === "error") {
      setLiveLogs((l) => [...l, { ts: Date.now() / 1000, level: "error", message: String(ev.message ?? "") }]);
    } else if (ev.type === "log") {
      setLiveLogs((l) => [...l, { ts: Date.now() / 1000, level: "info", message: String(ev.message ?? "") }]);
    }
  });

  async function stop() {
    try {
      await api.cancelEval();
      toast.message("Stopping evaluation…");
    } catch {
      toast.error("Could not stop");
    }
    setRunning(false);
    setPhase(null);
    qc.invalidateQueries({ queryKey: ["eval-status"] });
  }

  const toggleMetric = (k: string) =>
    setMetrics((m) => {
      const next = new Set(m);
      if (next.has(k)) next.delete(k); else next.add(k);
      return next;
    });

  async function run() {
    setResult(null); setProgress({}); setLiveLogs([]);
    try {
      const { eval_id } = await api.runEval({
        models: [...new Set(models.filter((m) => m.trim()))], // dedupe — repeated refs would collide in the scorecard
        dataset_id: datasetId,
        max_samples: maxSamples,
        metrics: [...metrics],
        judge: useJudge,
      });
      setEvalId(eval_id);
      setLastEvalId(eval_id);
      setRunning(true);
    } catch (e) {
      toast.error(e instanceof ApiError && e.status === 409 ? "GPU is busy — try again when the current job finishes." : "Could not start evaluation");
    }
  }

  async function loadPast(id: number) {
    try {
      const r = await api.evalResult(id);
      setLastEvalId(id);
      if (r.detail?.per_model) setResult({ per_model: r.detail.per_model, samples: r.detail.samples ?? [] });
      else {
        setResult(null);
        toast.message(r.detail?.error ? `That eval failed: ${r.detail.error}` : "No stored result for that eval.");
      }
    } catch {
      toast.error("Could not load result");
    }
  }

  const canRun = models.some((m) => m.trim()) && datasetId != null && metrics.size > 0 && !running;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      <div className="space-y-6">
        {/* Config */}
        <Card>
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle>Models to evaluate</CardTitle>
            {models.length < 3 && (
              <Button size="sm" variant="ghost" onClick={() => setModels((m) => [...m, "unsloth/SmolLM2-360M-Instruct"])}>
                <Plus /> Compare another
              </Button>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            {models.map((m, i) => (
              <div key={i} className="flex items-start gap-2">
                <div className="flex-1">
                  <BaseModelPicker value={m} onChange={(v) => setModels((ms) => ms.map((x, j) => (j === i ? v : x)))} />
                </div>
                {models.length > 1 && (
                  <Button size="icon" variant="ghost" className="mt-6" onClick={() => setModels((ms) => ms.filter((_, j) => j !== i))}>
                    <Trash2 className="text-muted-foreground" />
                  </Button>
                )}
              </div>
            ))}
            <DatasetPicker value={datasetId} onChange={setDatasetId} kinds={["eval", "instruction"]} />
            <div className="grid grid-cols-2 gap-4">
              <NumberField label="Max samples" value={maxSamples} min={1} max={500} onChange={(v) => setMaxSamples(v ?? 25)} hint="keep small for quick iterations" />
              <Field label="Metrics">
                <div className="flex flex-wrap gap-x-4 gap-y-1.5 pt-1">
                  {METRIC_OPTIONS.map((m) => (
                    <label key={m.key} className="flex cursor-pointer items-center gap-1.5 text-xs">
                      <input type="checkbox" checked={metrics.has(m.key)} onChange={() => toggleMetric(m.key)} className="accent-[hsl(var(--primary))]" />
                      {m.label}
                    </label>
                  ))}
                  <label className={cn("flex items-center gap-1.5 text-xs", judgeAvailable ? "cursor-pointer" : "opacity-50")}>
                    <input type="checkbox" checked={useJudge} disabled={!judgeAvailable} onChange={(e) => setUseJudge(e.target.checked)} className="accent-[hsl(var(--primary))]" />
                    LLM-as-judge {!judgeAvailable && "(needs API key)"}
                  </label>
                </div>
              </Field>
            </div>
            {busyElsewhere && (
              <div className="flex items-center justify-between gap-3 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-warning">
                <span>GPU busy{status.data?.current ? ` (${status.data.current.kind})` : ""} — a job is running or stuck.</span>
                <Button size="sm" variant="outline" onClick={stop}>Stop / clear</Button>
              </div>
            )}
            <div className="flex gap-2">
              <Button className="flex-1" size="lg" disabled={!canRun || busyElsewhere} onClick={run}>
                <FlaskConical /> {running ? "Evaluating…" : "Run evaluation"}
              </Button>
              {running && (
                <Button size="lg" variant="danger" onClick={stop}>
                  <Square className="size-4" /> Stop
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Progress */}
        {running && (
          <Card>
            <CardHeader><CardTitle>Progress</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              {phase && (
                <div className="flex items-center gap-2 rounded-md bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
                  <Loader2 className="size-3.5 animate-spin" />
                  {phase.phase === "loading_model"
                    ? `Loading ${phase.model?.split("/").pop() ?? "model"} (first load downloads it)…`
                    : "Working…"}
                  {phase.elapsed != null && ` · ${Math.round(phase.elapsed)}s`}
                </div>
              )}
              {models.filter((m) => m.trim()).map((m) => {
                const p = progress[m];
                return (
                  <div key={m} className="space-y-1">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span className="truncate font-mono">{m.split("/").pop()}</span>
                      <span>{p ? `${p.done}/${p.total}` : "loading…"}</span>
                    </div>
                    <Progress value={p ? (p.done / p.total) * 100 : 0} indeterminate={!p} />
                  </div>
                );
              })}
            </CardContent>
          </Card>
        )}

        {(running || lastEvalId) && (
          <Card>
            <CardContent className="pt-5">
              <LogPanel lines={logs} live={running} title="Evaluation logs" emptyHint="No log output yet." />
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {result && <Scorecard result={result} />}
        {result && result.samples.length > 0 && <SampleDiffs result={result} />}
      </div>

      {/* Past results */}
      <Card className="lg:sticky lg:top-4 lg:self-start">
        <CardHeader><CardTitle className="flex items-center gap-2"><History className="size-4" /> Past evaluations</CardTitle></CardHeader>
        <CardContent className="space-y-2">
          {past.data?.length ? (
            past.data.map((r) => (
              <button key={r.id} onClick={() => loadPast(r.id)}
                className="w-full rounded-md border px-3 py-2 text-left transition-colors hover:bg-muted/40">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-xs">{r.model_ref}</span>
                  <Badge variant={r.detail?.status === "done" ? "success" : r.detail?.status === "failed" ? "danger" : "warning"}>
                    {r.detail?.status ?? "?"}
                  </Badge>
                </div>
                <div className="text-[11px] text-muted-foreground">#{r.id} · {relativeTime(r.created_at)}</div>
              </button>
            ))
          ) : (
            <EmptyState icon={FlaskConical} title="No evaluations yet" description="Results are stored and can be reopened here." className="border-0" />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function Scorecard({ result }: { result: ResultView }) {
  const modelNames = Object.keys(result.per_model);
  const metricKeys = [...new Set(modelNames.flatMap((m) => Object.keys(result.per_model[m].scores)))];

  const bestFor = (metric: string): string | null => {
    if (modelNames.length < 2) return null;
    const vals = modelNames.map((m) => result.per_model[m].scores[metric]).filter((v) => v != null);
    if (vals.length < 2) return null;
    const best = LOWER_BETTER.has(metric) ? Math.min(...vals) : Math.max(...vals);
    return modelNames.find((m) => result.per_model[m].scores[metric] === best) ?? null;
  };

  return (
    <Card>
      <CardHeader><CardTitle className="flex items-center gap-2"><Trophy className="size-4 text-primary" /> Scorecard</CardTitle></CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="py-2 pr-4 font-medium">Metric</th>
              {modelNames.map((m) => (
                <th key={m} className="py-2 pr-4 font-mono text-[11px] font-medium">{m.split("/").pop()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricKeys.map((k) => {
              const best = bestFor(k);
              return (
                <tr key={k} className="border-b border-border/50">
                  <td className="py-2 pr-4 text-muted-foreground">{k}</td>
                  {modelNames.map((m) => {
                    const v = result.per_model[m].scores[k];
                    return (
                      <td key={m} className={cn("py-2 pr-4 font-mono tabular-nums", best === m && "font-bold text-success")}>
                        {v != null ? v : "—"}
                        {best === m && " ★"}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
            <tr>
              <td className="py-2 pr-4 text-muted-foreground">eval time</td>
              {modelNames.map((m) => (
                <td key={m} className="py-2 pr-4 font-mono text-xs text-muted-foreground">{result.per_model[m].seconds}s</td>
              ))}
            </tr>
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function SampleDiffs({ result }: { result: ResultView }) {
  const modelNames = Object.keys(result.per_model);
  return (
    <Card>
      <CardHeader><CardTitle>Sample comparisons</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        {result.samples.slice(0, 6).map((s, i) => (
          <div key={i} className="space-y-2 rounded-md border p-3">
            <div className="text-sm font-medium">{s.prompt}</div>
            <div className="rounded bg-muted/40 px-2 py-1.5 text-xs">
              <span className="text-muted-foreground">Reference: </span>{s.reference}
            </div>
            <div className={cn("grid gap-2", modelNames.length > 1 ? "sm:grid-cols-2" : "")}>
              {modelNames.map((m) => (
                <div key={m} className="rounded border border-primary/20 bg-primary/5 px-2 py-1.5 text-xs">
                  <div className="mb-1 font-mono text-[10px] text-primary">{m.split("/").pop()}</div>
                  <span className="text-muted-foreground">{s.preds[m] || "—"}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
