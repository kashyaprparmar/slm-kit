import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Send, Square, Bot, Loader2, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { useWebSocket } from "@/lib/ws";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { NumberField } from "@/components/ui/field";
import { BaseModelPicker } from "@/components/studio/BaseModelPicker";
import { EmptyState } from "@/components/ui/empty-state";
import { Badge } from "@/components/ui/badge";
import { LogPanel, type LogLine } from "@/components/LogPanel";
import { useWorkflow } from "@/lib/workflow";
import { duration } from "@/lib/format";

type GenEvent =
  | { type: "token"; text: string }
  | { type: "phase"; phase: string; elapsed?: number }
  | { type: "log"; level?: string; message: string }
  | { type: "error"; message: string }
  | { type: "done" };
type GenerationMetrics = { seconds: number; tokens: number; tokens_per_second: number; peak_vram_mb?: number };

const PHASE_LABEL: Record<string, string> = {
  importing: "Loading libraries…",
  loading_model: "Loading model (first load downloads it)…",
  generating: "Generating…",
};

export function Playground() {
  const qc = useQueryClient();
  const [search] = useSearchParams();
  const [modelRef, setModelRef] = useWorkflow("playground.model", "unsloth/Qwen2.5-0.5B-Instruct");
  const [prompt, setPrompt] = useWorkflow("playground.prompt", "Explain what gross margin is in one paragraph.");
  const [params, setParams] = useWorkflow("playground.params", { max_new_tokens: 256, temperature: 0.7, top_p: 0.95, top_k: 50, repetition_penalty: 1.1 });
  const [genId, setGenId] = useState<string | null>(null);
  const [lastGenId, setLastGenId] = useWorkflow<string | null>("playground.last", null, false); // kept after finish so logs stay viewable
  const [output, setOutput] = useState("");
  const [systemPrompt, setSystemPrompt] = useWorkflow("playground.system", "");
  const [seed, setSeed] = useWorkflow("playground.seed", 42);
  const [generationMetrics, setGenerationMetrics] = useState<GenerationMetrics | null>(null);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<string>("");
  const [liveLogs, setLiveLogs] = useState<LogLine[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef<number>(0);
  const setP = (k: keyof typeof params, v: number | null) => setParams((p) => ({ ...p, [k]: v ?? p[k] }));

  useEffect(() => {
    const requested = search.get("model");
    if (requested) setModelRef(requested);
  }, [search, setModelRef]);

  // Global eval/GPU status — so we can show + clear a stuck busy state.
  const status = useQuery({ queryKey: ["eval-status"], queryFn: api.evalStatus, refetchInterval: 3000 });
  const busyElsewhere = (status.data?.busy ?? false) && !running;

  useQuery({
    queryKey: ["generation-result", genId],
    queryFn: async () => {
      const result = await api.generationResult(genId!);
      if (["done", "failed", "cancelled"].includes(result.status)) {
        if (result.output && !output) setOutput(result.output);
        if (result.metrics) setGenerationMetrics(result.metrics as GenerationMetrics);
        if (result.status === "failed") toast.error(result.error || "Generation failed");
        finish();
      }
      return result;
    },
    enabled: !!genId && running,
    refetchInterval: 2000,
  });

  // Persisted logs for the most recent generation — lets the Logs panel stay
  // useful after the job finishes, not just while it's actively streaming.
  const logHistory = useQuery({
    queryKey: ["gen-logs", lastGenId],
    queryFn: () => api.jobLogs("gen", lastGenId!),
    enabled: !!lastGenId && !running,
  });
  const persistedLogs = logHistory.data?.lines ?? [];
  const logs: LogLine[] = running
    ? liveLogs
    : persistedLogs.length
      ? persistedLogs
      : liveLogs;

  // Local elapsed timer while running.
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setElapsed((Date.now() - startedAt.current) / 1000), 250);
    return () => clearInterval(t);
  }, [running]);

  function finish() {
    setRunning(false);
    setGenId(null);
    setPhase("");
    qc.invalidateQueries({ queryKey: ["eval-status"] });
    qc.invalidateQueries({ queryKey: ["gen-logs"] });
  }

  useWebSocket<GenEvent | ({ type: "metrics" } & GenerationMetrics)>(genId ? `/ws/gen/${genId}` : null, (ev) => {
    if (ev.type === "token") {
      setPhase("generating");
      setOutput((o) => o + ev.text);
    } else if (ev.type === "phase") {
      setPhase(ev.phase);
      setLiveLogs((l) => [...l.slice(-1999), { ts: Date.now() / 1000, level: "info", message: `[${ev.phase}] elapsed=${ev.elapsed ?? 0}s` }]);
    } else if (ev.type === "log") {
      setLiveLogs((l) => [...l, { ts: Date.now() / 1000, level: ev.level ?? "info", message: ev.message }]);
    } else if (ev.type === "error") {
      toast.error(ev.message || "Generation failed");
      setLiveLogs((l) => [...l, { ts: Date.now() / 1000, level: "error", message: ev.message }]);
      finish();
    } else if (ev.type === "metrics") {
      setGenerationMetrics(ev);
    } else if (ev.type === "done") {
      finish();
    }
  });

  async function generate() {
    setOutput("");
    setGenerationMetrics(null);
    setLiveLogs([]);
    setPhase("importing");
    setElapsed(0);
    startedAt.current = Date.now();
    setRunning(true);
    try {
      const { gen_id } = await api.generate({ model_ref: modelRef, prompt, system_prompt: systemPrompt, seed, ...params });
      setGenId(gen_id);
      setLastGenId(gen_id);
    } catch (e) {
      setRunning(false);
      setPhase("");
      toast.error(e instanceof ApiError && e.status === 409 ? "GPU is busy — cancel the current job or wait." : e instanceof Error ? e.message : "Could not start generation");
    }
  }

  async function stop() {
    try {
      await api.cancelEval();
      toast.message("Stopping…");
    } catch {
      toast.error("Could not stop");
    }
    // The WS will deliver the terminal event, but release the UI immediately too.
    finish();
  }

  async function clearStuck() {
    try {
      await api.cancelEval();
      toast.success("Cleared the busy GPU");
      qc.invalidateQueries({ queryKey: ["eval-status"] });
    } catch {
      toast.error("Could not clear");
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <div className="space-y-4">
        {busyElsewhere && (
          <div className="flex items-center justify-between gap-3 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-warning">
            <span className="flex items-center gap-2">
              <AlertTriangle className="size-4" />
              GPU is busy{status.data?.current ? ` (${status.data.current.kind})` : ""} — a job is running or stuck.
            </span>
            <Button size="sm" variant="outline" onClick={clearStuck}>Stop / clear</Button>
          </div>
        )}

        <Card>
          <CardHeader><CardTitle>Prompt</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <BaseModelPicker value={modelRef} onChange={setModelRef} />
            <Textarea aria-label="System prompt" placeholder="Optional system instructions" value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)} />
            <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} className="min-h-[120px]" placeholder="Ask the model something…" />
            <div className="flex items-center gap-2">
              <Button onClick={generate} disabled={running || busyElsewhere || !modelRef.trim() || !prompt.trim()}>
                <Send /> Generate
              </Button>
              {running && (
                <Button variant="danger" onClick={stop}>
                  <Square className="size-3.5" /> Stop
                </Button>
              )}
              {running && (
                <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Loader2 className="size-3.5 animate-spin" />
                  {PHASE_LABEL[phase] ?? "Working…"} · {duration(elapsed)}
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2"><Bot className="size-4 text-primary" /> Output</CardTitle>
            {running && phase === "loading_model" && <Badge variant="warning">downloading / loading</Badge>}
          </CardHeader>
          <CardContent>
            {generationMetrics && <p className="mb-3 text-xs text-muted-foreground">{generationMetrics.tokens} tokens · {generationMetrics.seconds}s · {generationMetrics.tokens_per_second} tokens/s{generationMetrics.peak_vram_mb != null ? ` · ${generationMetrics.peak_vram_mb} MB peak VRAM` : ""}</p>}
            {output ? (
              <div className="whitespace-pre-wrap rounded-md border bg-background/40 p-4 text-sm leading-relaxed">
                {output}
                {running && <span className="ml-0.5 inline-block h-4 w-2 animate-pulse-dot bg-primary align-middle" />}
              </div>
            ) : running ? (
              <div className="space-y-2 p-6">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="size-4 animate-spin" /> {PHASE_LABEL[phase] ?? "Starting…"} ({duration(elapsed)})
                </div>
                <p className="text-[11px] text-muted-foreground">
                  First run for a model downloads its weights — a 0.5B model is ~400 MB, a 7B model several GB.
                  You can press <span className="font-medium">Stop</span> any time. Full detail is in the Logs panel below.
                </p>
              </div>
            ) : (
              <EmptyState icon={Bot} title="No output yet" description="Enter a prompt and hit Generate to stream a response." className="border-0" />
            )}
          </CardContent>
        </Card>

        {(running || lastGenId) && (
          <Card>
            <CardContent className="pt-5">
              <LogPanel lines={logs} live={running} title="Generation logs" emptyHint="No log output yet." />
            </CardContent>
          </Card>
        )}
      </div>

      <Card className="lg:sticky lg:top-4 lg:self-start">
        <CardHeader><CardTitle>Generation params</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <NumberField label="Random Seed" value={seed} min={0} onChange={v => setSeed(v ?? 42)} hint="Use the same seed and settings for repeatable sampling." />
          <NumberField label="Max new tokens" value={params.max_new_tokens} min={1} step={16} onChange={(v) => setP("max_new_tokens", v)} />
          <NumberField label="Temperature" value={params.temperature} min={0} max={2} step={0.05} onChange={(v) => setP("temperature", v)} hint="0 = greedy/deterministic" />
          <NumberField label="Top-p" value={params.top_p} min={0} max={1} step={0.05} onChange={(v) => setP("top_p", v)} />
          <NumberField label="Top-k" value={params.top_k} min={0} step={1} onChange={(v) => setP("top_k", v)} />
          <NumberField label="Repetition penalty" value={params.repetition_penalty} min={1} max={2} step={0.05} onChange={(v) => setP("repetition_penalty", v)} />
        </CardContent>
      </Card>
    </div>
  );
}
