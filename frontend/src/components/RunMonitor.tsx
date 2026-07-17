import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip as RTooltip, CartesianGrid,
} from "recharts";
import { Ban, Activity, Gauge, Clock } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { useWebSocket } from "@/lib/ws";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { StatusBadge } from "@/components/StatusBadge";
import { EmptyState } from "@/components/ui/empty-state";
import { LogPanel, type LogLine } from "@/components/LogPanel";
import { duration } from "@/lib/format";
import type { RunStatus, TrainingEvent } from "@/lib/types";

interface Point {
  step: number;
  loss?: number;
  tokens_per_sec?: number;
}

export function RunMonitor({ runId }: { runId: number }) {
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.getRun(runId), refetchInterval: 8000 });
  const [series, setSeries] = useState<Point[]>([]);
  const [liveLogs, setLiveLogs] = useState<LogLine[]>([]);
  const [latest, setLatest] = useState<{ step: number; total?: number; tps?: number; eta?: number }>({ step: 0 });
  const [samples, setSamples] = useState<string[]>([]);
  const [liveStatus, setLiveStatus] = useState<RunStatus | null>(null);

  // Reset when switching runs.
  useEffect(() => {
    setSeries([]); setLiveLogs([]); setLatest({ step: 0 }); setSamples([]); setLiveStatus(null);
  }, [runId]);

  // Seed the chart from persisted history so finished runs still plot their curve.
  const history = useQuery({ queryKey: ["run-metrics", runId], queryFn: () => api.runMetrics(runId) });
  useEffect(() => {
    if (!history.data?.metrics.length) return;
    const hist: Point[] = history.data.metrics.map((m) => ({
      step: m.step, loss: m.loss, tokens_per_sec: m.tokens_per_sec,
    }));
    setSeries((prev) => {
      const bySt = new Map<number, Point>();
      for (const p of hist) bySt.set(p.step, p);
      for (const p of prev) bySt.set(p.step, p); // live points win over history
      return [...bySt.values()].sort((a, b) => a.step - b.step).slice(-600);
    });
    const last = history.data.metrics[history.data.metrics.length - 1];
    setLatest((l) => (l.step >= last.step ? l : { step: last.step, tps: last.tokens_per_sec, eta: last.eta_seconds }));
  }, [history.data]);

  // Persisted, comprehensive log history — makes the Logs panel useful for
  // finished/past runs too, not just while the WebSocket is live.
  const logHistory = useQuery({ queryKey: ["run-logs", runId], queryFn: () => api.runLogs(runId) });
  const persistedLogs = logHistory.data?.lines ?? [];
  // Live lines arrive over the WS while the run is active; persisted history
  // covers everything up to the last poll. Merge and de-dupe by (ts,message).
  const logs: LogLine[] = (() => {
    const seen = new Set(persistedLogs.map((l) => `${l.ts}|${l.message}`));
    const extra = liveLogs.filter((l) => !seen.has(`${l.ts}|${l.message}`));
    return [...persistedLogs, ...extra];
  })();

  const { connected } = useWebSocket<TrainingEvent>(`/ws/runs/${runId}`, (ev) => {
    if (ev.type === "metric") {
      const loss = ev.metrics.loss;
      const tps = ev.metrics.tokens_per_sec;
      setSeries((s) => [...s.slice(-400), { step: ev.step, loss, tokens_per_sec: tps }]);
      setLatest({ step: ev.step, total: ev.total_steps, tps, eta: ev.metrics.eta_seconds });
    } else if (ev.type === "log") {
      setLiveLogs((l) => [...l.slice(-3000), { ts: ev.ts, level: ev.level, message: ev.message }]);
    } else if (ev.type === "sample") {
      setSamples((s) => [ev.text, ...s].slice(0, 5));
    } else if (ev.type === "status") {
      setLiveStatus(ev.status as RunStatus);
      if (ev.status === "done") toast.success(`Run #${runId} finished`);
      if (ev.status === "failed") toast.error(`Run #${runId} failed`);
      run.refetch();
    }
  });

  const status: RunStatus = liveStatus ?? (run.data?.run.status ?? "queued");
  const running = status === "running";
  const queued = status === "queued";
  const progress = latest.total ? (latest.step / latest.total) * 100 : 0;
  const finalLoss = series.length ? series[series.length - 1].loss : run.data?.run.metrics?.loss;

  async function cancel() {
    try {
      await api.cancelRun(runId);
      toast.message("Cancellation requested — freeing VRAM…");
    } catch {
      toast.error("Could not cancel");
    }
  }

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <CardTitle>{run.data?.run.name ?? `Run #${runId}`}</CardTitle>
          <StatusBadge status={status} />
          {connected && running && <span className="text-[10px] uppercase tracking-wider text-success">live</span>}
        </div>
        {(running || queued) && (
          <Button size="sm" variant="danger" onClick={cancel}>
            <Ban /> Cancel
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {queued && latest.step === 0 ? (
          <EmptyState icon={Clock} title="Queued" description="Waiting for the GPU. Live metrics will stream here once it starts." className="border-0" />
        ) : (
          <>
            {latest.total ? (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Step {latest.step} / {latest.total}</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} tone={running ? "primary" : status === "done" ? "success" : "warning"} />
              </div>
            ) : running ? (
              <Progress indeterminate />
            ) : null}

            <div className="grid grid-cols-3 gap-3">
              <Metric icon={Activity} label="Loss" value={finalLoss != null ? finalLoss.toFixed(4) : "—"} />
              <Metric icon={Gauge} label="Tokens/s" value={latest.tps != null ? Math.round(latest.tps).toString() : "—"} />
              <Metric icon={Clock} label="ETA" value={duration(latest.eta)} />
            </div>

            <LossChart series={series} />

            {samples.length > 0 && (
              <div className="space-y-1.5">
                <div className="text-xs font-medium text-muted-foreground">Latest sample generation</div>
                <pre className="max-h-28 overflow-y-auto whitespace-pre-wrap rounded-md border bg-background/40 p-3 font-mono text-[11px] text-muted-foreground">
                  {samples[0]}
                </pre>
              </div>
            )}

            <LogPanel lines={logs} live={connected && running} title="Training logs" />

            {run.data?.run.error && (
              <div className="rounded-md border border-danger/40 bg-danger/10 p-3 text-xs text-danger">
                {run.data.run.error}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function Metric({ icon: Icon, label, value }: { icon: typeof Activity; label: string; value: string }) {
  return (
    <div className="rounded-md border bg-background/40 p-3">
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground"><Icon className="size-3" />{label}</div>
      <div className="mt-0.5 font-mono text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function LossChart({ series }: { series: Point[] }) {
  const data = series.filter((p) => p.loss != null);
  if (data.length < 2) {
    return (
      <div className="grid h-48 place-items-center rounded-md border border-dashed text-xs text-muted-foreground">
        Loss curve appears once training reports metrics…
      </div>
    );
  }
  return (
    <div className="h-48 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
          <XAxis dataKey="step" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickLine={false} axisLine={false} />
          <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickLine={false} axisLine={false} domain={["auto", "auto"]} width={44} />
          <RTooltip
            contentStyle={{ background: "hsl(var(--elevated))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "hsl(var(--muted-foreground))" }}
          />
          <Line type="monotone" dataKey="loss" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} isAnimationActive animationDuration={300} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

