import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Search, History, RotateCw, Download, Copy, ExternalLink, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible } from "@/components/ui/field";
import { EmptyState } from "@/components/ui/empty-state";
import { StatusBadge } from "@/components/StatusBadge";
import { RunMonitor } from "@/components/RunMonitor";
import { relativeTime, duration } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Run, RunStatus } from "@/lib/types";
import { useWorkflow, writeWorkflow } from "@/lib/workflow";
import { defaultPretrainForm, formFromConfig, type PretrainForm } from "@/lib/runconfig";
import { ErrorPanel } from "@/components/ErrorPanel";

const STATUSES: (RunStatus | "all")[] = ["all", "running", "queued", "done", "failed", "cancelled"];

function runDuration(r: Run): string {
  if (!r.started_at) return "—";
  const end = r.finished_at ? new Date(r.finished_at).getTime() : Date.now();
  return duration((end - new Date(r.started_at).getTime()) / 1000);
}

export default function RunHistory() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const [search, setSearch] = useWorkflow("runs.search", "");
  const [status, setStatus] = useWorkflow<RunStatus | "all">("runs.status", "all");
  const [method, setMethod] = useWorkflow("runs.method", "all");
  const [model, setModel] = useWorkflow("runs.model", "all");
  const [dataset, setDataset] = useWorkflow("runs.dataset", "all");
  const [sort, setSort] = useWorkflow("runs.sort", "newest");
  const [fromDate, setFromDate] = useWorkflow("runs.fromDate", "");
  const [limit, setLimit] = useState(25);
  const [compare, setCompare] = useState<number[]>([]);

  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, refetchInterval: 4000 });
  const selectedId = params.get("id") ? Number(params.get("id")) : runs.data?.[0]?.id ?? null;
  const select = (id: number) => setParams((p) => { p.set("id", String(id)); return p; }, { replace: true });

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return (runs.data ?? []).filter((r) => {
      if (status !== "all" && r.status !== status) return false;
      if (method !== "all" && r.method !== method) return false;
      if (model !== "all" && r.base_model !== model) return false;
      if (dataset !== "all" && String(r.dataset_id) !== dataset) return false;
      if (fromDate && r.created_at.slice(0, 10) < fromDate) return false;
      if (q && !`${r.name} ${r.base_model} ${r.method}`.toLowerCase().includes(q)) return false;
      return true;
    }).sort((a, b) => sort === "oldest" ? a.id - b.id : sort === "loss" ? (a.metrics?.loss ?? Infinity) - (b.metrics?.loss ?? Infinity) : b.id - a.id);
  }, [runs.data, search, status, method, model, dataset, sort, fromDate]);

  const selected = runs.data?.find((r) => r.id === selectedId) ?? null;

  return (
    <div className="space-y-6">
      <PageHeader title="Run History" description="Every run, searchable — with live and historical curves, logs, and one-click re-run." />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        {/* Run list */}
        <Card className="lg:sticky lg:top-4 lg:self-start">
          <CardHeader className="gap-3">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input placeholder="Search name, base model, method…" value={search} onChange={(e) => setSearch(e.target.value)} className="pl-9" />
            </div>
            <Select value={status} onChange={(e) => setStatus(e.target.value as RunStatus | "all")}>
              {STATUSES.map((s) => <option key={s} value={s}>{s === "all" ? "All statuses" : s}</option>)}
            </Select>
            <div className="grid grid-cols-2 gap-2">
              <Select aria-label="Filter training method" value={method} onChange={e => setMethod(e.target.value)}><option value="all">All methods</option>{[...new Set(runs.data?.map(r => r.method))].map(m => <option key={m}>{m}</option>)}</Select>
              <Select aria-label="Filter base model" value={model} onChange={e => setModel(e.target.value)}><option value="all">All models</option>{[...new Set(runs.data?.map(r => r.base_model))].filter(Boolean).map(m => <option key={m} value={m}>{m}</option>)}</Select>
              <Select aria-label="Filter dataset" value={dataset} onChange={e => setDataset(e.target.value)}><option value="all">All datasets</option>{[...new Set(runs.data?.map(r => r.dataset_id))].filter(Boolean).map(d => <option key={d} value={String(d)}>Dataset #{d}</option>)}</Select>
              <Input aria-label="Runs created since" type="date" value={fromDate} onChange={e => setFromDate(e.target.value)} />
              <Select aria-label="Sort runs" value={sort} onChange={e => setSort(e.target.value)}><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="loss">Lowest loss</option></Select>
            </div>
          </CardHeader>
          <CardContent className="max-h-[70vh] space-y-1.5 overflow-y-auto">
            {runs.error ? <ErrorPanel error={runs.error} retry={() => runs.refetch()} /> : runs.isLoading ? (
              [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-16" />)
            ) : filtered.length ? (
              filtered.slice(0, limit).map((r) => (
                <button
                  key={r.id}
                  onClick={() => select(r.id)}
                  className={cn(
                    "w-full rounded-md border px-3 py-2.5 text-left transition-colors",
                    r.id === selectedId ? "border-primary/50 bg-primary/5" : "border-border hover:bg-muted/40",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-medium">{r.name}</span>
                    <StatusBadge status={r.status} />
                  </div>
                  <div className="mt-0.5 truncate text-xs text-muted-foreground">
                    #{r.id} · {r.method.toUpperCase()} · {r.base_model || "scratch"} · {relativeTime(r.created_at)}
                  </div>
                </button>
              ))
            ) : (
              <EmptyState icon={History} title="No matching runs" description="Launch one from a studio, or clear the filters." className="border-0" />
            )}
          </CardContent>
          {filtered.length > limit && <Button variant="ghost" onClick={() => setLimit(l => l + 25)}>Show 25 more ({filtered.length - limit} remaining)</Button>}
        </Card>

        {/* Detail */}
        {selected ? (
          <div className="space-y-6">
            <Button variant="outline" onClick={() => setCompare(ids => ids.includes(selected.id) ? ids.filter(id => id !== selected.id) : [...ids.slice(-1), selected.id])}>{compare.includes(selected.id) ? "Remove from comparison" : "Compare this run"}</Button>
            {compare.length === 2 && <Card><CardHeader><CardTitle>Experiment comparison</CardTitle></CardHeader><CardContent className="grid grid-cols-2 gap-3">
              {compare.map(id => { const r = runs.data?.find(x => x.id === id); return r && <div key={id} className="min-w-0 space-y-2"><p className="text-sm font-medium">{r.name}</p><p className="text-xs">Loss: {r.metrics?.loss ?? "Not reported"} · {runDuration(r)}</p><p className="text-xs">Peak VRAM: {r.metrics?.peak_vram_mb ?? "Not measured"} MB</p><pre className="max-h-64 overflow-auto text-[10px]">{JSON.stringify(r.config, null, 2)}</pre></div>; })}
            </CardContent></Card>}
            <RunMonitor runId={selected.id} />
            <RunDetail run={selected} onChanged={() => qc.invalidateQueries({ queryKey: ["runs"] })} onSelect={select} />
          </div>
        ) : (
          <Card>
            <CardContent className="p-0">
              <EmptyState icon={History} title="Select a run" description="Pick a run on the left to see its curves, logs, config, and actions." className="border-0" />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

function RunDetail({ run, onChanged, onSelect }: { run: Run; onChanged: () => void; onSelect: (id: number) => void }) {
  const navigate = useNavigate();
  const [pending, setPending] = useState(false);
  async function clone() {
    setPending(true);
    try {
      const { config } = await api.cloneRun(run.id);
      if (run.task === "pretrain") {
        const arch = config.arch as Record<string, unknown>;
        const train = config.train as Record<string, unknown>;
        const optim = config.optim as Record<string, unknown>;
        writeWorkflow("pretrain.form", defaultPretrainForm({
          ...(arch as Partial<PretrainForm>),
          ...(train as Partial<PretrainForm>),
          learning_rate: typeof optim.learning_rate === "number" ? optim.learning_rate : 3e-4,
          output_name: typeof config.output_name === "string" ? config.output_name : "scratch-clone",
          dataset_id: typeof config.dataset_id === "number" ? config.dataset_id : null,
        }));
        navigate("/pretrain");
      } else {
        writeWorkflow(`training.${run.task}`, formFromConfig(config));
        navigate(run.task === "finetune" ? "/finetune" : "/domain");
      }
      toast.success("Configuration cloned into the studio");
    } catch (e) { toast.error(e instanceof Error ? e.message : "Could not clone run"); }
    finally { setPending(false); }
  }
  async function rerun() {
    if (pending) return;
    setPending(true);
    try {
      const res = await api.rerun(run.id);
      toast.success(`Re-queued as run #${res.run_id}`);
      onChanged();
      onSelect(res.run_id);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not re-run");
    } finally { setPending(false); }
  }

  async function exportConfig() {
    try {
      const cfg = await api.exportConfig(run.id);
      const blob = new Blob([cfg.content], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = cfg.filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      toast.error("Could not export config");
    }
  }

  async function copyConfig() {
    try { await navigator.clipboard.writeText(JSON.stringify(run.config, null, 2)); toast.success("Config copied"); }
    catch { toast.error("Clipboard unavailable. Export the configuration instead."); }
  }

  async function remove() {
    if (!window.confirm(`Delete run #${run.id} and its managed checkpoints/logs? Registered model artifacts prevent deletion.`)) return;
    setPending(true);
    try {
      await api.deleteRun(run.id);
      toast.success("Run deleted");
      onChanged();
      navigate("/runs", { replace: true });
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not delete run");
    } finally { setPending(false); }
  }

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>Run #{run.id} details</CardTitle>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant="secondary" disabled={pending} onClick={rerun}><RotateCw /> Re-run</Button>
          <Button size="sm" variant="outline" disabled={pending} onClick={clone}><Copy /> Clone</Button>
          <Button size="sm" variant="outline" onClick={exportConfig}><Download /> Export config</Button>
          {run.status !== "running" && run.status !== "queued" && <Button size="sm" variant="danger" disabled={pending} onClick={remove}><Trash2 /> Delete</Button>}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {run.status === "done" && <div className="flex flex-wrap gap-2">
          <Button asChild size="sm" variant="secondary"><Link to={`/eval?model=run:${run.id}`}>Test Model</Link></Button>
          <Button asChild size="sm" variant="outline"><Link to={`/eval?tab=evaluate&model=run:${run.id}`}>Evaluate</Link></Button>
          <Button asChild size="sm" variant="outline"><Link to={`/serving?model=run:${run.id}`}>Serve</Link></Button>
        </div>}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Detail label="Task" value={run.task} />
          <Detail label="Method" value={<Badge variant="neutral">{run.method.toUpperCase()}</Badge>} />
          <Detail label="Duration" value={runDuration(run)} />
          <Detail label="Backend" value={run.backend} />
          <Detail label="GPU" value={run.hardware?.gpu_name ?? "Not detected"} />
          <Detail label="Final loss" value={run.metrics?.loss != null ? run.metrics.loss.toFixed(4) : "—"} />
          <Detail label="Peak VRAM" value={run.metrics?.peak_vram_mb != null ? `${Math.round(run.metrics.peak_vram_mb)} MB` : "—"} />
          <Detail label="Dataset" value={run.dataset_id ? `#${run.dataset_id}` : "—"} />
        </div>
        <Detail label="Base model" value={<span className="font-mono text-xs">{run.base_model || "— (from scratch)"}</span>} />

        {run.hf_repo && (
          <a href={run.hf_repo} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-sm text-primary hover:underline">
            <ExternalLink className="size-3.5" /> {run.hf_repo}
          </a>
        )}

        {run.error && (
          <div className="rounded-md border border-danger/40 bg-danger/10 p-3 text-xs text-danger">{run.error}</div>
        )}

        {run.estimate && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Badge variant={run.estimate.fit === "fits" ? "success" : run.estimate.fit === "tight" ? "warning" : "danger"}>
              {run.estimate.fit}
            </Badge>
            predicted ~{Math.round(run.estimate.total_mb)} MB peak VRAM
          </div>
        )}

        <Collapsible title="Full configuration (config-as-data)">
          <div className="flex justify-end pb-2">
            <Button size="sm" variant="ghost" onClick={copyConfig}><Copy /> Copy</Button>
          </div>
          <pre className="max-h-80 overflow-auto rounded-md border bg-background/40 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
            {JSON.stringify(run.config, null, 2)}
          </pre>
        </Collapsible>
      </CardContent>
    </Card>
  );
}

function Detail({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-md border bg-background/40 p-2.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-sm font-medium">{value}</div>
    </div>
  );
}
