import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowRight,
  Database,
  Sparkles,
  SlidersHorizontal,
  Boxes,
  Layers,
  CircleSlash,
} from "lucide-react";
import { api } from "@/lib/api";
import { useHardware } from "@/components/hardware-context";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { StatusBadge } from "@/components/StatusBadge";
import { EmptyState } from "@/components/ui/empty-state";
import { mb, relativeTime } from "@/lib/format";
import type { Run } from "@/lib/types";

export default function Dashboard() {
  const { hw } = useHardware();
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, refetchInterval: 4000 });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
  const status = useQuery({ queryKey: ["status"], queryFn: api.status });

  const active = runs.data?.find((r) => r.status === "running" || r.status === "queued");
  const recent = runs.data?.slice(0, 6) ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard"
        description="Your local model-forging workstation at a glance."
        actions={
          <Button asChild>
            <Link to="/finetune">
              <SlidersHorizontal /> New fine-tune
            </Link>
          </Button>
        }
      />

      {/* Stat row */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="VRAM budget" value={mb(hw?.vram_total_mb ?? status.data?.vram_budget_mb)} hint={hw?.gpu_name ?? "GPU"} icon={Activity} />
        <Stat label="Datasets" value={datasets.data ? String(datasets.data.length) : undefined} hint="ready to use" icon={Database} />
        <Stat label="Total runs" value={runs.data ? String(runs.data.length) : undefined} hint="all time" icon={Boxes} />
        <Stat
          label="llmfit"
          value={status.data ? (status.data.llmfit_available ? "Active" : "Fallback") : undefined}
          hint={status.data?.llmfit_available ? "hardware-aware" : "internal estimator"}
          icon={Sparkles}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Active job */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle>Active job</CardTitle>
            {active && <StatusBadge status={active.status} />}
          </CardHeader>
          <CardContent>
            {runs.isLoading ? (
              <Skeleton className="h-24" />
            ) : active ? (
              <ActiveJob run={active} />
            ) : (
              <EmptyState
                icon={CircleSlash}
                title="No active job"
                description="The GPU is idle. Launch a run from a studio to see live metrics here."
                action={
                  <Button asChild variant="secondary" size="sm">
                    <Link to="/finetune">Open Fine-Tuning Studio</Link>
                  </Button>
                }
              />
            )}
          </CardContent>
        </Card>

        {/* Advisor teaser */}
        <Card className="relative overflow-hidden">
          <div className="pointer-events-none absolute -right-8 -top-8 size-32 rounded-full bg-primary/20 blur-2xl" />
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="size-4 text-primary" /> Model Advisor
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Let SLM Kit rank the best base-model + method combos for your hardware and goal.
            </p>
            <Button asChild variant="secondary" className="w-full">
              <Link to="/finetune?advisor=1">
                Get a recommendation <ArrowRight />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Recent runs + quick links */}
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle>Recent runs</CardTitle>
            <Button asChild variant="ghost" size="sm">
              <Link to="/runs">View all <ArrowRight /></Link>
            </Button>
          </CardHeader>
          <CardContent>
            {runs.isLoading ? (
              <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-12" />)}</div>
            ) : recent.length ? (
              <div className="divide-y divide-border/60">
                {recent.map((r) => <RunRow key={r.id} run={r} />)}
              </div>
            ) : (
              <p className="py-8 text-center text-sm text-muted-foreground">No runs yet.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Quick start</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            <QuickLink to="/datasets" icon={Database} label="Prepare a dataset" />
            <QuickLink to="/pretrain" icon={Boxes} label="Pretrain from scratch" />
            <QuickLink to="/domain" icon={Layers} label="Adapt to a domain" />
            <QuickLink to="/finetune" icon={SlidersHorizontal} label="Fine-tune a model" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Stat({ label, value, hint, icon: Icon }: { label: string; value?: string; hint: string; icon: typeof Activity }) {
  return (
    <Card>
      <CardContent className="flex items-start justify-between p-5">
        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">{label}</p>
          {value === undefined ? <Skeleton className="h-7 w-16" /> : <p className="text-2xl font-bold tracking-tight">{value}</p>}
          <p className="text-[11px] text-muted-foreground">{hint}</p>
        </div>
        <div className="grid size-9 place-items-center rounded-lg bg-primary/10 text-primary">
          <Icon className="size-4.5" />
        </div>
      </CardContent>
    </Card>
  );
}

function ActiveJob({ run }: { run: Run }) {
  const loss = run.metrics?.loss;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold">{run.name}</span>
        <Badge variant="neutral">{run.method.toUpperCase()}</Badge>
        <Badge variant="outline">{run.base_model}</Badge>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <Metric label="Loss" value={loss != null ? loss.toFixed(3) : "—"} />
        <Metric label="Tokens/s" value={run.metrics?.tokens_per_sec?.toFixed(0) ?? "—"} />
        <Metric label="Started" value={relativeTime(run.started_at)} />
      </div>
      <Button asChild variant="secondary" size="sm">
        <Link to={`/runs?id=${run.id}`}>Open live monitor <ArrowRight /></Link>
      </Button>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-background/40 p-3">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="font-mono text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function RunRow({ run }: { run: Run }) {
  return (
    <Link to={`/runs?id=${run.id}`} className="flex items-center justify-between gap-3 py-3 transition-colors hover:bg-muted/30 -mx-2 px-2 rounded-md">
      <div className="min-w-0">
        <div className="truncate text-sm font-medium">{run.name}</div>
        <div className="truncate text-xs text-muted-foreground">
          {run.method.toUpperCase()} · {run.base_model} · {relativeTime(run.created_at)}
        </div>
      </div>
      <StatusBadge status={run.status} />
    </Link>
  );
}

function QuickLink({ to, icon: Icon, label }: { to: string; icon: typeof Database; label: string }) {
  return (
    <Link
      to={to}
      className="flex items-center gap-3 rounded-md border border-transparent px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:border-border hover:bg-muted/40 hover:text-foreground"
    >
      <Icon className="size-4 text-primary" />
      {label}
      <ArrowRight className="ml-auto size-4 opacity-0 transition-opacity group-hover:opacity-100" />
    </Link>
  );
}
