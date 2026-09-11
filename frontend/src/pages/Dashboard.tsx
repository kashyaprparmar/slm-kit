import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Database, Boxes, SlidersHorizontal, Layers, FlaskConical, Server, MessageSquare, History, Library, Activity } from "lucide-react";
import { api } from "@/lib/api";
import { useHardware } from "@/components/hardware-context";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/StatusBadge";
import { ErrorPanel } from "@/components/ErrorPanel";
import { relativeTime } from "@/lib/format";

const actions = [
  ["/datasets", "Upload Dataset", Database], ["/finetune", "Fine-Tune Model", SlidersHorizontal],
  ["/domain", "Continue Pretraining", Layers], ["/pretrain", "Train From Scratch", Boxes],
  ["/eval?tab=evaluate", "Evaluate Model", FlaskConical], ["/eval", "Test Model", MessageSquare],
  ["/serving", "Serve Model", Server], ["/registry", "Model Registry", Library], ["/runs", "Run History", History],
] as const;
export default function Dashboard() {
  const { hw, connected } = useHardware();
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, refetchInterval: 4000 });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
  const models = useQuery({ queryKey: ["model-options"], queryFn: api.modelOptions });
  const evaluations = useQuery({ queryKey: ["eval-results"], queryFn: api.evalResults, refetchInterval: 6000 });
  const status = useQuery({ queryKey: ["status"], queryFn: api.status, refetchInterval: 3000 });
  const deployment = useQuery({ queryKey: ["deployment-status"], queryFn: api.deploymentStatus, refetchInterval: 4000 });
  const active = runs.data?.find(r => r.id === status.data?.current_run);
  const queued = runs.data?.filter(r => r.status === "queued") ?? [];
  const failures = runs.data?.filter(r => r.status === "failed").slice(0, 2) ?? [];
  const resource = status.data?.resource;
  return <div className="space-y-6">
    <PageHeader title="Workstation" description="Prepare data, train, evaluate and serve — on your hardware."
      actions={<Badge variant={connected ? "success" : "danger"}>{connected ? "Backend connected" : "Backend offline"}</Badge>} />
    {status.error && <ErrorPanel error={status.error} retry={() => status.refetch()} />}
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {[
        ["Datasets", datasets.data?.length, "Local training and evaluation data", "/datasets"],
        ["Trained models", models.data?.models.length, "Loadable local checkpoints", "/registry"],
        ["Completed runs", runs.data?.filter(r => r.status === "done").length, "Recorded experiments", "/runs"],
        ["Evaluations", evaluations.data?.length, "Stored model comparisons", "/eval?tab=evaluate"],
      ].map(([label, value, hint, link]) => <Link key={String(label)} to={String(link)} className="rounded-lg border bg-card p-5 transition-colors hover:border-primary/50">
        <p className="text-xs text-muted-foreground">{label}</p><p className="mt-1 text-3xl font-semibold tabular-nums">{value ?? "—"}</p>
        <p className="mt-2 text-[11px] text-muted-foreground">{hint}</p>
      </Link>)}
    </div>
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-2"><CardHeader className="flex-row items-center justify-between">
        <CardTitle className="flex gap-2"><Activity className="size-4 text-primary" />GPU workload</CardTitle>
        <Badge variant={resource?.kind && resource.kind !== "idle" ? "warning" : "neutral"}>{resource?.kind ?? "Checking"}</Badge>
      </CardHeader><CardContent className="space-y-4">
        {active ? <><p className="font-semibold">{active.name}</p><p className="font-mono text-xs text-muted-foreground">{active.base_model || "From scratch"} · {active.method}</p>
          <Button asChild variant="secondary" size="sm"><Link to={`/runs?id=${active.id}`}>Open live monitor <ArrowRight /></Link></Button></>
          : resource?.kind !== "idle" && resource?.id ? <><p className="font-mono text-sm">{resource.id}</p>
            <Button asChild size="sm" variant="secondary"><Link to={resource.kind === "serving" ? "/serving" : "/eval"}>Open active workload <ArrowRight /></Link></Button></>
          : <p className="py-3 text-sm text-muted-foreground">{status.isLoading ? "Checking workload…" : "No managed workload is using the GPU."}</p>}
        <div className="border-t pt-3"><p className="mb-2 text-xs font-medium">{queued.length} queued training jobs</p>
          {queued.slice(0, 5).map(r => <Link className="flex justify-between py-1.5 text-xs text-muted-foreground hover:text-primary" key={r.id} to={`/runs?id=${r.id}`}><span>#{r.id} {r.name}</span><span>Waiting for GPU</span></Link>)}
          {!!queued.length && resource?.kind !== "training" && resource?.kind !== "idle" && <p className="text-xs text-warning">Stop serving or finish the current evaluation to let the queue continue.</p>}
        </div>
      </CardContent></Card>
      <Card><CardHeader><CardTitle>Current model server</CardTitle></CardHeader><CardContent className="space-y-3">
        <Badge variant={deployment.data?.active ? "success" : "neutral"}>{deployment.data?.state ?? "Checking"}</Badge>
        <p className="break-all font-mono text-xs">{deployment.data?.model_ref ?? "No Transformers model loaded"}</p>
        {deployment.data?.endpoint && <p className="break-all font-mono text-xs text-primary">{deployment.data.endpoint}</p>}
        <Button asChild variant="outline" size="sm"><Link to="/serving">Manage serving <ArrowRight /></Link></Button>
        <p className="text-xs text-muted-foreground">Ollama and Transformers provider status is available in Model Serving.</p>
      </CardContent></Card>
    </div>
    <div className="grid grid-cols-2 gap-2 md:grid-cols-3">{actions.map(([to, label, Icon]) =>
      <Link to={to} key={label} className="flex items-center gap-3 rounded-md border bg-card px-4 py-3 text-sm hover:border-primary/50"><Icon className="size-4 shrink-0 text-primary" />{label}<ArrowRight className="ml-auto size-3 text-muted-foreground" /></Link>)}</div>
    <div className="grid gap-4 lg:grid-cols-2">
      <Card><CardHeader><CardTitle>Recent runs</CardTitle></CardHeader><CardContent>
        {runs.error && <ErrorPanel error={runs.error} retry={() => runs.refetch()} />}
        {!runs.isLoading && !runs.data?.length && <p className="text-sm text-muted-foreground">No experiments yet. Start with a dataset and the Fine-Tuning Studio.</p>}
        {runs.data?.slice(0, 5).map(r => <Link key={r.id} to={`/runs?id=${r.id}`} className="flex items-center justify-between gap-3 border-b py-3 last:border-0">
          <div className="min-w-0"><p className="truncate text-sm font-medium">{r.name}</p><p className="text-xs text-muted-foreground">{r.method} · {relativeTime(r.created_at)}</p></div><StatusBadge status={r.status} />
        </Link>)}
      </CardContent></Card>
      <Card><CardHeader><CardTitle>Recent evaluations</CardTitle></CardHeader><CardContent>
        {evaluations.error && <ErrorPanel error={evaluations.error} retry={() => evaluations.refetch()} />}
        {!evaluations.isLoading && !evaluations.data?.length && <p className="text-sm text-muted-foreground">No evaluations yet. Compare a base model with your trained model in Eval Lab.</p>}
        {evaluations.data?.slice(0, 5).map(e => <Link key={e.id} to={`/eval?tab=evaluate&result=${e.id}`} className="flex justify-between gap-3 border-b py-3 last:border-0">
          <div className="min-w-0"><p className="truncate font-mono text-xs">{e.model_ref}</p><p className="text-xs text-muted-foreground">{relativeTime(e.created_at)}</p></div>
          <Badge variant={e.detail?.status === "done" ? "success" : e.detail?.status === "failed" ? "danger" : "neutral"}>{e.detail?.status ?? "Unknown"}</Badge>
        </Link>)}
      </CardContent></Card>
    </div>
    <Card><CardHeader><CardTitle>System readiness</CardTitle></CardHeader><CardContent className="space-y-3">
      <div className="flex flex-wrap gap-2"><Badge variant={hw?.gpu_name ? "success" : "warning"}>{hw?.gpu_name ?? "No NVIDIA GPU detected"}</Badge>
        <Badge variant={status.data?.hf_token_set ? "success" : "neutral"}>Private Hub access {status.data?.hf_token_set ? "configured" : "optional"}</Badge>
        <Badge variant={status.data?.judge_configured ? "success" : "neutral"}>Judge {status.data?.judge_configured ? "configured" : "optional"}</Badge></div>
      {failures.map(r => <Link className="block text-xs text-danger" key={r.id} to={`/runs?id=${r.id}`}>Run #{r.id}: {r.error || "Failed — inspect its logs"}</Link>)}
      <Button asChild variant="ghost" size="sm"><Link to="/system">Open System & Diagnostics <ArrowRight /></Link></Button>
    </CardContent></Card>
  </div>;
}
