import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ErrorPanel } from "@/components/ErrorPanel";
export default function System() {
  const health = useQuery({ queryKey: ["diagnostics"], queryFn: api.diagnostics, staleTime: 30000 });
  const events = useQuery({ queryKey: ["activity"], queryFn: api.activity, refetchInterval: 5000 });
  const services = useQuery({ queryKey: ["services"], queryFn: api.services, refetchInterval: 5000 });
  return <div className="space-y-6"><PageHeader title="System & Diagnostics" description="Local configuration checks and request timings. No external observability service required."
    actions={<Button variant="secondary" disabled={health.isFetching} onClick={() => health.refetch()}>{health.isFetching ? "Checking…" : "Refresh Checks"}</Button>} />
    {health.error && <ErrorPanel error={health.error} retry={() => health.refetch()} />}
    <Card><CardHeader><CardTitle>Runtime services</CardTitle></CardHeader><CardContent className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <Service name="API" status={services.data?.api.status === "ready" ? "ready" : "error"} detail={services.data?.api.status} />
      <Service name="Database" status={services.data?.database.status === "ready" ? "ready" : "error"} detail={services.data?.database.error || services.data?.database.engine} />
      <Service name="Training worker" status={services.data?.training_worker.status === "busy" ? "busy" : services.data?.training_worker.status === "ready" ? "ready" : "error"} detail={services.data?.training_worker.run_id ? `Run #${services.data.training_worker.run_id}` : services.data?.training_worker.status} />
      {Object.entries(services.data?.providers ?? {}).map(([name, provider]) => <Service key={name} name={name} status={provider.active || provider.running ? "busy" : provider.available ? "ready" : "unavailable"} detail={provider.active || provider.running ? "Active" : provider.guidance || provider.error || "Stopped"} />)}
    </CardContent></Card>
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{health.data?.checks.map(c => <Card key={c.name}><CardContent className="space-y-2 p-4">
      <div className="flex justify-between gap-2"><span className="font-medium">{c.name}</span><Badge variant={c.status === "ready" ? "success" : c.status === "optional" ? "neutral" : "warning"}>{c.status}</Badge></div>
      <p className="break-all font-mono text-xs">{c.detail}</p><p className="text-xs text-muted-foreground">{c.guidance}</p>
    </CardContent></Card>)}</div>
    <Card><CardHeader><CardTitle>Recent requests & failures</CardTitle></CardHeader><CardContent className="space-y-2">
      {events.error && <ErrorPanel error={events.error} retry={() => events.refetch()} />}
      {events.data?.events.filter(e => e.duration_ms != null || e.level === "ERROR").slice(-40).reverse().map(e =>
        <details key={e.id} className="rounded-md border p-3 text-xs"><summary className="cursor-pointer"><span className={e.level === "ERROR" ? "text-danger" : "text-foreground"}>{e.message}</span> <span className="text-muted-foreground">{e.duration_ms != null ? `${e.duration_ms} ms` : ""}</span></summary>
          <pre className="mt-2 overflow-auto text-[11px] text-muted-foreground">{JSON.stringify(e, null, 2)}</pre></details>)}
      {!events.data?.events.length && <p className="text-sm text-muted-foreground">Activity appears as you use the application.</p>}
    </CardContent></Card>
  </div>;
}

function Service({ name, status, detail }: { name: string; status: "ready" | "busy" | "error" | "unavailable"; detail?: string }) {
  const variant = status === "ready" ? "success" : status === "busy" ? "warning" : status === "error" ? "danger" : "neutral";
  return <div className="rounded-md border p-3"><div className="mb-1 flex items-center justify-between gap-2"><span className="font-medium capitalize">{name}</span><Badge variant={variant}>{status}</Badge></div><p className="line-clamp-3 text-xs text-muted-foreground">{detail || "Checking…"}</p></div>;
}
