import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  UploadCloud, DownloadCloud, Package, ExternalLink, Boxes, Lock, Globe,
  Cpu, CheckCircle2, XCircle, Terminal,
} from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { useWebSocket } from "@/lib/ws";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { EmptyState } from "@/components/ui/empty-state";
import { LogPanel, type LogLine } from "@/components/LogPanel";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { relativeTime } from "@/lib/format";
import type { ModelArtifact, UnpublishedRun } from "@/lib/types";

export default function Registry() {
  const qc = useQueryClient();
  const local = useQuery({ queryKey: ["registry-local"], queryFn: api.localModels, refetchInterval: 3000 });
  const caps = useQuery({ queryKey: ["registry-caps"], queryFn: api.registryCapabilities });
  const hf = useQuery({ queryKey: ["registry-hf"], queryFn: api.hfModels });
  const invalidate = () => qc.invalidateQueries({ queryKey: ["registry-local"] });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Registry"
        description="Your locally-trained artifacts and your Hugging Face repos, in one place."
      />

      {/* Capability banner */}
      <div className="flex flex-wrap gap-2">
        <Badge variant={caps.data?.hf_token_set ? "success" : "neutral"}>
          {caps.data?.hf_token_set ? <CheckCircle2 className="size-3" /> : <XCircle className="size-3" />}
          HF token {caps.data?.hf_token_set ? "configured" : "not set"}
        </Badge>
        <Badge variant={caps.data?.gguf_available ? "success" : "neutral"}>
          {caps.data?.gguf_available ? <CheckCircle2 className="size-3" /> : <XCircle className="size-3" />}
          GGUF export {caps.data?.gguf_available ? "available" : "needs llama.cpp"}
        </Badge>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Local / trained here */}
        <Card>
          <CardHeader><CardTitle>Trained here</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {local.isLoading ? (
              [0, 1].map((i) => <Skeleton key={i} className="h-16" />)
            ) : (
              <>
                <Section title="Ready to publish">
                  {local.data?.unpublished_runs.length ? (
                    local.data.unpublished_runs.map((r) => (
                      <UnpublishedRow key={r.run_id} run={r} caps={caps.data} onChanged={invalidate} />
                    ))
                  ) : (
                    <p className="px-1 py-2 text-xs text-muted-foreground">No unpublished finished runs.</p>
                  )}
                </Section>

                <Section title="Artifacts">
                  {local.data?.artifacts.length ? (
                    local.data.artifacts.map((a) => <ArtifactRow key={a.id} art={a} />)
                  ) : (
                    <p className="px-1 py-2 text-xs text-muted-foreground">No published or exported artifacts yet.</p>
                  )}
                </Section>

                {!local.data?.unpublished_runs.length && !local.data?.artifacts.length && (
                  <EmptyState icon={Boxes} title="Nothing here yet" description="Finish a training run, then publish it or export it to GGUF." className="border-0" />
                )}
              </>
            )}
          </CardContent>
        </Card>

        {/* Hugging Face */}
        <Card>
          <CardHeader><CardTitle>Hugging Face</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <ImportBox onImported={invalidate} />
            <Section title="Your repos">
              {hf.isLoading ? (
                [0, 1].map((i) => <Skeleton key={i} className="h-12" />)
              ) : !hf.data?.token_set ? (
                <div className="rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-warning">
                  Set <span className="font-mono">SLMKIT_HF_TOKEN</span> to list and publish to your account.
                </div>
              ) : hf.data.models.length ? (
                hf.data.models.map((m) => (
                  <a key={m.repo_id} href={`https://huggingface.co/${m.repo_id}`} target="_blank" rel="noreferrer"
                     className="flex items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm transition-colors hover:bg-muted/40">
                    <span className="flex items-center gap-2 truncate">
                      {m.private ? <Lock className="size-3.5 text-muted-foreground" /> : <Globe className="size-3.5 text-muted-foreground" />}
                      <span className="truncate font-mono text-xs">{m.repo_id}</span>
                    </span>
                    <ExternalLink className="size-3.5 shrink-0 text-muted-foreground" />
                  </a>
                ))
              ) : (
                <p className="px-1 py-2 text-xs text-muted-foreground">No repos found on your account.</p>
              )}
            </Section>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/70">{title}</div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function UnpublishedRow({ run, caps, onChanged }: { run: UnpublishedRun; caps?: { gguf_available: boolean; quant_types: string[] }; onChanged: () => void }) {
  return (
    <div className="rounded-md border px-3 py-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-sm font-medium">{run.name}</span>
        <Badge variant="neutral">{run.method.toUpperCase()}</Badge>
      </div>
      <div className="truncate text-xs text-muted-foreground">{run.base_model || "from scratch"}</div>
      <div className="mt-2 flex flex-wrap gap-2">
        <PublishDialog run={run} onPublished={onChanged} />
        <QuantizeControl run={run} caps={caps} onStarted={onChanged} />
      </div>
    </div>
  );
}

function PublishDialog({ run, onPublished }: { run: UnpublishedRun; onPublished: () => void }) {
  const [open, setOpen] = useState(false);
  const [repoId, setRepoId] = useState(run.name);
  const [priv, setPriv] = useState(true);

  const publish = useMutation({
    mutationFn: () => api.publishModel({ run_id: run.run_id, repo_id: repoId.trim(), private: priv }),
    onSuccess: (res) => { toast.success("Published to Hugging Face"); setOpen(false); onPublished(); window.open(res.hf_repo, "_blank"); },
    onError: (e) => toast.error(e instanceof ApiError ? String(e.detail) : "Publish failed"),
  });

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm"><UploadCloud /> Publish</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">Publish to Hugging Face</DialogTitle>
          <p className="text-sm text-muted-foreground">Auto-generates a model card from this run's metadata.</p>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label>Repo id</Label>
            <Input value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="username/model-name" className="font-mono" />
            <p className="text-[11px] text-muted-foreground">Use <span className="font-mono">username/name</span> to target your namespace.</p>
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="checkbox" checked={priv} onChange={(e) => setPriv(e.target.checked)} className="accent-[hsl(var(--primary))]" />
            {priv ? <Lock className="size-3.5" /> : <Globe className="size-3.5" />} Private repository
          </label>
          <Button className="w-full" disabled={publish.isPending || !repoId.trim()} onClick={() => publish.mutate()}>
            {publish.isPending ? <><Spinner className="text-primary-foreground" /> Uploading…</> : <><UploadCloud /> Publish</>}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function QuantizeControl({ run, caps, onStarted }: { run: UnpublishedRun; caps?: { gguf_available: boolean; quant_types: string[] }; onStarted: () => void }) {
  const [qt, setQt] = useState("q4_k_m");
  const quantize = useMutation({
    mutationFn: () => api.quantize({ run_id: run.run_id, quant_type: qt }),
    onSuccess: () => { toast.message("GGUF export started — watch the Artifacts list."); onStarted(); },
    onError: () => toast.error("Could not start quantization"),
  });
  const available = caps?.gguf_available;
  return (
    <div className="flex items-center gap-1.5">
      <Select value={qt} onChange={(e) => setQt(e.target.value)} className="h-8 w-28 text-xs" disabled={!available}>
        {(caps?.quant_types ?? ["q4_k_m"]).map((t) => <option key={t} value={t}>{t}</option>)}
      </Select>
      <Button size="sm" variant="outline" disabled={!available || quantize.isPending}
        onClick={() => quantize.mutate()}
        title={available ? "Convert to GGUF via llama.cpp" : "Set SLMKIT_LLAMACPP_DIR to enable"}>
        <Cpu /> GGUF
      </Button>
    </div>
  );
}

function ArtifactRow({ art }: { art: ModelArtifact }) {
  const isGguf = art.kind === "gguf";
  const quantizing = art.status === "quantizing";
  const [showLogs, setShowLogs] = useState(quantizing);

  return (
    <div className="rounded-md border px-3 py-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2 truncate">
          <Package className="size-4 shrink-0 text-primary" />
          <span className="truncate text-sm font-medium">{art.name}</span>
        </span>
        {quantizing ? (
          <Badge variant="warning"><Spinner className="size-3 text-warning" /> quantizing</Badge>
        ) : art.status === "failed" ? (
          <Badge variant="danger">failed</Badge>
        ) : isGguf ? (
          <Badge variant="success">GGUF</Badge>
        ) : art.published ? (
          <Badge variant="default">published</Badge>
        ) : (
          <Badge variant="neutral">local</Badge>
        )}
      </div>
      <div className="mt-0.5 truncate text-xs text-muted-foreground">
        {art.base_model || "—"} · {relativeTime(art.created_at)}
      </div>
      {art.error && <div className="mt-1.5 rounded border border-danger/40 bg-danger/10 px-2 py-1 text-[11px] text-danger">{art.error}</div>}
      {art.hf_repo && (
        <a href={art.hf_repo} target="_blank" rel="noreferrer" className="mt-1 inline-flex items-center gap-1 text-[11px] text-primary hover:underline">
          <ExternalLink className="size-3" /> {art.hf_repo.replace("https://huggingface.co/", "")}
        </a>
      )}
      {isGguf && art.local_path && art.status === "ready" && (
        <div className="mt-1 truncate font-mono text-[10px] text-muted-foreground">{art.local_path}</div>
      )}
      {isGguf && (
        <>
          <button
            onClick={() => setShowLogs((s) => !s)}
            className="mt-2 flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
          >
            <Terminal className="size-3" /> {showLogs ? "Hide" : "View"} export logs
          </button>
          {showLogs && <GgufLogViewer artifactId={art.id} live={quantizing} />}
        </>
      )}
    </div>
  );
}

function GgufLogViewer({ artifactId, live }: { artifactId: number; live: boolean }) {
  const [liveLogs, setLiveLogs] = useState<LogLine[]>([]);
  const history = useQuery({
    queryKey: ["gguf-logs", artifactId],
    queryFn: () => api.quantizeLogs(artifactId),
    enabled: !live,
  });
  useWebSocket<{ type: string; message?: string }>(live ? `/ws/gguf/${artifactId}` : null, (ev) => {
    if (ev.type === "log") {
      setLiveLogs((l) => [...l, { ts: Date.now() / 1000, level: "info", message: ev.message ?? "" }]);
    }
  });
  // Avoid an empty flash right as `live` flips false (WS disconnects) before
  // the persisted-log fetch resolves — same pattern used in Playground/EvalHarness.
  const persisted = history.data?.lines ?? [];
  const logs = live ? liveLogs : persisted.length ? persisted : liveLogs;
  return (
    <div className="mt-2">
      <LogPanel lines={logs} live={live} title="GGUF export logs" emptyHint="No log output yet." />
    </div>
  );
}

function ImportBox({ onImported }: { onImported: () => void }) {
  const [repo, setRepo] = useState("");
  const importMut = useMutation({
    mutationFn: () => api.importModel(repo.trim()),
    onSuccess: () => { toast.success("Imported from Hugging Face"); setRepo(""); onImported(); },
    onError: () => toast.error("Import failed"),
  });
  return (
    <div className="space-y-1.5">
      <Label>Import a model from HF</Label>
      <div className="flex gap-2">
        <Input placeholder="username/model-name" value={repo} onChange={(e) => setRepo(e.target.value)} className="font-mono" />
        <Button variant="secondary" disabled={!repo.trim() || importMut.isPending} onClick={() => importMut.mutate()}>
          {importMut.isPending ? <Spinner /> : <DownloadCloud />} Import
        </Button>
      </div>
      <p className="text-[11px] text-muted-foreground">Downloads into your local models dir for further fine-tuning or testing.</p>
    </div>
  );
}
