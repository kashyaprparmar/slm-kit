import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BarChart, Bar, ResponsiveContainer, XAxis, YAxis, Tooltip as RTooltip, Cell,
} from "recharts";
import {
  Database, Upload, Sparkles, Trash2, CheckCircle2, AlertTriangle, XCircle, FileText, Cloud,
} from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Label } from "@/components/ui/input";
import { Field, NumberField } from "@/components/ui/field";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/ui/empty-state";
import { compactNum, mb } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Dataset, DatasetKind, ValidationReport } from "@/lib/types";
import { DatasetPreparation } from "@/components/DatasetPreparation";
import { ErrorPanel } from "@/components/ErrorPanel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { QualityProfileResult, TokenizerProfileResult } from "@/lib/types";

const KINDS: { value: DatasetKind; label: string }[] = [
  { value: "instruction", label: "Instruction / chat" },
  { value: "domain_corpus", label: "Domain corpus (text)" },
  { value: "pretrain_corpus", label: "Pretraining corpus (text)" },
  { value: "eval", label: "Evaluation set" },
  { value: "preference", label: "Preference pairs" },
  { value: "kto", label: "KTO labels" },
];

const KIND_BADGE: Record<DatasetKind, string> = {
  instruction: "Instruction",
  domain_corpus: "Domain",
  pretrain_corpus: "Pretrain",
  eval: "Eval",
  preference: "Preference",
  kto: "KTO",
};

export default function Datasets() {
  const qc = useQueryClient();
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
  const [selected, setSelected] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  const selectedId = selected ?? datasets.data?.[0]?.id ?? null;

  async function installSamples() {
    setBusy(true);
    try {
      const { installed } = await api.installSamples();
      toast.success(`Installed ${installed.length} sample datasets`);
      qc.invalidateQueries({ queryKey: ["datasets"] });
    } catch {
      toast.error("Could not install samples");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dataset Manager"
        description="Upload, validate, and preview your data before it feeds a run."
        actions={
          <Button variant="secondary" onClick={installSamples} disabled={busy}>
            <Sparkles /> Install sample datasets
          </Button>
        }
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <div className="space-y-6">
          <UploadCard onUploaded={() => qc.invalidateQueries({ queryKey: ["datasets"] })} />
          <HFImportCard onImported={() => qc.invalidateQueries({ queryKey: ["datasets"] })} />

          <Card>
            <CardHeader><CardTitle>Your datasets</CardTitle></CardHeader>
            <CardContent className="space-y-2">
              {datasets.error ? <ErrorPanel error={datasets.error} retry={() => datasets.refetch()} /> : datasets.isLoading ? (
                [0, 1, 2].map((i) => <Skeleton key={i} className="h-14" />)
              ) : datasets.data?.length ? (
                datasets.data.map((d) => (
                  <DatasetRow
                    key={d.id}
                    ds={d}
                    active={d.id === selectedId}
                    onClick={() => setSelected(d.id)}
                    onDeleted={() => {
                      qc.invalidateQueries({ queryKey: ["datasets"] });
                      if (selected === d.id) setSelected(null);
                    }}
                  />
                ))
              ) : (
                <EmptyState
                  icon={Database}
                  title="No datasets yet"
                  description="Install the bundled samples to run the full pipeline on day one, or upload your own."
                  action={<Button size="sm" onClick={installSamples} disabled={busy}><Sparkles /> Install samples</Button>}
                />
              )}
            </CardContent>
          </Card>
        </div>

        <PreviewPanel datasetId={selectedId} />
      </div>
    </div>
  );
}

function UploadCard({ onUploaded }: { onUploaded: () => void }) {
  const [kind, setKind] = useState<DatasetKind>("instruction");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    if (uploading) return;
    setUploading(true);
    try {
      const { validation } = await api.uploadDataset(file, kind);
      if (validation.ok) toast.success(`Uploaded ${file.name} — validation passed`);
      else toast.warning(`Uploaded ${file.name} — validation found issues`);
      onUploaded();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <Card>
      <CardHeader><CardTitle>Upload dataset</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <Label>Dataset type</Label>
          <Select value={kind} onChange={(e) => setKind(e.target.value as DatasetKind)}>
            {KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}
          </Select>
        </div>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files?.[0];
            if (f) handleFile(f);
          }}
          role="button" tabIndex={0} aria-label="Upload dataset file" aria-disabled={uploading}
          onKeyDown={e => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
          onClick={() => { if (!uploading) inputRef.current?.click(); }}
          className={cn(
            "flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed px-6 py-8 text-center transition-colors",
            dragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-muted/30",
          )}
        >
          <Upload className={cn("size-6", dragging ? "text-primary" : "text-muted-foreground")} />
          <div className="text-sm font-medium">{uploading ? "Uploading…" : "Drop a file or click to browse"}</div>
          <div className="text-xs text-muted-foreground">.jsonl · .json · .csv · .txt · .parquet</div>
          <input
            ref={inputRef}
            type="file"
            disabled={uploading}
            className="hidden"
            accept=".jsonl,.json,.csv,.txt,.parquet"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function HFImportCard({ onImported }: { onImported: () => void }) {
  const caps = useQuery({ queryKey: ["hf-dataset-caps"], queryFn: api.hfDatasetCapabilities });
  const [repoId, setRepoId] = useState("");
  const [config, setConfig] = useState("");
  const [split, setSplit] = useState("train");
  const [kind, setKind] = useState<DatasetKind>("instruction");
  const [maxRows, setMaxRows] = useState(1000);
  const [importing, setImporting] = useState(false);

  const available = caps.data?.available ?? true; // assume available until we know otherwise (avoid flash-of-disabled)

  async function doImport() {
    if (!repoId.trim()) return;
    setImporting(true);
    try {
      const res = await api.importFromHF({
        repo_id: repoId.trim(),
        config: config.trim() || undefined,
        split: split.trim() || "train",
        kind,
        max_rows: maxRows,
      });
      if (res.validation.ok) toast.success(`Imported ${res.rows_imported} rows from ${repoId}`);
      else toast.warning(`Imported ${res.rows_imported} rows — validation found issues`);
      setRepoId("");
      onImported();
    } catch (e) {
      const msg = e instanceof ApiError ? String(e.detail) : "Import failed";
      toast.error(msg);
    } finally {
      setImporting(false);
    }
  }

  return (
    <Card>
      <CardHeader><CardTitle className="flex items-center gap-2"><Cloud className="size-4 text-primary" /> Import from Hugging Face</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        {!available && (
          <div className="rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-warning">
            The <span className="font-mono">datasets</span> package isn't installed on the backend.
            Run <span className="font-mono">pip install datasets</span> to enable this.
          </div>
        )}
        <div className="space-y-1.5">
          <Label>Dataset repo id</Label>
          <Input
            className="font-mono"
            placeholder="e.g. tatsu-lab/alpaca"
            value={repoId}
            onChange={(e) => setRepoId(e.target.value)}
            disabled={!available}
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label>Config (optional)</Label>
            <Input placeholder="default" value={config} onChange={(e) => setConfig(e.target.value)} disabled={!available} />
          </div>
          <div className="space-y-1.5">
            <Label>Split</Label>
            <Input value={split} onChange={(e) => setSplit(e.target.value)} disabled={!available} />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label>Dataset type</Label>
            <Select value={kind} onChange={(e) => setKind(e.target.value as DatasetKind)} disabled={!available}>
              {KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}
            </Select>
          </div>
          <NumberField label="Max rows" value={maxRows} min={1} max={50000} step={100} onChange={(v) => setMaxRows(v ?? 1000)} />
        </div>
        <Button className="w-full" onClick={doImport} disabled={!available || importing || !repoId.trim()}>
          <Cloud /> {importing ? "Importing…" : "Import dataset"}
        </Button>
        <p className="text-[11px] text-muted-foreground">
          Streams the first N rows (doesn't download the full dataset) and runs it through the same validation as an upload.
        </p>
      </CardContent>
    </Card>
  );
}

function DatasetRow({ ds, active, onClick, onDeleted }: {
  ds: Dataset; active: boolean; onClick: () => void; onDeleted: () => void;
}) {
  const ok = ds.validation?.ok ?? true;
  return (
    <div
      onClick={onClick}
      className={cn(
        "group flex cursor-pointer items-center gap-3 rounded-md border px-3 py-2.5 transition-colors",
        active ? "border-primary/50 bg-primary/5" : "border-border hover:bg-muted/40",
      )}
    >
      <div className={cn("grid size-8 shrink-0 place-items-center rounded-md", ok ? "bg-primary/10 text-primary" : "bg-danger/10 text-danger")}>
        <FileText className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium">{ds.name}</span>
          {ds.is_sample && <Badge variant="neutral" className="shrink-0">sample</Badge>}
        </div>
        <div className="truncate text-xs text-muted-foreground">
          {KIND_BADGE[ds.kind]} · {ds.fmt} · {compactNum(ds.num_rows)} rows · ~{compactNum(ds.num_tokens_est)} tok
        </div>
      </div>
      {!ok && <AlertTriangle className="size-4 shrink-0 text-danger" />}
      {!ds.is_sample && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (window.confirm(`Delete dataset "${ds.name}"? Its uploaded file will be removed.`))
              api.deleteDataset(ds.id).then(() => { toast.success("Dataset deleted"); onDeleted(); }).catch((err: Error) => toast.error(err.message));
          }}
          className="shrink-0 text-muted-foreground opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
          aria-label={`Delete ${ds.name}`}
        >
          <Trash2 className="size-4" />
        </button>
      )}
    </div>
  );
}

function PreviewPanel({ datasetId }: { datasetId: number | null }) {
  const preview = useQuery({
    queryKey: ["preview", datasetId],
    queryFn: () => api.previewDataset(datasetId!),
    enabled: datasetId != null,
  });

  if (datasetId == null) {
    return (
      <Card>
        <CardContent className="p-0">
          <EmptyState icon={FileText} title="Select a dataset" description="Pick a dataset to see its validation report, token distribution, and sample rows." className="border-0" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader><CardTitle>Data Lab</CardTitle></CardHeader>
      <CardContent>
        {preview.isLoading ? (
          <>
            <Skeleton className="h-20" />
            <Skeleton className="h-40" />
          </>
        ) : preview.data ? (
          <Tabs defaultValue="schema">
            <TabsList className="w-full justify-start overflow-x-auto">
              <TabsTrigger value="schema">Schema</TabsTrigger>
              <TabsTrigger value="prepare">Prepare & lineage</TabsTrigger>
              <TabsTrigger value="tokenizer">Tokenizer</TabsTrigger>
              <TabsTrigger value="quality">Quality</TabsTrigger>
            </TabsList>
            <TabsContent value="schema" className="space-y-5">
              <ValidationSummary report={preview.data.validation} stats={preview.data.stats} />
              <TokenHistogram values={preview.data.stats.token_histogram} />
              <SampleRows rows={preview.data.stats.sample_rows} />
              <p className="text-xs text-muted-foreground">Columns: {preview.data.stats.columns?.join(", ") || "text"} · Duplicates: {preview.data.stats.duplicate_rows ?? 0} · Invalid: {preview.data.stats.invalid_rows ?? 0} · Empty: {preview.data.stats.empty_rows ?? 0}</p>
            </TabsContent>
            <TabsContent value="prepare" className="space-y-5">
              <DatasetPreparation key={datasetId} datasetId={datasetId} columns={preview.data.stats.columns ?? []} />
              <LineagePanel datasetId={datasetId} />
            </TabsContent>
            <TabsContent value="tokenizer"><TokenizerPanel datasetId={datasetId} /></TabsContent>
            <TabsContent value="quality"><QualityPanel datasetId={datasetId} /></TabsContent>
          </Tabs>
        ) : (
          <ErrorPanel error={preview.error ?? "Could not load dataset preview."} retry={() => preview.refetch()} />
        )}
      </CardContent>
    </Card>
  );
}

function useCurrentVersion(datasetId: number) {
  const versions = useQuery({
    queryKey: ["dataset-versions", datasetId],
    queryFn: () => api.datasetVersions(datasetId),
  });
  return { ...versions, current: versions.data?.[0] };
}

function LineagePanel({ datasetId }: { datasetId: number }) {
  const qc = useQueryClient();
  const versions = useCurrentVersion(datasetId);
  const recipes = useQuery({ queryKey: ["dataset-recipes", datasetId], queryFn: () => api.datasetRecipes(datasetId) });
  const rerun = useMutation({
    mutationFn: api.rerunDatasetRecipe,
    onSuccess: (result) => {
      toast.success(result.reused ? `Recipe #${result.recipe_id} reproduced the existing outputs.` : `Recipe #${result.recipe_id} published new outputs.`);
      qc.invalidateQueries({ queryKey: ["datasets"] });
      qc.invalidateQueries({ queryKey: ["dataset-versions", datasetId] });
      qc.invalidateQueries({ queryKey: ["dataset-recipes", datasetId] });
    },
  });
  if (versions.error || recipes.error || rerun.error) return <ErrorPanel error={versions.error ?? recipes.error ?? rerun.error} />;
  return <div className="space-y-4 border-t pt-4">
    <div>
      <h3 className="text-sm font-medium">Immutable versions</h3>
      <p className="text-xs text-muted-foreground">Files are fingerprinted at import or publication. Preparation never overwrites them.</p>
    </div>
    {versions.isLoading ? <Skeleton className="h-16" /> : <div className="space-y-2">{versions.data?.map(version =>
      <div key={version.id} className="flex items-center justify-between rounded-md border p-2 text-xs">
        <div><span className="font-medium">v{version.id}</span> <Badge variant="neutral">{version.split}</Badge><div className="mt-1 font-mono text-[10px] text-muted-foreground">{version.fingerprint.slice(0, 16)}…</div></div>
        <div className="text-right text-muted-foreground">{compactNum(version.num_rows)} rows<br />{new Date(version.created_at).toLocaleString()}</div>
      </div>)}</div>}
    {recipes.data?.length ? <div className="space-y-2"><h3 className="text-sm font-medium">Recipes</h3>{recipes.data.map(recipe =>
      <div key={recipe.id} className="flex items-center justify-between rounded-md border p-2 text-xs">
        <div><span className="font-medium">#{recipe.id} {recipe.name}</span><div className="text-muted-foreground">source v{recipe.source_version_id} · {recipe.fingerprint.slice(0, 12)}…</div></div>
        <Button size="sm" variant="outline" disabled={rerun.isPending} onClick={() => rerun.mutate(recipe.id)}>Replay</Button>
      </div>)}</div> : null}
  </div>;
}

function TokenizerPanel({ datasetId }: { datasetId: number }) {
  const versions = useCurrentVersion(datasetId);
  const [modelRef, setModelRef] = useState("HuggingFaceTB/SmolLM2-135M-Instruct");
  const [revision, setRevision] = useState("");
  const [maxLength, setMaxLength] = useState(1024);
  const [sampleLimit, setSampleLimit] = useState(250);
  const [lossPolicy, setLossPolicy] = useState("full_sequence");
  const [chatTemplate, setChatTemplate] = useState("");
  const profile = useMutation({
    mutationFn: () => api.tokenizerProfile(versions.current!.id, {
      model_ref: modelRef.trim(), revision: revision.trim() || undefined,
      max_length: maxLength, sample_limit: sampleLimit, loss_policy: lossPolicy,
      chat_template: chatTemplate.trim() || undefined,
    }),
  });
  const result: TokenizerProfileResult | undefined = profile.data?.result;
  if (versions.error) return <ErrorPanel error={versions.error} retry={() => versions.refetch()} />;
  return <div className="space-y-4">
    <p className="text-xs text-muted-foreground">Loads only the selected tokenizer in an isolated worker. The cached profile is bound to the dataset fingerprint, resolved tokenizer revision, template, and loss policy.</p>
    <Field label="Tokenizer model"><Input className="font-mono" value={modelRef} onChange={e => setModelRef(e.target.value)} /></Field>
    <div className="grid grid-cols-2 gap-3">
      <Field label="Revision (optional)"><Input value={revision} onChange={e => setRevision(e.target.value)} placeholder="commit, tag, or branch" /></Field>
      <Field label="Loss policy"><Select value={lossPolicy} onChange={e => setLossPolicy(e.target.value)}><option value="full_sequence">Full sequence</option><option value="completion_only">Completion only</option><option value="assistant_only">Assistant turns only</option></Select></Field>
      <NumberField label="Maximum length" value={maxLength} min={16} max={1048576} onChange={value => setMaxLength(value ?? 1024)} />
      <NumberField label="Rows to sample" value={sampleLimit} min={1} max={5000} onChange={value => setSampleLimit(value ?? 250)} />
    </div>
    <Field label="Custom chat template (optional)" hint="Jinja template used when the tokenizer has no native template."><textarea className="min-h-20 w-full rounded-md border bg-background px-3 py-2 font-mono text-xs" value={chatTemplate} onChange={e => setChatTemplate(e.target.value)} /></Field>
    {profile.error && <ErrorPanel error={profile.error} />}
    <Button disabled={!versions.current || !modelRef.trim() || profile.isPending} onClick={() => profile.mutate()}>{profile.isPending ? "Profiling…" : "Profile tokenizer"}</Button>
    {result && <TokenizerResults result={result} cached={profile.data?.cached ?? false} />}
  </div>;
}

function TokenizerResults({ result, cached }: { result: TokenizerProfileResult; cached: boolean }) {
  return <div className="space-y-4 border-t pt-4">
    <div className="flex flex-wrap items-center gap-2"><Badge variant="success">{cached ? "cached" : "profiled"}</Badge><span className="font-mono text-xs">{result.tokenizer.class} · {compactNum(result.tokenizer.vocab_size)} tokens</span>{result.tokenizer.chat_template && <Badge variant="neutral">native chat template</Badge>}</div>
    <div className="grid grid-cols-3 gap-2">
      <MiniStat label="Mean" value={String(result.stats.mean)} /><MiniStat label="P95" value={String(result.stats.p95)} /><MiniStat label="Max" value={String(result.stats.max)} />
      <MiniStat label="Truncated" value={`${result.stats.truncation_percentage}%`} /><MiniStat label="Padding" value={`${result.stats.padding_overhead_percentage}%`} /><MiniStat label="Packing" value={`${result.stats.packing_efficiency_percentage}%`} />
    </div>
    {result.errors.length > 0 && <div className="rounded-md border border-warning/40 bg-warning/10 p-3 text-xs text-warning">{result.errors.length} sampled row(s) could not be rendered. First: line {result.errors[0].line}: {result.errors[0].message}</div>}
    {result.previews[0] && <div className="space-y-2"><Label>Rendered preview and loss mask</Label><pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-md border bg-background/40 p-3 text-[11px]">{result.previews[0].rendered}</pre><p className="break-all font-mono text-[10px] text-muted-foreground">mask: {result.previews[0].loss_mask.slice(0, 128).join("")}</p></div>}
  </div>;
}

function QualityPanel({ datasetId }: { datasetId: number }) {
  const versions = useCurrentVersion(datasetId);
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
  const [compareDatasetId, setCompareDatasetId] = useState<number | null>(null);
  const comparisonVersions = useQuery({
    queryKey: ["dataset-versions", compareDatasetId],
    queryFn: () => api.datasetVersions(compareDatasetId!),
    enabled: compareDatasetId != null,
  });
  const [maxRows, setMaxRows] = useState(100000);
  const [threshold, setThreshold] = useState(.9);
  const quality = useMutation({ mutationFn: () => api.qualityProfile(versions.current!.id, { max_rows: maxRows, near_duplicate_threshold: threshold, compare_version_ids: comparisonVersions.data?.[0] ? [comparisonVersions.data[0].id] : [] }) });
  const result: QualityProfileResult | undefined = quality.data?.result;
  if (versions.error) return <ErrorPanel error={versions.error} retry={() => versions.refetch()} />;
  return <div className="space-y-4">
    <p className="text-xs text-muted-foreground">Runs read-only diagnostics for duplicates, conflicting answers, Unicode anomalies, markup, boilerplate, possible PII, and schema problems. No source rows are changed.</p>
    <div className="grid grid-cols-2 gap-3"><NumberField label="Maximum rows" value={maxRows} min={1} max={1000000} onChange={value => setMaxRows(value ?? 100000)} /><NumberField label="Near-duplicate threshold" value={threshold} min={.7} max={1} step={.01} onChange={value => setThreshold(value ?? .9)} /></div>
    <Field label="Leakage comparison (optional)" hint="Reports normalized rows also present in the selected dataset version."><Select value={compareDatasetId ?? ""} onChange={event => setCompareDatasetId(event.target.value ? Number(event.target.value) : null)}><option value="">Do not compare</option>{datasets.data?.filter(item => item.id !== datasetId).map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</Select></Field>
    {quality.error && <ErrorPanel error={quality.error} />}
    <Button disabled={!versions.current || quality.isPending || (compareDatasetId != null && !comparisonVersions.data?.[0])} onClick={() => quality.mutate()}>{quality.isPending ? "Scanning…" : "Run quality scan"}</Button>
    {result && <div className="space-y-3 border-t pt-4">
      <div className="grid grid-cols-3 gap-2"><MiniStat label="Sampled" value={compactNum(result.sampled_rows)} /><MiniStat label="Valid" value={compactNum(result.valid_rows)} /><MiniStat label="Warnings" value={compactNum(result.warnings.reduce((sum, item) => sum + item.count, 0))} /></div>
      {result.warnings.length === 0 ? <div className="rounded-md border border-success/40 bg-success/10 p-3 text-sm text-success">No issues found in the scanned rows.</div> : result.warnings.map(item => <details key={item.code} className="rounded-md border p-3"><summary className="cursor-pointer text-sm font-medium">{item.code.replace(/_/g, " ")} <Badge variant="warning">{item.count}</Badge></summary>{item.examples.map((example, index) => <pre key={index} className="mt-2 whitespace-pre-wrap border-t pt-2 text-[11px] text-muted-foreground">Line {example.line}: {example.text}</pre>)}</details>)}
    </div>}
  </div>;
}

function ValidationSummary({ report, stats }: { report: ValidationReport; stats: { num_rows: number; num_tokens_est: number; size_bytes: number } }) {
  const errors = report.issues.filter((i) => i.level === "error");
  const warnings = report.issues.filter((i) => i.level === "warning");
  const infos = report.issues.filter((i) => i.level === "info");

  return (
    <div className="space-y-3">
      <div className={cn(
        "flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium",
        report.ok ? "border-success/40 bg-success/10 text-success" : "border-danger/40 bg-danger/10 text-danger",
      )}>
        {report.ok ? <CheckCircle2 className="size-4" /> : <XCircle className="size-4" />}
        {report.ok ? "Validation passed — ready to use" : `${errors.length} error${errors.length === 1 ? "" : "s"} — fix before launching`}
      </div>

      <div className="grid grid-cols-3 gap-2">
        <MiniStat label="Rows" value={compactNum(stats.num_rows)} />
        <MiniStat label="Est. tokens" value={`~${compactNum(stats.num_tokens_est)}`} />
        <MiniStat label="Size" value={mb(stats.size_bytes / (1024 * 1024))} />
      </div>

      {(errors.length > 0 || warnings.length > 0 || infos.length > 0) && (
        <ul className="space-y-1.5">
          {[...errors, ...warnings, ...infos].slice(0, 12).map((issue, i) => (
            <li key={i} className="flex items-start gap-2 text-xs">
              {issue.level === "error" && <XCircle className="mt-0.5 size-3.5 shrink-0 text-danger" />}
              {issue.level === "warning" && <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-warning" />}
              {issue.level === "info" && <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />}
              <span className="text-muted-foreground">
                {issue.line != null && <span className="font-mono text-foreground/70">L{issue.line}: </span>}
                {issue.message}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-background/40 p-2.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="font-mono text-sm font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function TokenHistogram({ values }: { values: number[] }) {
  const bins = useMemo(() => {
    if (!values.length) return [];
    const max = Math.max(...values);
    const n = 12;
    const size = Math.max(1, Math.ceil(max / n));
    const buckets = Array.from({ length: n }, (_, i) => ({ label: `${i * size}`, count: 0 }));
    for (const v of values) {
      const idx = Math.min(n - 1, Math.floor(v / size));
      buckets[idx].count++;
    }
    return buckets;
  }, [values]);

  if (!bins.length) return null;

  return (
    <div className="space-y-2">
      <Label>Token length distribution (per row)</Label>
      <div className="h-40 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
            <XAxis dataKey="label" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickLine={false} axisLine={false} allowDecimals={false} />
            <RTooltip
              cursor={{ fill: "hsl(var(--muted) / 0.4)" }}
              contentStyle={{ background: "hsl(var(--elevated))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
            />
            <Bar dataKey="count" radius={[3, 3, 0, 0]}>
              {bins.map((_, i) => <Cell key={i} fill="hsl(var(--primary))" />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function SampleRows({ rows }: { rows: unknown[] }) {
  if (!rows.length) return null;
  return (
    <div className="space-y-2">
      <Label>Sample rows</Label>
      <div className="max-h-64 space-y-2 overflow-y-auto rounded-md border bg-background/40 p-3">
        {rows.slice(0, 6).map((row, i) => (
          <pre key={i} className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-muted-foreground border-b border-border/40 pb-2 last:border-0">
            {typeof row === "string" ? row : JSON.stringify(row, null, 2)}
          </pre>
        ))}
      </div>
    </div>
  );
}
