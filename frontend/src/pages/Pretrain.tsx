import { useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Rocket, XCircle, AlertTriangle, Info, Check } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, NumberField, Collapsible } from "@/components/ui/field";
import { Skeleton } from "@/components/ui/skeleton";
import { DatasetPicker } from "@/components/studio/DatasetPicker";
import { FitIndicator } from "@/components/FitIndicator";
import { RunMonitor } from "@/components/RunMonitor";
import { ARCH_PRESETS, type ArchPreset } from "@/lib/constants";
import { defaultPretrainForm, toPretrainPayload, type PretrainForm } from "@/lib/runconfig";
import { useWorkflow } from "@/lib/workflow";
import { ErrorPanel } from "@/components/ErrorPanel";
import { useDebouncedValue } from "@/lib/hooks";
import { compactNum } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ValidationReport } from "@/lib/types";

export default function Pretrain() {
  const qc = useQueryClient();
  const [form, setForm] = useWorkflow<PretrainForm>("pretrain.form", defaultPretrainForm());
  const [presetKey, setPresetKey] = useWorkflow("pretrain.preset", "tiny");
  const [launchedRunId, setLaunchedRunId] = useWorkflow<number | null>("pretrain.run", null, false);
  const set = <K extends keyof PretrainForm>(k: K, v: PretrainForm[K]) => setForm((f) => ({ ...f, [k]: v }));

  function applyPreset(p: ArchPreset) {
    setPresetKey(p.key);
    setForm((f) => ({ ...f, vocab_size: p.vocab_size, n_layers: p.n_layers, n_heads: p.n_heads, n_embd: p.n_embd, block_size: p.block_size }));
  }

  const approxParams = useMemo(
    () => form.vocab_size * form.n_embd + 12 * form.n_layers * form.n_embd ** 2,
    [form.vocab_size, form.n_embd, form.n_layers],
  );

  const payload = useMemo(() => toPretrainPayload(form), [form]);
  const debounced = useDebouncedValue(JSON.stringify(payload), 450);
  const estimate = useQuery({ queryKey: ["estimate", debounced], queryFn: () => api.estimate(JSON.parse(debounced)) });

  const launch = useMutation({
    mutationFn: () => api.createRun(payload),
    onSuccess: (res) => {
      setLaunchedRunId(res.run_id);
      toast.success(`Launched run #${res.run_id}`);
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
    onError: (err) => {
      if (err instanceof ApiError && err.status === 422) toast.error("Config failed validation — see the fit panel.");
      else toast.error(err instanceof Error ? err.message : "Could not launch run");
    },
  });

  const report: ValidationReport | undefined = estimate.data?.validation;
  const headOk = form.n_embd % form.n_heads === 0;
  const canLaunch = form.dataset_id != null && headOk && (report?.ok ?? false) && !launch.isPending && debounced === JSON.stringify(payload) && !estimate.isFetching;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Pretraining Studio"
        description="Train a small GPT from scratch with a custom tokenizer trained on your corpus."
      />

      {launchedRunId != null && (
        <div className="space-y-2">
          <RunMonitor runId={launchedRunId} />
          <div className="flex justify-end">
            <Button variant="ghost" size="sm" onClick={() => setLaunchedRunId(null)}>Configure another run</Button>
          </div>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <div className="space-y-6">
          <Card>
            <CardHeader><CardTitle>Architecture preset</CardTitle></CardHeader>
            <CardContent className="grid grid-cols-3 gap-3">
              {ARCH_PRESETS.map((p) => {
                const active = p.key === presetKey;
                return (
                  <button
                    key={p.key}
                    type="button"
                    onClick={() => applyPreset(p)}
                    className={cn(
                      "relative rounded-lg border p-3 text-left transition-all",
                      active ? "border-primary bg-primary/5 shadow-glow" : "border-border hover:border-primary/40 hover:bg-muted/30",
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold">{p.label}</span>
                      {active && <span className="grid size-4 place-items-center rounded-full bg-primary text-primary-foreground"><Check className="size-3" /></span>}
                    </div>
                    <p className="mt-1 text-[11px] text-muted-foreground">{p.blurb}</p>
                  </button>
                );
              })}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model architecture</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <NumberField label="Layers" value={form.n_layers} min={1} onChange={(v) => set("n_layers", v ?? 6)} />
                <NumberField label="Heads" value={form.n_heads} min={1} onChange={(v) => set("n_heads", v ?? 6)} />
                <NumberField label="Embedding dim" value={form.n_embd} step={64} min={64} onChange={(v) => set("n_embd", v ?? 384)} />
                <NumberField label="Context length" value={form.block_size} step={64} min={64} onChange={(v) => set("block_size", v ?? 256)} />
                <NumberField label="Vocab size" value={form.vocab_size} step={1024} min={512} onChange={(v) => set("vocab_size", v ?? 8192)} hint="tokenizer trained on corpus" />
                <NumberField label="Dropout" value={form.dropout} step={0.05} min={0} max={1} onChange={(v) => set("dropout", v ?? 0.1)} />
              </div>
              {!headOk && (
                <div className="flex items-center gap-2 rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger">
                  <XCircle className="size-4 shrink-0" /> Embedding dim ({form.n_embd}) must be divisible by heads ({form.n_heads}).
                </div>
              )}
              <div className="rounded-md border bg-background/40 px-3 py-2 text-xs text-muted-foreground">
                ≈ <span className="font-mono font-semibold text-foreground">{compactNum(approxParams)}</span> parameters
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Corpus & training</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <DatasetPicker value={form.dataset_id} onChange={(id) => set("dataset_id", id)} kinds={["pretrain_corpus", "domain_corpus"]} />
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <NumberField label="Max steps" value={form.max_steps} min={1} step={100} onChange={(v) => set("max_steps", v ?? 1000)} />
                <NumberField label="Batch size" value={form.per_device_batch_size} min={1} onChange={(v) => set("per_device_batch_size", v ?? 8)} />
                <NumberField label="Learning rate" value={form.learning_rate} step={0.00001} onChange={(v) => set("learning_rate", v ?? 3e-4)} />
              </div>
              <Collapsible title="Advanced (checkpoint / logging cadence)">
                <div className="grid grid-cols-2 gap-4">
                  <NumberField label="Save every (steps)" value={form.save_steps} min={1} onChange={(v) => set("save_steps", v ?? 200)} />
                  <NumberField label="Log every (steps)" value={form.logging_steps} min={1} onChange={(v) => set("logging_steps", v ?? 10)} />
                </div>
              </Collapsible>
              <Field label="Output name">
                <Input value={form.output_name} onChange={(e) => set("output_name", e.target.value)} />
              </Field>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6 lg:sticky lg:top-4 lg:self-start">
          <Card>
            <CardHeader><CardTitle>Predicted footprint</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              {estimate.error ? <ErrorPanel error={estimate.error} retry={() => estimate.refetch()} /> : estimate.isFetching && !estimate.data ? (
                <Skeleton className="h-40" />
              ) : estimate.data ? (
                <>
                  <FitIndicator estimate={estimate.data.estimate} />
                  {report && report.issues.length > 0 && (
                    <ul className="space-y-1.5">
                      {report.issues.map((issue, i) => (
                        <li key={i} className="flex items-start gap-2 text-xs">
                          {issue.level === "error" && <XCircle className="mt-0.5 size-3.5 shrink-0 text-danger" />}
                          {issue.level === "warning" && <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-warning" />}
                          {issue.level === "info" && <Info className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />}
                          <span className="text-muted-foreground">{issue.message}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              ) : (
                <p className="text-sm text-muted-foreground">Adjust the architecture to see the fit estimate.</p>
              )}
              <Button className="w-full" size="lg" disabled={!canLaunch} onClick={() => launch.mutate()}>
                <Rocket /> {launch.isPending ? "Launching…" : "Launch pretraining"}
              </Button>
              {form.dataset_id == null && <p className="text-center text-[11px] text-muted-foreground">Select a corpus to enable launch.</p>}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
