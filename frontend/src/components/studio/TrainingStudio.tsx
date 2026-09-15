import { useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Rocket, Sparkles, XCircle, AlertTriangle, Info } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Field, NumberField, Collapsible } from "@/components/ui/field";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { MethodSelector } from "@/components/studio/MethodSelector";
import { BaseModelPicker } from "@/components/studio/BaseModelPicker";
import { DatasetPicker } from "@/components/studio/DatasetPicker";
import { ModelAdvisor } from "@/components/studio/ModelAdvisor";
import { FitIndicator } from "@/components/FitIndicator";
import { RunMonitor } from "@/components/RunMonitor";
import { SCHEDULERS, OPTIMIZERS } from "@/lib/constants";
import { defaultForm, toPayload, type RunForm } from "@/lib/runconfig";
import { useWorkflow } from "@/lib/workflow";
import { ErrorPanel } from "@/components/ErrorPanel";
import { SettingsRecommendations } from "./SettingsRecommendations";
import { useDebouncedValue } from "@/lib/hooks";
import type { DatasetKind, TaskType, ValidationReport } from "@/lib/types";

export interface TrainingStudioProps {
  task: TaskType;
  title: string;
  description: string;
  datasetKinds: DatasetKind[];
  datasetLabel?: string;
  defaultOutputName: string;
  note?: string;
}

export function TrainingStudio({
  task, title, description, datasetKinds, defaultOutputName, note,
}: TrainingStudioProps) {
  const qc = useQueryClient();
  const [search] = useSearchParams();
  const [form, setForm] = useWorkflow<RunForm>(`training.${task}`, defaultForm({ task, output_name: defaultOutputName }));
  const [launchedRunId, setLaunchedRunId] = useWorkflow<number | null>(`training.${task}.run`, null, false);
  const [activeProject] = useWorkflow<number | null>("active-project", null);
  const set = <K extends keyof RunForm>(k: K, v: RunForm[K]) => setForm((f) => ({ ...f, [k]: v }));

  useEffect(() => {
    const requested = search.get("model");
    if (requested) setForm((current) => current.base_model === requested ? current : { ...current, base_model: requested });
  }, [search, setForm]);

  const payload = useMemo(() => ({ ...toPayload(form), extra: activeProject ? { project_id: activeProject } : {} }), [form, activeProject]);
  const debounced = useDebouncedValue(JSON.stringify(payload), 450);

  const estimate = useQuery({
    queryKey: ["estimate", debounced],
    queryFn: () => api.estimate(JSON.parse(debounced)),
    enabled: !!form.base_model?.trim(),
  });
  const backends = useQuery({ queryKey: ["training-backends"], queryFn: api.backends, staleTime: 60_000 });
  const selectedBackend = backends.data?.find((backend) => backend.name === form.backend);

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
  const canLaunch =
    !!form.base_model?.trim() && form.dataset_id != null && (report?.ok ?? false) && !launch.isPending && debounced === JSON.stringify(payload) && !estimate.isFetching;

  return (
    <div className="space-y-6">
      <PageHeader title={title} description={description} />

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
            <CardHeader><CardTitle>Method</CardTitle></CardHeader>
            <CardContent>
              <MethodSelector
                value={form.method}
                onChange={(m) => set("method", m)}
                capabilities={selectedBackend?.method_capabilities}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Base model & data</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <Field label="Training engine" hint="Unsloth is optimized for supported GPUs; Transformers is the broad compatibility fallback.">
                <Select value={form.backend} onChange={(e) => set("backend", e.target.value)}>
                  {(backends.data ?? []).filter((item) => item.name !== "scratch").map((item) => (
                    <option
                      key={item.name}
                      value={item.name}
                      disabled={
                        !["supported", "experimental"].includes(item.availability.state)
                        || !["supported", "experimental"].includes(item.task_capabilities[task]?.state ?? "unsupported")
                      }
                    >
                      {item.display_name}
                    </option>
                  ))}
                </Select>
              </Field>
              <BaseModelPicker value={form.base_model} revision={form.revision} onChange={(v) => set("base_model", v)} />
              <DatasetPicker value={form.dataset_id} onChange={(id) => set("dataset_id", id)} kinds={datasetKinds} />
              <Field label="Model revision" hint="Optional Hugging Face branch, tag or commit hash. Pin a commit for reproducibility."><Input value={form.revision ?? ""} onChange={e => set("revision", e.target.value)} placeholder="main" /></Field>
              <Field label="Output name" hint="Used for the local artifact and default HF repo name.">
                <Input value={form.output_name} onChange={(e) => set("output_name", e.target.value)} />
              </Field>
              {note && (
                <div className="flex items-start gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-xs text-muted-foreground">
                  <Info className="mt-0.5 size-3.5 shrink-0 text-primary" /> {note}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Training settings</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <SettingsRecommendations form={form} onApply={setForm} />
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <NumberField label="Epochs" value={form.epochs} step={0.5} min={0} onChange={(v) => set("epochs", v ?? 1)} />
                <NumberField label="Max steps" value={form.max_steps} min={1} hint="Optional step limit; overrides epochs." onChange={(v) => set("max_steps", v)} />
                <NumberField label="Learning Rate" hint="How quickly the model learns. Lower values are safer." value={form.learning_rate} step={0.00001} onChange={(v) => set("learning_rate", v ?? 2e-4)} />
                <NumberField label="Batch Size" hint="Examples processed by the GPU at once." value={form.per_device_batch_size} min={1} onChange={(v) => set("per_device_batch_size", v ?? 2)} />
                <NumberField label="Batch Accumulation" hint="Combines small batches before updating the model." value={form.gradient_accumulation} min={1} onChange={(v) => set("gradient_accumulation", v ?? 4)} />
                <NumberField label="Context Length" hint="Maximum tokens used from each example." value={form.max_seq_length} step={128} min={128} onChange={(v) => set("max_seq_length", v ?? 1024)} />
              </div>

              <Collapsible title="Advanced (LoRA, tokenizer, optimizer, checkpointing)">
                <div className="space-y-5">
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                  {form.method !== "full" && (
                    <>
                      <NumberField label="LoRA rank (r)" value={form.r} min={1} onChange={(v) => set("r", v ?? 16)} />
                      <NumberField label="LoRA alpha" value={form.alpha} min={1} onChange={(v) => set("alpha", v ?? 16)} />
                      <NumberField label="LoRA dropout" value={form.dropout} step={0.01} min={0} max={1} onChange={(v) => set("dropout", v ?? 0)} />
                    </>
                  )}
                  <NumberField label="Random Seed" value={form.seed ?? 42} min={0} hint="Keeps data shuffling and initialization repeatable." onChange={v => set("seed", v ?? 42)} />
                  <label className="text-xs"><input type="checkbox" checked={form.gradient_checkpointing ?? true} onChange={e => set("gradient_checkpointing", e.target.checked)} /> Gradient checkpointing — trades speed for lower memory</label>
                  <NumberField label="Warmup ratio" value={form.warmup_ratio} step={0.01} min={0} max={1} onChange={(v) => set("warmup_ratio", v ?? 0.03)} />
                  <Field label="LR scheduler">
                    <Select value={form.lr_scheduler} onChange={(e) => set("lr_scheduler", e.target.value)}>
                      {SCHEDULERS.map((s) => <option key={s} value={s}>{s}</option>)}
                    </Select>
                  </Field>
                  <Field label="Optimizer">
                    <Select value={form.optimizer} onChange={(e) => set("optimizer", e.target.value)}>
                      {OPTIMIZERS.map((s) => <option key={s} value={s}>{s}</option>)}
                    </Select>
                  </Field>
                  <NumberField label="Save every (steps)" value={form.save_steps} min={1} onChange={(v) => set("save_steps", v ?? 100)} />
                  <NumberField label="Log every (steps)" value={form.logging_steps} min={1} onChange={(v) => set("logging_steps", v ?? 5)} />
                </div>
                {task === "finetune" && <div className="space-y-3 border-t pt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <Field label="Tokenizer" hint="Reuse keeps model embeddings and tokenizer vocabulary aligned."><Select value={form.tokenizer_mode ?? "reuse"} onChange={e => set("tokenizer_mode", e.target.value as RunForm["tokenizer_mode"])}><option value="reuse">Reuse base tokenizer</option></Select></Field>
                    <Field label="Loss policy" hint="Masked policies train only target tokens and require a compatible template."><Select value={form.loss_policy ?? "full_sequence"} onChange={e => set("loss_policy", e.target.value as RunForm["loss_policy"])}><option value="full_sequence">Full sequence</option><option value="completion_only">Completion only</option><option value="assistant_only">Assistant turns only</option></Select></Field>
                  </div>
                  <Field label="Custom chat template (optional)" hint="Jinja template used only when the model tokenizer has no native chat template."><textarea className="min-h-20 w-full rounded-md border bg-background px-3 py-2 font-mono text-xs" value={form.chat_template ?? ""} onChange={e => set("chat_template", e.target.value)} /></Field>
                </div>}
                </div>
              </Collapsible>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6 lg:sticky lg:top-4 lg:self-start">
          <Card>
            <CardHeader className="flex-row items-center justify-between">
              <CardTitle>Predicted footprint</CardTitle>
              <Badge variant="neutral">{form.method.toUpperCase()}</Badge>
            </CardHeader>
            <CardContent className="space-y-3">
              {estimate.error ? <ErrorPanel error={estimate.error} retry={() => estimate.refetch()} /> : estimate.isFetching && !estimate.data ? (
                <Skeleton className="h-40" />
              ) : estimate.data ? (
                <>
                  <FitIndicator estimate={estimate.data.estimate} />
                  {report && <ValidationList report={report} />}
                  <details className="rounded-md border p-3 text-xs">
                    <summary className="cursor-pointer font-medium">Run plan</summary>
                    <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2">
                      {Object.entries(estimate.data.plan).filter(([, value]) => value != null && !Array.isArray(value)).map(([key, value]) => <div key={key}><dt className="text-muted-foreground">{key.replace(/_/g, " ")}</dt><dd className="break-words font-mono">{String(value)}</dd></div>)}
                    </dl>
                  </details>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">Pick a base model to see the fit estimate.</p>
              )}

              <Button className="w-full" size="lg" disabled={!canLaunch} onClick={() => launch.mutate()}>
                <Rocket /> {launch.isPending ? "Launching…" : "Start Training"}
              </Button>
              {form.dataset_id == null && (
                <p className="text-center text-[11px] text-muted-foreground">Select a dataset to enable launch.</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Sparkles className="size-4 text-primary" /> Model Advisor</CardTitle>
            </CardHeader>
            <CardContent>
              <ModelAdvisor
                task={task}
                method={form.method}
                maxSeqLength={form.max_seq_length}
                onAccept={(repo) => { set("base_model", repo); toast.success("Base model set from recommendation"); }}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function ValidationList({ report }: { report: ValidationReport }) {
  if (report.issues.length === 0) return null;
  return (
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
  );
}
