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
import type { Capability, DatasetKind, TaskType, ValidationReport } from "@/lib/types";

function capabilityAvailable(capability?: Capability): boolean {
  return capability != null && ["supported", "experimental"].includes(capability.state);
}

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
  const selectedBackend = backends.data?.find((backend) => backend.name === form.backend)
    ?? (form.backend === "auto" ? backends.data?.find((backend) => backend.name === "transformers") : undefined);
  const optionAvailable = (group: Record<string, Capability> | undefined, name: string) => capabilityAvailable(group?.[name]);
  const quantizationAvailable = (name: string) => name === "none" || capabilityAvailable(selectedBackend?.quantization_capabilities[name]?.support);
  const optimizationAvailable = (name: string) => capabilityAvailable(selectedBackend?.optimizations?.[name]?.support);

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
                onChange={(m) => setForm((current) => ({
                  ...current,
                  method: m,
                  reference_strategy: current.reference_strategy === "adapter_disabled" && !["lora", "qlora", "dora"].includes(m)
                    ? "base_model"
                    : current.reference_strategy,
                  optimizer: ["full", "freeze"].includes(m) && current.optimizer.includes("8bit")
                    ? "adamw_torch"
                    : current.optimizer,
                }))}
                capabilities={selectedBackend?.method_capabilities}
                allowedMethods={task === "alignment" ? ["full", "lora", "qlora", "dora"] : undefined}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Base model & data</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <Field label="Training engine" hint="Unsloth is optimized for supported GPUs; Transformers is the broad compatibility fallback.">
                <Select value={form.backend} onChange={(e) => set("backend", e.target.value)}>
                  <option value="auto">Auto — best compatible backend</option>
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

          {task === "alignment" && (
            <Card>
              <CardHeader><CardTitle>Preference objective</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                  <Field label="Objective">
                    <Select value={form.alignment_objective} onChange={(e) => {
                      const objective = e.target.value as RunForm["alignment_objective"];
                      setForm((current) => ({
                        ...current,
                        alignment_objective: objective,
                        reference_strategy: ["orpo", "simpo", "reward_model"].includes(objective)
                          ? "none"
                          : current.reference_strategy === "none" ? "base_model" : current.reference_strategy,
                      }));
                    }}>
                      <option value="dpo" disabled={!optionAvailable(selectedBackend?.optional_features, "objective:dpo")}>DPO</option><option value="ipo" disabled={!optionAvailable(selectedBackend?.optional_features, "objective:ipo")}>IPO</option>
                      <option value="orpo" disabled={!optionAvailable(selectedBackend?.optional_features, "objective:orpo")}>ORPO</option><option value="simpo" disabled={!optionAvailable(selectedBackend?.optional_features, "objective:simpo")}>SimPO</option><option value="kto" disabled={!optionAvailable(selectedBackend?.optional_features, "objective:kto")}>KTO</option><option value="reward_model" disabled={!optionAvailable(selectedBackend?.optional_features, "objective:reward_model")}>Reward model</option>
                    </Select>
                  </Field>
                  {form.alignment_objective !== "reward_model" && <NumberField label="Beta" value={form.alignment_beta} min={0.0001} step={0.05} onChange={(value) => set("alignment_beta", value ?? 0.1)} />}
                  {form.alignment_objective === "dpo" && <><Field label="DPO loss"><Select value={form.alignment_dpo_loss_variant} onChange={(e) => set("alignment_dpo_loss_variant", e.target.value as RunForm["alignment_dpo_loss_variant"])}><option value="sigmoid" disabled={!optionAvailable(selectedBackend?.optional_features, "dpo_loss:sigmoid")}>Sigmoid</option><option value="hinge" disabled={!optionAvailable(selectedBackend?.optional_features, "dpo_loss:hinge")}>Hinge</option><option value="robust" disabled={!optionAvailable(selectedBackend?.optional_features, "dpo_loss:robust")}>Robust</option><option value="exo_pair" disabled={!optionAvailable(selectedBackend?.optional_features, "dpo_loss:exo_pair")}>EXO pair</option></Select></Field>{form.alignment_dpo_loss_variant !== "hinge" && <NumberField label="Label smoothing" value={form.alignment_label_smoothing} min={0} max={0.49} step={0.01} onChange={(value) => set("alignment_label_smoothing", value ?? 0)} />}</>}
                  {form.alignment_objective === "simpo" && <NumberField label="SimPO gamma" value={form.alignment_simpo_gamma} min={0} step={0.1} onChange={(value) => set("alignment_simpo_gamma", value ?? 0.5)} />}
                  {form.alignment_objective === "kto" && <><NumberField label="Desirable weight" value={form.alignment_desirable_weight} min={0.01} step={0.1} onChange={(value) => set("alignment_desirable_weight", value ?? 1)} /><NumberField label="Undesirable weight" value={form.alignment_undesirable_weight} min={0.01} step={0.1} onChange={(value) => set("alignment_undesirable_weight", value ?? 1)} /></>}
                  {!(["orpo", "simpo", "reward_model"].includes(form.alignment_objective)) && <Field label="Reference strategy"><Select value={form.reference_strategy} onChange={(e) => set("reference_strategy", e.target.value as RunForm["reference_strategy"])}><option value="base_model" disabled={!optionAvailable(selectedBackend?.optional_features, "reference:base_model")}>Base model</option><option value="separate_model" disabled={!optionAvailable(selectedBackend?.optional_features, "reference:separate_model")}>Separate HF model</option>{["lora", "qlora", "dora"].includes(form.method) && <option value="adapter_disabled" disabled={!optionAvailable(selectedBackend?.optional_features, "reference:adapter_disabled")}>Current model, adapter disabled</option>}</Select></Field>}
                  {form.reference_strategy === "separate_model" && <><Field label="Reference model"><Input value={form.reference_model} onChange={(e) => set("reference_model", e.target.value)} placeholder="organization/model" /></Field><Field label="Reference revision"><Input value={form.reference_revision} onChange={(e) => set("reference_revision", e.target.value)} placeholder="commit or tag" /></Field></>}
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader><CardTitle>Training settings</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <SettingsRecommendations form={form} onApply={setForm} />
              <p className="text-xs text-muted-foreground"><strong>Recommended:</strong> start with the selected method and defaults. Advanced and expert options appear below only when the chosen backend can validate them.</p>
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
                  {["lora", "qlora", "dora"].includes(form.method) && (
                    <>
                      <NumberField label="LoRA rank (r)" value={form.r} min={1} onChange={(v) => set("r", v ?? 16)} />
                      <NumberField label="LoRA alpha" value={form.alpha} min={1} onChange={(v) => set("alpha", v ?? 16)} />
                      <NumberField label="LoRA dropout" value={form.dropout} step={0.01} min={0} max={1} onChange={(v) => set("dropout", v ?? 0)} />
                      <Field label="LoRA targets"><Select value={form.lora_target_strategy} onChange={e => set("lora_target_strategy", e.target.value as RunForm["lora_target_strategy"])}><option value="auto">Auto inspect model</option><option value="all_linear">All linear layers</option><option value="attention">Attention only</option><option value="mlp">MLP only</option><option value="custom">Custom</option></Select></Field>
                      {form.lora_target_strategy === "custom" && <Field label="Target modules" hint="Comma-separated module names; validated against the loaded architecture."><Input value={form.lora_target_modules} onChange={e => set("lora_target_modules", e.target.value)} /></Field>}
                      <label className="text-xs"><input type="checkbox" checked={form.use_rslora} disabled={!optimizationAvailable("rslora")} onChange={e => set("use_rslora", e.target.checked)} /> rsLoRA</label>
                      <Field label="LoRA initialization" hint={selectedBackend?.optimizations?.pissa?.impact?.performance as string | undefined}><Select value={form.lora_init_method} onChange={e => set("lora_init_method", e.target.value as RunForm["lora_init_method"])}><option value="standard">Standard</option><option value="pissa" disabled={!optimizationAvailable("pissa") || form.method === "qlora" || form.quantization_mode !== "none"}>PiSSA</option></Select></Field>
                      {optimizationAvailable("lora_plus") && <NumberField label="LoRA+ LR ratio" hint="Requires AdamW (PyTorch). Leave blank to disable." value={form.lora_plus_lr_ratio} min={1} step={1} onChange={v => set("lora_plus_lr_ratio", v)} />}
                    </>
                  )}
                  {form.method === "freeze" && <>
                    <NumberField label="Last transformer layers" value={form.freeze_last_n_layers} min={0} onChange={v => set("freeze_last_n_layers", v ?? 1)} />
                    <Field label="Selected modules" hint="Optional comma-separated module names."><Input value={form.freeze_modules} onChange={e => set("freeze_modules", e.target.value)} /></Field>
                    <label className="text-xs"><input type="checkbox" checked={form.freeze_embeddings} onChange={e => set("freeze_embeddings", e.target.checked)} /> Train embeddings</label>
                    <label className="text-xs"><input type="checkbox" checked={form.freeze_lm_head} onChange={e => set("freeze_lm_head", e.target.checked)} /> Train LM head</label>
                    <label className="text-xs"><input type="checkbox" checked={form.freeze_norms} onChange={e => set("freeze_norms", e.target.checked)} /> Train normalization layers</label>
                  </>}
                  <NumberField label="Random Seed" value={form.seed ?? 42} min={0} hint="Keeps data shuffling and initialization repeatable." onChange={v => set("seed", v ?? 42)} />
                  <Field label="Precision"><Select value={form.precision} onChange={e => set("precision", e.target.value as RunForm["precision"])}><option value="auto" disabled={!optionAvailable(selectedBackend?.precision, "auto")}>Auto</option><option value="bf16" disabled={!optionAvailable(selectedBackend?.precision, "bf16")}>BF16</option><option value="fp16" disabled={!optionAvailable(selectedBackend?.precision, "fp16")}>FP16</option><option value="fp32" disabled={!optionAvailable(selectedBackend?.precision, "fp32")}>FP32</option></Select></Field>
                  <Field label="Attention"><Select value={form.attention} onChange={e => set("attention", e.target.value as RunForm["attention"])}><option value="auto" disabled={!optionAvailable(selectedBackend?.attention, "auto")}>Auto</option><option value="sdpa" disabled={!optionAvailable(selectedBackend?.attention, "sdpa")}>SDPA</option><option value="flash_attention_2" disabled={!optionAvailable(selectedBackend?.attention, "flash_attention_2")}>FlashAttention 2</option><option value="eager" disabled={!optionAvailable(selectedBackend?.attention, "eager")}>Eager</option></Select></Field>
                  <Field label="Gradient checkpointing"><Select value={form.checkpointing_mode} onChange={e => set("checkpointing_mode", e.target.value as RunForm["checkpointing_mode"])}><option value="auto" disabled={!optionAvailable(selectedBackend?.gradient_checkpointing, "auto")}>Auto</option><option value="off" disabled={!optionAvailable(selectedBackend?.gradient_checkpointing, "off")}>Off</option><option value="standard" disabled={!optionAvailable(selectedBackend?.gradient_checkpointing, "standard")}>Standard</option><option value="non_reentrant" disabled={!optionAvailable(selectedBackend?.gradient_checkpointing, "non_reentrant")}>Non-reentrant</option><option value="backend_optimized" disabled={!optionAvailable(selectedBackend?.gradient_checkpointing, "backend_optimized")}>Backend optimized</option></Select></Field>
                  {["lora", "qlora", "dora"].includes(form.method) && <>
                    <Field label="Training quantization"><Select value={form.quantization_mode} onChange={e => set("quantization_mode", e.target.value as RunForm["quantization_mode"])}><option value="none">None</option><option value="nf4" disabled={!quantizationAvailable("nf4")}>4-bit NF4</option><option value="fp4" disabled={!quantizationAvailable("fp4")}>4-bit FP4</option><option value="int8" disabled={!quantizationAvailable("int8")}>8-bit</option></Select></Field>
                    {form.quantization_mode !== "none" && <Field label="Quant compute"><Select value={form.quantization_compute_dtype} onChange={e => set("quantization_compute_dtype", e.target.value as RunForm["quantization_compute_dtype"])}><option value="auto">Auto</option><option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp32">FP32</option></Select></Field>}
                    {["nf4", "fp4"].includes(form.quantization_mode) && <><Field label="Quant storage"><Select value={form.quantization_storage_dtype} onChange={e => set("quantization_storage_dtype", e.target.value as RunForm["quantization_storage_dtype"])}><option value="auto">Auto</option><option value="uint8">UInt8</option><option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp32">FP32</option></Select></Field><label className="text-xs"><input type="checkbox" checked={form.quantization_double} onChange={e => set("quantization_double", e.target.checked)} /> Nested/double quantization</label></>}
                  </>}
                  <label className="text-xs"><input type="checkbox" checked={form.rope_enabled} onChange={e => set("rope_enabled", e.target.checked)} /> Explicit RoPE extension</label>
                  {form.rope_enabled && <><NumberField label="RoPE factor" value={form.rope_factor} min={1.01} max={64} step={0.25} onChange={v => set("rope_factor", v ?? 2)} /><Field label="RoPE type"><Select value={form.rope_type} onChange={e => set("rope_type", e.target.value as RunForm["rope_type"])}><option value="linear" disabled={!optionAvailable(selectedBackend?.rope, "linear")}>Linear</option><option value="dynamic" disabled={!optionAvailable(selectedBackend?.rope, "dynamic")}>Dynamic</option><option value="yarn" disabled={!optionAvailable(selectedBackend?.rope, "yarn")}>YaRN</option></Select></Field></>}
                  <NumberField label="Warmup ratio" value={form.warmup_ratio} step={0.01} min={0} max={1} onChange={(v) => set("warmup_ratio", v ?? 0.03)} />
                  <Field label="LR scheduler">
                    <Select value={form.lr_scheduler} onChange={(e) => set("lr_scheduler", e.target.value)}>
                      {SCHEDULERS.map((s) => <option key={s} value={s}>{s}</option>)}
                    </Select>
                  </Field>
                  <Field label="Optimizer">
                    <Select value={form.optimizer} onChange={(e) => set("optimizer", e.target.value)}>
                      {OPTIMIZERS.map((s) => <option key={s} value={s} disabled={s.includes("8bit") && !optionAvailable(selectedBackend?.optional_features, "bitsandbytes_optimizer")}>{s}</option>)}
                    </Select>
                  </Field>
                  {optimizationAvailable("neftune") && <NumberField label="NEFTune noise" hint="Leave blank to disable embedding noise." value={form.neftune_noise_alpha} min={0.000001} step={0.5} onChange={v => set("neftune_noise_alpha", v)} />}
                  <NumberField label="Save every (steps)" value={form.save_steps} min={1} onChange={(v) => set("save_steps", v ?? 100)} />
                  <NumberField label="Log every (steps)" value={form.logging_steps} min={1} onChange={(v) => set("logging_steps", v ?? 5)} />
                </div>
                {["finetune", "alignment"].includes(task) && <div className="space-y-3 border-t pt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <Field label="Tokenizer" hint="Reuse keeps model embeddings and tokenizer vocabulary aligned."><Select value={form.tokenizer_mode ?? "reuse"} onChange={e => set("tokenizer_mode", e.target.value as RunForm["tokenizer_mode"])}><option value="reuse">Reuse base tokenizer</option></Select></Field>
                    <Field label="Loss policy" hint="Masked policies train only target tokens and require a compatible template."><Select value={form.loss_policy ?? "full_sequence"} onChange={e => set("loss_policy", e.target.value as RunForm["loss_policy"])}><option value="full_sequence">Full sequence</option><option value="completion_only">Completion only</option><option value="assistant_only">Assistant turns only</option></Select></Field>
                  </div>
                  <Field label="Custom chat template (optional)" hint="Explicit Jinja override for this run. Leave blank to use the tokenizer's native template."><textarea className="min-h-20 w-full rounded-md border bg-background px-3 py-2 font-mono text-xs" value={form.chat_template ?? ""} onChange={e => set("chat_template", e.target.value)} /></Field>
                </div>}
                </div>
              </Collapsible>
              <Collapsible title="Expert optimization">
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                  <Field label="Optimizer strategy" hint="Only verified optional optimizers can be selected."><Select value={form.optimizer_strategy} onChange={e => set("optimizer_strategy", e.target.value as RunForm["optimizer_strategy"])}><option value="default">Default optimizer</option>{Object.values(selectedBackend?.optimizations ?? {}).filter((option) => option.category === "optimizer").map((option) => <option key={option.id} value={option.id} disabled={!capabilityAvailable(option.support)}>{option.label}</option>)}</Select></Field>
                  {form.optimizer_strategy !== "default" && <><Field label="Optimizer target modules" hint="Comma-separated module names."><Input value={form.optimizer_target_modules} onChange={e => set("optimizer_target_modules", e.target.value)} /></Field><NumberField label="Low-rank rank" value={form.optimizer_low_rank_rank} min={1} onChange={v => set("optimizer_low_rank_rank", v ?? 128)} /><NumberField label="Projection update interval" value={form.optimizer_update_interval} min={1} onChange={v => set("optimizer_update_interval", v ?? 200)} /></>}
                  <label className="text-xs"><input type="checkbox" checked={form.use_liger} disabled={!optimizationAvailable("liger")} onChange={e => set("use_liger", e.target.checked)} /> Liger kernels</label>
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
                      {Object.entries(estimate.data.plan).filter(([, value]) => value != null && typeof value !== "object").map(([key, value]) => <div key={key}><dt className="text-muted-foreground">{key.replace(/_/g, " ")}</dt><dd className="break-words font-mono">{String(value)}</dd></div>)}
                    </dl>
                    {typeof estimate.data.plan.backend_resolution === "object" && estimate.data.plan.backend_resolution !== null && <div className="mt-3 border-t pt-3"><p className="font-medium">Backend decision</p><pre className="mt-1 whitespace-pre-wrap text-[11px]">{JSON.stringify(estimate.data.plan.backend_resolution, null, 2)}</pre></div>}
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
