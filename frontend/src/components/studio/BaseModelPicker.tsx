import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { CURATED_MODELS } from "@/lib/constants";
import { api } from "@/lib/api";
import { Select } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Field } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ErrorPanel } from "@/components/ErrorPanel";
import type { Capability, SupportState } from "@/lib/types";

const CUSTOM = "__custom__";

export function BaseModelPicker({ value = "", onChange, revision, allowRewardModels = false }: { value?: string; onChange: (v: string) => void; revision?: string; allowRewardModels?: boolean }) {
  // Past runs trained in this app — selectable as a base for further training/testing.
  const local = useQuery({ queryKey: ["model-options"], queryFn: api.modelOptions, staleTime: 30_000 });
  const trained = useMemo(
    () =>
      (local.data?.models ?? [])
        .filter((model) => allowRewardModels || model.model_category !== "reward_model")
        .map((model) => ({ repo: model.ref, label: model.label, kind: model.kind })),
    [allowRewardModels, local.data],
  );

  const families = useMemo(() => {
    const groups = new Map<string, typeof CURATED_MODELS>();
    for (const m of CURATED_MODELS) {
      const key = m.family ?? "Other";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(m);
    }
    return groups;
  }, []);

  const safeValue = value ?? "";
  const isKnown = CURATED_MODELS.some((m) => m.repo === safeValue) || trained.some((m) => m.repo === safeValue);
  const [custom, setCustom] = useState(!isKnown && safeValue !== "");
  const [inspectedRef, setInspectedRef] = useState(safeValue.trim());
  useEffect(() => {
    const timer = window.setTimeout(() => setInspectedRef((value ?? "").trim()), custom ? 700 : 50);
    return () => window.clearTimeout(timer);
  }, [value, custom]);

  const inspection = useQuery({
    queryKey: ["model-inspection", inspectedRef, revision],
    queryFn: () => api.inspectModel(inspectedRef, revision),
    enabled: inspectedRef.length > 2,
    staleTime: 5 * 60_000,
    retry: false,
  });

  const preflight = useMutation({
    mutationFn: () =>
      api.preflightModel({
        model_ref: inspectedRef,
        revision: revision || null,
      }),
  });

  useEffect(() => {
    preflight.reset();
  }, [inspectedRef, revision]);

  return (
    <Field label="Base model" hint="Pick a curated 8GB-friendly model, one of your past runs, or paste any HF repo id / local checkpoint path.">
      <Select
        value={custom ? CUSTOM : value}
        onChange={(e) => {
          if (e.target.value === CUSTOM) {
            setCustom(true);
            onChange("");
          } else {
            setCustom(false);
            onChange(e.target.value);
          }
        }}
      >
        {trained.length > 0 && (
          <optgroup label="Your trained models">
            {trained.map((m) => (
              <option key={m.repo} value={m.repo}>{m.label} ({m.kind})</option>
            ))}
          </optgroup>
        )}
        {[...families.entries()].map(([family, models]) => (
          <optgroup key={family} label={family}>
            {models.map((m) => (
              <option key={m.repo} value={m.repo}>
                {m.latest ? "★ " : ""}{m.label} · {m.params_b}B{m.note ? ` — ${m.note}` : ""}
              </option>
            ))}
          </optgroup>
        ))}
        <option value={CUSTOM}>Custom (HF repo id / local path)…</option>
      </Select>
      {custom && (
        <Input
          className="mt-2 font-mono"
          placeholder="e.g. Qwen/Qwen2.5-1.5B-Instruct or /home/you/models/base"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Button type="button" size="sm" variant="outline" disabled={!(value ?? "").trim() || inspection.isFetching} onClick={() => inspection.refetch()}>
          {inspection.isFetching ? "Inspecting…" : "Refresh compatibility"}
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={!(value ?? "").trim() || preflight.isPending}
          onClick={() => preflight.mutate()}
        >
          {preflight.isPending ? "Preflighting…" : "Run runtime preflight"}
        </Button>
        {inspection.data && (
          <>
            <Badge variant={inspection.data.supports_causal_lm ? "success" : "danger"}>{inspection.data.kind ?? "model"}</Badge>
            {inspection.data.parameters ? <span className="text-xs text-muted-foreground">{(inspection.data.parameters / 1e9).toFixed(2)}B params</span> : null}
            {inspection.data.context_length ? <span className="text-xs text-muted-foreground">{inspection.data.context_length.toLocaleString()} token context</span> : null}
          </>
        )}
      </div>
      {inspection.data && (inspection.data.resolved_commit || inspection.data.config_fingerprint || inspection.data.dependencies) && (
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-muted-foreground font-mono">
          {inspection.data.resolved_commit && (
            <span>commit: <strong className="text-foreground">{inspection.data.resolved_commit.slice(0, 10)}</strong></span>
          )}
          {inspection.data.config_fingerprint && (
            <span>cfg: <span className="text-foreground">{inspection.data.config_fingerprint.slice(0, 8)}</span></span>
          )}
          {inspection.data.tokenizer_fingerprint && (
            <span>tok: <span className="text-foreground">{inspection.data.tokenizer_fingerprint.slice(0, 8)}</span></span>
          )}
          {inspection.data.dependencies && (
            <div className="flex flex-wrap items-center gap-1">
              {Object.entries(inspection.data.dependencies).map(([pkg, info]) => (
                <Badge key={pkg} variant={info.installed ? "neutral" : "warning"} className="text-[10px] py-0 px-1">
                  {pkg}{info.version ? `@${info.version.split("+")[0]}` : ""}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}
      {inspection.error && <div className="mt-2"><ErrorPanel error={inspection.error} /></div>}
      {inspection.data?.warnings?.length ? (
        <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-warning">
          {inspection.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
      {inspection.data?.capabilities && <CapabilitySummary capabilities={inspection.data.capabilities} />}
      {preflight.data && (
        <div className="mt-3 rounded-md border bg-muted/20 p-3 text-xs space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-foreground flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" />
              Runtime Preflight Verified
            </span>
            <Badge variant="success">Backend: {preflight.data.backend_verified || preflight.data.recommended_backend || "verified"}</Badge>
          </div>
          <div className="grid grid-cols-2 gap-2 text-muted-foreground pt-1">
            <div>Architecture: <span className="font-mono text-foreground">{preflight.data.architecture || "decoder_only"}</span></div>
            <div>Context: <span className="font-mono text-foreground">{preflight.data.context_length ? `${preflight.data.context_length.toLocaleString()} tokens` : "unknown"}</span></div>
            <div>Chat template: <span className="font-mono text-foreground">{preflight.data.has_chat_template ? "present" : "none"}</span></div>
            {typeof preflight.data.vram_peak_mb === "number" && (
              <div>VRAM probe: <span className="font-mono text-foreground">{preflight.data.vram_peak_mb} MB</span></div>
            )}
          </div>
          {preflight.data.warnings && preflight.data.warnings.length > 0 && (
            <ul className="list-disc pl-4 text-warning space-y-0.5 pt-1">
              {preflight.data.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      {preflight.error && <div className="mt-2"><ErrorPanel error={preflight.error} /></div>}
    </Field>
  );
}

function CapabilitySummary({ capabilities }: { capabilities: NonNullable<import("@/lib/types").ModelInspection["capabilities"]> }) {
  const visible = [
    ["Training", capabilities.training.qlora],
    ["Transformers", capabilities.inference.transformers],
    ["vLLM", capabilities.inference.vllm],
    ["GGUF", capabilities.export.gguf],
  ] as [string, Capability][];
  return <div className="mt-3 rounded-md border bg-muted/20 p-3">
    <div className="mb-2 flex flex-wrap items-center gap-2 text-xs"><span className="font-semibold">{capabilities.family}</span><Badge variant="neutral">{capabilities.architecture_kind.replace(/_/g, " ")}</Badge>{capabilities.is_moe && <Badge variant="warning">MoE</Badge>}{capabilities.is_multimodal && <Badge variant="warning">multimodal</Badge>}</div>
    <div className="grid gap-2 sm:grid-cols-2">{visible.map(([label, capability]) => <div key={label} title={capability.reason} className="flex items-center justify-between gap-2 text-xs"><span className="text-muted-foreground">{label}</span><CapabilityBadge state={capability.state} /></div>)}</div>
  </div>;
}

function CapabilityBadge({ state }: { state: SupportState }) {
  const variant = state === "supported" ? "success" : state === "experimental" || state === "requires_conversion" ? "warning" : "neutral";
  return <Badge variant={variant}>{state.replace(/_/g, " ")}</Badge>;
}
