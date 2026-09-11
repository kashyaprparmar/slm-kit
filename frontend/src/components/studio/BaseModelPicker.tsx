import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { CURATED_MODELS } from "@/lib/constants";
import { api } from "@/lib/api";
import { Select } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Field } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ErrorPanel } from "@/components/ErrorPanel";

const CUSTOM = "__custom__";

export function BaseModelPicker({ value, onChange, revision }: { value: string; onChange: (v: string) => void; revision?: string }) {
  // Past runs trained in this app — selectable as a base for further training/testing.
  const local = useQuery({ queryKey: ["model-options"], queryFn: api.modelOptions, staleTime: 30_000 });
  const trained = useMemo(
    () =>
      (local.data?.models ?? [])
        .map((model) => ({ repo: model.ref, label: model.label, kind: model.kind })),
    [local.data],
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

  const isKnown = CURATED_MODELS.some((m) => m.repo === value) || trained.some((m) => m.repo === value);
  const [custom, setCustom] = useState(!isKnown && value !== "");
  const inspection = useMutation({ mutationFn: () => api.inspectModel(value.trim(), revision) });

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
        <Button type="button" size="sm" variant="outline" disabled={!value.trim() || inspection.isPending} onClick={() => inspection.mutate()}>
          {inspection.isPending ? "Inspecting…" : "Inspect compatibility"}
        </Button>
        {inspection.data && (
          <>
            <Badge variant={inspection.data.supports_causal_lm ? "success" : "danger"}>{inspection.data.kind ?? "model"}</Badge>
            {inspection.data.parameters ? <span className="text-xs text-muted-foreground">{(inspection.data.parameters / 1e9).toFixed(2)}B params</span> : null}
            {inspection.data.context_length ? <span className="text-xs text-muted-foreground">{inspection.data.context_length.toLocaleString()} token context</span> : null}
          </>
        )}
      </div>
      {inspection.error && <div className="mt-2"><ErrorPanel error={inspection.error} /></div>}
      {inspection.data?.warnings?.length ? (
        <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-warning">
          {inspection.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
    </Field>
  );
}
