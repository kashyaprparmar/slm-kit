import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CURATED_MODELS } from "@/lib/constants";
import { api } from "@/lib/api";
import { Select } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Field } from "@/components/ui/field";

const CUSTOM = "__custom__";

export function BaseModelPicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  // Past runs trained in this app — selectable as a base for further training/testing.
  const local = useQuery({ queryKey: ["registry-local"], queryFn: api.localModels, staleTime: 30_000 });
  const trained = useMemo(
    () =>
      (local.data?.artifacts ?? [])
        .filter((a) => a.local_path && a.status === "ready")
        .map((a) => ({ repo: a.local_path as string, label: a.name, kind: a.kind })),
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
    </Field>
  );
}
