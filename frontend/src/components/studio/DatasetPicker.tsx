import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import { api } from "@/lib/api";
import { Select } from "@/components/ui/select";
import { Field } from "@/components/ui/field";
import { Spinner } from "@/components/ui/spinner";
import { compactNum } from "@/lib/format";
import type { DatasetKind } from "@/lib/types";

export function DatasetPicker({
  value,
  onChange,
  kinds,
}: {
  value: number | null;
  onChange: (id: number | null) => void;
  kinds: DatasetKind[];
}) {
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
  const options = (datasets.data ?? []).filter((d) => kinds.includes(d.kind));
  const selected = options.find((d) => d.id === value);

  return (
    <Field label="Dataset" hint={selected ? `${compactNum(selected.num_rows)} rows · ~${compactNum(selected.num_tokens_est)} tokens` : undefined}>
      {datasets.isLoading ? (
        <div className="flex h-9 items-center gap-2 text-sm text-muted-foreground"><Spinner /> Loading…</div>
      ) : options.length === 0 ? (
        <div className="flex items-center gap-2 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-warning">
          <AlertTriangle className="size-4 shrink-0" />
          No matching datasets.{" "}
          <Link to="/datasets" className="underline">Add one in the Dataset Manager.</Link>
        </div>
      ) : (
        <Select value={value ?? ""} onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}>
          <option value="">Select a dataset…</option>
          {options.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}{d.validation && !d.validation.ok ? "  ⚠ has errors" : ""}
            </option>
          ))}
        </Select>
      )}
    </Field>
  );
}
