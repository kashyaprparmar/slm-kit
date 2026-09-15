import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { Field, NumberField } from "@/components/ui/field";
import { ErrorPanel } from "@/components/ErrorPanel";
export function DatasetPreparation({ datasetId, columns }: { datasetId: number; columns: string[] }) {
  const qc = useQueryClient();
  const [prompt, setPrompt] = useState("");
  const [answer, setAnswer] = useState("");
  const [testFraction, setTestFraction] = useState(0.1);
  const [validationFraction, setValidationFraction] = useState(0.1);
  const [seed, setSeed] = useState(42);
  const [skipInvalid, setSkipInvalid] = useState(false);
  const [shuffle, setShuffle] = useState(true);
  const [dedup, setDedup] = useState(true);
  const prepare = useMutation({
    mutationFn: () => api.prepareDataset(datasetId, { columns: prompt && answer ? { prompt, response: answer } : {},
      test_fraction: testFraction, validation_fraction: validationFraction,
      drop_invalid: skipInvalid, shuffle, deduplicate: dedup, seed }),
    onSuccess: r => {
      toast.success(r.reused ? `Reused recipe #${r.recipe_id}; outputs are unchanged.` : `Created ${r.datasets.length} immutable split(s); removed ${r.dropped_rows} rows.`);
      qc.invalidateQueries({ queryKey: ["datasets"] });
      qc.invalidateQueries({ queryKey: ["dataset-versions", datasetId] });
      qc.invalidateQueries({ queryKey: ["dataset-recipes", datasetId] });
    },
  });
  return <div className="space-y-3"><p className="text-xs text-muted-foreground">Creates versioned train, validation, and test outputs atomically while preserving the source. Repeating an identical recipe reuses its immutable outputs.</p>
      <div className="grid grid-cols-2 gap-3">{([["User prompt column", prompt, setPrompt], ["Expected answer column", answer, setAnswer]] as const).map(([label, value, setter]) =>
        <Field key={label} label={label}><Select value={value} onChange={e => setter(e.target.value)}><option value="">Auto-detect</option>{columns.map(c => <option key={c}>{c}</option>)}</Select></Field>)}</div>
      <div className="grid grid-cols-3 gap-3">
        <NumberField label="Validation fraction" min={0} max={0.4} step={0.05} value={validationFraction} onChange={v => setValidationFraction(v ?? .1)} />
        <NumberField label="Test fraction" min={0} max={0.4} step={0.05} value={testFraction} onChange={v => setTestFraction(v ?? .1)} />
        <NumberField label="Seed" hint="Controls deterministic ordering." min={0} value={seed} onChange={v => setSeed(v ?? 42)} />
      </div>
      <div className="flex flex-wrap gap-3 text-xs">
        <label><input type="checkbox" checked={shuffle} onChange={e => setShuffle(e.target.checked)} /> Shuffle</label>
        <label><input type="checkbox" checked={dedup} onChange={e => setDedup(e.target.checked)} /> Remove exact duplicates</label>
        <label><input type="checkbox" checked={skipInvalid} onChange={e => setSkipInvalid(e.target.checked)} /> Skip invalid rows</label>
      </div>
      {prepare.error && <ErrorPanel error={prepare.error} />}
      <Button variant="secondary" disabled={prepare.isPending || (!!prompt !== !!answer) || testFraction + validationFraction > .8} onClick={() => prepare.mutate()}>{prepare.isPending ? "Preparing…" : "Publish immutable splits"}</Button>
      {testFraction + validationFraction > .8 && <p className="text-xs text-danger">Validation and test fractions must leave at least 20% for training.</p>}
    </div>;
}
