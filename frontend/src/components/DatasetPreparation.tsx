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
  const [fraction, setFraction] = useState(0.1);
  const [skipInvalid, setSkipInvalid] = useState(false);
  const [shuffle, setShuffle] = useState(true);
  const [dedup, setDedup] = useState(true);
  const prepare = useMutation({
    mutationFn: () => api.prepareDataset(datasetId, { columns: prompt && answer ? { prompt, response: answer } : {},
      test_fraction: fraction, drop_invalid: skipInvalid, shuffle, deduplicate: dedup, seed: 42 }),
    onSuccess: r => { toast.success(`Created ${r.datasets.length} prepared dataset(s); removed ${r.dropped_rows} rows.`); qc.invalidateQueries({ queryKey: ["datasets"] }); },
  });
  return <details className="rounded-md border p-3"><summary className="cursor-pointer text-sm font-medium">Prepare data & create a test split</summary>
    <div className="mt-4 space-y-3"><p className="text-xs text-muted-foreground">Creates new datasets and keeps the original. Auto-detection works for standard text, Q&A and chat schemas.</p>
      <div className="grid grid-cols-2 gap-3">{([["User prompt column", prompt, setPrompt], ["Expected answer column", answer, setAnswer]] as const).map(([label, value, setter]) =>
        <Field key={label} label={label}><Select value={value} onChange={e => setter(e.target.value)}><option value="">Auto-detect</option>{columns.map(c => <option key={c}>{c}</option>)}</Select></Field>)}</div>
      <NumberField label="Test fraction" hint="0.1 keeps 10% for evaluation. 0 creates only a training copy. Uses seed 42." min={0} max={0.5} step={0.05} value={fraction} onChange={v => setFraction(v ?? .1)} />
      <div className="flex flex-wrap gap-3 text-xs">
        <label><input type="checkbox" checked={shuffle} onChange={e => setShuffle(e.target.checked)} /> Shuffle</label>
        <label><input type="checkbox" checked={dedup} onChange={e => setDedup(e.target.checked)} /> Remove exact duplicates</label>
        <label><input type="checkbox" checked={skipInvalid} onChange={e => setSkipInvalid(e.target.checked)} /> Skip invalid rows</label>
      </div>
      {prepare.error && <ErrorPanel error={prepare.error} />}
      <Button variant="secondary" disabled={prepare.isPending || (!!prompt !== !!answer)} onClick={() => prepare.mutate()}>{prepare.isPending ? "Preparing…" : "Prepare Dataset"}</Button>
    </div>
  </details>;
}
