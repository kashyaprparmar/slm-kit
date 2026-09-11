import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { formFromConfig, toPayload, type RunForm } from "@/lib/runconfig";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { ErrorPanel } from "@/components/ErrorPanel";

export function SettingsRecommendations({ form, onApply }: { form: RunForm; onApply: (f: RunForm) => void }) {
  const [preset, setPreset] = useState("balanced");
  const recommendation = useMutation({
    mutationFn: () => api.trainingSettings(toPayload(form), preset),
    onSuccess: result => { onApply(formFromConfig(result.config)); toast.success("Recommended settings applied. Review the fit estimate."); },
  });
  return <div className="space-y-3 rounded-md border bg-primary/5 p-3">
    <div className="flex flex-wrap gap-2"><Select className="min-w-32 flex-1" aria-label="Training preset" value={preset} onChange={e => setPreset(e.target.value)}>
      <option value="fast">Fast</option><option value="balanced">Balanced</option><option value="best_quality">Best Quality</option><option value="lowest_memory">Lowest Memory</option>
      <option value="custom">Custom / manual</option>
    </Select><Button variant="secondary" disabled={preset === "custom" || recommendation.isPending || !form.base_model.trim()} onClick={() => recommendation.mutate()}>{recommendation.isPending ? "Calculating…" : preset === "custom" ? "Manual settings" : "Recommend Settings"}</Button></div>
    <p className="text-xs text-muted-foreground">Uses model size, dataset length and current hardware. You can adjust every setting afterward.</p>
    {recommendation.error && <ErrorPanel error={recommendation.error} />}
    {recommendation.data && <details><summary className="cursor-pointer text-xs font-medium">Why these settings?</summary><div className="mt-2 space-y-2">
      {Object.entries(recommendation.data.reasons).map(([label, reason]) => <p key={label} className="text-xs text-muted-foreground"><span className="font-medium text-foreground">{label}: </span>{reason}</p>)}
    </div></details>}
  </div>;
}
