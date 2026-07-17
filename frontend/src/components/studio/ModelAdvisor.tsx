import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Sparkles, TrendingUp, Zap, Scale, Check } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { mb } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AdvisorRec, Method, TaskType } from "@/lib/types";

const PRIORITIES = [
  { value: "fastest", label: "Fastest iteration", icon: Zap },
  { value: "balanced", label: "Balanced", icon: Scale },
  { value: "best_quality", label: "Best quality", icon: TrendingUp },
];

const FIT_TONE = { fits: "success", tight: "warning", wont_fit: "danger" } as const;

export function ModelAdvisor({
  task,
  method,
  maxSeqLength,
  onAccept,
}: {
  task: TaskType;
  method: Method;
  maxSeqLength: number;
  onAccept: (repo: string) => void;
}) {
  const [priority, setPriority] = useState("balanced");
  const rec = useMutation({
    mutationFn: () => api.recommend({ task, method, priority, max_seq_length: maxSeqLength }),
  });

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Select value={priority} onChange={(e) => setPriority(e.target.value)} className="h-8 text-xs">
          {PRIORITIES.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
        </Select>
        <Button size="sm" onClick={() => rec.mutate()} disabled={rec.isPending}>
          {rec.isPending ? <Spinner className="text-primary-foreground" /> : <Sparkles />}
          Recommend
        </Button>
      </div>

      {rec.data && (
        <div className="space-y-2">
          <p className="text-[11px] text-muted-foreground">
            Ranked via {rec.data.source} for your {mb(rec.data.hardware.vram_total_mb)} GPU.
          </p>
          {rec.data.recommendations.slice(0, 4).map((r, i) => (
            <RecRow key={r.repo} rec={r} best={i === 0} onAccept={() => onAccept(r.repo)} />
          ))}
        </div>
      )}
    </div>
  );
}

function RecRow({ rec, best, onAccept }: { rec: AdvisorRec; best: boolean; onAccept: () => void }) {
  return (
    <div className={cn("rounded-md border p-3", best ? "border-primary/50 bg-primary/5" : "border-border")}>
      <div className="flex items-center justify-between gap-2">
        <span className="truncate font-mono text-xs font-medium">{rec.repo.split("/").pop()}</span>
        <div className="flex items-center gap-1.5">
          {best && <Badge variant="default">Top pick</Badge>}
          <Badge variant={FIT_TONE[rec.fit]}>{mb(rec.estimate.total_mb)}</Badge>
        </div>
      </div>
      <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{rec.reasoning}</p>
      <Button size="sm" variant="secondary" className="mt-2 h-7 w-full text-xs" onClick={onAccept}>
        <Check /> Use this model
      </Button>
    </div>
  );
}
