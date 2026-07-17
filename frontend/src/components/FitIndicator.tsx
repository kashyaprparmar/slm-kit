import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";
import type { MemoryEstimate } from "@/lib/types";
import { mb } from "@/lib/format";
import { cn } from "@/lib/utils";

const CONFIG = {
  fits: { icon: CheckCircle2, label: "Fits comfortably", tone: "text-success", bar: "bg-success", ring: "border-success/40 bg-success/10" },
  tight: { icon: AlertTriangle, label: "Tight fit", tone: "text-warning", bar: "bg-warning", ring: "border-warning/40 bg-warning/10" },
  wont_fit: { icon: XCircle, label: "Will not fit", tone: "text-danger", bar: "bg-danger", ring: "border-danger/40 bg-danger/10" },
} as const;

export function FitIndicator({ estimate }: { estimate: MemoryEstimate }) {
  const c = CONFIG[estimate.fit];
  const Icon = c.icon;
  const ratio = estimate.budget_mb ? estimate.total_mb / estimate.budget_mb : 0;

  const parts: { label: string; value: number; color: string }[] = [
    { label: "Weights", value: estimate.weights_mb, color: "bg-primary" },
    { label: "Optimizer", value: estimate.optimizer_mb, color: "bg-[hsl(252_60%_50%)]" },
    { label: "Activations", value: estimate.activations_mb, color: "bg-[hsl(200_70%_50%)]" },
    { label: "KV cache", value: estimate.kv_cache_mb, color: "bg-[hsl(170_60%_45%)]" },
    { label: "Overhead", value: estimate.overhead_mb, color: "bg-muted-foreground/60" },
  ].filter((p) => p.value > 0);

  return (
    <div className={cn("space-y-3 rounded-lg border p-4", c.ring)}>
      <div className="flex items-center justify-between">
        <div className={cn("flex items-center gap-2 text-sm font-semibold", c.tone)}>
          <Icon className="size-4.5" />
          {c.label}
        </div>
        <div className="text-right">
          <div className="font-mono text-sm font-semibold tabular-nums">{mb(estimate.total_mb)}</div>
          <div className="text-[10px] text-muted-foreground">of {mb(estimate.budget_mb)} budget</div>
        </div>
      </div>

      {/* Stacked footprint bar vs budget */}
      <div className="h-3 w-full overflow-hidden rounded-full bg-muted">
        <div className="flex h-full" style={{ width: `${Math.min(100, ratio * 100)}%` }}>
          {parts.map((p) => (
            <div
              key={p.label}
              className={p.color}
              style={{ width: `${(p.value / estimate.total_mb) * 100}%` }}
              title={`${p.label}: ${mb(p.value)}`}
            />
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {parts.map((p) => (
          <span key={p.label} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className={cn("size-2 rounded-sm", p.color)} />
            {p.label} {mb(p.value)}
          </span>
        ))}
      </div>

      {estimate.notes.length > 0 && (
        <ul className="space-y-0.5 border-t border-border/60 pt-2 text-[11px] text-muted-foreground">
          {estimate.notes.map((n, i) => (
            <li key={i}>• {n}</li>
          ))}
        </ul>
      )}
      <div className="text-right text-[10px] uppercase tracking-wider text-muted-foreground/60">
        via {estimate.source}
      </div>
    </div>
  );
}
