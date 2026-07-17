import { Badge } from "@/components/ui/badge";
import type { RunStatus } from "@/lib/types";

const MAP: Record<RunStatus, { variant: "default" | "success" | "warning" | "danger" | "neutral"; label: string }> = {
  queued: { variant: "warning", label: "Queued" },
  running: { variant: "default", label: "Running" },
  done: { variant: "success", label: "Done" },
  failed: { variant: "danger", label: "Failed" },
  cancelled: { variant: "neutral", label: "Cancelled" },
};

export function StatusBadge({ status }: { status: RunStatus }) {
  const c = MAP[status] ?? MAP.queued;
  return (
    <Badge variant={c.variant} className="gap-1.5">
      {status === "running" && <span className="size-1.5 rounded-full bg-current animate-pulse-dot" />}
      {c.label}
    </Badge>
  );
}
