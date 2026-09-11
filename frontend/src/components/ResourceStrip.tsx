import { Cpu, HardDrive, MemoryStick, Zap, Gauge } from "lucide-react";
import { useHardware } from "./hardware-context";
import { mb, pct } from "@/lib/format";
import { cn } from "@/lib/utils";

function toneFor(ratio: number): string {
  if (ratio >= 0.9) return "bg-danger";
  if (ratio >= 0.75) return "bg-warning";
  return "bg-primary";
}

function Meter({
  icon: Icon,
  label,
  used,
  total,
  valueText,
  utilPct,
}: {
  icon: typeof Cpu;
  label: string;
  used?: number | null;
  total?: number | null;
  valueText: string;
  utilPct?: number | null;
}) {
  const ratio = total && used != null ? used / total : (utilPct ?? 0) / 100;
  return (
    <div className="flex min-w-[9rem] flex-1 flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          <Icon className="size-3.5" />
          {label}
        </span>
        <span className="font-mono text-xs tabular-nums text-foreground/90">{valueText}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full transition-[width] duration-700 ease-out", toneFor(ratio))}
          style={{ width: `${Math.min(100, Math.max(0, ratio * 100))}%` }}
        />
      </div>
    </div>
  );
}

export function ResourceStrip() {
  const { hw, connected } = useHardware();
  const vramUsed = hw?.vram_total_mb != null && hw?.vram_free_mb != null ? hw.vram_total_mb - hw.vram_free_mb : null;
  const ramUsed = hw?.ram_total_mb != null && hw?.ram_free_mb != null ? hw.ram_total_mb - hw.ram_free_mb : null;

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-lg border bg-surface/60 px-4 py-2.5 backdrop-blur">
      <div className="flex items-center gap-2 pr-2">
        <span className={cn("relative flex size-2", connected ? "text-success" : "text-muted-foreground")}>
          <span className={cn("absolute inline-flex size-2 rounded-full", connected ? "bg-success animate-pulse-dot" : "bg-muted-foreground")} />
        </span>
        <span className="max-w-[10rem] truncate text-xs font-semibold" title={hw?.gpu_name ?? undefined}>
          {hw?.gpu_name ?? (connected ? "No GPU detected" : "Backend offline")}
        </span>
      </div>
      <Meter icon={Zap} label="VRAM" used={vramUsed} total={hw?.vram_total_mb}
        valueText={`${mb(vramUsed)} / ${mb(hw?.vram_total_mb)}`} />
      <Meter icon={Gauge} label="GPU" utilPct={hw?.gpu_util_pct} valueText={pct(hw?.gpu_util_pct)} />
      <Meter icon={MemoryStick} label="RAM" used={ramUsed} total={hw?.ram_total_mb}
        valueText={`${mb(ramUsed)} / ${mb(hw?.ram_total_mb)}`} />
      <Meter icon={Cpu} label="CPU" utilPct={hw?.cpu_util_pct} valueText={pct(hw?.cpu_util_pct)} />
      <Meter icon={HardDrive} label="Disk free" valueText={mb(hw?.disk_free_mb)}
        total={hw?.disk_total_mb} used={hw?.disk_total_mb != null && hw?.disk_free_mb != null ? hw.disk_total_mb - hw.disk_free_mb : null} />
    </div>
  );
}
