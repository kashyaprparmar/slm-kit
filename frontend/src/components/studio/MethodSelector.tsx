import { Check, Star } from "lucide-react";
import { METHODS } from "@/lib/constants";
import type { Method } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

export function MethodSelector({ value, onChange }: { value: Method; onChange: (m: Method) => void }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {METHODS.map((m) => {
        const active = m.value === value;
        return (
          <button
            key={m.value}
            type="button"
            onClick={() => onChange(m.value)}
            className={cn(
              "relative rounded-lg border p-3 text-left transition-all",
              active ? "border-primary bg-primary/5 shadow-glow" : "border-border hover:border-primary/40 hover:bg-muted/30",
            )}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold">{m.label}</span>
              {active ? (
                <span className="grid size-4 place-items-center rounded-full bg-primary text-primary-foreground">
                  <Check className="size-3" />
                </span>
              ) : m.recommended ? (
                <Badge variant="default" className="gap-1 px-1.5 py-0"><Star className="size-2.5" /></Badge>
              ) : null}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{m.blurb}</p>
            <p className={cn("mt-1.5 text-[11px] font-medium", active ? "text-primary" : "text-muted-foreground/80")}>
              {m.vramNote}
            </p>
          </button>
        );
      })}
    </div>
  );
}
