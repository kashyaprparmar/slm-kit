import * as React from "react";
import { cn } from "@/lib/utils";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number; // 0..100
  indeterminate?: boolean;
  tone?: "primary" | "success" | "warning" | "danger";
}

const tones: Record<string, string> = {
  primary: "bg-primary",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
};

export function Progress({ value = 0, indeterminate, tone = "primary", className, ...props }: ProgressProps) {
  return (
    <div
      className={cn("relative h-2 w-full overflow-hidden rounded-full bg-muted", className)}
      role="progressbar"
      aria-valuenow={indeterminate ? undefined : Math.round(value)}
      {...props}
    >
      <div
        className={cn("h-full rounded-full transition-[width] duration-500 ease-out", tones[tone], indeterminate && "w-1/3 animate-[shimmer_1.2s_infinite]")}
        style={indeterminate ? undefined : { width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  );
}
