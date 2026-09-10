import { useEffect, useMemo, useRef, useState } from "react";
import {
  Terminal, Search, Copy, Download, ArrowDownToLine, Maximize2, Minimize2, X,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { LogLine } from "@/lib/types";

export type { LogLine };

const LEVELS = ["info", "warning", "error"] as const;

function fmtTime(ts?: number | null): string {
  if (ts == null) return "";
  const d = new Date(ts * 1000);
  if (isNaN(d.getTime())) return "";
  return d.toLocaleTimeString(undefined, { hour12: false });
}

/**
 * Reusable, comprehensive log viewer — used by RunMonitor (Pretraining / Domain
 * Adaptation / Fine-Tuning / Run History), the Eval Lab (Playground + Evaluate),
 * and the Model Registry (GGUF export). Shows timestamps, supports search +
 * level filtering, copy/download, and a pausable auto-scroll so you can read
 * history without fighting new lines.
 */
export function LogPanel({
  lines,
  live,
  title = "Logs",
  emptyHint = "Waiting for output…",
  className,
}: {
  lines: LogLine[];
  live?: boolean;
  title?: string;
  emptyHint?: string;
  className?: string;
}) {
  const [query, setQuery] = useState("");
  const [levelFilter, setLevelFilter] = useState<Set<string>>(new Set(LEVELS));
  const [autoScroll, setAutoScroll] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return lines.filter((l) => {
      const lvl = LEVELS.includes(l.level as typeof LEVELS[number]) ? l.level : "info";
      if (!levelFilter.has(lvl)) return false;
      if (q && !l.message.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [lines, query, levelFilter]);

  // Auto-scroll to bottom on new lines, unless the user scrolled up to read history.
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filtered.length, autoScroll]);

  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    setAutoScroll(nearBottom);
  }

  function toggleLevel(lvl: string) {
    setLevelFilter((prev) => {
      const next = new Set(prev);
      if (next.has(lvl)) next.delete(lvl); else next.add(lvl);
      return next;
    });
  }

  function copyAll() {
    const text = filtered.map((l) => `${fmtTime(l.ts)}${l.ts ? "  " : ""}[${l.level}] ${l.message}`).join("\n");
    navigator.clipboard.writeText(text);
    toast.success(`Copied ${filtered.length} log lines`);
  }

  function download() {
    const text = filtered.map((l) => `${fmtTime(l.ts)}${l.ts ? "  " : ""}[${l.level}] ${l.message}`).join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.toLowerCase().replace(/\s+/g, "-")}-${Date.now()}.log`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const body = (
    <div className={cn("flex min-h-0 flex-col gap-2", expanded && "h-full")}>
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          <Terminal className="size-3.5" /> {title}
          {live && <span className="size-1.5 rounded-full bg-success animate-pulse-dot" />}
        </div>
        <Badge variant="neutral" className="text-[10px]">{filtered.length}{filtered.length !== lines.length ? ` / ${lines.length}` : ""} lines</Badge>
        <div className="ml-auto flex items-center gap-1">
          {LEVELS.map((lvl) => (
            <button
              key={lvl}
              onClick={() => toggleLevel(lvl)}
              className={cn(
                "rounded-full border px-2 py-0.5 text-[10px] font-medium capitalize transition-colors",
                levelFilter.has(lvl)
                  ? lvl === "error" ? "border-danger/40 bg-danger/10 text-danger"
                    : lvl === "warning" ? "border-warning/40 bg-warning/10 text-warning"
                    : "border-primary/40 bg-primary/10 text-primary"
                  : "border-border text-muted-foreground/50",
              )}
            >
              {lvl}
            </button>
          ))}
        </div>
      </div>

      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter log lines…"
          className="h-7 pl-8 text-xs"
        />
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className={cn(
          "dark overflow-y-auto rounded-md border bg-[hsl(222_32%_4%)] p-3 font-mono text-[11px] leading-relaxed text-foreground",
          expanded ? "flex-1" : "h-64",
        )}
      >
        {filtered.length === 0 ? (
          <span className="text-muted-foreground">{lines.length === 0 ? emptyHint : "No lines match the current filter."}</span>
        ) : (
          filtered.map((l, i) => (
            <div key={i} className="flex gap-2 whitespace-pre-wrap break-words">
              {l.ts != null && <span className="shrink-0 text-muted-foreground/50">{fmtTime(l.ts)}</span>}
              <span className={cn(
                "shrink-0",
                l.level === "error" ? "text-danger" : l.level === "warning" ? "text-warning" : "text-muted-foreground/70",
              )}>
                [{l.level}]
              </span>
              <span className={cn(l.level === "error" ? "text-danger" : "text-foreground/90")}>{l.message}</span>
            </div>
          ))
        )}
      </div>

      <div className="flex items-center justify-between">
        <button
          onClick={() => setAutoScroll((s) => !s)}
          className={cn(
            "flex items-center gap-1 text-[11px] transition-colors",
            autoScroll ? "text-primary" : "text-muted-foreground hover:text-foreground",
          )}
        >
          <ArrowDownToLine className="size-3" /> {autoScroll ? "Auto-scrolling" : "Scroll paused — click to resume"}
        </button>
        <div className="flex items-center gap-1">
          <Button size="sm" variant="ghost" className="h-6 px-2 text-[11px]" onClick={copyAll} disabled={filtered.length === 0}>
            <Copy className="size-3" /> Copy
          </Button>
          <Button size="sm" variant="ghost" className="h-6 px-2 text-[11px]" onClick={download} disabled={filtered.length === 0}>
            <Download className="size-3" /> Download
          </Button>
          <Button size="sm" variant="ghost" className="h-6 px-2 text-[11px]" onClick={() => setExpanded((e) => !e)}>
            {expanded ? <><Minimize2 className="size-3" /> Collapse</> : <><Maximize2 className="size-3" /> Expand</>}
          </Button>
        </div>
      </div>
    </div>
  );

  if (!expanded) return <div className={className}>{body}</div>;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={() => setExpanded(false)}>
      <div
        className="flex h-full w-full max-w-4xl flex-col rounded-lg border bg-elevated p-4 shadow-card"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-1 flex items-center justify-between">
          <span className="text-sm font-semibold">{title} — expanded view</span>
          <button onClick={() => setExpanded(false)} className="text-muted-foreground hover:text-foreground">
            <X className="size-4" />
          </button>
        </div>
        {body}
      </div>
    </div>
  );
}
