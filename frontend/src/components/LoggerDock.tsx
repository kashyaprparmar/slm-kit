import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Copy, Pause, Play, Terminal, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { getLogBuffer, subscribeLogs, clearLogs, ingestLog, type LogEntry } from "@/lib/logger";
import { useWebSocket } from "@/lib/ws";
import { useWorkflow } from "@/lib/workflow";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select } from "./ui/select";

interface ServerEvent { id: string; ts: number; level: string; source: string; message: string; correlation_id?: string; duration_ms?: number }
export function LoggerDock() {
  const [open, setOpen] = useWorkflow("logger.open", false);
  const [height, setHeight] = useWorkflow("logger.height", 220);
  const [level, setLevel] = useWorkflow("logger.level", "all");
  const [source, setSource] = useWorkflow("logger.source", "all");
  const [search, setSearch] = useState("");
  const [paused, setPaused] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>(getLogBuffer);
  const scroll = useRef<HTMLDivElement>(null);
  const dirty = useRef(false);
  const seen = useRef(new Set<string>());
  const { connected } = useWebSocket<{ events: ServerEvent[] }>("/ws/activity", msg => {
    for (const e of msg.events ?? []) {
      if (seen.current.has(e.id)) continue;
      seen.current.add(e.id);
      if (seen.current.size > 4000) seen.current.delete(seen.current.values().next().value!);
      ingestLog(e);
    }
  });
  useEffect(() => {
    const unsub = subscribeLogs(() => { dirty.current = true; });
    const timer = setInterval(() => { if (dirty.current) { dirty.current = false; setLogs(getLogBuffer()); } }, 200);
    return () => { unsub(); clearInterval(timer); };
  }, []);
  const filtered = useMemo(() => logs.filter(e =>
    (level === "all" || e.level === level) &&
    (source === "all" || e.tag === source) &&
    (!search || e.message.toLowerCase().includes(search.toLowerCase()))), [logs, level, source, search]);
  // Render a bounded tail; the complete 2,000-event buffer remains searchable/copyable.
  const visible = filtered.slice(-300);
  useEffect(() => { if (!paused && scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight; }, [logs, paused, open]);
  function resize(e: React.PointerEvent<HTMLDivElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
  }
  return <section aria-label="Application logger" className="shrink-0 border-t bg-surface">
    <div role="separator" aria-label="Resize logger" aria-orientation="horizontal" tabIndex={0}
      onPointerDown={resize}
      onPointerMove={e => { if (e.currentTarget.hasPointerCapture(e.pointerId)) setHeight(Math.min(window.innerHeight * 0.55, Math.max(120, window.innerHeight - e.clientY))); }}
      onKeyDown={e => { if (e.key === "ArrowUp") setHeight(Math.min(420, height + 20)); if (e.key === "ArrowDown") setHeight(Math.max(120, height - 20)); }}
      className={open ? "h-1 cursor-row-resize touch-none hover:bg-primary/50" : "hidden"} />
    <div className="flex h-10 items-center gap-2 px-4">
      <button className="flex items-center gap-2 text-xs font-semibold" onClick={() => setOpen(!open)} aria-expanded={open}>
        <Terminal className="size-4 text-primary" /> Application log {open ? <ChevronDown className="size-3" /> : <ChevronUp className="size-3" />}
      </button>
      <span className="text-[10px] text-muted-foreground">{connected ? "Live" : "Reconnecting"} · {logs.length} events</span>
      <div className="ml-auto flex gap-1">
        <Button variant="ghost" size="icon" aria-label={paused ? "Resume scrolling" : "Pause scrolling"} onClick={() => setPaused(!paused)}>{paused ? <Play /> : <Pause />}</Button>
        <Button variant="ghost" size="icon" aria-label="Copy logs" onClick={async () => {
          try { await navigator.clipboard.writeText(filtered.map(e => `${new Date(e.ts).toISOString()} ${e.level.toUpperCase()} [${e.tag}] ${e.message}`).join("\n")); toast.success("Logs copied"); }
          catch { toast.error("Clipboard unavailable. Select and copy the log text."); }
        }}><Copy /></Button>
        <Button variant="ghost" size="icon" aria-label="Clear visible logs" onClick={() => { clearLogs(); setLogs([]); }}><Trash2 /></Button>
      </div>
    </div>
    {open && <div style={{ height: Math.min(height, window.innerHeight * 0.55) }} className="flex flex-col">
      <div className="flex flex-wrap gap-2 border-y px-4 py-2">
        <Input aria-label="Search logs" placeholder="Search logs…" className="h-8 min-w-32 flex-1" value={search} onChange={e => setSearch(e.target.value)} />
        <Select aria-label="Log severity" className="h-8 w-28 text-xs" value={level} onChange={e => setLevel(e.target.value)}>
          {["all", "debug", "info", "success", "warn", "error"].map(l => <option key={l} value={l}>{l === "all" ? "All levels" : l === "warn" ? "WARNING" : l.toUpperCase()}</option>)}
        </Select>
        <Select aria-label="Log subsystem" className="h-8 w-36 text-xs" value={source} onChange={e => setSource(e.target.value)}>
          <option value="all">All sources</option>{[...new Set(logs.map(l => l.tag))].sort().map(s => <option key={s}>{s}</option>)}
        </Select>
      </div>
      <div ref={scroll} className="min-h-0 flex-1 overflow-auto px-4 py-2 font-mono text-[11px] leading-5">
        {filtered.length > visible.length && <p className="text-muted-foreground">Showing latest 300 matches. Search to narrow or copy all matches.</p>}
        {visible.map((e, i) => <div key={i} className="flex gap-3">
          <span className="shrink-0 text-muted-foreground">{new Date(e.ts).toLocaleTimeString()}</span>
          <span className={e.level === "error" ? "text-danger" : e.level === "warn" ? "text-warning" : e.level === "success" ? "text-success" : "text-muted-foreground"}>{e.level.toUpperCase()}</span>
          <span className="max-w-36 shrink-0 truncate text-primary" title={e.tag}>[{e.tag}]</span>
          <span className="min-w-0 whitespace-pre-wrap break-words">{e.message}</span>
        </div>)}
        {!visible.length && <p className="text-muted-foreground">No matching events.</p>}
      </div>
    </div>}
  </section>;
}
