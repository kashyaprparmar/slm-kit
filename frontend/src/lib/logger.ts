/**
 * Lightweight frontend logger. Logs to the browser console with levels +
 * timestamps + a component tag, and keeps a rolling in-memory buffer the UI can
 * surface (see the "Frontend logs" viewer). Enabled by default in dev.
 */

export type LogLevel = "debug" | "info" | "success" | "warn" | "error";

export interface LogEntry {
  ts: number;
  level: LogLevel;
  tag: string;
  message: string;
}

const BUFFER_MAX = 2000;
const buffer: LogEntry[] = [];
const listeners = new Set<(e: LogEntry) => void>();

const LEVEL_STYLE: Record<LogLevel, string> = {
  debug: "color:#8b8b9a",
  info: "color:#7c5cff",
  success: "color:#22c55e",
  warn: "color:#f5a524",
  error: "color:#f5556d",
};

function emit(level: LogLevel, tag: string, args: unknown[]) {
  const message = args
    .map((a) => (typeof a === "string" ? a : safeStringify(a)))
    .join(" ");
  const entry: LogEntry = { ts: Date.now(), level, tag, message };
  buffer.push(entry);
  if (buffer.length > BUFFER_MAX) buffer.shift();
  listeners.forEach((fn) => fn(entry));

  const time = new Date(entry.ts).toLocaleTimeString();
  const fn = level === "error" ? console.error : level === "warn" ? console.warn : console.log;
  fn(`%c${time} %c[${tag}]%c ${message}`, "color:#8b8b9a", LEVEL_STYLE[level], "color:inherit");
}

function safeStringify(v: unknown): string {
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

export function createLogger(tag: string) {
  return {
    debug: (...a: unknown[]) => emit("debug", tag, a),
    info: (...a: unknown[]) => emit("info", tag, a),
    success: (...a: unknown[]) => emit("success", tag, a),
    warn: (...a: unknown[]) => emit("warn", tag, a),
    error: (...a: unknown[]) => emit("error", tag, a),
  };
}

export function getLogBuffer(): LogEntry[] {
  return [...buffer];
}

export function clearLogs() { buffer.length = 0; }
export function ingestLog(e: { ts: number; level: string; source: string; message: string; correlation_id?: string; duration_ms?: number }) {
  const raw = e.level.toLowerCase();
  const level: LogLevel = raw === "warning" ? "warn" : raw in LEVEL_STYLE ? raw as LogLevel : "info";
  const entry: LogEntry = { ts: e.ts * 1000, level, tag: e.source, message: e.message +
    (e.duration_ms != null ? ` (${e.duration_ms}ms)` : "") + (e.correlation_id ? ` [${e.correlation_id}]` : "") };
  buffer.push(entry);
  if (buffer.length > BUFFER_MAX) buffer.splice(0, buffer.length - BUFFER_MAX);
  listeners.forEach(fn => fn(entry));
}

export function subscribeLogs(fn: (e: LogEntry) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// Surface uncaught errors + promise rejections into the buffer too.
if (typeof window !== "undefined") {
  window.addEventListener("error", (e) =>
    emit("error", "window", [e.message || String(e.error)]),
  );
  window.addEventListener("unhandledrejection", (e) =>
    emit("error", "promise", [safeStringify(e.reason)]),
  );
}
