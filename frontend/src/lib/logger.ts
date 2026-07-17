/**
 * Lightweight frontend logger. Logs to the browser console with levels +
 * timestamps + a component tag, and keeps a rolling in-memory buffer the UI can
 * surface (see the "Frontend logs" viewer). Enabled by default in dev.
 */

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface LogEntry {
  ts: number;
  level: LogLevel;
  tag: string;
  message: string;
}

const BUFFER_MAX = 500;
const buffer: LogEntry[] = [];
const listeners = new Set<(e: LogEntry) => void>();

const LEVEL_STYLE: Record<LogLevel, string> = {
  debug: "color:#8b8b9a",
  info: "color:#7c5cff",
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
  // eslint-disable-next-line no-console
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
    warn: (...a: unknown[]) => emit("warn", tag, a),
    error: (...a: unknown[]) => emit("error", tag, a),
  };
}

export function getLogBuffer(): LogEntry[] {
  return [...buffer];
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
