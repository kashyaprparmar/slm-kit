import { useCallback, useRef, useSyncExternalStore } from "react";

// Only user drafts/preferences are persisted. Runtime status remains server-owned.
const values = new Map<string, unknown>();
const listeners = new Set<() => void>();
const prefix = "slmkit.workflow.v1.";
export function readWorkflow<T>(key: string, fallback: T): T {
  if (!values.has(key)) {
    try {
      const raw = localStorage.getItem(prefix + key);
      values.set(key, raw == null ? fallback : JSON.parse(raw));
    } catch { values.set(key, fallback); }
  }
  return values.get(key) as T;
}
export function writeWorkflow<T>(key: string, value: T, persist = true) {
  values.set(key, value);
  if (persist) {
    try { localStorage.setItem(prefix + key, JSON.stringify(value)); }
    catch { /* Draft remains available in memory if storage is full/disabled. */ }
  }
  listeners.forEach(fn => fn());
}
export function useWorkflow<T>(key: string, fallback: T, persist = true): [T, (value: T | ((previous: T) => T)) => void] {
  const fallbackRef = useRef(fallback);
  const subscribe = useCallback((fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn); }; }, []);
  const value = useSyncExternalStore(subscribe, () => readWorkflow(key, fallback));
  const setValue = useCallback((next: T | ((previous: T) => T)) => {
    const previous = readWorkflow(key, fallbackRef.current);
    writeWorkflow(key, typeof next === "function" ? (next as (p: T) => T)(previous) : next, persist);
  }, [key, persist]);
  return [value, setValue];
}
