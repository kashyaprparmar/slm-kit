import { useCallback, useRef, useSyncExternalStore } from "react";

// Only user drafts/preferences are persisted. Runtime status remains server-owned.
const values = new Map<string, unknown>();
const listeners = new Set<() => void>();
const prefix = "slmkit.workflow.v1.";
function mergeWithFallback<T>(val: unknown, fallback: T): T {
  if (
    typeof fallback === "object" &&
    fallback !== null &&
    !Array.isArray(fallback) &&
    typeof val === "object" &&
    val !== null &&
    !Array.isArray(val)
  ) {
    const vObj = val as Record<string, unknown>;
    const fObj = fallback as Record<string, unknown>;
    let hasMissing = false;
    for (const k in fObj) {
      if (fObj[k] !== undefined && vObj[k] === undefined) {
        hasMissing = true;
        break;
      }
    }
    if (hasMissing) {
      const merged: Record<string, unknown> = { ...fObj };
      for (const k in vObj) {
        if (vObj[k] !== undefined) {
          merged[k] = vObj[k];
        }
      }
      return merged as T;
    }
  }
  return val as T;
}

export function readWorkflow<T>(key: string, fallback: T): T {
  if (!values.has(key)) {
    try {
      const raw = localStorage.getItem(prefix + key);
      if (raw == null) {
        values.set(key, fallback);
      } else {
        const parsed = JSON.parse(raw);
        values.set(key, mergeWithFallback(parsed, fallback));
      }
    } catch {
      values.set(key, fallback);
    }
  } else {
    const current = values.get(key);
    const merged = mergeWithFallback(current, fallback);
    if (merged !== current) {
      values.set(key, merged);
    }
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
  fallbackRef.current = fallback;
  const subscribe = useCallback((fn: () => void) => {
    listeners.add(fn);
    return () => { listeners.delete(fn); };
  }, []);
  const getSnapshot = useCallback(() => readWorkflow(key, fallbackRef.current), [key]);
  const value = useSyncExternalStore(subscribe, getSnapshot);
  const setValue = useCallback((next: T | ((previous: T) => T)) => {
    const previous = readWorkflow(key, fallbackRef.current);
    writeWorkflow(key, typeof next === "function" ? (next as (p: T) => T)(previous) : next, persist);
  }, [key, persist]);
  return [value, setValue];
}
