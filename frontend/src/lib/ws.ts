import { useEffect, useRef, useState } from "react";
import { createLogger } from "./logger";

const log = createLogger("ws");

function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${path}`;
}

/**
 * Subscribe to a backend WebSocket topic. Auto-reconnects with backoff.
 * `onMessage` is kept in a ref so re-renders don't tear down the socket.
 */
export function useWebSocket<T = unknown>(
  path: string | null,
  onMessage: (data: T) => void,
) {
  const [connected, setConnected] = useState(false);
  const cbRef = useRef(onMessage);
  cbRef.current = onMessage;

  useEffect(() => {
    if (!path) return;
    let ws: WebSocket | null = null;
    let retry = 0;
    let closed = false;
    let timer: ReturnType<typeof setTimeout>;

    const connect = () => {
      ws = new WebSocket(wsUrl(path));
      ws.onopen = () => {
        if (retry > 0) log.info(`connected ${path} (after ${retry} retries)`);
        else log.debug(`connected ${path}`);
        retry = 0;
        setConnected(true);
      };
      ws.onmessage = (ev) => {
        try {
          cbRef.current(JSON.parse(ev.data) as T);
        } catch {
          /* ignore non-JSON frames */
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) {
          retry = Math.min(retry + 1, 6);
          if (retry === 1) log.warn(`disconnected ${path} — reconnecting…`);
          timer = setTimeout(connect, 400 * retry);
        }
      };
      ws.onerror = () => ws?.close();
    };
    connect();

    return () => {
      closed = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, [path]);

  return { connected };
}
