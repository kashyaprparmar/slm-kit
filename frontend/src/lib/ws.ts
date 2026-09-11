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
    setConnected(false);
    if (!path) return;
    let ws: WebSocket | null = null;
    let retry = 0;
    let closed = false;
    let timer: ReturnType<typeof setTimeout>;
    let heartbeat: ReturnType<typeof setInterval>;
    const received = new Set<string>();
    const deliver = (data: T & { _event_id?: string; type?: string; events?: T[] }) => {
      if (data.type === "replay") { data.events?.forEach(e => deliver(e as T & { _event_id?: string })); return; }
      if (data._event_id) {
        if (received.has(data._event_id)) return;
        received.add(data._event_id);
        if (received.size > 3000) received.delete(received.values().next().value!);
      }
      cbRef.current(data);
    };

    const connect = () => {
      ws = new WebSocket(wsUrl(path));
      ws.onopen = () => {
        if (retry > 0) log.info(`connected ${path} (after ${retry} retries)`);
        else log.debug(`connected ${path}`);
        retry = 0;
        setConnected(true);
        heartbeat = setInterval(() => { if (ws?.readyState === WebSocket.OPEN) ws.send("ping"); }, 15000);
      };
      ws.onmessage = (ev) => {
        try {
          deliver(JSON.parse(ev.data));
        } catch (e) {
          log.error("Could not process live update", String(e));
        }
      };
      ws.onclose = () => {
        clearInterval(heartbeat);
        setConnected(false);
        if (!closed) {
          retry = Math.min(retry + 1, 6);
          if (retry === 1) log.warn(`disconnected ${path} — reconnecting…`);
          timer = setTimeout(connect, Math.min(10000, 400 * 2 ** retry) + Math.random() * 300);
        }
      };
      ws.onerror = () => ws?.close();
    };
    connect();

    return () => {
      closed = true;
      clearTimeout(timer);
      clearInterval(heartbeat);
      ws?.close();
    };
  }, [path]);

  return { connected };
}
