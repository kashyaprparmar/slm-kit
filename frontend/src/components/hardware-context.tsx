import { createContext, useContext, useState } from "react";
import { useWebSocket } from "@/lib/ws";
import type { HardwareProfile } from "@/lib/types";

interface QueueState {
  current: number | null;
  queued: number[];
}

interface HardwareState {
  hw: HardwareProfile | null;
  queue: QueueState;
  connected: boolean;
}

const Ctx = createContext<HardwareState>({
  hw: null,
  queue: { current: null, queued: [] },
  connected: false,
});

export function HardwareProvider({ children }: { children: React.ReactNode }) {
  const [hw, setHw] = useState<HardwareProfile | null>(null);
  const [queue, setQueue] = useState<QueueState>({ current: null, queued: [] });

  const { connected } = useWebSocket<Record<string, unknown>>("/ws/system", (msg) => {
    if (msg.type === "queue") {
      setQueue({ current: (msg.current as number) ?? null, queued: (msg.queued as number[]) ?? [] });
    } else if ("source" in msg) {
      setHw(msg as unknown as HardwareProfile);
    }
  });

  return <Ctx.Provider value={{ hw, queue, connected }}>{children}</Ctx.Provider>;
}

export const useHardware = () => useContext(Ctx);
