import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

const apiTarget = process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:8000";
const wsTarget = apiTarget.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      // Backend REST + WebSocket, so the app is same-origin in dev.
      "/api": { target: apiTarget, changeOrigin: true },
      "/ws": { target: wsTarget, ws: true },
    },
  },
});
