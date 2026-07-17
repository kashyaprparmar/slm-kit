import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: { "@": path.resolve(__dirname, "./src") },
    },
    server: {
        port: 5173,
        proxy: {
            // Backend REST + WebSocket, so the app is same-origin in dev.
            "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
            "/ws": { target: "ws://127.0.0.1:8000", ws: true },
        },
    },
});
