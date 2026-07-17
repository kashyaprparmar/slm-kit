import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./index.css";
import App from "./App";
import { ThemeProvider } from "@/components/theme";
import { HardwareProvider } from "@/components/hardware-context";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1, staleTime: 5000 } },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <HardwareProvider>
          <BrowserRouter>
            <App />
          </BrowserRouter>
          <Toaster theme="dark" position="bottom-right" richColors closeButton />
        </HardwareProvider>
      </QueryClientProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
