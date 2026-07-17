import { useState } from "react";
import { Menu, Moon, Sun } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { ResourceStrip } from "@/components/ResourceStrip";
import { useTheme } from "@/components/theme";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const { theme, toggle } = useTheme();

  return (
    <div className="flex h-dvh overflow-hidden">
      {/* Desktop sidebar */}
      <aside className="hidden lg:block shrink-0">
        <Sidebar />
      </aside>

      {/* Mobile drawer */}
      <div className={cn("fixed inset-0 z-50 lg:hidden", open ? "" : "pointer-events-none")}>
        <div
          className={cn("absolute inset-0 bg-black/50 transition-opacity", open ? "opacity-100" : "opacity-0")}
          onClick={() => setOpen(false)}
        />
        <div className={cn("absolute inset-y-0 left-0 transition-transform", open ? "translate-x-0" : "-translate-x-full")}>
          <Sidebar onNavigate={() => setOpen(false)} />
        </div>
      </div>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center gap-3 border-b bg-surface/50 px-4 backdrop-blur md:px-6">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setOpen(true)}>
            <Menu className="size-5" />
          </Button>
          <div className="min-w-0 flex-1">
            <ResourceStrip />
          </div>
          <Button variant="ghost" size="icon" onClick={toggle} aria-label="Toggle theme">
            {theme === "dark" ? <Sun className="size-4.5" /> : <Moon className="size-4.5" />}
          </Button>
        </header>

        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-7xl px-4 py-6 md:px-8 md:py-8">{children}</div>
        </main>
      </div>
    </div>
  );
}
