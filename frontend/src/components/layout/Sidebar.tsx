import { NavLink } from "react-router-dom";
import { Flame, X } from "lucide-react";
import { NAV } from "./nav";
import { cn } from "@/lib/utils";

const GROUPS = ["Overview", "Studios", "Analysis"];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full w-60 flex-col border-r bg-surface/70 backdrop-blur">
      <div className="flex h-16 items-center justify-between px-5">
        <div className="flex items-center gap-2.5">
          <div className="grid size-8 place-items-center rounded-lg bg-primary text-primary-foreground shadow-glow">
            <Flame className="size-4.5" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-bold tracking-tight">SLM Kit</div>
            <div className="text-[10px] uppercase tracking-widest text-muted-foreground">lifecycle studio</div>
          </div>
        </div>
        {onNavigate && (
          <button onClick={onNavigate} className="lg:hidden text-muted-foreground hover:text-foreground">
            <X className="size-5" />
          </button>
        )}
      </div>

      <nav className="flex-1 space-y-6 overflow-y-auto px-3 py-4">
        {GROUPS.map((group) => (
          <div key={group}>
            <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/70">
              {group}
            </div>
            <div className="space-y-1">
              {NAV.filter((n) => n.group === group).map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === "/"}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn(
                      "group flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-primary/15 text-primary"
                        : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                    )
                  }
                >
                  {({ isActive }) => (
                    <>
                      <Icon className={cn("size-4 transition-transform group-hover:scale-110", isActive && "text-primary")} />
                      {label}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t px-4 py-3 text-[10px] text-muted-foreground">
        Tuned for RTX 4060 · 8GB · v0.1.0
      </div>
    </div>
  );
}
