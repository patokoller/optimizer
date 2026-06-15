"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FolderOpen,
  BarChart3,
  RefreshCw,
  TrendingUp,
  Download,
  Settings,
  Circle,
  Activity,
  Zap,
  Search,
  FileText,
} from "lucide-react";
import { cn } from "@/components/ui";
import { useStore } from "@/store";

const NAV_ITEMS = [
  { href: "/dashboard",  label: "Dashboard",       icon: LayoutDashboard },
  { href: "/portfolio",  label: "Portfolio",        icon: FolderOpen      },
  { href: "/scoring",    label: "Scoring Engine",   icon: BarChart3       },
  { href: "/discovery",  label: "Discovery",        icon: Zap             },
  { href: "/search",     label: "Stock Search",     icon: Search          },
  { href: "/rebalance",  label: "Rebalance",        icon: RefreshCw       },
  { href: "/backtest",   label: "Backtest",         icon: TrendingUp      },
  { href: "/report",     label: "Analysis Report",  icon: FileText        },
  { href: "/export",     label: "Trade Export",     icon: Download        },
  { href: "/settings",   label: "Settings",         icon: Settings        },
];

export function Sidebar() {
  const pathname = usePathname();
  const { activeJobs, llmFallbackActive } = useStore((s) => ({
    activeJobs: s.activeJobs,
    llmFallbackActive: s.llmFallbackActive,
  }));

  const runningJobs = activeJobs.filter(
    (j) => j.status === "running" || j.status === "pending"
  );

  return (
    <aside
      className="w-[220px] shrink-0 flex flex-col border-r border-border"
      style={{ background: "var(--color-surface)", height: "100vh", position: "sticky", top: 0 }}
    >
      {/* Logo */}
      <div className="px-4 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          <div>
            <p className="text-xs font-bold tracking-widest text-primary">ALPHA·LENS</p>
            <p className="text-2xs text-muted tracking-widest">PORTFOLIO DECISION</p>
          </div>
        </div>
      </div>

      {/* Status bar */}
      <div className="px-4 py-2.5 border-b border-border">
        <p className="label-sm mb-1.5">System Status</p>
        <div className="flex items-center gap-1.5">
          <Circle
            className="w-1.5 h-1.5 fill-current"
            style={{ color: runningJobs.length > 0 ? "#f5a623" : "#3ecf8e" }}
          />
          <span className="text-xs" style={{ color: runningJobs.length > 0 ? "#f5a623" : "#3ecf8e" }}>
            {runningJobs.length > 0
              ? `${runningJobs.length} job${runningJobs.length > 1 ? "s" : ""} running`
              : "Ready"}
          </span>
        </div>
        {llmFallbackActive && (
          <div className="flex items-center gap-1.5 mt-1">
            <Circle className="w-1.5 h-1.5 fill-current text-warning" />
            <span className="text-2xs text-warning">LLM fallback active</span>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-2 overflow-y-auto">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname?.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2.5 px-4 py-2.5 text-sm transition-all duration-100",
                "border-l-2",
                active
                  ? "text-primary bg-primary/8 border-primary font-medium"
                  : "text-muted hover:text-text hover:bg-surface2 border-transparent"
              )}
            >
              <Icon className="w-3.5 h-3.5 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* User footer */}
      <div className="px-4 py-3 border-t border-border">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center text-2xs font-bold text-primary">
            PM
          </div>
          <div>
            <p className="text-xs font-medium text-text">Portfolio Manager</p>
            <p className="text-2xs text-muted">NASDAQ-100</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
