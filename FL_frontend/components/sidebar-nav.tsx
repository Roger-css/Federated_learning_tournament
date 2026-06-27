"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { BarChart2, Activity, GitCompare, Wifi } from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "FL Dashboard", icon: Activity },
  { href: "/fault-detection", label: "Fault Detection", icon: Wifi },
  { href: "/local-vs-global", label: "Local vs Global", icon: GitCompare },
] as const;

export function SidebarNav() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-56 flex-col border-r border-border bg-card text-card-foreground shrink-0">
      {/* Logo / title */}
      <div className="flex items-center gap-2 border-b border-border px-4 py-4">
        <BarChart2 className="h-5 w-5 text-primary" aria-hidden="true" />
        <div className="min-w-0">
          <p className="text-sm font-semibold leading-tight text-foreground">
            FL Monitor
          </p>
          <p className="truncate text-[10px] text-muted-foreground">
            DHSV Fault Detection
          </p>
        </div>
      </div>

      {/* Navigation */}
      <nav
        className="flex flex-col gap-0.5 p-2 flex-1"
        aria-label="Main navigation"
      >
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-primary text-primary-foreground font-medium"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )}
              aria-current={active ? "page" : undefined}
            >
              <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border px-4 py-3">
        <p className="text-[10px] text-muted-foreground">
          Backend: {process.env.NEXT_PUBLIC_FL_API_BASE}
        </p>
      </div>
    </aside>
  );
}
