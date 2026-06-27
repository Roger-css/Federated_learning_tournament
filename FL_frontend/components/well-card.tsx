'use client'

import { getHealthStatus, CLIENT_COLORS, type ClientMetrics } from '@/lib/fl-constants'
import { cn } from '@/lib/utils'

interface Props {
  clientId: string
  label: string
  metrics: ClientMetrics | null
}

const HEALTH_STYLES = {
  green: {
    badge: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300',
    dot: 'bg-emerald-500',
    label: 'Healthy',
  },
  yellow: {
    badge: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
    dot: 'bg-amber-400',
    label: 'Degraded',
  },
  red: {
    badge: 'bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300',
    dot: 'bg-rose-500',
    label: 'Fault Risk',
  },
} as const

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-lg font-semibold text-foreground">{value}</span>
    </div>
  )
}

export function WellCard({ clientId, label, metrics }: Props) {
  const clientColor = CLIENT_COLORS[clientId as keyof typeof CLIENT_COLORS] ?? '#94a3b8'
  const health = metrics ? getHealthStatus(metrics.testF1) : null
  const healthStyle = health ? HEALTH_STYLES[health] : null

  return (
    <div className="flex flex-col rounded-xl border border-border bg-card overflow-hidden">
      {/* Color accent top bar */}
      <div className="h-1.5 w-full" style={{ backgroundColor: clientColor }} />

      <div className="flex flex-col gap-4 p-5">
        {/* Header */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span
              className="h-3 w-3 rounded-full shrink-0"
              style={{ backgroundColor: clientColor }}
              aria-hidden="true"
            />
            <h2 className="font-semibold text-foreground">{label}</h2>
            <span className="text-xs text-muted-foreground">({clientId})</span>
          </div>

          {healthStyle ? (
            <span
              className={cn(
                'flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium',
                healthStyle.badge
              )}
            >
              <span className={cn('h-1.5 w-1.5 rounded-full', healthStyle.dot)} aria-hidden="true" />
              {healthStyle.label}
            </span>
          ) : (
            <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs text-muted-foreground">
              No data
            </span>
          )}
        </div>

        {/* Metrics */}
        {metrics ? (
          <div className="grid grid-cols-2 gap-4">
            <Metric label="Test F1" value={metrics.testF1.toFixed(4)} />
            <Metric label="Accuracy" value={metrics.accuracy.toFixed(4)} />
            <Metric label="Train F1" value={metrics.trainF1.toFixed(4)} />
            <Metric label="Examples" value={metrics.numExamples.toLocaleString()} />
          </div>
        ) : (
          <p className="py-4 text-center text-sm text-muted-foreground">
            Waiting for first training round…
          </p>
        )}

        {/* Threshold legend */}
        {metrics && (
          <p className="text-[10px] text-muted-foreground border-t border-border pt-2">
            Health: F1 &gt; 0.70 = Healthy · 0.40–0.70 = Degraded · &lt; 0.40 = Fault Risk
          </p>
        )}
      </div>
    </div>
  )
}
