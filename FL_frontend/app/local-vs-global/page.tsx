'use client'

import { useEffect, useState } from 'react'
import { useFlSocket } from '@/hooks/use-fl-socket'
import { ConnectionBadge } from '@/components/connection-badge'
import { LocalGlobalChart } from '@/components/charts/local-global-chart'
import {
  FL_API_BASE,
  CLIENT_IDS,
  CLIENT_LABELS,
  type ClientMetrics,
  type RoundResult,
} from '@/lib/fl-constants'
import { AlertCircle, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ComparisonRow {
  clientId: string
  label: string
  localF1: number
  globalF1: number
  delta: number
}

function fmt(n: number) {
  return n.toFixed(4)
}

export default function LocalVsGlobalPage() {
  const { localBaseline: liveBaseline, rounds: liveRounds, connectionState } = useFlSocket()

  const [baseline, setBaseline] = useState<ClientMetrics[]>([])
  const [latestGlobal, setLatestGlobal] = useState<Record<string, ClientMetrics>>({})
  const [fetchError, setFetchError] = useState(false)
  const [loaded, setLoaded] = useState(false)

  function roundAvgF1(r: RoundResult) {
    if (r.clients.length === 0) return 0
    return r.clients.reduce((sum, c) => sum + c.testF1, 0) / r.clients.length
  }

  function applyRounds(rounds: RoundResult[]) {
    if (rounds.length === 0) return
    const best = rounds.reduce((a, b) => (roundAvgF1(a) > roundAvgF1(b) ? a : b))
    const map: Record<string, ClientMetrics> = {}
    for (const c of best.clients) {
      map[c.clientId] = c
    }
    setLatestGlobal((prev) => ({ ...prev, ...map }))
  }

  // Initial REST fetches — both in parallel
  useEffect(() => {
    Promise.all([
      fetch(`${FL_API_BASE}/api/fl/local-baseline`).then((r) => r.json()),
      fetch(`${FL_API_BASE}/api/fl/rounds`).then((r) => r.json()),
    ])
      .then(([baseData, roundData]) => {
        setBaseline((baseData as { clients: ClientMetrics[] }).clients ?? [])
        applyRounds((roundData as { rounds: RoundResult[] }).rounds ?? [])
        setLoaded(true)
      })
      .catch(() => {
        setFetchError(true)
        setLoaded(true)
      })
  }, [])

  // Live baseline updates
  useEffect(() => {
    if (liveBaseline.length > 0) setBaseline(liveBaseline)
  }, [liveBaseline])

  // Live round updates
  useEffect(() => {
    applyRounds(liveRounds)
  }, [liveRounds])

  // Build comparison rows
  const rows: ComparisonRow[] = CLIENT_IDS.map((id) => {
    const local = baseline.find((c) => c.clientId === id)
    const global = latestGlobal[id]
    const localF1 = local?.testF1 ?? 0
    const globalF1 = global?.testF1 ?? 0
    return {
      clientId: id,
      label: CLIENT_LABELS[id],
      localF1,
      globalF1,
      delta: globalF1 - localF1,
    }
  })

  const hasBaseline = baseline.length > 0
  const hasGlobal = Object.keys(latestGlobal).length > 0
  const hasAny = hasBaseline || hasGlobal

  // Averages
  const localRows = rows.filter((r) => r.localF1 > 0)
  const globalRows = rows.filter((r) => r.globalF1 > 0)
  const completedRows = rows.filter((r) => r.localF1 > 0 && r.globalF1 > 0)
  const avgLocal =
    localRows.length > 0
      ? localRows.reduce((sum, r) => sum + r.localF1, 0) / localRows.length
      : null
  const avgGlobal =
    globalRows.length > 0
      ? globalRows.reduce((sum, r) => sum + r.globalF1, 0) / globalRows.length
      : null
  const avgDelta =
    completedRows.length > 0
      ? completedRows.reduce((sum, r) => sum + r.delta, 0) / completedRows.length
      : null

  return (
    <div className="flex flex-col gap-6 p-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-foreground">Local vs Global</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Phase 1 (local-only training) compared to Phase 2 (federated) results
          </p>
        </div>
        <ConnectionBadge state={connectionState} />
      </header>

      {/* Backend unreachable */}
      {fetchError && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <div>
            <p className="font-medium">Cannot connect to FL backend</p>
            <p className="mt-0.5 opacity-80">
              Is the .NET server running on{' '}
              <code className="rounded bg-destructive/20 px-1 font-mono text-xs">{FL_API_BASE}</code>?
            </p>
          </div>
        </div>
      )}

      {/* Loading state */}
      {!loaded && !fetchError && (
        <div className="flex items-center justify-center py-20 text-sm text-muted-foreground">
          Loading…
        </div>
      )}

      {/* Missing data notice */}
      {loaded && !fetchError && !hasAny && (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20 text-center">
          <p className="font-medium text-foreground">No comparison data yet</p>
          <p className="mt-1 max-w-sm text-sm text-muted-foreground">
            Start the FL simulation to populate this view.
          </p>
        </div>
      )}

      {/* Average comparison cards — visible as soon as any data exists */}
      {hasAny && (
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-xl border border-border bg-card p-4 text-center">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Local Avg F1</p>
            <p className={cn('mt-1 text-2xl font-bold', avgLocal !== null ? 'text-foreground' : 'text-muted-foreground')}>
              {avgLocal !== null ? avgLocal.toFixed(4) : '—'}
            </p>
          </div>
          <div className="rounded-xl border border-border bg-card p-4 text-center">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Global Avg F1</p>
            <p className={cn('mt-1 text-2xl font-bold', avgGlobal !== null ? 'text-foreground' : 'text-muted-foreground')}>
              {avgGlobal !== null ? avgGlobal.toFixed(4) : '—'}
            </p>
          </div>
          <div className="rounded-xl border border-border bg-card p-4 text-center">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Change</p>
            <p className={cn('mt-1 text-2xl font-bold', avgDelta !== null ? (avgDelta > 0 ? 'text-emerald-700 dark:text-emerald-400' : avgDelta < 0 ? 'text-rose-700 dark:text-rose-400' : 'text-foreground') : 'text-muted-foreground')}>
              {avgDelta !== null ? (avgDelta >= 0 ? '+' : '') + avgDelta.toFixed(4) : '—'}
            </p>
          </div>
        </div>
      )}

      {/* Bar chart + table — visible as soon as any data exists */}
      {hasAny && (
        <>
          <section aria-label="Local vs global F1 comparison chart">
            <h2 className="mb-3 text-sm font-medium text-foreground">
              Test F1: Local vs Federated
              {hasBaseline && !hasGlobal && <span className="ml-2 text-xs font-normal text-muted-foreground">(awaiting Phase 2…)</span>}
              {!hasBaseline && hasGlobal && <span className="ml-2 text-xs font-normal text-muted-foreground">(no Phase 1 baseline)</span>}
            </h2>
            <div className="rounded-xl border border-border bg-card p-4">
              <LocalGlobalChart rows={rows} />
            </div>
          </section>

          {/* Comparison table */}
          <section aria-label="Per-client comparison table">
            <h2 className="mb-3 text-sm font-medium text-foreground">Per-client Breakdown</h2>
            <div className="rounded-xl border border-border bg-card overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-muted/40">
                      <th className="px-4 py-2.5 text-left font-medium text-muted-foreground">Client</th>
                      <th className="px-4 py-2.5 text-right font-medium text-muted-foreground">Local F1</th>
                      <th className="px-4 py-2.5 text-right font-medium text-muted-foreground">Global F1</th>
                      <th className="px-4 py-2.5 text-right font-medium text-muted-foreground">Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, i) => {
                      const noData = row.localF1 === 0 && row.globalF1 === 0
                      const improved = row.delta > 0.001
                      const hurt = row.delta < -0.001
                      return (
                        <tr key={row.clientId} className={i < rows.length - 1 ? 'border-b border-border' : ''}>
                          <td className="px-4 py-2.5 font-medium text-foreground">{row.label}</td>
                          <td className="px-4 py-2.5 text-right font-mono text-foreground">
                            {row.localF1 > 0 ? fmt(row.localF1) : <span className="text-muted-foreground">—</span>}
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-foreground">
                            {row.globalF1 > 0 ? fmt(row.globalF1) : <span className="text-muted-foreground">—</span>}
                          </td>
                          <td className="px-4 py-2.5 text-right">
                            {noData ? (
                              <span className="text-muted-foreground">—</span>
                            ) : (
                              <span className={cn('inline-flex items-center justify-end gap-1 font-mono font-semibold', improved ? 'text-emerald-700 dark:text-emerald-400' : hurt ? 'text-rose-700 dark:text-rose-400' : 'text-muted-foreground')}>
                                {improved ? <TrendingUp className="h-3.5 w-3.5" aria-hidden="true" /> : hurt ? <TrendingDown className="h-3.5 w-3.5" aria-hidden="true" /> : <Minus className="h-3.5 w-3.5" aria-hidden="true" />}
                                {row.delta >= 0 ? '+' : ''}{fmt(row.delta)}
                              </span>
                            )}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  )
}
