'use client'

import { useEffect, useState } from 'react'
import { useFlSocket } from '@/hooks/use-fl-socket'
import { ConnectionBadge } from '@/components/connection-badge'
import { WellCard } from '@/components/well-card'
import {
  FL_API_BASE,
  CLIENT_IDS,
  CLIENT_LABELS,
  type ClientMetrics,
  type RoundResult,
} from '@/lib/fl-constants'
import { AlertCircle } from 'lucide-react'

export default function FaultDetectionPage() {
  const { rounds: liveRounds, connectionState } = useFlSocket()
  const [latestMetrics, setLatestMetrics] = useState<Record<string, ClientMetrics>>({})
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
    setLatestMetrics((prev) => ({ ...prev, ...map }))
  }

  // Initial REST fetch
  useEffect(() => {
    fetch(`${FL_API_BASE}/api/fl/rounds`)
      .then((r) => r.json())
      .then((data: { rounds: RoundResult[] }) => {
        applyRounds(data.rounds ?? [])
        setLoaded(true)
      })
      .catch(() => {
        setFetchError(true)
        setLoaded(true)
      })
  }, [])

  // Merge live rounds
  useEffect(() => {
    applyRounds(liveRounds)
  }, [liveRounds])

  return (
    <div className="flex flex-col gap-6 p-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-foreground">Fault Detection</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Per-well valve health — based on latest federated model metrics
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

      {/* 2×2 grid of well cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {CLIENT_IDS.map((id) => (
          <WellCard
            key={id}
            clientId={id}
            label={CLIENT_LABELS[id]}
            metrics={latestMetrics[id] ?? null}
          />
        ))}
      </div>

      {/* Empty state hint */}
      {loaded && !fetchError && Object.keys(latestMetrics).length === 0 && (
        <p className="text-center text-sm text-muted-foreground">
          No training data yet — start the FL simulation to populate the well cards.
        </p>
      )}
    </div>
  )
}
