'use client'

import { CLIENT_COLORS, CLIENT_LABELS, CLIENT_IDS, type ClientMetrics } from '@/lib/fl-constants'

interface Props {
  clients: ClientMetrics[]
}

function fmt(n: number) {
  return n.toFixed(4)
}

export function LatestRoundTable({ clients }: Props) {
  // Sort clients by the canonical order defined in CLIENT_IDS
  const sorted = CLIENT_IDS.map((id) => clients.find((c) => c.clientId === id)).filter(
    Boolean
  ) as ClientMetrics[]

  if (sorted.length === 0) return null

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-muted/40">
            <th className="px-4 py-2.5 text-left font-medium text-muted-foreground">Client</th>
            <th className="px-4 py-2.5 text-right font-medium text-muted-foreground">Train F1</th>
            <th className="px-4 py-2.5 text-right font-medium text-muted-foreground">Test F1</th>
            <th className="px-4 py-2.5 text-right font-medium text-muted-foreground">Accuracy</th>
            <th className="px-4 py-2.5 text-right font-medium text-muted-foreground">Examples</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((c, i) => (
            <tr
              key={c.clientId}
              className={i < sorted.length - 1 ? 'border-b border-border' : ''}
            >
              <td className="flex items-center gap-2 px-4 py-2.5 font-medium text-foreground">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full shrink-0"
                  style={{ backgroundColor: CLIENT_COLORS[c.clientId as keyof typeof CLIENT_COLORS] }}
                  aria-hidden="true"
                />
                {CLIENT_LABELS[c.clientId as keyof typeof CLIENT_LABELS] ?? c.clientId}
              </td>
              <td className="px-4 py-2.5 text-right font-mono text-foreground">{fmt(c.trainF1)}</td>
              <td className="px-4 py-2.5 text-right font-mono text-foreground">{fmt(c.testF1)}</td>
              <td className="px-4 py-2.5 text-right font-mono text-foreground">{fmt(c.accuracy)}</td>
              <td className="px-4 py-2.5 text-right font-mono text-muted-foreground">
                {c.numExamples.toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
