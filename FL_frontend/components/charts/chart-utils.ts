import { CLIENT_IDS, CLIENT_COLORS, CLIENT_LABELS, type RoundResult } from '@/lib/fl-constants'

/**
 * Converts an array of RoundResult into a flat array of row objects
 * that Recharts can consume, keyed by clientId for each metric.
 *
 * Output shape per row: { round: 1, client_1: 0.85, client_2: 0.73, ... }
 */
export function buildRoundRows(
  rounds: RoundResult[],
  metric: 'testF1' | 'accuracy' | 'trainF1'
): Record<string, number | string>[] {
  return rounds.map((r) => {
    const row: Record<string, number | string> = { round: r.roundNumber }
    for (const c of r.clients) {
      row[c.clientId] = Number(c[metric].toFixed(4))
    }
    return row
  })
}

export { CLIENT_IDS, CLIENT_COLORS, CLIENT_LABELS }
