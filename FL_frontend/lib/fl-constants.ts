// Federated Learning dashboard — shared constants

export const FL_API_BASE = process.env.NEXT_PUBLIC_FL_API_BASE ?? 'http://localhost:5036'
export const FL_HUB_URL = `${FL_API_BASE}/hubs/fl`

// Client IDs exactly as returned by the backend
export const CLIENT_IDS = ['client_1', 'client_2', 'client_3'] as const
export type ClientId = (typeof CLIENT_IDS)[number]

// Friendly display names for each client/well
export const CLIENT_LABELS: Record<ClientId, string> = {
  client_1: 'Well 1',
  client_2: 'Well 2',
  client_3: 'Well 3',
}

// Consistent color mapping — same client always gets the same color across all pages
export const CLIENT_COLORS: Record<ClientId, string> = {
  client_1: '#3b82f6',  // blue-500
  client_2: '#f59e0b',  // amber-500
  client_3: '#10b981',  // emerald-500
}

// Health thresholds for Fault Detection page (based on test F1)
export const HEALTH_THRESHOLDS = {
  green: 0.7,
  yellow: 0.4,
} as const

export function getHealthStatus(testF1: number): 'green' | 'yellow' | 'red' {
  if (testF1 >= HEALTH_THRESHOLDS.green) return 'green'
  if (testF1 >= HEALTH_THRESHOLDS.yellow) return 'yellow'
  return 'red'
}

// ─── Shared data types ──────────────────────────────────────────────────────

export interface ClientMetrics {
  clientId: string
  trainF1: number
  testF1: number
  accuracy: number
  numExamples: number
}

export interface RoundResult {
  roundNumber: number
  clients: ClientMetrics[]
}
