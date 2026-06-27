'use client'

import { type ConnectionState } from '@/hooks/use-fl-socket'
import { cn } from '@/lib/utils'

interface ConnectionBadgeProps {
  state: ConnectionState
}

const STATE_CONFIG: Record<ConnectionState, { label: string; dotClass: string; textClass: string }> = {
  connected: {
    label: 'Live',
    dotClass: 'bg-emerald-500',
    textClass: 'text-emerald-700 dark:text-emerald-400',
  },
  connecting: {
    label: 'Connecting…',
    dotClass: 'bg-amber-400 animate-pulse',
    textClass: 'text-amber-700 dark:text-amber-400',
  },
  disconnected: {
    label: 'Disconnected',
    dotClass: 'bg-rose-500',
    textClass: 'text-rose-700 dark:text-rose-400',
  },
  error: {
    label: 'No live updates',
    dotClass: 'bg-rose-500',
    textClass: 'text-rose-700 dark:text-rose-400',
  },
}

export function ConnectionBadge({ state }: ConnectionBadgeProps) {
  const { label, dotClass, textClass } = STATE_CONFIG[state]
  return (
    <span className={cn('flex items-center gap-1.5 text-xs font-medium', textClass)}>
      <span className={cn('inline-block h-2 w-2 rounded-full', dotClass)} aria-hidden="true" />
      {label}
    </span>
  )
}
