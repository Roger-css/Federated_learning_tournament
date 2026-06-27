'use client'

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { buildRoundRows, CLIENT_IDS, CLIENT_COLORS, CLIENT_LABELS } from './chart-utils'
import type { RoundResult } from '@/lib/fl-constants'

interface Props {
  rounds: RoundResult[]
}

export function RoundsF1Chart({ rounds }: Props) {
  const data = buildRoundRows(rounds, 'testF1')

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis
          dataKey="round"
          label={{ value: 'Round', position: 'insideBottomRight', offset: -8, fontSize: 11 }}
          tick={{ fontSize: 11 }}
          stroke="var(--muted-foreground)"
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => v.toFixed(2)}
          tick={{ fontSize: 11 }}
          stroke="var(--muted-foreground)"
          label={{ value: 'Test F1', angle: -90, position: 'insideLeft', fontSize: 11, offset: 8 }}
        />
        <Tooltip
          formatter={(value: number, name: string) => [
            value.toFixed(4),
            CLIENT_LABELS[name as keyof typeof CLIENT_LABELS] ?? name,
          ]}
          labelFormatter={(l) => `Round ${l}`}
          contentStyle={{
            fontSize: 12,
            borderRadius: 8,
            border: '1px solid var(--border)',
            background: 'var(--card)',
            color: 'var(--card-foreground)',
          }}
        />
        <Legend
          formatter={(value) => CLIENT_LABELS[value as keyof typeof CLIENT_LABELS] ?? value}
          wrapperStyle={{ fontSize: 12 }}
        />
        {CLIENT_IDS.map((id) => (
          <Line
            key={id}
            type="monotone"
            dataKey={id}
            stroke={CLIENT_COLORS[id]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
