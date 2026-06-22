"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface ChartRow {
  clientId: string;
  label: string;
  localF1: number;
  globalF1: number;
}

interface Props {
  rows: ChartRow[];
}

export function LocalGlobalChart({ rows }: Props) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart
        data={rows}
        margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
        barCategoryGap="30%"
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="var(--border)"
          vertical={false}
        />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 12 }}
          stroke="var(--muted-foreground)"
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => v.toFixed(2)}
          tick={{ fontSize: 11 }}
          stroke="var(--muted-foreground)"
          label={{
            value: "Test F1",
            angle: -90,
            position: "insideLeft",
            fontSize: 11,
            offset: 8,
          }}
        />
        <Tooltip
          formatter={(value: number, name: string) => [
            value.toFixed(4),
            name === "localF1" ? "Local (Phase 1)" : "Global / FL (Phase 2)",
          ]}
          contentStyle={{
            fontSize: 12,
            borderRadius: 8,
            border: "1px solid var(--border)",
            background: "var(--card)",
            color: "var(--card-foreground)",
          }}
        />
        <Legend
          formatter={(value) =>
            value === "localF1" ? "Local (Phase 1)" : "Global / FL (Phase 2)"
          }
          wrapperStyle={{ fontSize: 12 }}
        />
        <Bar
          dataKey="localF1"
          name="localF1"
          fill="#94a3b8"
          radius={[3, 3, 0, 0]}
        />
        <Bar
          dataKey="globalF1"
          name="globalF1"
          fill="#3b82f6"
          radius={[3, 3, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
