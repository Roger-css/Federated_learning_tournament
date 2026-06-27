"use client";

import { useEffect, useState } from "react";
import { useFlSocket } from "@/hooks/use-fl-socket";
import { ConnectionBadge } from "@/components/connection-badge";
import { RoundsF1Chart } from "@/components/charts/rounds-f1-chart";
import { RoundsAccuracyChart } from "@/components/charts/rounds-accuracy-chart";
import { LatestRoundTable } from "@/components/latest-round-table";
import { FL_API_BASE, type RoundResult } from "@/lib/fl-constants";
import { AlertCircle, Radio } from "lucide-react";

export default function DashboardPage() {
  const { rounds: liveRounds, connectionState } = useFlSocket();
  const [rounds, setRounds] = useState<RoundResult[]>([]);
  const [fetchError, setFetchError] = useState(false);
  const [loaded, setLoaded] = useState(false);

  // Initial REST fetch for historical data
  useEffect(() => {
    fetch(`${FL_API_BASE}/api/fl/rounds`)
      .then((r) => r.json())
      .then((data: { rounds: RoundResult[] }) => {
        setRounds(data.rounds ?? []);
        setLoaded(true);
      })
      .catch((e) => {
        console.log(e);

        setFetchError(true);
        setLoaded(true);
      });
  }, []);

  // Merge live SignalR rounds on top
  useEffect(() => {
    if (liveRounds.length === 0) return;
    setRounds((prev) => {
      const merged = [...prev];
      for (const round of liveRounds) {
        const idx = merged.findIndex(
          (r) => r.roundNumber === round.roundNumber,
        );
        if (idx >= 0) merged[idx] = round;
        else merged.push(round);
      }
      return merged.sort((a, b) => a.roundNumber - b.roundNumber);
    });
  }, [liveRounds]);

  function roundAvgF1(r: RoundResult) {
    if (r.clients.length === 0) return 0;
    return r.clients.reduce((sum, c) => sum + c.testF1, 0) / r.clients.length;
  }

  const bestRound = rounds.length > 0
    ? rounds.reduce((best, r) => (roundAvgF1(r) > roundAvgF1(best) ? r : best))
    : null;

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Page header */}
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-foreground">
            FL Dashboard
          </h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Training collaboration across 4 oil wells — per-round metrics
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {bestRound && (
            <span className="flex items-center gap-1.5 rounded-md border border-border bg-muted px-2.5 py-1 text-xs font-medium text-foreground">
              <Radio className="h-3.5 w-3.5" aria-hidden="true" />
              Best Round {bestRound.roundNumber}
            </span>
          )}
          <ConnectionBadge state={connectionState} />
        </div>
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
              Is the .NET server running on{" "}
              <code className="rounded bg-destructive/20 px-1 font-mono text-xs">
                {FL_API_BASE}
              </code>
              ?
            </p>
          </div>
        </div>
      )}

      {/* Empty state */}
      {loaded && !fetchError && rounds.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20 text-center">
          <Radio
            className="mb-3 h-8 w-8 text-muted-foreground"
            aria-hidden="true"
          />
          <p className="font-medium text-foreground">No training data yet</p>
          <p className="mt-1 max-w-xs text-sm text-muted-foreground">
            Start the FL simulation to see live round-by-round results here.
          </p>
        </div>
      )}

      {/* Charts + table */}
      {rounds.length > 0 && (
        <>
          <section aria-label="Test F1 over rounds">
            <h2 className="mb-3 text-sm font-medium text-foreground">
              Test F1 Score — per Round
            </h2>
            <div className="rounded-xl border border-border bg-card p-4">
              <RoundsF1Chart rounds={rounds} />
            </div>
          </section>

          <section aria-label="Accuracy over rounds">
            <h2 className="mb-3 text-sm font-medium text-foreground">
              Accuracy — per Round
            </h2>
            <div className="rounded-xl border border-border bg-card p-4">
              <RoundsAccuracyChart rounds={rounds} />
            </div>
          </section>

          <section aria-label="Best round summary">
            <h2 className="mb-3 text-sm font-medium text-foreground">
              Best Round Summary
              {bestRound && (
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  (Round {bestRound.roundNumber})
                </span>
              )}
            </h2>
            <div className="rounded-xl border border-border bg-card overflow-hidden">
              <LatestRoundTable clients={bestRound?.clients ?? []} />
            </div>
          </section>
        </>
      )}
    </div>
  );
}
