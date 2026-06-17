import { readCsv } from "@/lib/csv-reader";
import type { Prediction, ValueBet, PastResult } from "@/lib/types";
import { RoundView } from "@/components/round-view";

export const dynamic = "force-dynamic";

function roundSortKey(name: string): number {
  if (!name) return 999;
  if (name.toLowerCase().startsWith("opening")) return 0;
  const m = name.match(/(\d+)/);
  if (m) return parseInt(m[1], 10);
  // Finals etc. — push to the end
  if (/final/i.test(name)) return 900;
  return 500;
}

export default function Home() {
  const predictions = readCsv<Prediction>("master/upcoming_predictions.csv");
  const valueBets = readCsv<ValueBet>("master/value_bets.csv");
  const results = readCsv<PastResult>("master/current_season_results.csv");

  const pastRounds = [
    ...new Set(results.map((r) => r.roundname).filter(Boolean)),
  ];
  const upcomingRounds = [
    ...new Set(predictions.map((p) => p.roundname).filter(Boolean)),
  ];

  // Build a unified, sorted list of (round, kind), preferring upcoming for
  // any round that appears in both (e.g. mid-round when some games played).
  const upcomingSet = new Set(upcomingRounds);
  const merged = [
    ...pastRounds
      .filter((r) => !upcomingSet.has(r))
      .map((r) => ({ name: r, kind: "past" as const })),
    ...upcomingRounds.map((r) => ({ name: r, kind: "upcoming" as const })),
  ].sort((a, b) => roundSortKey(a.name) - roundSortKey(b.name));

  // Default to the first upcoming round (the "current" round)
  const defaultRound =
    merged.find((r) => r.kind === "upcoming")?.name ??
    merged[merged.length - 1]?.name ??
    "";

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Round Overview</h2>
      <RoundView
        predictions={predictions}
        results={results}
        valueBets={valueBets}
        rounds={merged}
        defaultRound={defaultRound}
      />
    </div>
  );
}
