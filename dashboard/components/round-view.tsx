"use client";

import { useState } from "react";
import type { Prediction, ValueBet, PastResult } from "@/lib/types";
import { MatchCard } from "./match-card";
import { ResultCard } from "./result-card";
import { ValueBetTable } from "./value-bet-table";

interface RoundViewProps {
  predictions: Prediction[];
  results: PastResult[];
  valueBets: ValueBet[];
  rounds: { name: string; kind: "past" | "upcoming" }[];
  defaultRound: string;
}

export function RoundView({
  predictions,
  results,
  valueBets,
  rounds,
  defaultRound,
}: RoundViewProps) {
  const [selectedRound, setSelectedRound] = useState(defaultRound);

  const selected = rounds.find((r) => r.name === selectedRound);
  const isPast = selected?.kind === "past";

  const filteredResults = isPast
    ? results
        .filter((r) => r.roundname === selectedRound)
        .sort(
          (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime(),
        )
    : [];

  const filteredPredictions = !isPast
    ? predictions.filter((p) => p.roundname === selectedRound)
    : [];

  // Compute simple round stats for past rounds
  const stats = isPast
    ? {
        total: filteredResults.length,
        scored: filteredResults.filter((r) => r.predicted_winner != null && r.predicted_winner !== "")
          .length,
        correct: filteredResults.filter(
          (r) =>
            r.predicted_winner != null &&
            r.predicted_winner !== "" &&
            r.predicted_winner === r.winner,
        ).length,
      }
    : null;

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <select
          value={selectedRound}
          onChange={(e) => setSelectedRound(e.target.value)}
          className="bg-card border border-border rounded-md px-3 py-2 text-sm"
        >
          {rounds.map((r) => (
            <option key={r.name} value={r.name}>
              {r.name}
              {r.kind === "past" ? " (played)" : ""}
            </option>
          ))}
        </select>
        <span className="text-sm text-muted-foreground">
          {isPast
            ? `${filteredResults.length} matches`
            : `${filteredPredictions.length} matches`}
        </span>
        {stats && stats.scored > 0 && (
          <span className="text-sm text-muted-foreground">
            • Model: {stats.correct}/{stats.scored} (
            {Math.round((stats.correct / stats.scored) * 100)}%)
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mb-8">
        {isPast
          ? filteredResults.map((r) => (
              <ResultCard key={r.game_id} result={r} />
            ))
          : filteredPredictions.map((p) => (
              <MatchCard key={p.game_id} prediction={p} />
            ))}
      </div>

      {!isPast && valueBets.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Value Bets</h3>
          <ValueBetTable bets={valueBets} />
        </div>
      )}
    </div>
  );
}
