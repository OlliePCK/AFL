"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { PastResult } from "@/lib/types";
import { TeamLogo } from "./team-logo";
import { ExplanationPanel } from "./explanation-panel";

function TeamRow({
  team,
  score,
  isWinner,
}: {
  team: string;
  score: number;
  isWinner: boolean;
}) {
  return (
    <div
      className={`flex items-center justify-between py-1 ${
        isWinner ? "" : "opacity-60"
      }`}
    >
      <div className="flex items-center gap-2">
        <TeamLogo team={team} size={32} />
        <span className="text-sm font-medium">{team}</span>
      </div>
      <span className={`text-lg ${isWinner ? "font-bold" : "font-medium"}`}>
        {score}
      </span>
    </div>
  );
}

export function ResultCard({ result: r }: { result: PastResult }) {
  const [expanded, setExpanded] = useState(false);
  const homeWin = r.home_score > r.away_score;
  const isDraw = r.home_score === r.away_score;
  const winner = isDraw ? "Draw" : r.winner;

  // Compare prediction to actual (if a snapshot exists)
  const hasPrediction =
    r.predicted_winner != null && r.predicted_winner !== "";
  const correct = hasPrediction && r.predicted_winner === winner;

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex justify-between items-start mb-3">
          <span className="text-xs text-muted-foreground">
            {r.date
              ? new Date(r.date).toLocaleDateString("en-AU", {
                  weekday: "short",
                  day: "numeric",
                  month: "short",
                })
              : ""}
          </span>
          <span className="text-xs text-muted-foreground">{r.venue}</span>
        </div>

        <div className="space-y-1">
          <TeamRow
            team={r.home_team}
            score={r.home_score}
            isWinner={homeWin || isDraw}
          />
          <TeamRow
            team={r.away_team}
            score={r.away_score}
            isWinner={!homeWin || isDraw}
          />
        </div>

        <div className="mt-3 flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            {isDraw ? "Drawn" : `${winner} won`}
          </span>
          {hasPrediction && (
            <Badge
              className={
                correct
                  ? "bg-emerald-600 text-white text-xs"
                  : "bg-red-600 text-white text-xs"
              }
            >
              {correct ? "✓" : "✗"} Predicted {r.predicted_winner}
              {r.home_win_prob != null && (
                <span className="ml-1 opacity-80">
                  (
                  {Math.round(
                    (r.predicted_winner === r.home_team
                      ? r.home_win_prob
                      : 1 - r.home_win_prob) * 100,
                  )}
                  %)
                </span>
              )}
            </Badge>
          )}
        </div>

        {hasPrediction && (
          <>
            <button
              type="button"
              onClick={() => setExpanded((x) => !x)}
              className="mt-3 w-full text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center justify-center gap-1"
            >
              <span>{expanded ? "Hide reasoning" : "Why this pick?"}</span>
              <span className={`transition-transform ${expanded ? "rotate-180" : ""}`}>▾</span>
            </button>

            {expanded && (
              <ExplanationPanel
                gameId={r.game_id}
                homeTeam={r.home_team}
                awayTeam={r.away_team}
                snapshotDate={r.snapshot_date}
              />
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
