"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Prediction } from "@/lib/types";
import { TeamLogo } from "./team-logo";
import { ExplanationPanel } from "./explanation-panel";

function TeamBadge({ team }: { team: string }) {
  return (
    <div className="flex items-center gap-2">
      <TeamLogo team={team} size={32} />
      <span className="text-sm font-medium">{team}</span>
    </div>
  );
}

function ConfidenceBar({
  homeProb,
  height = "h-2",
  dim = false,
}: {
  homeProb: number;
  height?: string;
  dim?: boolean;
}) {
  const awayProb = 1 - homeProb;
  const conf = Math.max(homeProb, awayProb);
  const baseColor =
    conf > 0.7
      ? "bg-emerald-500"
      : conf > 0.55
        ? "bg-amber-500"
        : "bg-red-500";
  const color = dim ? `${baseColor} opacity-60` : baseColor;

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground w-10 text-right tabular-nums">
        {(homeProb * 100).toFixed(0)}%
      </span>
      <div className={`flex-1 ${height} rounded-full bg-muted overflow-hidden flex`}>
        <div className={`${color} transition-all`} style={{ width: `${homeProb * 100}%` }} />
      </div>
      <span className="text-xs text-muted-foreground w-10 tabular-nums">
        {(awayProb * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export function MatchCard({ prediction: p }: { prediction: Prediction }) {
  const [expanded, setExpanded] = useState(false);
  const isValue = p.value_team != null && p.edge != null && p.edge > 0.15;

  // Primary display uses the analytical probability when available
  // (best forecast — includes market signal). Falls back to betting prob.
  const hasAnalytical =
    p.analytical_home_prob != null && !Number.isNaN(p.analytical_home_prob);
  const primaryHomeProb = hasAnalytical
    ? (p.analytical_home_prob as number)
    : p.home_win_prob;
  const primaryAwayProb = 1 - primaryHomeProb;
  const primaryWinner =
    primaryHomeProb > primaryAwayProb
      ? p.home_team
      : primaryAwayProb > primaryHomeProb
        ? p.away_team
        : null;
  const conf = Math.max(primaryHomeProb, primaryAwayProb);

  const absMargin = p.predicted_margin ? Math.abs(p.predicted_margin) : 0;
  const winnerLabel = primaryWinner || p.predicted_winner || "Toss-up";
  const margin = absMargin >= 0.5
    ? `by ${Math.round(absMargin)} pts`
    : absMargin > 0
      ? "by <1 pt"
      : "";

  // Market implied probability from home odds (de-vigged approximation)
  const impliedHome =
    p.home_odds && p.away_odds
      ? (1 / p.home_odds) / (1 / p.home_odds + 1 / p.away_odds)
      : null;

  return (
    <Card className={`${isValue ? "ring-1 ring-emerald-500/50" : ""}`}>
      <CardContent className="p-4">
        <div className="flex justify-between items-start mb-3">
          <span className="text-xs text-muted-foreground">
            {p.date ? new Date(p.date).toLocaleDateString("en-AU", { weekday: "short", day: "numeric", month: "short" }) : ""}
          </span>
          <span className="text-xs text-muted-foreground">{p.venue}</span>
        </div>

        <div className="flex justify-between items-center mb-1">
          <TeamBadge team={p.home_team} />
          <span className="text-xs text-muted-foreground mx-2">vs</span>
          <TeamBadge team={p.away_team} />
        </div>

        <div className="mt-2 space-y-1.5">
          <div>
            <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-0.5">
              <span className="uppercase tracking-wide">
                {hasAnalytical ? "Forecast" : "Model"}
              </span>
              {hasAnalytical && (
                <span title="Market-informed forecast: includes bookmaker odds as features. Best estimate for tipping.">
                  with market ⓘ
                </span>
              )}
            </div>
            <ConfidenceBar homeProb={primaryHomeProb} />
          </div>

          {hasAnalytical && (
            <div>
              <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-0.5">
                <span className="uppercase tracking-wide">Independent</span>
                <span title="Independent model: no odds features. Compared to market to find value bets.">
                  no market ⓘ
                </span>
              </div>
              <ConfidenceBar homeProb={p.home_win_prob} height="h-1" dim />
            </div>
          )}

          {impliedHome != null && (
            <div className="flex items-center justify-between text-[10px] text-muted-foreground pt-0.5">
              <span>Market: {(impliedHome * 100).toFixed(0)}% / {((1 - impliedHome) * 100).toFixed(0)}%</span>
              {hasAnalytical && (
                <span>
                  Δ {((p.home_win_prob - impliedHome) * 100 >= 0 ? "+" : "")}
                  {((p.home_win_prob - impliedHome) * 100).toFixed(0)}pp vs market
                </span>
              )}
            </div>
          )}
        </div>

        <div className="mt-3 text-center">
          <span className="text-sm font-semibold">{winnerLabel}</span>
          <span className="text-xs text-muted-foreground ml-1">
            ({(conf * 100).toFixed(0)}%) {margin}
          </span>
        </div>

        {(p.home_odds || p.away_odds) && (
          <div className="flex justify-between mt-3 text-xs text-muted-foreground">
            <span>H: ${p.home_odds?.toFixed(2)}</span>
            <span>A: ${p.away_odds?.toFixed(2)}</span>
            {p.edge != null && p.edge > 0 && (
              <span className="text-emerald-400">
                Edge: {(p.edge * 100).toFixed(1)}%
              </span>
            )}
          </div>
        )}

        {isValue && (
          <div className="mt-2 flex justify-center">
            <Badge className="bg-emerald-600 text-white text-xs">
              VALUE: {p.value_team} @ {p.value_odds?.toFixed(2)}
            </Badge>
          </div>
        )}

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
            gameId={p.game_id}
            homeTeam={p.home_team}
            awayTeam={p.away_team}
          />
        )}
      </CardContent>
    </Card>
  );
}
