"use client";

import { useEffect, useState } from "react";
import type { PredictionExplanation } from "@/lib/types";
import { labelFor } from "@/lib/feature-labels";

interface ExplanationPanelProps {
  gameId: number;
  homeTeam: string;
  awayTeam: string;
  snapshotDate?: string | null;
}

function formatValue(name: string, value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  // Boolean-ish features
  if (
    name.startsWith("home_") ||
    name.startsWith("away_") ||
    name === "is_wet" ||
    name === "wind_strong" ||
    name === "is_roofed"
  ) {
    if (value === 0 || value === 1) return value === 1 ? "Yes" : "No";
  }
  // Integer-like
  if (Math.abs(value) >= 100 || Number.isInteger(value)) {
    return value.toFixed(0);
  }
  return value.toFixed(2);
}

export function ExplanationPanel({
  gameId,
  homeTeam,
  awayTeam,
  snapshotDate,
}: ExplanationPanelProps) {
  const [data, setData] = useState<PredictionExplanation | null | undefined>(
    undefined
  );

  useEffect(() => {
    let cancelled = false;
    const params = new URLSearchParams({ game_id: String(gameId) });
    if (snapshotDate) {
      params.set("snapshot_date", snapshotDate);
    }
    fetch(`/api/predictions/explanations?${params.toString()}`)
      .then((r) => r.json())
      .then((json) => {
        if (!cancelled) setData(json ?? null);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [gameId, snapshotDate]);

  if (data === undefined) {
    return (
      <div className="mt-3 text-xs text-muted-foreground">Loading explanation…</div>
    );
  }
  if (data === null) {
    return (
      <div className="mt-3 text-xs text-muted-foreground">
        No explanation available for this match.
      </div>
    );
  }

  const maxAbs = Math.max(...data.features.map((f) => Math.abs(f.shap)), 0.0001);

  return (
    <div className="mt-3 border-t border-muted pt-3">
      {data.summary && (
        <p className="text-sm mb-3 leading-relaxed text-foreground/90">
          {data.summary}
        </p>
      )}
      <p className="text-xs text-muted-foreground mb-2">
        Top drivers — positive bar favours{" "}
        <span className="text-emerald-400">{homeTeam}</span>, negative favours{" "}
        <span className="text-red-400">{awayTeam}</span>
      </p>
      <div className="space-y-1.5">
        {data.features.map((f) => {
          const pct = (Math.abs(f.shap) / maxAbs) * 100;
          const positive = f.shap >= 0;
          return (
            <div key={f.name} className="text-xs">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground truncate pr-2">
                  {labelFor(f.name)}
                </span>
                <span className="font-mono text-[10px] text-muted-foreground shrink-0">
                  {formatValue(f.name, f.value)}
                </span>
              </div>
              <div className="relative h-1.5 rounded-full bg-muted mt-0.5 overflow-hidden">
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-muted-foreground/40" />
                <div
                  className={`absolute top-0 bottom-0 ${
                    positive ? "bg-emerald-500 left-1/2" : "bg-red-500 right-1/2"
                  }`}
                  style={{ width: `${pct / 2}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
