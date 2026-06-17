import { NextResponse } from "next/server";
import { readJson } from "@/lib/csv-reader";
import type {
  PredictionExplanations,
  PredictionExplanation,
  PredictionExplanationHistory,
} from "@/lib/types";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const gameId = searchParams.get("game_id");
  const snapshotDate = searchParams.get("snapshot_date");

  const live = readJson<PredictionExplanations>(
    "master/prediction_explanations.json"
  ) ?? {};
  const history = readJson<PredictionExplanationHistory>(
    "master/prediction_explanations_history.json"
  ) ?? {};

  if (gameId) {
    if (snapshotDate) {
      const exact = history[gameId]?.[snapshotDate];
      if (exact) {
        return NextResponse.json(exact);
      }
    }

    const liveSingle: PredictionExplanation | undefined = live[gameId];
    if (liveSingle) {
      return NextResponse.json(liveSingle);
    }

    const historical = history[gameId];
    if (historical) {
      const latestDate = Object.keys(historical).sort().at(-1);
      if (latestDate) {
        return NextResponse.json(historical[latestDate]);
      }
    }

    return NextResponse.json(null);
  }

  return NextResponse.json(live);
}
