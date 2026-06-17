import { NextResponse } from "next/server";
import { readCsv } from "@/lib/csv-reader";

const FEATURE_COLS = [
  "home_team", "away_team", "roundname", "year",
  "elo_diff", "elo_expected",
  "home_interstate", "away_interstate",
  "home_at_home_ground", "away_at_home_ground",
  "venue_win_rate_diff",
  "win_rate_diff_5", "avg_margin_diff_5",
  "score_for_diff_5", "score_against_diff_5",
  "lineup_changes_diff", "lineup_continuity_diff",
  "home_ruck_missing", "away_ruck_missing",
  "missing_rating_diff", "net_quality_change_diff",
  "missing_mid_rating_diff", "missing_fwd_rating_diff",
  "missing_def_rating_diff",
  "home_score",
];

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const round = searchParams.get("round") || "Round 4";

  const raw = readCsv<Record<string, unknown>>("master/afl_featured_dataset.csv");
  // Upcoming matches have null home_score
  const upcoming = raw.filter(
    (r) => r.home_score === null && r.roundname === round
  );

  const data = upcoming.map((r) => {
    const obj: Record<string, unknown> = {};
    for (const col of FEATURE_COLS) {
      obj[col] = r[col] ?? null;
    }
    return obj;
  });

  return NextResponse.json(data);
}
