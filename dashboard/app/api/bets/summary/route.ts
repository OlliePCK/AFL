import { NextResponse } from "next/server";
import { readCsv } from "@/lib/csv-reader";
import type { Bet, BettingSummary } from "@/lib/types";

export async function GET() {
  const bets = readCsv<Bet>("betting/bet_log.csv");
  const settled = bets.filter((b) => b.status === "won" || b.status === "lost");
  const won = bets.filter((b) => b.status === "won");
  const lost = bets.filter((b) => b.status === "lost");
  const pending = bets.filter((b) => b.status === "pending");

  const totalStaked = settled.reduce((s, b) => s + (b.bet_amount || 0), 0);
  const totalPl = settled.reduce((s, b) => s + (b.profit_loss || 0), 0);
  const pendingExposure = pending.reduce((s, b) => s + (b.bet_amount || 0), 0);

  const summary: BettingSummary = {
    total_pl: totalPl,
    roi_pct: totalStaked > 0 ? (totalPl / totalStaked) * 100 : 0,
    win_rate: settled.length > 0 ? won.length / settled.length : 0,
    record: { won: won.length, lost: lost.length, pending: pending.length },
    total_staked: totalStaked,
    pending_exposure: pendingExposure,
  };

  return NextResponse.json(summary);
}
