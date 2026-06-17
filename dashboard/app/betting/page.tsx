import { readCsv } from "@/lib/csv-reader";
import type { Bet } from "@/lib/types";
import { BettingDashboard } from "@/components/betting-dashboard";

export const dynamic = "force-dynamic";

export default function BettingPage() {
  const bets = readCsv<Bet>("betting/bet_log.csv");

  const settled = bets.filter((b) => b.status === "won" || b.status === "lost");
  const won = bets.filter((b) => b.status === "won");
  const pending = bets.filter((b) => b.status === "pending");
  const totalStaked = settled.reduce((s, b) => s + (b.bet_amount || 0), 0);
  const totalPl = settled.reduce((s, b) => s + (b.profit_loss || 0), 0);
  const pendingExposure = pending.reduce((s, b) => s + (b.bet_amount || 0), 0);

  const summary = {
    total_pl: totalPl,
    roi_pct: totalStaked > 0 ? (totalPl / totalStaked) * 100 : 0,
    win_rate: settled.length > 0 ? won.length / settled.length : 0,
    record: { won: won.length, lost: settled.length - won.length, pending: pending.length },
    total_staked: totalStaked,
    pending_exposure: pendingExposure,
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Betting Tracker</h2>
      <BettingDashboard bets={bets} summary={summary} />
    </div>
  );
}
