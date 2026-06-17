"use client";

import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import type { Bet, BettingSummary } from "@/lib/types";

function StatCard({
  label,
  value,
  sub,
  valueClass,
}: {
  label: string;
  value: string;
  sub?: string;
  valueClass?: string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className={`text-2xl font-bold mt-1 ${valueClass ?? ""}`}>{value}</p>
        {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
      </CardContent>
    </Card>
  );
}

const STATUS_STYLES: Record<string, string> = {
  won: "bg-emerald-600 text-white",
  lost: "bg-red-600 text-white",
  pending: "bg-amber-600 text-white",
  void: "bg-zinc-600 text-white",
};

interface EdgeBucket {
  label: string;
  min: number;
  max: number;
  n: number;
  wins: number;
  staked: number;
  pl: number;
}

function buildEdgeBuckets(settled: Bet[]): EdgeBucket[] {
  const buckets: EdgeBucket[] = [
    { label: "15-20%", min: 0.15, max: 0.20, n: 0, wins: 0, staked: 0, pl: 0 },
    { label: "20-25%", min: 0.20, max: 0.25, n: 0, wins: 0, staked: 0, pl: 0 },
    { label: "25-30%", min: 0.25, max: 0.30, n: 0, wins: 0, staked: 0, pl: 0 },
    { label: "30%+",   min: 0.30, max: 10,   n: 0, wins: 0, staked: 0, pl: 0 },
  ];
  for (const b of settled) {
    const e = b.edge ?? 0;
    const bucket = buckets.find((x) => e >= x.min && e < x.max);
    if (!bucket) continue;
    bucket.n += 1;
    bucket.wins += b.status === "won" ? 1 : 0;
    bucket.staked += b.bet_amount || 0;
    bucket.pl += b.profit_loss || 0;
  }
  return buckets;
}

export function BettingDashboard({
  bets,
  summary,
}: {
  bets: Bet[];
  summary: BettingSummary;
}) {
  const plColor = summary.total_pl >= 0 ? "text-emerald-400" : "text-red-400";
  const roiColor = summary.roi_pct >= 0 ? "text-emerald-400" : "text-red-400";

  const settled = useMemo(
    () => bets.filter((b) => b.status === "won" || b.status === "lost"),
    [bets],
  );
  const pending = useMemo(
    () => bets.filter((b) => b.status === "pending"),
    [bets],
  );

  // Sort settled bets by reconciliation date (or date_placed as fallback)
  // for the cumulative P&L curve.
  const plCurve = useMemo(() => {
    const sorted = [...settled].sort((a, b) => {
      const ad = a.reconciled_at || a.date_placed || "";
      const bd = b.reconciled_at || b.date_placed || "";
      return ad.localeCompare(bd);
    });
    let cum = 0;
    return sorted.map((b, i) => {
      cum += b.profit_loss ?? 0;
      return {
        idx: i + 1,
        label: `#${i + 1}`,
        date: b.reconciled_at || b.date_placed || "",
        cumulative: Math.round(cum * 100) / 100,
        pl: b.profit_loss ?? 0,
        bet: `${b.home_team} vs ${b.away_team}`,
      };
    });
  }, [settled]);

  const edgeBuckets = useMemo(() => buildEdgeBuckets(settled), [settled]);
  const hasEdgeData = edgeBuckets.some((b) => b.n > 0);

  const avgEdge = useMemo(() => {
    const e = settled
      .map((b) => b.edge)
      .filter((x): x is number => x != null);
    if (!e.length) return null;
    return e.reduce((s, x) => s + x, 0) / e.length;
  }, [settled]);

  const bestBet = useMemo(() => {
    if (!settled.length) return null;
    return [...settled].sort(
      (a, b) => (b.profit_loss ?? 0) - (a.profit_loss ?? 0),
    )[0];
  }, [settled]);

  const worstBet = useMemo(() => {
    if (!settled.length) return null;
    return [...settled].sort(
      (a, b) => (a.profit_loss ?? 0) - (b.profit_loss ?? 0),
    )[0];
  }, [settled]);

  return (
    <div>
      {/* Summary stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        <StatCard
          label="Total P&L"
          value={`$${summary.total_pl >= 0 ? "+" : ""}${summary.total_pl.toFixed(0)}`}
          valueClass={plColor}
        />
        <StatCard
          label="ROI"
          value={`${summary.roi_pct >= 0 ? "+" : ""}${summary.roi_pct.toFixed(1)}%`}
          valueClass={roiColor}
          sub={`on $${summary.total_staked.toFixed(0)} staked`}
        />
        <StatCard
          label="Win Rate"
          value={`${(summary.win_rate * 100).toFixed(0)}%`}
          sub={`${summary.record.won}W - ${summary.record.lost}L`}
        />
        <StatCard
          label="Avg Edge"
          value={avgEdge != null ? `${(avgEdge * 100).toFixed(1)}%` : "—"}
          sub={`${settled.length} settled`}
        />
        <StatCard
          label="Pending"
          value={`$${summary.pending_exposure.toFixed(0)}`}
          sub={`${summary.record.pending} bet(s)`}
        />
        <StatCard
          label="Total Bets"
          value={`${bets.length}`}
          sub={`${settled.length} settled, ${summary.record.pending} live`}
        />
      </div>

      {bets.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center text-muted-foreground">
            No bets recorded yet. Run{" "}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">
              python run_predictions.py
            </code>{" "}
            to detect and log value bets.
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Charts row: P&L curve + edge analysis */}
          {settled.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-baseline justify-between mb-3">
                    <h3 className="text-sm font-semibold">Cumulative P&L</h3>
                    <span className="text-xs text-muted-foreground">
                      {settled.length} settled bet{settled.length === 1 ? "" : "s"}
                    </span>
                  </div>
                  <div style={{ width: "100%", height: 220 }}>
                    <ResponsiveContainer>
                      <LineChart
                        data={plCurve}
                        margin={{ top: 4, right: 8, bottom: 4, left: 0 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="hsl(var(--border))"
                          opacity={0.4}
                        />
                        <XAxis
                          dataKey="label"
                          tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                          stroke="hsl(var(--border))"
                        />
                        <YAxis
                          tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                          stroke="hsl(var(--border))"
                          tickFormatter={(v) => `$${v}`}
                          width={48}
                        />
                        <ReferenceLine
                          y={0}
                          stroke="hsl(var(--muted-foreground))"
                          strokeDasharray="2 2"
                        />
                        <Tooltip
                          contentStyle={{
                            background: "hsl(var(--popover))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: 6,
                            fontSize: 12,
                          }}
                          labelFormatter={(_, payload) => {
                            const p = payload?.[0]?.payload as
                              | { bet: string; date: string }
                              | undefined;
                            if (!p) return "";
                            const d = p.date
                              ? new Date(p.date).toLocaleDateString("en-AU")
                              : "";
                            return `${p.bet}${d ? ` · ${d}` : ""}`;
                          }}
                          formatter={(value, name) => {
                            const v = typeof value === "number" ? value : Number(value);
                            if (name === "cumulative")
                              return [`$${v >= 0 ? "+" : ""}${v.toFixed(2)}`, "Cumulative"];
                            return [`$${v.toFixed(2)}`, String(name)];
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="cumulative"
                          stroke={summary.total_pl >= 0 ? "#10b981" : "#ef4444"}
                          strokeWidth={2}
                          dot={{ r: 3 }}
                          activeDot={{ r: 5 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-baseline justify-between mb-3">
                    <h3 className="text-sm font-semibold">ROI by Edge Band</h3>
                    <span className="text-xs text-muted-foreground">
                      ROI % per edge bucket
                    </span>
                  </div>
                  {hasEdgeData ? (
                    <div style={{ width: "100%", height: 220 }}>
                      <ResponsiveContainer>
                        <BarChart
                          data={edgeBuckets.map((b) => ({
                            label: b.label,
                            roi: b.staked > 0 ? (b.pl / b.staked) * 100 : 0,
                            n: b.n,
                            winRate: b.n > 0 ? (b.wins / b.n) * 100 : 0,
                            pl: b.pl,
                          }))}
                          margin={{ top: 4, right: 8, bottom: 4, left: 0 }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="hsl(var(--border))"
                            opacity={0.4}
                          />
                          <XAxis
                            dataKey="label"
                            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                            stroke="hsl(var(--border))"
                          />
                          <YAxis
                            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                            stroke="hsl(var(--border))"
                            tickFormatter={(v) => `${v}%`}
                            width={48}
                          />
                          <ReferenceLine
                            y={0}
                            stroke="hsl(var(--muted-foreground))"
                            strokeDasharray="2 2"
                          />
                          <Tooltip
                            contentStyle={{
                              background: "hsl(var(--popover))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: 6,
                              fontSize: 12,
                            }}
                            formatter={(value, name, item) => {
                              const v = typeof value === "number" ? value : Number(value);
                              const p = item?.payload as
                                | { n: number; winRate: number; pl: number }
                                | undefined;
                              if (name === "roi") {
                                return [
                                  `${v.toFixed(1)}% · ${p?.n ?? 0} bet${p?.n === 1 ? "" : "s"} · ${p?.winRate.toFixed(0) ?? 0}% win · $${(p?.pl ?? 0).toFixed(0)}`,
                                  "ROI",
                                ];
                              }
                              return [String(value), String(name)];
                            }}
                          />
                          <Bar dataKey="roi" radius={[4, 4, 0, 0]}>
                            {edgeBuckets.map((b, i) => {
                              const roi = b.staked > 0 ? b.pl / b.staked : 0;
                              return (
                                <Cell
                                  key={i}
                                  fill={b.n === 0 ? "#3f3f46" : roi >= 0 ? "#10b981" : "#ef4444"}
                                  opacity={b.n === 0 ? 0.3 : 1}
                                />
                              );
                            })}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="h-[220px] flex items-center justify-center text-xs text-muted-foreground">
                      Edge analysis available after more bets settle
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}

          {/* Best / worst callout */}
          {(bestBet || worstBet) && settled.length > 1 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
              {bestBet && (
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Best Bet</p>
                    <p className="text-sm font-medium mt-1">
                      {bestBet.bet_team}
                      <span className="text-muted-foreground ml-1">
                        ({bestBet.home_team} vs {bestBet.away_team})
                      </span>
                    </p>
                    <p className="text-lg font-bold text-emerald-400 mt-1">
                      +${(bestBet.profit_loss ?? 0).toFixed(0)}
                      <span className="text-xs font-normal text-muted-foreground ml-2">
                        @ {bestBet.odds?.toFixed(2)}, edge {((bestBet.edge ?? 0) * 100).toFixed(1)}%
                      </span>
                    </p>
                  </CardContent>
                </Card>
              )}
              {worstBet && (bestBet?.bet_id !== worstBet.bet_id) && (
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Worst Bet</p>
                    <p className="text-sm font-medium mt-1">
                      {worstBet.bet_team}
                      <span className="text-muted-foreground ml-1">
                        ({worstBet.home_team} vs {worstBet.away_team})
                      </span>
                    </p>
                    <p
                      className={`text-lg font-bold mt-1 ${(worstBet.profit_loss ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}
                    >
                      {(worstBet.profit_loss ?? 0) >= 0 ? "+" : ""}${(worstBet.profit_loss ?? 0).toFixed(0)}
                      <span className="text-xs font-normal text-muted-foreground ml-2">
                        @ {worstBet.odds?.toFixed(2)}, edge {((worstBet.edge ?? 0) * 100).toFixed(1)}%
                      </span>
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {/* Pending bets callout */}
          {pending.length > 0 && (
            <Card className="mb-6 ring-1 ring-amber-500/40">
              <CardContent className="p-4">
                <div className="flex items-baseline justify-between mb-2">
                  <h3 className="text-sm font-semibold">Live Bets</h3>
                  <span className="text-xs text-muted-foreground">
                    ${summary.pending_exposure.toFixed(0)} exposed · {pending.length} open
                  </span>
                </div>
                <div className="space-y-1.5">
                  {pending.map((b) => (
                    <div
                      key={b.bet_id}
                      className="flex items-center justify-between text-sm"
                    >
                      <span className="truncate">
                        <span className="font-medium">{b.bet_team}</span>
                        <span className="text-muted-foreground ml-2 text-xs">
                          {b.home_team} vs {b.away_team} · {b.round}
                        </span>
                      </span>
                      <span className="text-xs text-muted-foreground whitespace-nowrap ml-2">
                        ${b.bet_amount.toFixed(0)} @ {b.odds?.toFixed(2)} · edge {((b.edge ?? 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Bet log */}
          <div className="rounded-md border border-border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead>Round</TableHead>
                  <TableHead>Match</TableHead>
                  <TableHead>Bet</TableHead>
                  <TableHead className="text-right">Odds</TableHead>
                  <TableHead className="text-right">Edge</TableHead>
                  <TableHead className="text-right">Amount</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">P&L</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[...bets]
                  .sort((a, b) =>
                    (b.date_placed || "").localeCompare(a.date_placed || ""),
                  )
                  .map((b) => (
                    <TableRow key={b.bet_id}>
                      <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                        {b.date_placed
                          ? new Date(b.date_placed).toLocaleDateString("en-AU", {
                              day: "numeric",
                              month: "short",
                            })
                          : "—"}
                      </TableCell>
                      <TableCell className="text-xs">{b.round}</TableCell>
                      <TableCell className="text-sm">
                        {b.home_team} vs {b.away_team}
                      </TableCell>
                      <TableCell className="font-medium">{b.bet_team}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {b.odds?.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-right text-emerald-400 tabular-nums">
                        {((b.edge || 0) * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        ${b.bet_amount?.toFixed(0)}
                      </TableCell>
                      <TableCell>
                        <Badge className={STATUS_STYLES[b.status] || ""}>
                          {b.status}
                        </Badge>
                      </TableCell>
                      <TableCell
                        className={`text-right font-medium tabular-nums ${
                          b.profit_loss != null
                            ? b.profit_loss >= 0
                              ? "text-emerald-400"
                              : "text-red-400"
                            : "text-muted-foreground"
                        }`}
                      >
                        {b.profit_loss != null
                          ? `$${b.profit_loss >= 0 ? "+" : ""}${b.profit_loss.toFixed(0)}`
                          : "—"}
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </div>
        </>
      )}
    </div>
  );
}
