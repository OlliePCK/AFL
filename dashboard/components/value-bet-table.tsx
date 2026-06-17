"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import type { ValueBet } from "@/lib/types";

const MOVE_STYLES: Record<string, string> = {
  AGREE: "bg-emerald-600 text-white",
  DISAGREE: "bg-red-600 text-white",
  NEUTRAL: "bg-zinc-600 text-white",
  unknown: "bg-zinc-700 text-zinc-300",
};

export function ValueBetTable({ bets }: { bets: ValueBet[] }) {
  const total = bets.reduce((s, b) => s + b.bet_amount, 0);

  return (
    <div className="rounded-md border border-border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Team</TableHead>
            <TableHead className="text-right">Edge</TableHead>
            <TableHead className="text-right">Odds</TableHead>
            <TableHead className="text-right">EV/$</TableHead>
            <TableHead className="text-right">Kelly</TableHead>
            <TableHead className="text-right">Bet</TableHead>
            <TableHead>Movement</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {bets.map((b, i) => (
            <TableRow key={i}>
              <TableCell className="font-medium">{b.bet_team}</TableCell>
              <TableCell className="text-right text-emerald-400">
                {(b.edge * 100).toFixed(1)}%
              </TableCell>
              <TableCell className="text-right">{b.odds.toFixed(2)}</TableCell>
              <TableCell className="text-right">
                +${b.ev_per_dollar.toFixed(2)}
              </TableCell>
              <TableCell className="text-right">
                {(b.kelly_pct * 100).toFixed(1)}%
              </TableCell>
              <TableCell className="text-right font-medium">
                ${b.bet_amount.toFixed(0)}
              </TableCell>
              <TableCell>
                <Badge className={MOVE_STYLES[b.movement] || MOVE_STYLES.unknown}>
                  {b.movement}
                </Badge>
              </TableCell>
            </TableRow>
          ))}
          <TableRow>
            <TableCell colSpan={5} className="text-right text-sm text-muted-foreground">
              Total ({bets.length} bets):
            </TableCell>
            <TableCell className="text-right font-bold">${total.toFixed(0)}</TableCell>
            <TableCell />
          </TableRow>
        </TableBody>
      </Table>
    </div>
  );
}
