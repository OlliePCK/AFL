import { NextResponse } from "next/server";
import { readCsv } from "@/lib/csv-reader";
import type { LiveOdds } from "@/lib/types";

export async function GET() {
  const data = readCsv<LiveOdds>("live_odds/current_odds.csv");
  return NextResponse.json(data);
}
