import { NextResponse } from "next/server";
import { readCsv } from "@/lib/csv-reader";
import type { ValueBet } from "@/lib/types";

export async function GET() {
  const data = readCsv<ValueBet>("master/value_bets.csv");
  return NextResponse.json(data);
}
