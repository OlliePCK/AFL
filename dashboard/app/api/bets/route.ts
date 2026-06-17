import { NextResponse } from "next/server";
import { readCsv } from "@/lib/csv-reader";
import type { Bet } from "@/lib/types";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const status = searchParams.get("status");
  let data = readCsv<Bet>("betting/bet_log.csv");
  if (status) {
    data = data.filter((d) => d.status === status);
  }
  return NextResponse.json(data);
}
