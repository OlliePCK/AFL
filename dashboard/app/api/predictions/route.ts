import { NextResponse } from "next/server";
import { readCsv } from "@/lib/csv-reader";
import type { Prediction } from "@/lib/types";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const round = searchParams.get("round");
  let data = readCsv<Prediction>("master/upcoming_predictions.csv");
  if (round) {
    data = data.filter((d) => d.roundname === round);
  }
  return NextResponse.json(data);
}
