import { NextResponse } from "next/server";
import { readJson } from "@/lib/csv-reader";

export async function GET() {
  const data = readJson<{ yearly: unknown[]; summary: unknown }>(
    "model/walk_forward_results.json"
  );
  return NextResponse.json(data ?? { yearly: [], summary: {} });
}
