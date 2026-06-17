import { NextResponse } from "next/server";
import { readJson } from "@/lib/csv-reader";
import type { ModelMetrics } from "@/lib/types";

export async function GET() {
  const data = readJson<ModelMetrics>("model/model_metrics.json");
  return NextResponse.json(data ?? {});
}
