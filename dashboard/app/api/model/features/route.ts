import { NextResponse } from "next/server";
import { readJson } from "@/lib/csv-reader";
import type { FeatureImportance } from "@/lib/types";

export async function GET() {
  const data = readJson<FeatureImportance[]>("model/feature_importance.json");
  return NextResponse.json(data ?? []);
}
