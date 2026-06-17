import { readJson } from "@/lib/csv-reader";
import type {
  CalibrationCurve,
  FeatureImportance,
  ModelMetrics,
} from "@/lib/types";
import { InsightsDashboard } from "@/components/insights-dashboard";

export const dynamic = "force-dynamic";

export default function InsightsPage() {
  const features = readJson<FeatureImportance[]>("model/feature_importance.json") ?? [];
  const metrics = readJson<ModelMetrics>("model/model_metrics.json");
  const walkForward = readJson<{ yearly: { year: number; n_bets: number; win_rate: number; roi: number; total_profit: number }[]; summary: Record<string, unknown> }>("model/walk_forward_results.json");
  const calibration = readJson<CalibrationCurve>("model/calibration_curve.json");

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Model Insights</h2>
      <InsightsDashboard
        features={features}
        metrics={metrics}
        walkForward={walkForward}
        calibration={calibration}
      />
    </div>
  );
}
