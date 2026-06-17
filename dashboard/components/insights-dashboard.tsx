"use client";

import { Card, CardContent } from "@/components/ui/card";
import type {
  CalibrationCurve,
  FeatureImportance,
  ModelMetrics,
} from "@/lib/types";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  CartesianGrid, ReferenceLine,
  ComposedChart, Line, Scatter,
} from "recharts";

const FEATURE_LABELS: Record<string, string> = {
  elo_diff: "Elo Rating Diff",
  elo_expected: "Elo Win Expectation",
  home_interstate: "Home Interstate",
  away_interstate: "Away Interstate",
  home_at_home_ground: "Home at Home Ground",
  away_at_home_ground: "Away at Home Ground",
  venue_win_rate_diff: "Venue Win Rate Diff",
  win_rate_diff_5: "Win Rate Diff (5g)",
  avg_margin_diff_5: "Avg Margin Diff (5g)",
  score_for_diff_5: "Score For Diff (5g)",
  score_against_diff_5: "Score Against Diff (5g)",
  lineup_changes_diff: "Lineup Changes Diff",
  lineup_continuity_diff: "Lineup Continuity Diff",
  home_ruck_missing: "Home Ruck Missing",
  away_ruck_missing: "Away Ruck Missing",
  missing_rating_diff: "Missing Rating Diff",
  net_quality_change_diff: "Net Quality Change Diff",
  missing_mid_rating_diff: "Missing Mid Rating Diff",
  missing_fwd_rating_diff: "Missing Fwd Rating Diff",
  missing_def_rating_diff: "Missing Def Rating Diff",
  // Extended features (from optimization)
  win_rate_diff_10: "Win Rate Diff (10g)",
  avg_margin_diff_10: "Avg Margin Diff (10g)",
  score_for_diff_10: "Score For Diff (10g)",
  score_against_diff_10: "Score Against Diff (10g)",
  rest_diff: "Rest Days Diff",
  home_had_bye: "Home Had Bye",
  away_had_bye: "Away Had Bye",
  ladder_rank_diff: "Ladder Rank Diff",
  percentage_diff: "Percentage Diff",
  home_top4: "Home Top 4",
  away_top4: "Away Top 4",
  home_top8: "Home Top 8",
  away_top8: "Away Top 8",
  h2h_home_win_rate: "H2H Home Win Rate",
  h2h_meetings: "H2H Meetings",
  season_progress: "Season Progress",
  avg_D_diff_5: "Disposals Diff (5g)",
  avg_I50_diff_5: "Inside 50 Diff (5g)",
  avg_CL_diff_5: "Clearances Diff (5g)",
  avg_T_diff_5: "Tackles Diff (5g)",
  avg_HO_diff_5: "Hitouts Diff (5g)",
  avg_CG_diff_5: "Clangers Diff (5g)",
  avg_R50_diff_5: "Rebound 50 Diff (5g)",
  avg_M_diff_5: "Marks Diff (5g)",
  avg_FF_diff_5: "Frees For Diff (5g)",
  avg_FA_diff_5: "Frees Against Diff (5g)",
};

function MetricCard({
  label,
  oldVal,
  newVal,
  lowerBetter,
}: {
  label: string;
  oldVal: number;
  newVal: number;
  lowerBetter?: boolean;
}) {
  const delta = newVal - oldVal;
  const improved = lowerBetter ? delta < 0 : delta > 0;
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-2xl font-bold mt-1">{newVal.toFixed(3)}</p>
        <p
          className={`text-xs mt-1 ${improved ? "text-emerald-400" : "text-red-400"}`}
        >
          {delta >= 0 ? "+" : ""}
          {delta.toFixed(4)} vs old model
        </p>
      </CardContent>
    </Card>
  );
}

interface WFYear {
  year: number;
  n_bets: number;
  win_rate: number;
  roi: number;
  total_profit: number;
}

export function InsightsDashboard({
  features,
  metrics,
  walkForward,
  calibration,
}: {
  features: FeatureImportance[];
  metrics: ModelMetrics | null;
  walkForward: { yearly: WFYear[]; summary: Record<string, unknown> } | null;
  calibration?: CalibrationCurve | null;
}) {
  const chartData = features.map((f) => ({
    name: FEATURE_LABELS[f.feature] || f.feature,
    importance: f.importance,
  }));

  const wfData = walkForward?.yearly.map((y) => ({
    year: y.year.toString(),
    roi: y.roi,
    bets: y.n_bets,
    profit: y.total_profit,
  })) ?? [];

  return (
    <div className="space-y-6">
      {metrics && (
        <div>
          <h3 className="text-lg font-semibold mb-3">
            Model Performance ({metrics.train_period} train, {metrics.val_period} val)
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard label="Accuracy" oldVal={metrics.old_model.accuracy} newVal={metrics.new_model.accuracy} />
            <MetricCard label="AUC-ROC" oldVal={metrics.old_model.auc_roc} newVal={metrics.new_model.auc_roc} />
            <MetricCard label="Log Loss" oldVal={metrics.old_model.log_loss} newVal={metrics.new_model.log_loss} lowerBetter />
            <MetricCard label="Brier Score" oldVal={metrics.old_model.brier_score} newVal={metrics.new_model.brier_score} lowerBetter />
          </div>
          {metrics.optimization && (
            <p className="text-sm text-muted-foreground mt-2">
              Optimized: {metrics.optimization.n_trials} trials,{" "}
              {metrics.optimization.n_features} features,{" "}
              CV log loss {metrics.optimization.best_log_loss_cv.toFixed(4)},{" "}
              groups: {metrics.optimization.feature_groups.join(", ") || "base only"}
            </p>
          )}
        </div>
      )}

      {calibration && calibration.raw.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">
            Reliability Diagram ({calibration.val_period}, {calibration.n_samples} matches)
          </h3>
          <Card>
            <CardContent className="p-4">
              <p className="text-xs text-muted-foreground mb-3">
                Bars show how often the home team actually won for each predicted-probability bucket.
                Perfect calibration lies on the dashed diagonal — points above it mean the model is
                under-confident, below means over-confident.
              </p>
              <ResponsiveContainer width="100%" height={340}>
                <ComposedChart
                  margin={{ top: 10, right: 20, bottom: 10, left: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis
                    type="number"
                    dataKey="predicted"
                    domain={[0, 1]}
                    ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                    tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                    stroke="#888"
                    label={{
                      value: "Predicted probability",
                      position: "insideBottom",
                      offset: -5,
                      fill: "#888",
                      fontSize: 12,
                    }}
                  />
                  <YAxis
                    type="number"
                    dataKey="observed"
                    domain={[0, 1]}
                    ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                    tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                    stroke="#888"
                    label={{
                      value: "Actual win rate",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#888",
                      fontSize: 12,
                    }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#1c1c1c", border: "1px solid #333" }}
                    formatter={(value) => {
                      const n = typeof value === "number" ? value : Number(value);
                      return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : String(value);
                    }}
                    labelFormatter={() => ""}
                  />
                  {/* Perfect calibration diagonal */}
                  <Line
                    type="linear"
                    data={[{ predicted: 0, observed: 0 }, { predicted: 1, observed: 1 }]}
                    dataKey="observed"
                    stroke="#666"
                    strokeDasharray="4 4"
                    dot={false}
                    activeDot={false}
                    isAnimationActive={false}
                    name="Perfect"
                    legendType="none"
                  />
                  <Scatter
                    data={calibration.raw}
                    dataKey="observed"
                    fill="#3b82f6"
                    name="Raw"
                    line={{ stroke: "#3b82f6", strokeWidth: 2 }}
                  />
                  {calibration.calibrated && (
                    <Scatter
                      data={calibration.calibrated}
                      dataKey="observed"
                      fill="#10b981"
                      name="Calibrated"
                      line={{ stroke: "#10b981", strokeWidth: 2 }}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
              <div className="flex gap-4 justify-center text-xs text-muted-foreground mt-2">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-blue-500" /> Raw
                </span>
                {calibration.calibrated && (
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" /> Calibrated (isotonic)
                  </span>
                )}
                <span className="flex items-center gap-1">
                  <span className="w-3 h-px bg-muted-foreground" /> Perfect
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <div>
        <h3 className="text-lg font-semibold mb-3">Feature Importance</h3>
        <Card>
          <CardContent className="p-4">
            <ResponsiveContainer width="100%" height={450}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 150 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis type="number" stroke="#888" />
                <YAxis type="category" dataKey="name" stroke="#888" width={140} tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1c1c1c", border: "1px solid #333" }}
                />
                <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {wfData.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Walk-Forward Betting ROI by Year</h3>
          <Card>
            <CardContent className="p-4">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={wfData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="year" stroke="#888" />
                  <YAxis stroke="#888" tickFormatter={(v: number) => `${v}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#1c1c1c", border: "1px solid #333" }}
                    formatter={(value, name) => {
                      if (name === "roi" && typeof value === "number") {
                        return [`${value.toFixed(1)}%`, "ROI"];
                      }
                      return [String(value), String(name)];
                    }}
                  />
                  <ReferenceLine y={0} stroke="#666" />
                  <Bar dataKey="roi" radius={[4, 4, 0, 0]}>
                    {wfData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.roi >= 0 ? "#10b981" : "#ef4444"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
