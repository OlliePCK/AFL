// Human-readable labels for model features, shared by the SHAP panel
// and the insights dashboard. Keep in sync with src/features.py outputs.

export const FEATURE_LABELS: Record<string, string> = {
  // Elo
  elo_diff: "Elo rating gap",
  elo_expected: "Elo win expectation",
  // Venue / travel
  home_interstate: "Home travelling interstate",
  away_interstate: "Away travelling interstate",
  home_at_home_ground: "Home at own ground",
  away_at_home_ground: "Away at own ground",
  venue_win_rate_diff: "Venue win rate gap",
  // Rolling form
  win_rate_diff_5: "Win rate gap (last 5)",
  avg_margin_diff_5: "Avg margin gap (last 5)",
  score_for_diff_5: "Scoring gap (last 5)",
  score_against_diff_5: "Defence gap (last 5)",
  win_rate_diff_10: "Win rate gap (last 10)",
  avg_margin_diff_10: "Avg margin gap (last 10)",
  score_for_diff_10: "Scoring gap (last 10)",
  score_against_diff_10: "Defence gap (last 10)",
  // Lineups / availability
  lineup_changes_diff: "Lineup changes gap",
  lineup_continuity_diff: "Lineup continuity gap",
  home_ruck_missing: "Home ruck missing",
  away_ruck_missing: "Away ruck missing",
  missing_rating_diff: "Missing player quality gap",
  net_quality_change_diff: "Net quality change",
  missing_mid_rating_diff: "Missing midfield quality",
  missing_fwd_rating_diff: "Missing forward quality",
  missing_def_rating_diff: "Missing defensive quality",
  // Ladder
  ladder_rank_diff: "Ladder position gap",
  percentage_diff: "Percentage gap",
  home_top4: "Home in top 4",
  away_top4: "Away in top 4",
  home_top8: "Home in top 8",
  away_top8: "Away in top 8",
  // Rest / bye
  rest_diff: "Rest days gap",
  home_had_bye: "Home had bye",
  away_had_bye: "Away had bye",
  bye_advantage: "Bye advantage",
  // H2H
  h2h_home_win_rate: "H2H home win rate",
  h2h_meetings: "H2H meetings",
  // Season context
  season_progress: "Season progress",
  // Detailed stats (5g)
  avg_D_diff_5: "Disposals gap (last 5)",
  avg_I50_diff_5: "Inside-50s gap (last 5)",
  avg_CL_diff_5: "Clearances gap (last 5)",
  avg_T_diff_5: "Tackles gap (last 5)",
  avg_HO_diff_5: "Hitouts gap (last 5)",
  avg_CG_diff_5: "Clangers gap (last 5)",
  avg_R50_diff_5: "Rebound-50s gap (last 5)",
  avg_M_diff_5: "Marks gap (last 5)",
  avg_FF_diff_5: "Frees-for gap (last 5)",
  avg_FA_diff_5: "Frees-against gap (last 5)",
  // Weather
  rain_mm: "Rain (mm)",
  wind_kmh: "Wind speed (km/h)",
  wind_gust_kmh: "Wind gusts (km/h)",
  humidity: "Humidity (%)",
  temp_c: "Temperature (°C)",
  is_wet: "Wet conditions",
  wind_strong: "Strong wind",
  is_roofed: "Roofed stadium",
};

export function labelFor(feature: string): string {
  return FEATURE_LABELS[feature] ?? feature;
}
