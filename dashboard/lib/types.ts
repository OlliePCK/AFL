export interface Prediction {
  game_id: number;
  date: string;
  roundname: string;
  venue: string;
  home_team: string;
  away_team: string;
  // Betting model: no odds features, independent of market.
  // Use this for value betting — compare to market to find mispriced lines.
  home_win_prob: number;
  away_win_prob: number;
  // Analytical model: includes market odds as features.
  // Use this for tipping/forecasting — best estimate using all available info.
  analytical_home_prob?: number | null;
  analytical_away_prob?: number | null;
  predicted_winner: string | null;
  confidence: number;
  predicted_margin: number;
  home_elo: number;
  away_elo: number;
  elo_diff: number;
  home_odds: number | null;
  away_odds: number | null;
  edge: number | null;
  value_team: string | null;
  value_odds: number | null;
  ev_per_dollar: number | null;
  kelly_pct: number | null;
}

export interface PastResult {
  game_id: number;
  year: number;
  round: number;
  roundname: string;
  date: string;
  venue: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  winner: string;
  // Optional snapshot fields if a prediction was saved at the time
  predicted_winner?: string | null;
  home_win_prob?: number | null;
  away_win_prob?: number | null;
  predicted_margin?: number | null;
  snapshot_date?: string | null;
}

export interface ValueBet {
  match: string;
  venue: string;
  date: string;
  bet_team: string;
  bet_side: "home" | "away";
  model_prob: number;
  market_prob: number;
  edge: number;
  odds: number;
  ev_per_dollar: number;
  kelly_pct: number;
  bet_amount: number;
  movement: "AGREE" | "DISAGREE" | "NEUTRAL" | "unknown";
  move_value: number | null;
}

export interface Bet {
  bet_id: string;
  date_placed: string;
  round: string;
  home_team: string;
  away_team: string;
  bet_team: string;
  bet_side: "home" | "away";
  odds: number;
  model_prob: number;
  edge: number;
  kelly_pct: number;
  bet_amount: number;
  status: "pending" | "won" | "lost" | "void";
  profit_loss: number | null;
  reconciled_at: string | null;
}

export interface LiveOdds {
  home_team: string;
  away_team: string;
  home_odds: number;
  away_odds: number;
  home_odds_best: number;
  away_odds_best: number;
  n_bookmakers: number;
  fetched_at: string;
}

export interface BettingSummary {
  total_pl: number;
  roi_pct: number;
  win_rate: number;
  record: { won: number; lost: number; pending: number };
  total_staked: number;
  pending_exposure: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface MatchFeatures {
  home_team: string;
  away_team: string;
  elo_diff: number;
  elo_expected: number;
  home_interstate: number;
  away_interstate: number;
  home_at_home_ground: number;
  away_at_home_ground: number;
  venue_win_rate_diff: number;
  win_rate_diff_5: number;
  avg_margin_diff_5: number;
  score_for_diff_5: number;
  score_against_diff_5: number;
  [key: string]: string | number;
}

export interface WalkForwardYear {
  year: number;
  n_bets: number;
  win_rate: number;
  roi: number;
  total_profit: number;
}

export interface ModelMetrics {
  new_model: { accuracy: number; log_loss: number; auc_roc: number; brier_score: number };
  old_model: { accuracy: number; log_loss: number; auc_roc: number; brier_score: number };
  train_period: string;
  val_period: string;
  optimization?: {
    n_trials: number;
    best_log_loss_cv: number;
    feature_groups: string[];
    n_features: number;
  };
}

export interface CalibrationBin {
  bin_mid: number;
  predicted: number;
  observed: number;
  count: number;
}

export interface CalibrationCurve {
  val_period: string;
  n_samples: number;
  raw: CalibrationBin[];
  calibrated?: CalibrationBin[];
}

// Per-match SHAP explanations from CatBoost. Positive shap = pushed toward home win.
export interface FeatureContribution {
  name: string;
  value: number | null;
  shap: number;
}

export interface PredictionExplanation {
  home_team: string;
  away_team: string;
  home_win_prob?: number;
  base_value: number;
  features: FeatureContribution[];
  summary?: string | null;
}

// Map of game_id (string) -> explanation
export type PredictionExplanations = Record<string, PredictionExplanation>;
export type PredictionExplanationHistory = Record<string, Record<string, PredictionExplanation>>;
