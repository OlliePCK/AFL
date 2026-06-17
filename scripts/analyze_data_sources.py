"""
Data sources analysis: systematic evaluation of untapped signal in the AFL model.

Tier 1 — Unused data already in the master dataset:
  1A. Betting odds (25 columns, 92.6% coverage from 2013+)
  1B. Tipster consensus (6 columns, 100% from 2017+)
  1C. Weather interaction (rain * not-roofed)

Tier 2 — Derivable from existing data:
  2A. Strength of schedule (avg opponent Elo last 5)
  2B. Cumulative travel burden (total km last 3 games)
  2C. Consecutive away games

Tier 3 — Qualitative assessment of external data sources.

Uses 5-seed walk-forward ensemble protocol. Noise floor +/-0.00174 LL.
Writes: data/analysis/data_sources_report.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    build_features, TEAM_HOME_COORDS, _haversine_km,
)
from src.model import BETTING_FEATURE_COLS, FEATURE_GROUPS, TARGET  # noqa: E402
from src.weather import VENUE_COORDINATES  # noqa: E402

VAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEEDS = [42, 123, 256, 789, 1337]


# ---------------------------------------------------------------------------
# Shared infrastructure (same as analyze_new_features.py)
# ---------------------------------------------------------------------------

def load_best_params() -> tuple[dict, list[str]]:
    with open(PROJECT_ROOT / "data" / "optimization_results.json") as f:
        opt = json.load(f)
    params = dict(opt["best_params"])
    params.update({
        "iterations": 1500,
        "early_stopping_rounds": 80,
        "verbose": 0,
        "eval_metric": "Logloss",
        "use_best_model": True,
    })
    if params.get("subsample", 1.0) < 1.0:
        params["bootstrap_type"] = "Bernoulli"
    return params, list(opt["best_feature_groups"])


def features_for_groups(groups: list[str]) -> list[str]:
    cols = list(BETTING_FEATURE_COLS)
    for g in groups:
        cols.extend(FEATURE_GROUPS[g])
    return cols


def run_wf_cv_ensemble(df: pd.DataFrame, feature_cols: list[str],
                       base_params: dict) -> dict:
    """Walk-forward CV with 5-seed ensemble averaging per fold."""
    available = [c for c in feature_cols if c in df.columns]
    fold_rows = []
    pooled_ensemble_probs, pooled_labels = [], []
    pooled_single_seed_probs: dict[int, list] = {s: [] for s in SEEDS}

    for val_year in VAL_YEARS:
        train_df = df[df["year"] < val_year].dropna(subset=[TARGET])
        val_df = df[df["year"] == val_year].dropna(subset=[TARGET])
        if train_df.empty or val_df.empty:
            continue

        y = val_df[TARGET].values.astype(float)
        seed_probs: list[np.ndarray] = []

        for seed in SEEDS:
            p = {**base_params, "random_seed": seed}
            m = CatBoostClassifier(**p)
            m.fit(
                Pool(train_df[available], train_df[TARGET]),
                eval_set=Pool(val_df[available], val_df[TARGET]),
            )
            probs = m.predict_proba(val_df[available])[:, 1]
            seed_probs.append(probs)
            pooled_single_seed_probs[seed].extend(probs.tolist())

        ensemble_probs = np.mean(seed_probs, axis=0)
        fold_seed_lls = [float(log_loss(y, p)) for p in seed_probs]
        fold_ensemble_ll = float(log_loss(y, ensemble_probs))

        fold_rows.append({
            "year": int(val_year),
            "n": int(len(val_df)),
            "seed_mean_ll": float(np.mean(fold_seed_lls)),
            "seed_std_ll": float(np.std(fold_seed_lls)),
            "ensemble_ll": fold_ensemble_ll,
        })

        pooled_ensemble_probs.extend(ensemble_probs.tolist())
        pooled_labels.extend(y.tolist())

    pooled_ensemble_probs = np.asarray(pooled_ensemble_probs)
    pooled_labels = np.asarray(pooled_labels)

    per_seed_pooled = {}
    for s in SEEDS:
        if pooled_single_seed_probs[s]:
            per_seed_pooled[s] = float(log_loss(
                pooled_labels, np.asarray(pooled_single_seed_probs[s])
            ))

    seed_mean = float(np.mean(list(per_seed_pooled.values())))
    seed_std = float(np.std(list(per_seed_pooled.values())))

    return {
        "n_features": len(available),
        "folds": fold_rows,
        "pooled_ensemble_log_loss": float(log_loss(pooled_labels, pooled_ensemble_probs)),
        "pooled_ensemble_brier": float(brier_score_loss(pooled_labels, pooled_ensemble_probs)),
        "pooled_ensemble_accuracy": float(
            ((pooled_ensemble_probs >= 0.5).astype(int) == pooled_labels).mean()
        ),
        "per_seed_pooled_log_loss": per_seed_pooled,
        "seed_mean_pooled_log_loss": seed_mean,
        "seed_std_pooled_log_loss": seed_std,
    }


# ---------------------------------------------------------------------------
# Tier 1+2: Derive extra features inline (no changes to src/features.py)
# ---------------------------------------------------------------------------

def derive_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all Tier 1+2 features that aren't produced by build_features()."""

    # -- Odds-derived --
    df["home_odds_range"] = df["home_odds_max"] - df["home_odds_min"]
    df["away_odds_range"] = df["away_odds_max"] - df["away_odds_min"]
    df["odds_range_diff"] = df["home_odds_range"] - df["away_odds_range"]

    # -- Tipster-derived --
    # mean_hconfidence is raw (0-100 scale), convert to probability
    df["tipster_prob"] = df["mean_hconfidence"] / 100.0
    df["tipster_disagreement"] = 100.0 - df["mean_confidence"]
    # Divergence: tipster consensus vs our Elo expected probability
    df["tipster_vs_elo"] = df["tipster_prob"] - df["elo_expected"]
    df["tipster_margin"] = df["mean_hmargin"]

    # -- Weather interaction --
    df["rain_x_open"] = df["rain_mm"] * (1 - df["is_roofed"].fillna(0))

    # -- Strength of schedule (avg opponent Elo last 5 games) --
    df = _add_strength_of_schedule(df, window=5)

    # -- Cumulative travel (total km last 3 games) --
    df = _add_cumulative_travel(df, window=3)

    # -- Consecutive away games --
    df = _add_consecutive_away(df)

    return df


def _add_strength_of_schedule(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Average opponent Elo over the last N games for each team."""
    home_sos = np.full(len(df), np.nan)
    away_sos = np.full(len(df), np.nan)

    # Track each team's recent opponent Elo values
    team_opp_elos: dict[str, deque] = {}

    for i, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        h_elo, a_elo = row.get("home_elo", np.nan), row.get("away_elo", np.nan)

        # Read before updating
        if ht in team_opp_elos and len(team_opp_elos[ht]) > 0:
            home_sos[i] = float(np.mean(team_opp_elos[ht]))
        if at in team_opp_elos and len(team_opp_elos[at]) > 0:
            away_sos[i] = float(np.mean(team_opp_elos[at]))

        # Update: home team's opponent was away team and vice versa
        if ht not in team_opp_elos:
            team_opp_elos[ht] = deque(maxlen=window)
        if at not in team_opp_elos:
            team_opp_elos[at] = deque(maxlen=window)

        if not np.isnan(a_elo):
            team_opp_elos[ht].append(a_elo)
        if not np.isnan(h_elo):
            team_opp_elos[at].append(h_elo)

    df["home_sos_5"] = home_sos
    df["away_sos_5"] = away_sos
    df["sos_diff_5"] = home_sos - away_sos
    return df


def _add_cumulative_travel(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Total km traveled over the last N games for each team."""
    home_cum = np.full(len(df), np.nan)
    away_cum = np.full(len(df), np.nan)

    team_travel: dict[str, deque] = {}

    for i, row in df.iterrows():
        ht, at, venue = row["home_team"], row["away_team"], row["venue"]

        # Read before updating
        if ht in team_travel and len(team_travel[ht]) > 0:
            home_cum[i] = float(sum(team_travel[ht]))
        if at in team_travel and len(team_travel[at]) > 0:
            away_cum[i] = float(sum(team_travel[at]))

        # Compute this game's travel for each team
        v_coords = VENUE_COORDINATES.get(venue)
        h_coords = TEAM_HOME_COORDS.get(ht)
        a_coords = TEAM_HOME_COORDS.get(at)

        if ht not in team_travel:
            team_travel[ht] = deque(maxlen=window)
        if at not in team_travel:
            team_travel[at] = deque(maxlen=window)

        if v_coords and h_coords:
            km = _haversine_km(h_coords[0], h_coords[1], v_coords[0], v_coords[1])
            team_travel[ht].append(km)
        else:
            team_travel[ht].append(0.0)

        if v_coords and a_coords:
            km = _haversine_km(a_coords[0], a_coords[1], v_coords[0], v_coords[1])
            team_travel[at].append(km)
        else:
            team_travel[at].append(0.0)

    df["home_cumulative_travel_3"] = home_cum
    df["away_cumulative_travel_3"] = away_cum
    df["cumulative_travel_diff_3"] = home_cum - away_cum
    return df


def _add_consecutive_away(df: pd.DataFrame) -> pd.DataFrame:
    """Count consecutive away games for each team (resets on home game)."""
    home_consec = np.full(len(df), np.nan)
    away_consec = np.full(len(df), np.nan)

    team_away_streak: dict[str, int] = {}

    for i, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]

        # Read before updating
        home_consec[i] = team_away_streak.get(ht, 0)
        away_consec[i] = team_away_streak.get(at, 0)

        # Home team is playing at home -> reset streak
        team_away_streak[ht] = 0
        # Away team is playing away -> increment streak
        team_away_streak[at] = team_away_streak.get(at, 0) + 1

    df["home_consecutive_away"] = home_consec
    df["away_consecutive_away"] = away_consec
    df["consecutive_away_diff"] = home_consec - away_consec
    return df


# ---------------------------------------------------------------------------
# Tier 3: Qualitative assessment
# ---------------------------------------------------------------------------

def assess_tier3() -> dict:
    """Qualitative assessment of external data sources not yet collected."""
    return {
        "injury_lists": {
            "description": "Detailed injury/suspension lists from AFL.com.au",
            "expected_signal": "Moderate -- we already have 9 player features covering "
                              "lineup changes, ruck absence, and quality-weighted missing "
                              "players. Injury lists would add injury type (hamstring vs "
                              "managed rest) and return timeline, which may sharpen the "
                              "existing player features.",
            "collection_effort": "Moderate -- weekly scrape of AFL injury list page, "
                                "need to match player names to teams and positions.",
            "historical_coverage": "2018+ reliably, spotty before that.",
            "feature_count": "2-4 (e.g. total_weeks_injured_diff, key_player_injury_diff)",
            "risk_of_null": "High -- existing player features already capture the bulk "
                           "of lineup disruption signal.",
            "priority": "Medium",
        },
        "coach_changes": {
            "description": "Coach tenure and mid-season changes",
            "expected_signal": "Low -- only 2-3 coaching changes per year across 18 teams. "
                              "Very sparse signal. New coach effect is real but hard to "
                              "model with so few data points.",
            "collection_effort": "Low -- manual CSV, ~30 entries since 2012.",
            "historical_coverage": "Full (2012+).",
            "feature_count": "2 (coach_tenure_games, is_new_coach)",
            "risk_of_null": "Very high -- too sparse for tree-based learners.",
            "priority": "Low",
        },
        "quarter_scoring": {
            "description": "Quarter-by-quarter scoring patterns from Squiggle",
            "expected_signal": "Low-moderate -- could capture comeback ability or fast-start "
                              "tendency. But rolling margin already captures scoring "
                              "consistency indirectly.",
            "collection_effort": "Low -- Squiggle API already provides quarter scores, "
                                "just need to add columns to fetch_games.",
            "historical_coverage": "Full (2012+).",
            "feature_count": "3-4 (q1_margin_diff, second_half_surge_diff)",
            "risk_of_null": "High -- form consistency features (margin volatility) "
                           "were already null.",
            "priority": "Low-Medium",
        },
        "brownlow_votes": {
            "description": "Brownlow Medal votes as individual player form proxy",
            "expected_signal": "Low -- highly correlated with disposal counts and "
                              "inside 50s which we already feature. Votes are also "
                              "announced with a delay (only known publicly at season end).",
            "collection_effort": "Moderate -- scrape from AFL Tables, match to games.",
            "historical_coverage": "Full but delayed release.",
            "feature_count": "1-2 (team_brownlow_votes_diff_5)",
            "risk_of_null": "Very high -- delayed availability makes it impractical "
                           "for in-season prediction.",
            "priority": "Very Low",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_path = PROJECT_ROOT / "data" / "analysis" / "data_sources_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 80)
    print("DATA SOURCES ANALYSIS -- 5-seed ensemble walk-forward")
    print("=" * 80)

    # Load and feature the dataset
    print("\nLoading master dataset + building features...")
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    df = build_features(df)
    print(f"  After build_features: {len(df)} rows, {len(df.columns)} cols")

    # Derive extra features for this analysis
    print("Deriving extra features (odds, tipster, SoS, travel, away streak)...")
    df = derive_extra_features(df)
    print(f"  After derive_extra: {len(df.columns)} cols")

    # Coverage check
    check_cols = [
        "implied_home_close", "implied_home_open", "odds_move",
        "tipster_prob", "tipster_disagreement", "tipster_vs_elo",
        "rain_x_open", "sos_diff_5", "cumulative_travel_diff_3",
        "consecutive_away_diff",
    ]
    print("\nFeature coverage (non-null in WF years 2019-2025):")
    wf_mask = df["year"].between(2019, 2025)
    wf_n = wf_mask.sum()
    for c in check_cols:
        if c in df.columns:
            n = df.loc[wf_mask, c].notna().sum()
            print(f"  {c:35s}: {n:5d}/{wf_n} ({n / wf_n * 100:.0f}%)")

    base_params, best_groups = load_best_params()
    baseline_cols = features_for_groups(best_groups)
    print(f"\nBaseline: {len(baseline_cols)} features, groups={best_groups}")
    print(f"Seeds: {SEEDS}, Val years: {VAL_YEARS}")

    # -----------------------------------------------------------------------
    # Run all configs
    # -----------------------------------------------------------------------
    configs: dict[str, list[str]] = {}

    # Baseline
    configs["baseline"] = baseline_cols

    # -- Tier 1A: Odds --
    configs["+implied_close"] = baseline_cols + [
        "implied_home_close", "overround_close",
    ]
    configs["+implied_open"] = baseline_cols + [
        "implied_home_open", "overround_open",
    ]
    configs["+odds_movement"] = baseline_cols + [
        "odds_move", "odds_move_magnitude", "overround_change",
    ]
    configs["+line"] = baseline_cols + [
        "home_line_close",
    ]
    configs["+all_odds"] = baseline_cols + [
        "implied_home_close", "overround_close",
        "odds_move", "odds_move_magnitude", "overround_change",
        "home_line_close",
    ]
    # Odds-only: minimal feature set to establish market ceiling
    configs["odds_only"] = [
        "implied_home_close", "overround_close",
        "home_line_close",
        "odds_move", "odds_move_magnitude",
    ]

    # -- Tier 1B: Tipster consensus --
    configs["+tipster_raw"] = baseline_cols + [
        "tipster_prob", "tipster_margin", "n_models",
    ]
    configs["+tipster_derived"] = baseline_cols + [
        "tipster_disagreement", "tipster_vs_elo",
    ]
    configs["+tipster_all"] = baseline_cols + [
        "tipster_prob", "tipster_margin", "n_models",
        "tipster_disagreement", "tipster_vs_elo",
    ]

    # -- Tier 1C: Weather interaction --
    configs["+rain_interaction"] = baseline_cols + [
        "rain_x_open",
    ]

    # -- Tier 2A: Strength of schedule --
    configs["+sos"] = baseline_cols + [
        "sos_diff_5",
    ]

    # -- Tier 2B: Cumulative travel --
    configs["+cumulative_travel"] = baseline_cols + [
        "cumulative_travel_diff_3",
    ]

    # -- Tier 2C: Consecutive away --
    configs["+consecutive_away"] = baseline_cols + [
        "consecutive_away_diff",
    ]

    print(f"\n{len(configs)} configs to test")
    results = {}

    for name, cols in configs.items():
        t0 = time.time()
        # De-duplicate columns
        cols_dedup = list(dict.fromkeys(cols))
        print(f"\n==> {name}  ({len(cols_dedup)} features)")
        cv = run_wf_cv_ensemble(df, cols_dedup, base_params)
        results[name] = {**cv}
        elapsed = time.time() - t0
        print(
            f"   ensemble pooled LL: {cv['pooled_ensemble_log_loss']:.5f}  "
            f"Brier: {cv['pooled_ensemble_brier']:.5f}  "
            f"acc: {cv['pooled_ensemble_accuracy']:.3f}  "
            f"({cv['n_features']} feats, {elapsed:.0f}s)"
        )
        print(
            f"   seed pooled LL: mean={cv['seed_mean_pooled_log_loss']:.5f}  "
            f"std={cv['seed_std_pooled_log_loss']:.5f}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    baseline_ll = results["baseline"]["pooled_ensemble_log_loss"]
    noise_floor = results["baseline"]["seed_std_pooled_log_loss"]

    print("\n" + "=" * 80)
    print("SUMMARY: DELTAS vs BASELINE (ensemble log loss -- NEGATIVE = better)")
    print(f"Noise floor (baseline seed std): +/- {noise_floor:.5f}")
    print("=" * 80)

    tier_labels = {
        "+implied_close": "1A", "+implied_open": "1A", "+odds_movement": "1A",
        "+line": "1A", "+all_odds": "1A", "odds_only": "1A",
        "+tipster_raw": "1B", "+tipster_derived": "1B", "+tipster_all": "1B",
        "+rain_interaction": "1C",
        "+sos": "2A", "+cumulative_travel": "2B", "+consecutive_away": "2C",
    }

    for name, r in results.items():
        delta = r["pooled_ensemble_log_loss"] - baseline_ll
        if abs(delta) < noise_floor:
            marker = "NOISE"
        elif delta < 0:
            marker = "BETTER"
        else:
            marker = "WORSE"
        tier = tier_labels.get(name, "--")
        print(f"  [{tier:2s}] {name:28s} {delta:+.5f}  [{marker}]")

    # Per-fold breakdown for interesting configs
    print("\n" + "=" * 80)
    print("PER-FOLD DELTAS (configs with |delta| > noise floor)")
    print("=" * 80)
    baseline_folds = {f["year"]: f["ensemble_ll"] for f in results["baseline"]["folds"]}
    for name, r in results.items():
        if name == "baseline":
            continue
        delta = r["pooled_ensemble_log_loss"] - baseline_ll
        if abs(delta) <= noise_floor:
            continue
        folds = r["folds"]
        print(f"\n  {name} (pooled delta: {delta:+.5f}):")
        for f in folds:
            bl = baseline_folds.get(f["year"], 0)
            fd = f["ensemble_ll"] - bl
            print(f"    {f['year']}  n={f['n']:3d}  "
                  f"baseline={bl:.4f}  this={f['ensemble_ll']:.4f}  "
                  f"delta={fd:+.4f}")

    # Tier 3
    tier3 = assess_tier3()
    print("\n" + "=" * 80)
    print("TIER 3: QUALITATIVE ASSESSMENT (external data sources)")
    print("=" * 80)
    for name, info in tier3.items():
        print(f"\n  {name}: priority={info['priority']}")
        print(f"    {info['description']}")
        print(f"    Signal: {info['expected_signal'][:80]}...")
        print(f"    Effort: {info['collection_effort'][:80]}...")

    # Write report
    report = {
        "seeds": SEEDS,
        "val_years": VAL_YEARS,
        "baseline_groups": best_groups,
        "noise_floor_seed_std": noise_floor,
        "configs": results,
        "deltas_vs_baseline": {
            name: r["pooled_ensemble_log_loss"] - baseline_ll
            for name, r in results.items()
        },
        "tier3_qualitative": tier3,
        "runtime_sec": time.time() - t_start,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {out_path}")
    print(f"Total runtime: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
