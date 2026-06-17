"""
Prediction pipeline for upcoming AFL matches.

Fetches the next round's fixture from Squiggle, computes features
using historical data, and generates predictions with confidence scores.
Optionally fetches live odds and identifies value bets.
"""
import hashlib
import json
import os
from datetime import datetime

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from src.config import PROJECT_ROOT, SQUIGGLE_BASE_URL, SQUIGGLE_USER_AGENT
from src.features import build_features
from src.model import (
    get_feature_cols, get_analytical_feature_cols,
    load_betting_ensemble, load_analytical_ensemble,
    load_analytical_calibrator,
    ensemble_predict_proba,
)
from src.value import decimal_odds_to_implied_prob, calculate_edge, kelly_fraction, expected_value
from src.utils import fetch_url, setup_logging

log = setup_logging()


def fetch_upcoming_fixture(year: int = 2026) -> pd.DataFrame:
    """Fetch upcoming (incomplete) matches from Squiggle."""
    headers = {"User-Agent": SQUIGGLE_USER_AGENT}
    resp = fetch_url(SQUIGGLE_BASE_URL, params={"q": "games", "year": str(year), "complete": "!100"},
                     headers=headers)
    data = resp.json()
    games = data.get("games", [])

    if not games:
        log.info("No upcoming matches found")
        return pd.DataFrame()

    df = pd.DataFrame(games)
    df = df.rename(columns={
        "id": "game_id", "hteam": "home_team", "ateam": "away_team",
        "hscore": "home_score", "ascore": "away_score",
        "hgoals": "home_goals", "hbehinds": "home_behinds",
        "agoals": "away_goals", "abehinds": "away_behinds",
    })

    # Filter to only future games (complete == 0) with known teams
    df = df[df["complete"] == 0].copy()
    df = df[df["home_team"].notna() & df["away_team"].notna()]

    keep = ["game_id", "year", "round", "roundname", "date", "localtime", "venue",
            "home_team", "away_team", "is_final", "is_grand_final"]
    df = df[[c for c in keep if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])

    log.info(f"Found {len(df)} upcoming matches")
    return df


def _populate_analytical_odds(upcoming: pd.DataFrame, odds: pd.DataFrame) -> None:
    """Populate V3 analytical model features from the live odds snapshot.

    Maps live snapshot columns → historical training column names:
      home_odds / away_odds  → implied_home_open, overround_open
      home_line              → home_line_close

    Modifies `upcoming` in-place.
    """
    from src.value import decimal_odds_to_implied_prob

    for idx, row in upcoming.iterrows():
        match_odds = odds[
            (odds["home_team"] == row["home_team"])
            & (odds["away_team"] == row["away_team"])
        ]
        if match_odds.empty:
            continue
        o = match_odds.iloc[0]

        # h2h implied probability (vig-adjusted)
        ho = o.get("home_odds")
        ao = o.get("away_odds")
        if pd.notna(ho) and pd.notna(ao) and ho > 1 and ao > 1:
            imp_h, imp_a = decimal_odds_to_implied_prob(float(ho), float(ao))
            upcoming.at[idx, "implied_home_open"] = imp_h
            upcoming.at[idx, "overround_open"] = (1 / ho) + (1 / ao)

        # Spread / line
        line = o.get("home_line")
        if pd.notna(line):
            upcoming.at[idx, "home_line_close"] = float(line)

    n_pop = upcoming["implied_home_open"].notna().sum()
    n_line = upcoming["home_line_close"].notna().sum()
    log.info(f"Populated analytical odds: {n_pop} h2h, {n_line} lines "
             f"(of {len(upcoming)} upcoming)")


def fetch_live_odds(upcoming: pd.DataFrame) -> pd.DataFrame:
    """Fetch current odds for upcoming matches.

    Tries sources in order:
    1. The Odds API (if ODDS_API_KEY env var is set). Also refreshes the
       `data/live_odds/current_odds.csv` cache and appends a snapshot, so
       downstream callers (e.g. run_predictions.py value detection) see
       the same fresh odds we just fetched.
    2. Cached current_odds.csv (populated by a recent API fetch).
    3. Manual odds CSV (data/odds_manual.csv).
    4. aussportsbetting.com historical file (for recently completed rounds).

    Returns a DataFrame with home_team, away_team, home_odds, away_odds.
    """
    import os
    from src.team_mapping import normalize_team

    # 1. Try The Odds API via the shared odds_monitor cache (h2h + spreads).
    api_key = os.environ.get("ODDS_API_KEY")
    if api_key:
        try:
            from src.odds_monitor import fetch_and_save
            odds = fetch_and_save()
            if not odds.empty:
                n_line = odds["home_line"].notna().sum() if "home_line" in odds.columns else 0
                log.info(f"Fetched odds from The Odds API: {len(odds)} matches ({n_line} with spread lines)")
                return odds
        except Exception as e:
            log.info(f"Odds API failed: {e}")

    # 2. Fall back to the last saved snapshot.
    try:
        from src.odds_monitor import load_current_odds
        cached = load_current_odds()
        if not cached.empty:
            log.info(f"Using cached odds snapshot ({len(cached)} matches)")
            return cached
    except Exception as e:
        log.info(f"Could not load cached odds snapshot: {e}")

    # 2. Try manual odds CSV
    manual_path = PROJECT_ROOT / "data" / "odds_manual.csv"
    if manual_path.exists():
        try:
            manual = pd.read_csv(manual_path)
            if "home_team" in manual.columns and "home_odds" in manual.columns:
                log.info(f"Using manual odds from {manual_path}")
                return manual[["home_team", "away_team", "home_odds", "away_odds"]]
        except Exception as e:
            log.info(f"Could not load manual odds: {e}")

    # 3. Fall back to historical file
    from src.odds import load_odds
    try:
        odds = load_odds()
        odds_lookup = {}
        for _, row in odds.iterrows():
            key = (row["home_team"], row["away_team"])
            odds_lookup[key] = {
                "home_odds": row.get("home_odds_close") or row.get("home_odds_avg"),
                "away_odds": row.get("away_odds_close") or row.get("away_odds_avg"),
            }
        results = []
        for _, row in upcoming.iterrows():
            key = (row["home_team"], row["away_team"])
            if key in odds_lookup:
                results.append({"home_team": row["home_team"], "away_team": row["away_team"],
                                **odds_lookup[key]})
        if results:
            return pd.DataFrame(results)
    except Exception as e:
        log.info(f"Could not load historical odds: {e}")

    log.info("No odds available for upcoming matches")
    return pd.DataFrame()


def _fetch_odds_api(api_key: str, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Fetch AFL odds from The Odds API (the-odds-api.com)."""
    from src.team_mapping import normalize_team
    resp = fetch_url(
        "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds",
        params={"apiKey": api_key, "regions": "au", "markets": "h2h", "oddsFormat": "decimal"},
    )
    data = resp.json()
    if not data:
        return pd.DataFrame()

    # Build odds lookup from API response
    results = []
    for game in data:
        home = normalize_team(game.get("home_team", ""))
        away = normalize_team(game.get("away_team", ""))

        # Get best odds across bookmakers
        best_home, best_away = 0, 0
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        name = normalize_team(outcome["name"])
                        price = outcome["price"]
                        if name == home:
                            best_home = max(best_home, price)
                        elif name == away:
                            best_away = max(best_away, price)

        if best_home > 0 and best_away > 0:
            results.append({"home_team": home, "away_team": away,
                            "home_odds": best_home, "away_odds": best_away})

    if results:
        log.info(f"Fetched odds from The Odds API: {len(results)} matches")
    return pd.DataFrame(results)


def _add_value_detection(output: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Add value bet columns by comparing model probs to market odds."""
    if odds.empty:
        return output

    # Merge odds into output
    merged = output.merge(odds, on=["home_team", "away_team"], how="left")

    value_cols = []
    for idx, row in merged.iterrows():
        home_odds = row.get("home_odds")
        away_odds = row.get("away_odds")
        home_prob = row.get("home_win_prob")

        if pd.isna(home_odds) or pd.isna(away_odds) or pd.isna(home_prob):
            value_cols.append({})
            continue

        away_prob = 1.0 - home_prob
        fair_home, fair_away = decimal_odds_to_implied_prob(home_odds, away_odds)

        home_edge = calculate_edge(home_prob, fair_home)
        away_edge = calculate_edge(away_prob, fair_away)

        if home_edge >= away_edge and home_edge > 0:
            edge = home_edge
            bet_odds = home_odds
            bet_prob = home_prob
            bet_team = row["home_team"]
        elif away_edge > 0:
            edge = away_edge
            bet_odds = away_odds
            bet_prob = away_prob
            bet_team = row["away_team"]
        else:
            value_cols.append({"edge": 0, "value_team": None})
            continue

        # Kelly 0.15 (V2 sensitivity analysis) — halves max drawdown vs 0.25
        kelly_pct = kelly_fraction(bet_prob, bet_odds, fraction=0.15)
        ev = expected_value(bet_prob, bet_odds)

        value_cols.append({
            "edge": edge,
            "value_team": bet_team,
            "value_odds": bet_odds,
            "ev_per_dollar": ev,
            "kelly_pct": kelly_pct,
            "home_odds": home_odds,
            "away_odds": away_odds,
        })

    value_df = pd.DataFrame(value_cols, index=merged.index)
    for col in value_df.columns:
        merged[col] = value_df[col]

    return merged


def _load_selection_lineups(player_df: pd.DataFrame) -> dict[tuple, dict[str, set[str]]]:
    """Load official team selections, keyed to a specific season round."""
    selection_lineups: dict[tuple, dict[str, set[str]]] = {}
    try:
        from src.selection_scraper import scrape_team_selections
        from src.players import resolve_selection_names

        selections = scrape_team_selections()
        raw_lineups: dict[tuple, dict[str, set[str]]] = {}
        for match in selections:
            key = (
                int(match["year"]),
                int(match["round"]),
                match["home_team"],
                match["away_team"],
            )
            raw_lineups[key] = {
                match["home_team"]: match["home_players"],
                match["away_team"]: match["away_players"],
            }

        selection_lineups = resolve_selection_names(raw_lineups, player_df)
        log.info(f"Loaded official team selections for {len(selection_lineups)} matches")
    except Exception as e:
        log.info(f"Could not fetch team selections: {e}")

    return selection_lineups


def _add_projected_next_round_lineups(
    upcoming: pd.DataFrame,
    selection_lineups: dict[tuple, dict[str, set[str]]],
    player_df: pd.DataFrame,
) -> dict[tuple, dict[str, set[str]]]:
    """Fill next-round lineup gaps from each club's latest known lineup."""
    if upcoming.empty or player_df.empty:
        return selection_lineups

    from src.players import build_latest_team_lineups

    upcoming_sorted = upcoming.sort_values("date")
    next_round = upcoming_sorted["roundname"].dropna().iloc[0] if len(upcoming_sorted) > 0 else None
    if not next_round:
        return selection_lineups

    next_round_df = upcoming_sorted[upcoming_sorted["roundname"] == next_round].copy()
    if next_round_df.empty:
        return selection_lineups

    latest_team_lineups = build_latest_team_lineups(player_df)

    projected_count = 0

    for _, match in next_round_df.iterrows():
        ht, at = match["home_team"], match["away_team"]
        if pd.isna(match.get("year")) or pd.isna(match.get("round")):
            key = (ht, at)
        else:
            key = (int(match["year"]), int(match["round"]), ht, at)

        team_lineups = selection_lineups.get(key)
        if team_lineups is None:
            h_latest = latest_team_lineups.get(ht)
            a_latest = latest_team_lineups.get(at)
            if not h_latest or not a_latest:
                continue
            team_lineups = {ht: set(h_latest), at: set(a_latest)}
            projected_count += 1

        selection_lineups[key] = team_lineups

    log.info(
        "Prepared next-round lineup inputs for %s: %d official/projected matches",
        next_round,
        len(next_round_df),
    )
    if projected_count:
        log.info(f"Used projected latest lineups for {projected_count} next-round matches without official teams")

    return selection_lineups


def predict_upcoming(classifier_path: str | None = None,
                     margin_path: str | None = None) -> pd.DataFrame:
    """Generate predictions for upcoming matches.

    Uses the betting ensemble (odds-free model) for value detection,
    with isotonic calibration. Falls back to single model if no ensemble.

    1. Loads historical data + features
    2. Fetches upcoming fixture + team selections
    3. Computes features (rolling stats use only historical data)
    4. Runs ensemble predictions on upcoming matches
    5. Compares model probs to bookmaker odds for value detection (15-30% edge)
    """
    # Load betting ensemble (preferred) or single model
    ensemble = load_betting_ensemble()
    if ensemble:
        log.info(f"Using {len(ensemble)}-model betting ensemble")
    else:
        clf_path = classifier_path or str(PROJECT_ROOT / "data" / "model.cbm")
        clf = CatBoostClassifier()
        clf.load_model(clf_path)
        log.info("Using single model (no ensemble found)")

    from src.model import load_margin_ensemble, ensemble_predict_margin
    margin_ensemble = load_margin_ensemble()
    margin_model = None
    if margin_ensemble:
        log.info(f"Using {len(margin_ensemble)}-model margin ensemble")
    else:
        mrg_path = margin_path or str(PROJECT_ROOT / "data" / "margin_model.cbm")
        try:
            margin_model = CatBoostRegressor()
            margin_model.load_model(mrg_path)
            log.info("Using single margin model (no ensemble found)")
        except Exception:
            log.info("No margin model found — skipping margin predictions")

    # Load historical data
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    historical = pd.read_csv(master_path, parse_dates=["date"])
    log.info(f"Loaded {len(historical)} historical matches")

    # Fetch upcoming
    upcoming = fetch_upcoming_fixture()
    if upcoming.empty:
        return pd.DataFrame()

    # Set placeholder scores for upcoming (NaN — won't affect rolling calcs)
    for col in ["home_score", "away_score", "home_goals", "home_behinds",
                "away_goals", "away_behinds", "home_win", "margin"]:
        if col not in upcoming.columns:
            upcoming[col] = np.nan

    # Team availability inputs for upcoming matches:
    # 1) official selections when available
    # 2) latest known lineup projection for next round
    from src.players import load_player_data
    player_df = load_player_data()
    selection_lineups: dict[tuple, dict[str, set[str]]] = {}
    if not player_df.empty:
        selection_lineups = _load_selection_lineups(player_df)
        selection_lineups = _add_projected_next_round_lineups(
            upcoming, selection_lineups, player_df
        )
    else:
        log.info("No player data available â€” upcoming player features will be missing")

    # Combine and build features
    combined = pd.concat([historical, upcoming], ignore_index=True)
    combined = build_features(combined, selection_lineups=selection_lineups)

    # Extract upcoming matches (NaN scores)
    upcoming_featured = combined[combined["home_score"].isna()].copy()

    if upcoming_featured.empty:
        log.info("No upcoming matches to predict")
        return pd.DataFrame()

    # Use schema features (matches whatever the production model was trained with)
    schema_features = get_feature_cols()
    betting_features = [c for c in schema_features if c in upcoming_featured.columns]
    missing = [c for c in schema_features if c not in upcoming_featured.columns]
    if missing:
        log.warning(f"Schema features missing from data: {missing}")
    X_betting = upcoming_featured[betting_features]

    # Ensemble or single model predictions
    from src.model import load_calibrator
    if ensemble:
        win_probs = ensemble_predict_proba(ensemble, X_betting)
    else:
        win_probs = clf.predict_proba(X_betting)[:, 1]

    # Apply calibrator
    calibrator = load_calibrator()
    if calibrator is not None:
        upcoming_featured["home_win_prob_raw"] = win_probs
        win_probs = calibrator.predict(win_probs)
        win_probs = np.clip(win_probs, 0.02, 0.98)  # prevent 0/1 extremes
        log.info("Applied calibrated probabilities")
    upcoming_featured["home_win_prob"] = win_probs
    upcoming_featured["away_win_prob"] = 1 - win_probs
    upcoming_featured["predicted_winner"] = upcoming_featured.apply(
        lambda r: "" if np.isclose(r["home_win_prob"], 0.5)
        else (r["home_team"] if r["home_win_prob"] > 0.5 else r["away_team"]),
        axis=1,
    )
    upcoming_featured["confidence"] = upcoming_featured["home_win_prob"].apply(
        lambda p: max(p, 1 - p)
    )

    # Margin predictions (ensemble preferred, single-model fallback)
    if margin_ensemble:
        try:
            margin_features = [c for c in schema_features if c in upcoming_featured.columns]
            margins = ensemble_predict_margin(margin_ensemble, upcoming_featured[margin_features])
            upcoming_featured["predicted_margin"] = margins
        except Exception as e:
            log.info(f"Margin ensemble incompatible with current features: {e}")
            upcoming_featured["predicted_margin"] = np.nan
    elif margin_model is not None:
        try:
            margin_features = [c for c in schema_features if c in upcoming_featured.columns]
            margins = margin_model.predict(upcoming_featured[margin_features])
            upcoming_featured["predicted_margin"] = margins
        except Exception as e:
            log.info(f"Margin model incompatible with current features: {e}")
            upcoming_featured["predicted_margin"] = np.nan
    else:
        upcoming_featured["predicted_margin"] = np.nan

    # Fetch live odds early — needed for both the analytical model (V3 uses
    # odds features) and downstream value detection.
    odds = fetch_live_odds(upcoming_featured)

    # Populate V3 analytical features from live odds snapshot.
    # The V3 model needs: implied_home_open, overround_open, home_line_close.
    # In live mode the "open" proxy is the current snapshot's h2h prices, and
    # the "close line" proxy is the current snapshot's spread.
    if not odds.empty:
        _populate_analytical_odds(upcoming_featured, odds)

    # Analytical model predictions (V3: 3 odds-only features for tipping)
    analytical_ensemble = load_analytical_ensemble()
    if analytical_ensemble:
        analytical_features_list = get_analytical_feature_cols()
        if analytical_features_list:
            avail = [c for c in analytical_features_list if c in upcoming_featured.columns]
            missing_a = [c for c in analytical_features_list if c not in upcoming_featured.columns]
            if missing_a:
                log.warning(f"Analytical features missing: {missing_a}")
            analytical_probs = ensemble_predict_proba(analytical_ensemble, upcoming_featured[avail])
            analytical_cal = load_analytical_calibrator()
            if analytical_cal is not None:
                analytical_probs = analytical_cal.predict(analytical_probs)
                analytical_probs = np.clip(analytical_probs, 0.02, 0.98)
            upcoming_featured["analytical_home_prob"] = analytical_probs
            upcoming_featured["analytical_away_prob"] = 1 - analytical_probs
            log.info(f"Analytical model predictions added ({len(analytical_ensemble)} models, {len(avail)} features)")
        else:
            log.info("Analytical schema not found -- skipping")
    else:
        log.info("No analytical ensemble found -- skipping")

    # Format output
    output_cols = ["game_id", "date", "roundname", "venue",
                   "home_team", "away_team",
                   "home_win_prob", "away_win_prob",
                   "analytical_home_prob", "analytical_away_prob",
                   "predicted_winner", "confidence", "predicted_margin",
                   "home_elo", "away_elo", "elo_diff"]
    output = upcoming_featured[[c for c in output_cols if c in upcoming_featured.columns]]
    output = output.sort_values("date").reset_index(drop=True)

    # Value detection (odds already fetched above)
    output = _add_value_detection(output, odds)

    # Pretty print (next round only)
    next_round = output["roundname"].dropna().iloc[0] if len(output) > 0 else "Unknown"
    next_round_df = output[output["roundname"] == next_round]

    log.info(f"\n{'='*80}")
    log.info(f"AFL PREDICTIONS — {next_round}")
    log.info(f"{'='*80}")
    for _, row in next_round_df.iterrows():
        ht = str(row.get("home_team", "?"))
        at = str(row.get("away_team", "?"))
        winner = str(row.get("predicted_winner") or "Toss-up")
        conf = row.get("confidence", 0)
        margin = row.get("predicted_margin", float("nan"))
        margin_str = f"by {abs(margin):.0f} pts" if pd.notna(margin) else ""
        log.info(f"  {ht:>25s} vs {at:<25s} | {winner} ({conf:.0%}) {margin_str}")

    # Preliminary value bets list — edge-band only (15-30%, V2 baseline).
    # The smart-money AGREE filter lives in run_predictions.py::_detect_value_bets
    # which has access to live odds-snapshot movement data. This list is the
    # superset before the smart-money gate is applied.
    if "edge" in next_round_df.columns:
        value_bets = next_round_df[
            (next_round_df["edge"] >= 0.15) & (next_round_df["edge"] < 0.30)
        ].sort_values("edge", ascending=False)
        if not value_bets.empty:
            log.info(f"\n{'-'*80}")
            log.info(f"VALUE-CANDIDATES (15-30% edge, pre-smart-money filter)")
            log.info(f"{'-'*80}")
            for _, row in value_bets.iterrows():
                team = str(row.get("value_team", "?"))
                edge = row.get("edge", 0)
                odds_val = row.get("value_odds", 0)
                ev = row.get("ev_per_dollar", 0)
                kelly = row.get("kelly_pct", 0)
                log.info(f"  {team:>25s} @ {odds_val:.2f} | edge {edge:.1%} | EV ${ev:.2f} | kelly {kelly:.1%}")
        else:
            log.info(f"\nNo value candidates (15-30% edge band)")

    log.info(f"{'='*80}")

    # Save
    output.to_csv(PROJECT_ROOT / "data" / "master" / "upcoming_predictions.csv", index=False)
    log.info(f"Predictions saved to data/master/upcoming_predictions.csv")

    # Persist the feature-engineered upcoming frame so downstream sanity
    # checks can audit NaN coverage against the schemas without having to
    # rebuild features themselves.
    try:
        id_cols = [c for c in ("game_id", "date", "roundname", "venue",
                                "home_team", "away_team") if c in upcoming_featured.columns]
        betting_cols = [c for c in (get_feature_cols() or []) if c in upcoming_featured.columns]
        analytical_cols = [c for c in (get_analytical_feature_cols() or []) if c in upcoming_featured.columns]
        audit_cols = list(dict.fromkeys(id_cols + betting_cols + analytical_cols))
        if audit_cols:
            upcoming_featured[audit_cols].to_csv(
                PROJECT_ROOT / "data" / "master" / "upcoming_features.csv",
                index=False,
            )
    except Exception as e:
        log.warning(f"Could not save upcoming_features.csv for sanity audit: {e}")

    # Per-match SHAP explanations — which features pushed each prediction
    # toward home or away. Uses the single classifier if no ensemble is loaded.
    explanations: dict[str, dict] = {}
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    try:
        explainer_model = ensemble[0] if ensemble else clf
        explanations = _save_prediction_explanations(
            explainer_model, X_betting, betting_features,
            upcoming_featured, top_n=10,
        )
    except Exception as e:
        log.warning(f"SHAP explanation generation failed: {e}")

    # Append to history snapshot (one row per (game_id, snapshot_date))
    # so we can compare predicted vs actual after the match plays out.
    _append_predictions_history(output, snapshot_date=snapshot_date)
    try:
        _upsert_prediction_explanations_history(
            explanations,
            {game_id: snapshot_date for game_id in explanations},
        )
    except Exception as e:
        log.warning(f"Explanation history update failed: {e}")

    return output


# Plain-English labels for raw feature names. Must stay in sync with
# dashboard/lib/feature-labels.ts — kept here so the summary prompt reads
# human-friendly text instead of identifiers like ``avg_I50_diff_5``.
_FEATURE_LABELS: dict[str, str] = {
    "elo_diff": "Elo rating gap",
    "elo_expected": "Elo win expectation",
    "home_interstate": "Home team travelling interstate",
    "away_interstate": "Away team travelling interstate",
    "home_at_home_ground": "Home at own ground",
    "away_at_home_ground": "Away at own ground",
    "venue_win_rate_diff": "Venue win rate gap",
    "win_rate_diff_5": "Win rate gap (last 5)",
    "win_rate_diff_10": "Win rate gap (last 10)",
    "avg_margin_diff_5": "Avg margin gap (last 5)",
    "avg_margin_diff_10": "Avg margin gap (last 10)",
    "score_for_diff_5": "Scoring gap (last 5)",
    "score_for_diff_10": "Scoring gap (last 10)",
    "score_against_diff_5": "Defence gap (last 5)",
    "score_against_diff_10": "Defence gap (last 10)",
    "lineup_changes_diff": "Lineup change gap",
    "lineup_continuity_diff": "Lineup continuity gap",
    "missing_rating_diff": "Missing player quality gap",
    "missing_mid_rating_diff": "Missing midfielder quality gap",
    "missing_fwd_rating_diff": "Missing forward quality gap",
    "missing_def_rating_diff": "Missing defender quality gap",
    "rest_diff": "Rest day gap",
    "home_had_bye": "Home coming off bye",
    "away_had_bye": "Away coming off bye",
    "ladder_rank_diff": "Ladder position gap",
    "percentage_diff": "Percentage gap",
    "home_top4": "Home in top 4",
    "away_top4": "Away in top 4",
    "home_top8": "Home in top 8",
    "away_top8": "Away in top 8",
    "avg_D_diff_5": "Disposals gap (last 5)",
    "avg_I50_diff_5": "Inside-50s gap (last 5)",
    "avg_CL_diff_5": "Clearances gap (last 5)",
    "avg_T_diff_5": "Tackles gap (last 5)",
    "avg_HO_diff_5": "Hit-outs gap (last 5)",
    "avg_CG_diff_5": "Clangers gap (last 5)",
    "avg_R50_diff_5": "Rebound-50s gap (last 5)",
    "avg_M_diff_5": "Marks gap (last 5)",
    "avg_FF_diff_5": "Frees for gap (last 5)",
    "avg_FA_diff_5": "Frees against gap (last 5)",
    "season_progress": "Point in the season",
    "implied_home_close": "Market implied home win probability",
    "overround_close": "Bookmaker margin (overround)",
    "home_line_close": "Bookmaker point spread",
}


def _label_for(feature: str) -> str:
    return _FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _cache_key(
    game_id: str, home_team: str, away_team: str,
    home_prob: float, top_features: list[dict],
) -> str:
    """Hash of the inputs that should cause the summary to regenerate.

    We intentionally exclude the exact SHAP magnitudes so tiny reshuffles
    don't invalidate every cache entry. Instead we key on the set of
    top-5 feature names (direction-agnostic) plus the rounded probability.
    """
    names = tuple(sorted(f["name"] for f in top_features[:5]))
    bucket = round(home_prob * 20) / 20  # nearest 5%
    payload = f"{game_id}|{home_team}|{away_team}|{bucket:.2f}|{'|'.join(names)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _generate_summary_for_match(
    client,  # anthropic.Anthropic
    home_team: str,
    away_team: str,
    home_prob: float,
    top_features: list[dict],
) -> str | None:
    """Ask Claude Haiku to write a 2-3 sentence plain-English pick summary."""
    winner = home_team if home_prob >= 0.5 else away_team
    confidence = max(home_prob, 1 - home_prob)

    # Build a compact feature bullet list: label, direction, value
    lines = []
    for f in top_features[:8]:
        shap = float(f.get("shap", 0.0))
        if abs(shap) < 1e-4:
            continue
        direction = home_team if shap > 0 else away_team
        val = f.get("value")
        val_str = f" ({val:+.2f})" if isinstance(val, (int, float)) else ""
        lines.append(f"- {_label_for(f['name'])}{val_str} → favours {direction}")

    prompt = (
        f"You are an AFL analyst. In 2-3 short sentences (max 60 words), "
        f"explain in natural prose why the model picks {winner} at "
        f"{confidence:.0%} in {home_team} vs {away_team}. Reference the "
        f"strongest drivers from the list below — do not mention 'SHAP', "
        f"'features', or percentages for individual drivers. Write in a "
        f"confident, neutral tone.\n\nTop drivers:\n" + "\n".join(lines)
    )

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=160,
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ]
        summary = "".join(text_parts).strip()
        return summary or None
    except Exception as e:
        log.warning(f"Claude summary failed for {home_team} vs {away_team}: {e}")
        return None


def _load_prediction_explanations(path: str) -> dict:
    file_path = PROJECT_ROOT / "data" / "master" / path
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return {}


def _latest_explanation_entries(history: dict[str, dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for game_id, snapshots in history.items():
        if not isinstance(snapshots, dict) or not snapshots:
            continue
        latest_date = sorted(snapshots.keys())[-1]
        latest[game_id] = snapshots[latest_date]
    return latest


def _build_prediction_explanations(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    matches: pd.DataFrame,
    existing: dict[str, dict] | None = None,
    top_n: int = 10,
) -> dict[str, dict]:
    """Compute per-match SHAP contributions and return a JSON-ready mapping."""
    if X.empty:
        return {}

    shap_matrix = model.get_feature_importance(
        Pool(X), type="ShapValues"
    )
    base_values = shap_matrix[:, -1]
    feature_shaps = shap_matrix[:, :-1]

    existing = existing or {}

    client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
        except Exception as e:
            log.warning(f"Anthropic SDK unavailable, skipping summaries: {e}")

    explanations: dict[str, dict] = {}
    summaries_made = 0
    summaries_cached = 0
    for i, (_, row) in enumerate(matches.iterrows()):
        game_id = row.get("game_id")
        if pd.isna(game_id):
            continue
        game_id_str = str(int(game_id))

        contribs = []
        for j, fname in enumerate(feature_names):
            val = X.iloc[i, j]
            contribs.append({
                "name": fname,
                "value": None if pd.isna(val) else float(val),
                "shap": float(feature_shaps[i, j]),
            })
        contribs.sort(key=lambda c: abs(c["shap"]), reverse=True)
        top_features = contribs[:top_n]

        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        home_prob = float(row.get("home_win_prob", 0.5))

        cache_key = _cache_key(game_id_str, home_team, away_team, home_prob, top_features)

        prior = existing.get(game_id_str) or {}
        summary: str | None = None
        if prior.get("cache_key") == cache_key and prior.get("summary"):
            summary = prior["summary"]
            summaries_cached += 1
        elif client is not None:
            summary = _generate_summary_for_match(
                client, home_team, away_team, home_prob, top_features,
            )
            if summary:
                summaries_made += 1

        explanations[game_id_str] = {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": home_prob,
            "base_value": float(base_values[i]),
            "features": top_features,
            "summary": summary,
            "cache_key": cache_key,
        }

    log.info(
        f"Prepared SHAP explanations for {len(explanations)} matches "
        f"(summaries: {summaries_made} new, {summaries_cached} cached)"
    )
    return explanations


def _save_prediction_explanations(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    upcoming_featured: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, dict]:
    """Compute per-match SHAP contributions and write to JSON.

    Output shape:
        { "<game_id>": {
            "home_team": str, "away_team": str,
            "base_value": float,   # model expected log-odds
            "features": [
                {"name": str, "value": float|null, "shap": float}, ...
            ]
        }, ... }

    SHAP values are in log-odds space (pre-calibration). Positive values
    pushed the model toward a home win; negative toward an away win.
    CatBoost returns (n_samples, n_features + 1) where the last column is
    the base/expected value.
    """
    if X.empty:
        return {}

    out_path = PROJECT_ROOT / "data" / "master" / "prediction_explanations.json"
    existing = _load_prediction_explanations("prediction_explanations.json")
    explanations = _build_prediction_explanations(
        model, X, feature_names, upcoming_featured, existing=existing, top_n=top_n,
    )
    with open(out_path, "w") as f:
        json.dump(explanations, f, indent=2)
    log.info(f"Saved SHAP explanations for {len(explanations)} matches to {out_path}")
    return explanations


def _upsert_prediction_explanations_history(
    explanations: dict[str, dict],
    snapshot_dates: dict[str, str],
) -> None:
    """Persist explanation snapshots by game_id and snapshot_date."""
    if not explanations:
        return

    out_path = PROJECT_ROOT / "data" / "master" / "prediction_explanations_history.json"
    history = _load_prediction_explanations("prediction_explanations_history.json")

    saved = 0
    for game_id, explanation in explanations.items():
        snapshot_date = snapshot_dates.get(game_id)
        if not snapshot_date:
            continue
        history.setdefault(game_id, {})[snapshot_date] = explanation
        saved += 1

    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Updated explanation history with {saved} snapshot rows at {out_path}")


def backfill_prediction_explanations_history(year: int | None = None) -> int:
    """Backfill latest saved prediction explanations for completed matches."""
    history_path = PROJECT_ROOT / "data" / "master" / "predictions_history.csv"
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    if not history_path.exists() or not master_path.exists():
        return 0

    pred_history = pd.read_csv(history_path)
    if pred_history.empty or "game_id" not in pred_history.columns or "snapshot_date" not in pred_history.columns:
        return 0

    master = pd.read_csv(master_path, parse_dates=["date"])
    completed = master.dropna(subset=["home_score", "away_score"]).copy()
    if year is not None and "year" in completed.columns:
        completed = completed[completed["year"] == year].copy()
    if completed.empty:
        return 0

    completed_ids = set(completed["game_id"].dropna().astype(int))
    pred_history = pred_history[pred_history["game_id"].isin(completed_ids)].copy()
    if pred_history.empty:
        return 0

    pred_history["snapshot_date"] = pred_history["snapshot_date"].astype(str)
    pred_history = pred_history.sort_values(["game_id", "snapshot_date"]).drop_duplicates(
        "game_id", keep="last"
    )

    existing_history = _load_prediction_explanations("prediction_explanations_history.json")
    missing = pred_history[
        pred_history.apply(
            lambda r: str(r["snapshot_date"]) not in existing_history.get(str(int(r["game_id"])), {}),
            axis=1,
        )
    ].copy()
    if missing.empty:
        log.info("Historical explanation snapshots already up to date")
        return 0

    explainer_model = None
    ensemble = load_betting_ensemble()
    if ensemble:
        explainer_model = ensemble[0]
    else:
        clf_path = str(PROJECT_ROOT / "data" / "model.cbm")
        explainer_model = CatBoostClassifier()
        explainer_model.load_model(clf_path)

    featured = build_features(master).copy()
    targets = featured[featured["game_id"].isin(missing["game_id"])].copy()
    if targets.empty:
        return 0

    targets = targets.merge(
        missing[["game_id", "snapshot_date", "home_win_prob"]],
        on="game_id",
        how="inner",
        suffixes=("", "_snapshot"),
    )
    if targets.empty:
        return 0

    feature_names = [c for c in get_feature_cols() if c in targets.columns]
    explanations = _build_prediction_explanations(
        explainer_model,
        targets[feature_names],
        feature_names,
        targets,
        existing=_latest_explanation_entries(existing_history),
        top_n=10,
    )
    snapshot_dates = {
        str(int(row["game_id"])): str(row["snapshot_date"])
        for _, row in targets.iterrows()
        if pd.notna(row.get("game_id")) and pd.notna(row.get("snapshot_date"))
    }
    _upsert_prediction_explanations_history(explanations, snapshot_dates)
    log.info(f"Backfilled explanation history for {len(explanations)} completed matches")
    return len(explanations)


def _append_predictions_history(output: pd.DataFrame, snapshot_date: str | None = None):
    """Upsert today's predictions into predictions_history.csv.

    Each row gets a snapshot_date. If a row already exists for the same
    (game_id, snapshot_date), it is replaced — so re-running predict on
    the same day overwrites that day's snapshot rather than duplicating.
    """
    history_path = PROJECT_ROOT / "data" / "master" / "predictions_history.csv"
    snapshot_date = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    snapshot = output.copy()
    snapshot["snapshot_date"] = snapshot_date

    if history_path.exists():
        existing = pd.read_csv(history_path)
        # Drop any existing rows for game_ids in this snapshot taken today
        if "snapshot_date" in existing.columns and "game_id" in existing.columns:
            mask = ~(
                (existing["snapshot_date"] == snapshot_date)
                & (existing["game_id"].isin(snapshot["game_id"]))
            )
            existing = existing[mask]
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        combined = snapshot

    combined.to_csv(history_path, index=False)
    log.info(f"Predictions history updated ({len(snapshot)} new rows) at {history_path}")
