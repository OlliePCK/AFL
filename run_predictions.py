"""
AFL Match Predictions — Live Pipeline

Usage:
    python run_predictions.py              # Predict + odds + value bets + auto-log
    python run_predictions.py --no-odds    # Predict without fetching live odds
    python run_predictions.py --fetch-odds # Just fetch and save current odds snapshot
    python run_predictions.py --update     # Update dataset with latest results, then predict
    python run_predictions.py --retrain    # Retrain model on 2012-2024, validate on 2025
    python run_predictions.py --reconcile  # Reconcile pending bets against results + show P&L
    python run_predictions.py --performance # Show betting performance summary
    python run_predictions.py --bankroll N # Custom bankroll for bet sizing (default: 1000)
    python run_predictions.py --optimize   # Optuna hyperparameter + feature optimization, then retrain
    python run_predictions.py --optimize --n-trials 5  # Quick smoke test (5 trials)

Strategy (walk-forward validated 2019-2025, V2 H5 filter):
    - Odds-free betting model (20 features, no market odds)
    - 15-30% edge band (V2 baseline band)
    - REQUIRE_AGREE: only bet when market has moved TOWARD our pick by
      >=0.5% implied prob since opening. Strict AGREE signal from
      odds_monitor snapshots. NEUTRAL and DISAGREE bets are rejected.
    - No favorites-only gate. V2 showed H5 alone (any role) beats
      H1+H5 (fav+H5) on every metric — the alignment signal already
      filters out the underdog blowups that the favorite gate guarded
      against.
    - 0.15 Kelly bet sizing (down from 0.25), max 10% of bankroll.
      Kelly sensitivity: halves max drawdown (71% -> 56%) for only
      1.7pp less ROI.

V2 backtest calibration (scripts/backtest_value_v2.py, walk-forward 2019-2025):
    Baseline (15-30%, no filter)       : 301 bets, 53.5% win, +3.3% ROI, 4/7 yrs, 63% maxDD
    H1 favorites only                  : 143 bets, 72.0% win, +4.7% ROI, 3/7 yrs, 42% maxDD
    H4 V3 agrees (walk-forward)        : 163 bets, 65.6% win, -0.1% ROI, 4/7 yrs  (null)
    H5 smart-money aligned (THIS)      : 211 bets, 57.8% win, +12.8% ROI, 5/7 yrs, 71% maxDD
    H5 at 0.15 Kelly (DEPLOYED)        : 211 bets, 57.8% win, +11.1% ROI, 5/7 yrs, 56% maxDD

H5 missed strict CI bar by ~1pp (lower bound -1.0%). Monitor live ROI
for 2-3 rounds; revert to baseline or NOT_DISAGREE softening if live
performance diverges from backtest.
"""
import argparse
import sys
import pandas as pd
import numpy as np

from src.config import PROJECT_ROOT
from src.predict import predict_upcoming
from src.odds_monitor import fetch_and_save, load_current_odds, load_snapshots, compute_movement
from src.value import decimal_odds_to_implied_prob, calculate_edge, kelly_fraction, expected_value
from src.bet_tracker import log_bets, reconcile_results, get_performance
from src.utils import setup_logging

log = setup_logging()

MIN_EDGE = 0.15            # V2 baseline edge band (was 0.10 pre-V2)
MAX_EDGE = 0.30            # V2 baseline edge band (was 0.25 pre-V2)
REQUIRE_AGREE_BETS = True  # V2 H5: strict AGREE on market movement since open
MOVE_THRESH = 0.005        # 0.5% implied-prob move classifies AGREE/DISAGREE
KELLY_FRAC = 0.15          # V2 Kelly sensitivity: halves drawdown vs 0.25
MAX_BET_PCT = 0.10
VALUE_BET_COLUMNS = [
    "match",
    "venue",
    "date",
    "bet_team",
    "bet_side",
    "model_prob",
    "market_prob",
    "edge",
    "odds",
    "ev_per_dollar",
    "kelly_pct",
    "bet_amount",
    "movement",
    "move_value",
]


def run_full_pipeline(fetch_odds: bool = True, bankroll: float = 1000.0):
    """Run the complete prediction + value detection pipeline."""
    # Step 1: Generate model predictions
    log.info("=" * 80)
    log.info("STEP 1: GENERATING PREDICTIONS")
    log.info("=" * 80)
    predictions = predict_upcoming()
    if predictions.empty:
        log.info("No upcoming matches to predict")
        return

    # Step 2: Fetch live odds
    if fetch_odds:
        log.info("\n" + "=" * 80)
        log.info("STEP 2: FETCHING LIVE ODDS")
        log.info("=" * 80)
        odds = fetch_and_save()
        if odds.empty:
            log.info("No live odds available — using any cached odds")
            odds = load_current_odds()
    else:
        odds = load_current_odds()

    # Filter to next round only
    if "roundname" in predictions.columns:
        next_round = predictions["roundname"].dropna().iloc[0] if len(predictions) > 0 else None
        if next_round:
            predictions = predictions[predictions["roundname"] == next_round].copy()
            log.info(f"Focusing on {next_round} ({len(predictions)} matches)")

    if odds.empty:
        log.info("No odds available — showing predictions only (no value detection)")
        _write_value_bets([])
        _print_predictions_only(predictions)
        return

    # Step 3: Compute odds movement (if we have multiple snapshots)
    snapshots = load_snapshots()
    movement = compute_movement(snapshots)

    # Step 4: Value detection with movement-aware filtering
    log.info("\n" + "=" * 80)
    log.info("STEP 3: VALUE DETECTION")
    log.info("=" * 80)

    value_bets = _detect_value_bets(predictions, odds, movement, bankroll)

    # Step 5: Save and display
    output_path = PROJECT_ROOT / "data" / "master" / "upcoming_predictions.csv"
    predictions.to_csv(output_path, index=False)

    _write_value_bets(value_bets)
    if value_bets:
        vb_path = PROJECT_ROOT / "data" / "master" / "value_bets.csv"
        log.info(f"\nValue bets saved to {vb_path}")

        # Auto-log to bet tracker
        round_name = ""
        if "roundname" in predictions.columns:
            round_name = predictions["roundname"].dropna().iloc[0] if len(predictions) > 0 else ""
        log_bets(value_bets, round_name=str(round_name))

    return predictions


def _write_value_bets(value_bets: list[dict]):
    """Rewrite value_bets.csv on every run, even when no bets qualify."""
    vb_path = PROJECT_ROOT / "data" / "master" / "value_bets.csv"
    vb_df = pd.DataFrame(value_bets, columns=VALUE_BET_COLUMNS)
    vb_df.to_csv(vb_path, index=False)


def _detect_value_bets(predictions: pd.DataFrame, odds: pd.DataFrame,
                       movement: pd.DataFrame, bankroll: float) -> list[dict]:
    """Compare model probabilities to live odds, apply movement filter."""
    value_bets = []

    for _, pred in predictions.iterrows():
        ht, at = pred["home_team"], pred["away_team"]
        home_prob = pred.get("home_win_prob")
        if pd.isna(home_prob):
            continue
        away_prob = 1.0 - home_prob

        # Find odds for this match
        match_odds = odds[
            (odds["home_team"] == ht) & (odds["away_team"] == at)
        ]
        if match_odds.empty:
            continue

        row = match_odds.iloc[0]
        home_odds = float(row.get("home_odds_best", row.get("home_odds", 0)))
        away_odds = float(row.get("away_odds_best", row.get("away_odds", 0)))
        if home_odds <= 1 or away_odds <= 1:
            continue

        # Fair implied probabilities (vig-adjusted)
        fair_home, fair_away = decimal_odds_to_implied_prob(home_odds, away_odds)

        # Calculate edges
        home_edge = calculate_edge(home_prob, fair_home)
        away_edge = calculate_edge(away_prob, fair_away)

        # Pick best side — edge in [MIN_EDGE, MAX_EDGE). No favorites-only
        # gate in V2: the smart-money alignment filter below already handles
        # underdog blowouts (V2 H5 alone > H1+H5 on every metric).
        if home_edge >= away_edge and MIN_EDGE <= home_edge < MAX_EDGE:
            bet_side, edge, bet_odds, bet_prob = "home", home_edge, home_odds, home_prob
            bet_team = ht
        elif MIN_EDGE <= away_edge < MAX_EDGE:
            bet_side, edge, bet_odds, bet_prob = "away", away_edge, away_odds, away_prob
            bet_team = at
        else:
            continue

        # Kelly sizing
        kelly_pct = kelly_fraction(bet_prob, bet_odds, fraction=KELLY_FRAC)
        kelly_pct = min(kelly_pct, MAX_BET_PCT)
        bet_amount = bankroll * kelly_pct
        ev = expected_value(bet_prob, bet_odds)

        # Check odds movement agreement
        move_status = "unknown"
        move_value = None
        if not movement.empty:
            match_move = movement[
                (movement["home_team"] == ht) & (movement["away_team"] == at)
            ]
            if not match_move.empty:
                imp_move = float(match_move.iloc[0]["implied_move"])
                move_value = imp_move
                # imp_move > 0 = market moved toward home
                if bet_side == "home":
                    move_status = "AGREE" if imp_move > MOVE_THRESH else (
                        "DISAGREE" if imp_move < -MOVE_THRESH else "NEUTRAL")
                else:
                    move_status = "AGREE" if imp_move < -MOVE_THRESH else (
                        "DISAGREE" if imp_move > MOVE_THRESH else "NEUTRAL")

        # V2 H5 filter: require strict AGREE — reject NEUTRAL and DISAGREE.
        # Walk-forward 2019-2025: H5 lifted ROI from +3.3% (baseline) to
        # +12.8% with 5/7 profitable years. Gate relaxation: if no movement
        # data (first run of the round), move_status is "unknown" and the
        # bet is rejected — run predictions a second time after the market
        # has moved to get AGREE/DISAGREE classification.
        if REQUIRE_AGREE_BETS and move_status != "AGREE":
            if move_value is not None:
                log.info(
                    f"  SKIP {bet_team} @ {bet_odds:.2f}: movement "
                    f"{move_status} ({move_value:+.3f}, edge {edge:.1%})"
                )
            else:
                log.info(
                    f"  SKIP {bet_team} @ {bet_odds:.2f}: no movement data "
                    f"(edge {edge:.1%}) — fetch odds again once market moves"
                )
            continue

        bet_info = {
            "match": f"{ht} vs {at}",
            "venue": pred.get("venue", ""),
            "date": pred.get("date", ""),
            "bet_team": bet_team,
            "bet_side": bet_side,
            "model_prob": bet_prob,
            "market_prob": fair_home if bet_side == "home" else fair_away,
            "edge": edge,
            "odds": bet_odds,
            "ev_per_dollar": ev,
            "kelly_pct": kelly_pct,
            "bet_amount": bet_amount,
            "movement": move_status,
            "move_value": move_value,
        }
        value_bets.append(bet_info)

    # Sort by edge
    value_bets.sort(key=lambda x: x["edge"], reverse=True)

    # Display
    if value_bets:
        log.info(f"\n{'='*90}")
        log.info(f"VALUE BETS (15-30% edge, smart-money AGREE, 0.15 Kelly)")
        log.info(f"{'='*90}")
        log.info(f"{'Match':>40} {'Bet':>20} {'Edge':>6} {'Odds':>6} "
                 f"{'EV':>6} {'Kelly':>6} {'$Bet':>7} {'Move':>10}")
        log.info(f"{'-'*90}")

        total_bet = 0
        for b in value_bets:
            move_icon = {"AGREE": "++ AGREE", "DISAGREE": "-- DISAG",
                         "NEUTRAL": "~  NEUT", "unknown": "?  ----"}.get(b["movement"], "?")
            log.info(
                f"{b['match']:>40} {b['bet_team']:>20} "
                f"{b['edge']:>5.1%} {b['odds']:>5.2f} "
                f"{b['ev_per_dollar']:>+5.2f} {b['kelly_pct']:>5.1%} "
                f"${b['bet_amount']:>6.0f} {move_icon:>10}"
            )
            total_bet += b["bet_amount"]

        log.info(f"{'-'*90}")
        log.info(f"Total bets: {len(value_bets)}, Total staked: ${total_bet:.0f} "
                 f"({total_bet/bankroll:.0%} of bankroll)")

        # Flag movement state for reviewer clarity (DISAGREE are auto-skipped above)
        agree = [b for b in value_bets if b["movement"] == "AGREE"]
        neutral = [b for b in value_bets if b["movement"] == "NEUTRAL"]
        unknown = [b for b in value_bets if b["movement"] in ("unknown", "UNKNOWN")]
        if agree:
            log.info(f"  STRONG: {len(agree)} bet(s) AGREE with line movement "
                     f"(V2 H5 backtest: +11.1% ROI, 211 bets, 5/7 yrs, 0.15 Kelly).")
        if neutral:
            log.info(f"  OK:     {len(neutral)} bet(s) NEUTRAL movement.")
        if unknown:
            log.info(f"  NOTE:   {len(unknown)} bet(s) no movement data yet "
                     f"(snapshots still accumulating).")
    else:
        log.info("\nNo value bets found (15-30% edge, smart-money AGREE).")

    return value_bets


def _print_predictions_only(predictions: pd.DataFrame):
    """Print predictions without odds comparison."""
    log.info(f"\n{'='*80}")
    log.info("PREDICTIONS (no odds available for value detection)")
    log.info(f"{'='*80}")
    for _, row in predictions.iterrows():
        ht = str(row.get("home_team", "?"))
        at = str(row.get("away_team", "?"))
        winner = str(row.get("predicted_winner", "?"))
        conf = row.get("confidence", 0)
        log.info(f"  {ht:>25s} vs {at:<25s} | {winner} ({conf:.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFL Match Predictions")
    parser.add_argument("--fetch-odds", action="store_true",
                        help="Only fetch and save current odds (no predictions)")
    parser.add_argument("--no-odds", action="store_true",
                        help="Run predictions without fetching live odds")
    parser.add_argument("--update", action="store_true",
                        help="Update dataset with latest results before predicting")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain model on 2012-2024, validate on 2025")
    parser.add_argument("--reconcile", action="store_true",
                        help="Reconcile pending bets against completed results")
    parser.add_argument("--performance", action="store_true",
                        help="Show betting performance summary")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Current bankroll for bet sizing (default: 1000)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run Optuna hyperparameter + feature optimization, then retrain")
    parser.add_argument("--n-trials", type=int, default=80,
                        help="Number of Optuna trials for --optimize (default: 80)")
    args = parser.parse_args()

    if args.optimize:
        from src.model import optimize_model, retrain_production_model
        from src.features import build_features
        from src.config import PROJECT_ROOT

        master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
        df = pd.read_csv(master_path, parse_dates=["date"])
        df = build_features(df)
        optimize_model(df, n_trials=args.n_trials)

        log.info("Retraining with optimized parameters...")
        retrain_production_model()

        # Re-export dashboard data
        import subprocess
        subprocess.run([sys.executable,
                        str(PROJECT_ROOT / "scripts" / "export_dashboard_data.py")])
        log.info("Dashboard data exported.")
    elif args.retrain:
        from src.model import retrain_production_model
        retrain_production_model()
    elif args.reconcile:
        reconcile_results()
        get_performance()
    elif args.performance:
        get_performance()
    elif args.fetch_odds:
        odds = fetch_and_save()
        if not odds.empty:
            log.info(f"Fetched and saved {len(odds)} matches")
            for _, r in odds.iterrows():
                log.info(f"  {r['home_team']:>25} vs {r['away_team']:<25} "
                         f"H:{r['home_odds']:.2f} A:{r['away_odds']:.2f} "
                         f"(best H:{r['home_odds_best']:.2f} A:{r['away_odds_best']:.2f})")
    else:
        if args.update:
            from src.dataset_builder import update_master_dataset
            update_master_dataset()
        run_full_pipeline(fetch_odds=not args.no_odds, bankroll=args.bankroll)
