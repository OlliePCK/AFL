"""
AFL Prediction Pipeline — all-in-one CLI.

Usage:
    python scripts/run_pipeline.py collect     # Refresh data from all sources
    python scripts/run_pipeline.py train       # Train model + evaluate
    python scripts/run_pipeline.py predict     # Predict next round
    python scripts/run_pipeline.py simulate    # Backtest with real odds
    python scripts/run_pipeline.py odds        # Update odds data
    python scripts/run_pipeline.py all         # Full pipeline
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ensure_dirs, setup_logging


def cmd_collect(args):
    """Refresh data from all sources."""
    from src.config import FIRST_YEAR, CURRENT_YEAR
    from src.squiggle_client import fetch_all_games, fetch_all_tips
    from src.afltables_scraper import scrape_all_seasons
    from src.footywire_scraper import scrape_all_seasons as fw_scrape
    from src.dataset_builder import build_master_dataset
    from src.odds import process_odds

    log = setup_logging()
    start, end = args.start_year, args.end_year

    log.info(f"=== Collecting data: {start}-{end} ===")
    fetch_all_games(start, end)
    fetch_all_tips(start, end)
    scrape_all_seasons(start, end)
    if not args.skip_footywire:
        fw_scrape(start, end)
    build_master_dataset()
    process_odds()
    log.info("=== Collection complete ===")


def cmd_train(args):
    """Train model and evaluate."""
    from src.model import run_full_pipeline
    run_full_pipeline()


def cmd_predict(args):
    """Generate predictions for upcoming matches."""
    from src.predict import predict_upcoming
    predict_upcoming()


def cmd_simulate(args):
    """Run betting simulation with real odds."""
    import pandas as pd
    from catboost import CatBoostClassifier
    from src.config import PROJECT_ROOT
    from src.features import build_features
    from src.value import simulate_betting, plot_bankroll

    log = setup_logging()

    df = pd.read_csv(PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv", parse_dates=["date"])
    df = build_features(df)

    model = CatBoostClassifier()
    model.load_model(str(PROJECT_ROOT / "data" / "model.cbm"))

    test = df[(df["year"] >= args.start_year) & (df["year"] <= args.end_year)].copy()
    test = test.dropna(subset=["home_win"])

    log.info(f"Simulating on {len(test)} matches ({args.start_year}-{args.end_year})")

    bets = simulate_betting(test, model, odds_source=args.odds_source,
                            min_edge=args.min_edge, kelly_frac=args.kelly_frac)

    if not bets.empty:
        bets.to_csv(PROJECT_ROOT / "data" / "master" / "simulation_results.csv", index=False)
        plots_dir = PROJECT_ROOT / "data" / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_bankroll(bets, save_path=plots_dir / "bankroll_real_odds.png")


def cmd_odds(args):
    """Update odds data."""
    from src.odds import process_odds
    process_odds()


def main():
    parser = argparse.ArgumentParser(description="AFL Prediction Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Refresh data from all sources")
    p_collect.add_argument("--start-year", type=int, default=2012)
    p_collect.add_argument("--end-year", type=int, default=2026)
    p_collect.add_argument("--skip-footywire", action="store_true")
    p_collect.set_defaults(func=cmd_collect)

    # train
    p_train = sub.add_parser("train", help="Train model + evaluate")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_predict = sub.add_parser("predict", help="Predict next round")
    p_predict.set_defaults(func=cmd_predict)

    # simulate
    p_sim = sub.add_parser("simulate", help="Backtest with real odds")
    p_sim.add_argument("--start-year", type=int, default=2024)
    p_sim.add_argument("--end-year", type=int, default=2025)
    p_sim.add_argument("--odds-source", choices=["closing", "opening", "avg"], default="closing")
    p_sim.add_argument("--min-edge", type=float, default=0.07)
    p_sim.add_argument("--kelly-frac", type=float, default=0.25)
    p_sim.set_defaults(func=cmd_simulate)

    # odds
    p_odds = sub.add_parser("odds", help="Update odds data")
    p_odds.set_defaults(func=cmd_odds)

    # all
    p_all = sub.add_parser("all", help="Full pipeline: collect + train + predict")
    p_all.add_argument("--start-year", type=int, default=2012)
    p_all.add_argument("--end-year", type=int, default=2026)
    p_all.add_argument("--skip-footywire", action="store_true")
    p_all.set_defaults(func=lambda a: (cmd_collect(a), cmd_train(a), cmd_predict(a)))

    args = parser.parse_args()
    ensure_dirs()
    args.func(args)


if __name__ == "__main__":
    main()
