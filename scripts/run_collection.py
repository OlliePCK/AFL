"""
AFL Data Collection Pipeline

Usage:
    python scripts/run_collection.py [--start-year 2012] [--end-year 2026] [--skip-footywire] [--skip-afltables]
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FIRST_YEAR, CURRENT_YEAR
from src.utils import ensure_dirs, setup_logging


def main():
    parser = argparse.ArgumentParser(description="AFL Data Collection Pipeline")
    parser.add_argument("--start-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--end-year", type=int, default=CURRENT_YEAR)
    parser.add_argument("--skip-footywire", action="store_true", help="Skip slow Footywire scrape")
    parser.add_argument("--skip-afltables", action="store_true", help="Skip AFL Tables scrape")
    parser.add_argument("--skip-tips", action="store_true", help="Skip Squiggle tips (faster)")
    args = parser.parse_args()

    log = setup_logging()
    ensure_dirs()

    log.info(f"=== AFL Data Collection: {args.start_year}-{args.end_year} ===")

    # Step 1: Squiggle games (fast — one API call per year)
    log.info("--- Step 1: Squiggle Games ---")
    from src.squiggle_client import fetch_all_games
    games = fetch_all_games(args.start_year, args.end_year)
    log.info(f"Squiggle: {len(games)} matches collected")

    # Step 2: Squiggle tips
    if not args.skip_tips:
        log.info("--- Step 2: Squiggle Tips ---")
        from src.squiggle_client import fetch_all_tips
        tips = fetch_all_tips(args.start_year, args.end_year)
        log.info(f"Squiggle tips: {len(tips)} game aggregations")

    # Step 3: AFL Tables
    if not args.skip_afltables:
        log.info("--- Step 3: AFL Tables ---")
        from src.afltables_scraper import scrape_all_seasons
        aft = scrape_all_seasons(args.start_year, args.end_year)
        log.info(f"AFL Tables: {len(aft)} matches scraped")

    # Step 4: Footywire (slow)
    if not args.skip_footywire:
        log.info("--- Step 4: Footywire ---")
        from src.footywire_scraper import scrape_all_seasons as fw_scrape
        fw = fw_scrape(args.start_year, args.end_year)
        log.info(f"Footywire: {len(fw)} team-match stat rows")

    # Step 5: Build master dataset
    log.info("--- Step 5: Building Master Dataset ---")
    from src.dataset_builder import build_master_dataset
    master = build_master_dataset()

    # Summary
    log.info("=== Collection Complete ===")
    log.info(f"Total matches: {len(master)}")
    log.info(f"Years: {master['year'].min()}-{master['year'].max()}")
    log.info(f"Columns: {len(master.columns)}")
    log.info(f"Home win rate: {master['home_win'].mean():.1%}")
    if "margin" in master.columns:
        log.info(f"Mean margin: {master['margin'].mean():.1f} points")


if __name__ == "__main__":
    main()
