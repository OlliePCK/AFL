"""
Historical odds data processing.

Source: aussportsbetting.com — free Excel download with head-to-head,
line, and total-score odds from multiple bookmakers (2009-present).
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import PROJECT_ROOT, DATA_RAW, DATA_PROCESSED
from src.team_mapping import normalize_team
from src.utils import setup_logging

log = setup_logging()

# Map aussportsbetting team names → Squiggle canonical names
_ODDS_TEAM_MAP = {
    "Brisbane": "Brisbane Lions",
    "GWS Giants": "Greater Western Sydney",
}


def load_odds(path: Path | None = None) -> pd.DataFrame:
    """Load and clean the aussportsbetting.com historical odds file."""
    path = path or DATA_RAW / "afl_odds_historical.xlsx"
    df = pd.read_excel(path, header=1)

    # Normalize team names
    df["home_team"] = df["Home Team"].map(lambda t: _ODDS_TEAM_MAP.get(t, t))
    df["away_team"] = df["Away Team"].map(lambda t: _ODDS_TEAM_MAP.get(t, t))

    df["date"] = pd.to_datetime(df["Date"])
    df["year"] = df["date"].dt.year

    # Rename odds columns to snake_case
    rename = {
        "Home Score": "home_score_odds",
        "Away Score": "away_score_odds",
        "Home Odds": "home_odds_avg",
        "Away Odds": "away_odds_avg",
        "Bookmakers Surveyed": "bookmakers_surveyed",
        "Home Odds Open": "home_odds_open",
        "Home Odds Close": "home_odds_close",
        "Away Odds Open": "away_odds_open",
        "Away Odds Close": "away_odds_close",
        "Home Odds Min": "home_odds_min",
        "Home Odds Max": "home_odds_max",
        "Away Odds Min": "away_odds_min",
        "Away Odds Max": "away_odds_max",
        "Home Line Close": "home_line_close",
        "Away Line Close": "away_line_close",
        "Home Line Odds Close": "home_line_odds_close",
        "Away Line Odds Close": "away_line_odds_close",
        "Venue": "venue_odds",
        "Play Off Game?": "is_final_odds",
    }
    df = df.rename(columns=rename)

    # Select columns we care about
    keep = [
        "date", "year", "home_team", "away_team", "venue_odds",
        "home_odds_avg", "away_odds_avg", "bookmakers_surveyed",
        "home_odds_open", "home_odds_close",
        "away_odds_open", "away_odds_close",
        "home_odds_min", "home_odds_max",
        "away_odds_min", "away_odds_max",
        "home_line_close", "away_line_close",
        "home_line_odds_close", "away_line_odds_close",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Convert odds to numeric
    odds_cols = [c for c in df.columns if "odds" in c or "line" in c]
    for col in odds_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Loaded {len(df)} matches with odds data ({df['year'].min()}-{df['year'].max()})")
    return df


def add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Add vig-adjusted implied probabilities from closing odds."""
    for prefix in ["close", "open", "avg"]:
        home_col = f"home_odds_{prefix}"
        away_col = f"away_odds_{prefix}"
        if home_col in df.columns and away_col in df.columns:
            raw_home = 1 / df[home_col]
            raw_away = 1 / df[away_col]
            total = raw_home + raw_away  # overround / vig
            df[f"implied_home_{prefix}"] = raw_home / total
            df[f"implied_away_{prefix}"] = raw_away / total
            df[f"overround_{prefix}"] = total

    return df


def merge_odds_with_master(master: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Merge odds data into master dataset by matching date + teams."""
    odds = add_implied_probabilities(odds)

    # Create merge key: date + team pair (order-independent)
    master["_date_str"] = pd.to_datetime(master["date"]).dt.strftime("%Y-%m-%d")
    odds["_date_str"] = odds["date"].dt.strftime("%Y-%m-%d")

    # Merge on date + home/away teams (exact match)
    merged = master.merge(
        odds.drop(columns=["date", "year", "venue_odds"], errors="ignore"),
        on=["_date_str", "home_team", "away_team"],
        how="left",
    )
    merged = merged.drop(columns=["_date_str"])

    matched = merged["home_odds_close"].notna().sum()
    log.info(f"Odds merge: {matched}/{len(master)} matches matched ({matched/len(master):.1%})")

    return merged


def process_odds():
    """Full pipeline: load odds, merge with master, save."""
    odds = load_odds()
    odds = add_implied_probabilities(odds)

    # Save processed odds
    odds.to_csv(DATA_PROCESSED / "odds_historical.csv", index=False)
    log.info(f"Saved processed odds to {DATA_PROCESSED / 'odds_historical.csv'}")

    # Merge with master
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    if master_path.exists():
        master = pd.read_csv(master_path, parse_dates=["date"])
        merged = merge_odds_with_master(master, odds)
        merged.to_csv(master_path, index=False)
        log.info(f"Updated master dataset with odds data")
        return merged

    return odds
