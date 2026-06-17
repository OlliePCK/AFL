"""
Live odds monitoring — fetches, caches, and tracks odds movement over time.

Uses The Odds API (the-odds-api.com) to fetch current AFL head-to-head AND
spread (line) odds.  The line is the single most predictive feature for the
analytical model (+0.0137 LL over raw market implied, 8x noise floor).

Free tier: 500 requests/month.  Each fetch = 2 requests (h2h + spreads).
Strategy: fetch once when odds open (early week), once before game day.
~24 credits/month at 3 fetches/week << 500 free.
"""
import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from src.config import PROJECT_ROOT
from src.team_mapping import normalize_team
from src.utils import fetch_url, setup_logging

log = setup_logging()

ODDS_CACHE_DIR = PROJECT_ROOT / "data" / "live_odds"
SNAPSHOTS_PATH = ODDS_CACHE_DIR / "snapshots.csv"
CURRENT_PATH = ODDS_CACHE_DIR / "current_odds.csv"


def _get_api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY")


def fetch_current_odds() -> pd.DataFrame:
    """Fetch current AFL h2h + spread odds from The Odds API.

    Returns DataFrame with: home_team, away_team, home_odds, away_odds,
    home_odds_best, away_odds_best, home_line, away_line, home_line_odds,
    away_line_odds, n_bookmakers, n_line_bookmakers, commence_time, fetched_at.

    The line columns use the same sign convention as aussportsbetting.com:
    negative = favored (gives up points), positive = underdog (receives points).
    """
    api_key = _get_api_key()
    if not api_key:
        log.warning("No ODDS_API_KEY set — cannot fetch live odds")
        return pd.DataFrame()

    try:
        resp = fetch_url(
            "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds",
            params={
                "apiKey": api_key,
                "regions": "au",
                "markets": "h2h,spreads",
                "oddsFormat": "decimal",
            },
        )
    except Exception as e:
        log.error(f"Odds API request failed: {e}")
        return pd.DataFrame()

    # Check remaining quota from headers
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    log.info(f"Odds API quota: {used} used, {remaining} remaining (cost=2 for h2h+spreads)")

    data = resp.json()
    if not data:
        log.info("No odds data returned from API")
        return pd.DataFrame()

    now = datetime.now(timezone.utc).isoformat()
    results = []

    for game in data:
        try:
            home = normalize_team(game.get("home_team", ""))
            away = normalize_team(game.get("away_team", ""))
        except ValueError:
            continue

        commence = game.get("commence_time", "")
        bookmakers = game.get("bookmakers", [])

        # Collect h2h and spread data across all bookmakers
        home_prices, away_prices = [], []
        home_lines, away_lines = [], []
        home_line_prices, away_line_prices = [], []

        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        try:
                            name = normalize_team(outcome["name"])
                        except ValueError:
                            continue
                        price = outcome["price"]
                        if name == home:
                            home_prices.append(price)
                        elif name == away:
                            away_prices.append(price)

                elif market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        try:
                            name = normalize_team(outcome["name"])
                        except ValueError:
                            continue
                        point = outcome.get("point")
                        price = outcome.get("price")
                        if point is None or price is None:
                            continue
                        if name == home:
                            home_lines.append(float(point))
                            home_line_prices.append(float(price))
                        elif name == away:
                            away_lines.append(float(point))
                            away_line_prices.append(float(price))

        if not home_prices or not away_prices:
            continue

        row = {
            "home_team": home,
            "away_team": away,
            "home_odds": np.median(home_prices),
            "away_odds": np.median(away_prices),
            "home_odds_best": max(home_prices),
            "away_odds_best": max(away_prices),
            "n_bookmakers": len(bookmakers),
            "commence_time": commence,
            "fetched_at": now,
        }

        # Line data (may be absent if no bookmaker offers spreads)
        if home_lines:
            row["home_line"] = float(np.median(home_lines))
            row["away_line"] = float(np.median(away_lines)) if away_lines else np.nan
            row["home_line_odds"] = float(np.median(home_line_prices))
            row["away_line_odds"] = float(np.median(away_line_prices)) if away_line_prices else np.nan
            row["n_line_bookmakers"] = len(home_lines)
        else:
            row["home_line"] = np.nan
            row["away_line"] = np.nan
            row["home_line_odds"] = np.nan
            row["away_line_odds"] = np.nan
            row["n_line_bookmakers"] = 0

        results.append(row)

    df = pd.DataFrame(results)
    if not df.empty:
        n_with_line = (df["home_line"].notna()).sum()
        log.info(f"Fetched odds for {len(df)} matches from "
                 f"{df['n_bookmakers'].iloc[0]} bookmakers "
                 f"({n_with_line} with spread lines)")

    return df


def save_snapshot(odds: pd.DataFrame):
    """Append current odds to the snapshot history and update current_odds.csv."""
    ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if odds.empty:
        return

    # Save as current
    odds.to_csv(CURRENT_PATH, index=False)

    # Append to snapshots
    if SNAPSHOTS_PATH.exists():
        existing = pd.read_csv(SNAPSHOTS_PATH)
        combined = pd.concat([existing, odds], ignore_index=True)
    else:
        combined = odds

    combined.to_csv(SNAPSHOTS_PATH, index=False)
    log.info(f"Saved odds snapshot ({len(odds)} matches, {len(combined)} total snapshots)")


def load_current_odds() -> pd.DataFrame:
    """Load the most recent odds snapshot."""
    if CURRENT_PATH.exists():
        return pd.read_csv(CURRENT_PATH)
    return pd.DataFrame()


def load_snapshots() -> pd.DataFrame:
    """Load all historical odds snapshots."""
    if SNAPSHOTS_PATH.exists():
        return pd.read_csv(SNAPSHOTS_PATH)
    return pd.DataFrame()


def compute_movement(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Compute odds movement from first snapshot (opening) to latest (current).

    Returns DataFrame with: home_team, away_team, home_odds_open, home_odds_current,
    implied_move (positive = market moved toward home), move_direction.
    """
    if snapshots.empty:
        return pd.DataFrame()

    snapshots = snapshots.copy()
    snapshots["fetched_at"] = pd.to_datetime(snapshots["fetched_at"], format="ISO8601")

    results = []
    for (ht, at), group in snapshots.groupby(["home_team", "away_team"]):
        group = group.sort_values("fetched_at")
        first = group.iloc[0]
        last = group.iloc[-1]

        ho_open, ao_open = float(first["home_odds"]), float(first["away_odds"])
        ho_curr, ao_curr = float(last["home_odds"]), float(last["away_odds"])

        # Implied probability movement
        imp_open = (1 / ho_open) / (1 / ho_open + 1 / ao_open)
        imp_curr = (1 / ho_curr) / (1 / ho_curr + 1 / ao_curr)
        imp_move = imp_curr - imp_open  # positive = market moved toward home

        results.append({
            "home_team": ht,
            "away_team": at,
            "home_odds_open": ho_open,
            "away_odds_open": ao_open,
            "home_odds_current": ho_curr,
            "away_odds_current": ao_curr,
            "implied_home_open": imp_open,
            "implied_home_current": imp_curr,
            "implied_move": imp_move,
            "move_magnitude": abs(imp_move),
            "n_snapshots": len(group),
            "first_seen": first["fetched_at"],
            "last_seen": last["fetched_at"],
        })

    return pd.DataFrame(results)


def fetch_and_save() -> pd.DataFrame:
    """Convenience: fetch current odds and save snapshot. Returns the odds."""
    odds = fetch_current_odds()
    if not odds.empty:
        save_snapshot(odds)
    return odds
