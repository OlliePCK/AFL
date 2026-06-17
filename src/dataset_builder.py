import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED, DATA_MASTER
from src.utils import setup_logging

log = setup_logging()


def build_master_dataset() -> pd.DataFrame:
    """Merge all data sources into a single match-level dataset."""
    # 1. Load Squiggle games as the spine
    games_path = DATA_PROCESSED / "squiggle_games_clean.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"Run Squiggle collection first: {games_path}")
    games = pd.read_csv(games_path, parse_dates=["date"])
    log.info(f"Loaded {len(games)} Squiggle games")

    # 2. Load aggregated tips
    tips_path = DATA_PROCESSED / "squiggle_tips_agg.csv"
    if tips_path.exists():
        tips = pd.read_csv(tips_path)
        games = games.merge(tips, on="game_id", how="left")
        log.info(f"Merged tips: {tips['game_id'].nunique()} games with predictions")

    # 3. Load AFL Tables data (quarter scores, attendance)
    afltables_path = DATA_PROCESSED / "afltables_matches_clean.csv"
    if afltables_path.exists():
        aft = pd.read_csv(afltables_path)
        aft_cols = ["year", "round", "home_team", "away_team", "attendance"]
        aft_available = [c for c in aft_cols if c in aft.columns]
        if aft_available:
            aft_merge = aft[aft_available].drop_duplicates()
            if "round" in aft_merge.columns:
                aft_merge["round"] = aft_merge["round"].apply(_parse_round_number)
            games = games.merge(
                aft_merge, on=["year", "round", "home_team", "away_team"], how="left"
            )
            log.info(f"Merged AFL Tables data")

    # 4. Load Footywire team stats
    fw_path = DATA_PROCESSED / "footywire_team_stats.csv"
    if fw_path.exists():
        fw = pd.read_csv(fw_path)
        log.info(f"Loaded {len(fw)} Footywire team-match stat rows")
        games = _merge_footywire(games, fw)

    # 5. Merge historical odds (aussportsbetting.com)
    odds_path = DATA_RAW / "afl_odds_historical.xlsx"
    if odds_path.exists():
        from src.odds import load_odds, add_implied_probabilities
        odds = load_odds(odds_path)
        odds = add_implied_probabilities(odds)

        # Merge on date + home/away teams
        games["_date_str"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
        odds["_date_str"] = odds["date"].dt.strftime("%Y-%m-%d")
        odds_cols = [c for c in odds.columns if c not in ("date", "year", "venue_odds")]
        games = games.merge(odds[odds_cols], on=["_date_str", "home_team", "away_team"], how="left")
        games = games.drop(columns=["_date_str"])

        matched = games["home_odds_close"].notna().sum()
        log.info(f"Merged odds: {matched}/{len(games)} matches ({matched/len(games):.1%})")

        # Odds movement features (analytical model only — NOT used in betting model)
        if "implied_home_close" in games.columns and "implied_home_open" in games.columns:
            games["odds_move"] = games["implied_home_close"] - games["implied_home_open"]
            games["odds_move_magnitude"] = games["odds_move"].abs()
        if "overround_close" in games.columns and "overround_open" in games.columns:
            games["overround_change"] = games["overround_close"] - games["overround_open"]
        n_move = games["odds_move"].notna().sum() if "odds_move" in games.columns else 0
        log.info(f"Odds movement features: {n_move} matches with open+close implied probs")

    # 6. Merge weather data (if cache exists)
    from src.weather import merge_weather, load_weather_cache
    if not load_weather_cache().empty:
        games = merge_weather(games)

    # 7. Add target variables
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["margin"] = games["home_score"] - games["away_score"]
    games.loc[games["home_score"] == games["away_score"], "home_win"] = 0

    # 8. Sort chronologically
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)

    # 9. Save
    games.to_csv(DATA_MASTER / "afl_master_dataset.csv", index=False)
    log.info(f"Master dataset: {len(games)} matches, {len(games.columns)} columns")
    log.info(f"Date range: {games['date'].min()} to {games['date'].max()}")
    log.info(f"Home win rate: {games['home_win'].mean():.3f}")

    return games


def update_master_dataset(year: int = 2026) -> int:
    """Incrementally add new completed matches to the master dataset.

    Fetches completed games for the given year from Squiggle,
    appends any that aren't already in the dataset.
    Returns the number of new matches added.
    """
    from src.squiggle_client import fetch_games
    from src.team_mapping import normalize_team

    master_path = DATA_MASTER / "afl_master_dataset.csv"
    if not master_path.exists():
        log.error("Master dataset not found. Run build_master_dataset() first.")
        return 0

    existing = pd.read_csv(master_path, parse_dates=["date"])
    existing_ids = set(existing["game_id"].dropna().astype(int))
    log.info(f"Existing dataset: {len(existing)} matches ({len(existing_ids)} unique game_ids)")

    # Fetch completed games for this year
    new_games = fetch_games(year)
    if new_games.empty:
        log.info(f"No completed games found for {year}")
        return 0

    # Filter to truly new games
    new_games["game_id"] = new_games["game_id"].astype(int)
    new_games = new_games[~new_games["game_id"].isin(existing_ids)]

    if new_games.empty:
        log.info(f"Dataset already up to date for {year}")
        return 0

    # Normalize team names
    for col in ["home_team", "away_team"]:
        new_games[col] = new_games[col].apply(
            lambda t: normalize_team(t) if pd.notna(t) else t
        )

    # Add target variables
    new_games["home_win"] = (new_games["home_score"] > new_games["away_score"]).astype(int)
    new_games["margin"] = new_games["home_score"] - new_games["away_score"]
    new_games.loc[new_games["home_score"] == new_games["away_score"], "home_win"] = 0

    # Concat, dedup, sort
    combined = pd.concat([existing, new_games], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_id"], keep="first")
    combined = combined.sort_values(["date", "game_id"]).reset_index(drop=True)

    combined.to_csv(master_path, index=False)
    n_added = len(new_games)
    log.info(f"Added {n_added} new matches for {year}. "
             f"Dataset now: {len(combined)} total matches.")

    return n_added


def _merge_footywire(games: pd.DataFrame, fw: pd.DataFrame) -> pd.DataFrame:
    """Merge Footywire team-level stats into games by matching team pairs per mid.

    Each Footywire mid has exactly 2 rows (one per team). We match these to
    Squiggle games by finding which Squiggle game has the same two teams
    within the same year (date-aware matching to avoid cross-year misalignment).
    """
    stat_cols = [c for c in ["K", "HB", "D", "M", "G", "B", "T", "HO",
                              "GA", "I50", "CL", "CG", "R50", "FF", "FA", "AF", "SC"]
                 if c in fw.columns]

    if not stat_cols:
        log.warning("No stat columns found in Footywire data")
        return games

    # Group Footywire by mid to get both teams per match
    fw_grouped = fw.groupby("mid")

    mid_data = {}
    for mid, group in fw_grouped:
        if len(group) != 2:
            continue
        teams = group["team"].tolist()
        year = int(group["year"].iloc[0]) if "year" in group.columns else None
        stats = {}
        for _, row in group.iterrows():
            stats[row["team"]] = {c: row[c] for c in stat_cols if c in row.index}
        mid_data[mid] = {"teams": set(teams), "stats": stats, "year": year}

    # Build lookup by (team-pair, year) for date-aware matching
    pair_year_to_mids: dict[tuple, list] = {}
    for mid, data in mid_data.items():
        key = (frozenset(data["teams"]), data["year"])
        pair_year_to_mids.setdefault(key, []).append(mid)

    # Sort mids per key so we consume them in chronological order
    for key in pair_year_to_mids:
        pair_year_to_mids[key].sort()

    # Track which mids have been used
    used_mids = set()

    # Add stat columns + mid tracking
    games["mid"] = pd.NA
    for prefix in ["home", "away"]:
        for stat in stat_cols:
            games[f"{prefix}_{stat}"] = pd.NA

    matched = 0
    goal_mismatches = 0
    for idx, row in games.iterrows():
        ht, at = row["home_team"], row["away_team"]
        team_pair = frozenset({ht, at})
        game_year = int(row["year"]) if pd.notna(row.get("year")) else None

        # Try year-specific match first, then fall back to any year
        key = (team_pair, game_year)
        mids = pair_year_to_mids.get(key, [])
        if not mids:
            # Fallback: try without year constraint
            for k, v in pair_year_to_mids.items():
                if k[0] == team_pair:
                    mids = v
                    break

        mid = None
        for m in mids:
            if m not in used_mids:
                mid = m
                break

        if mid is None:
            continue

        used_mids.add(mid)
        games.at[idx, "mid"] = mid
        stats = mid_data[mid]["stats"]

        for stat in stat_cols:
            if ht in stats:
                games.at[idx, f"home_{stat}"] = stats[ht].get(stat)
            if at in stats:
                games.at[idx, f"away_{stat}"] = stats[at].get(stat)
        matched += 1

        # Diagnostic: cross-validate goals where available
        if "G" in stat_cols and pd.notna(row.get("home_goals")):
            fw_home_g = stats.get(ht, {}).get("G")
            sq_home_g = row["home_goals"]
            if fw_home_g is not None and pd.notna(fw_home_g) and pd.notna(sq_home_g):
                if int(fw_home_g) != int(sq_home_g):
                    goal_mismatches += 1

    log.info(f"Merged Footywire stats: {matched}/{len(games)} matches")
    if goal_mismatches > 0:
        log.warning(f"Footywire goal mismatches: {goal_mismatches} (possible swapped fixtures)")

    return games


def _parse_round_number(round_str) -> int | None:
    """Extract round number from strings like 'Round 5', 'R5', etc."""
    if pd.isna(round_str):
        return None
    import re
    m = re.search(r"(\d+)", str(round_str).strip())
    return int(m.group(1)) if m else None
