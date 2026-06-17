import pandas as pd
from tqdm import tqdm

from src.config import SQUIGGLE_BASE_URL, SQUIGGLE_USER_AGENT, DATA_RAW, DATA_PROCESSED
from src.utils import fetch_url, setup_logging
from src.team_mapping import normalize_team

log = setup_logging()


def _squiggle_get(params: dict) -> dict:
    headers = {"User-Agent": SQUIGGLE_USER_AGENT}
    resp = fetch_url(SQUIGGLE_BASE_URL, params=params, headers=headers)
    return resp.json()


def fetch_games(year: int) -> pd.DataFrame:
    data = _squiggle_get({"q": "games", "year": str(year), "complete": "100"})
    games = data.get("games", [])
    if not games:
        return pd.DataFrame()
    df = pd.DataFrame(games)
    df = df.rename(columns={
        "id": "game_id", "hteam": "home_team", "ateam": "away_team",
        "hscore": "home_score", "ascore": "away_score",
        "hgoals": "home_goals", "hbehinds": "home_behinds",
        "agoals": "away_goals", "abehinds": "away_behinds",
    })
    keep = ["game_id", "year", "round", "roundname", "date", "localtime", "venue",
            "home_team", "away_team", "home_score", "away_score",
            "home_goals", "home_behinds", "away_goals", "away_behinds",
            "winner", "is_final", "is_grand_final"]
    df = df[[c for c in keep if c in df.columns]]
    return df


def fetch_tips(year: int) -> pd.DataFrame:
    data = _squiggle_get({"q": "tips", "year": str(year)})
    tips = data.get("tips", [])
    if not tips:
        return pd.DataFrame()
    df = pd.DataFrame(tips)
    df = df.rename(columns={
        "gameid": "game_id", "hteam": "home_team", "ateam": "away_team",
    })
    keep = ["game_id", "year", "round", "source", "sourceid", "tip",
            "confidence", "hconfidence", "margin", "hmargin", "correct", "err"]
    df = df[[c for c in keep if c in df.columns]]
    return df


def fetch_standings(year: int, round_num: int | None = None) -> pd.DataFrame:
    params = {"q": "standings", "year": str(year)}
    if round_num is not None:
        params["round"] = str(round_num)
    data = _squiggle_get(params)
    standings = data.get("standings", [])
    if not standings:
        return pd.DataFrame()
    df = pd.DataFrame(standings)
    df["year"] = year
    if round_num is not None:
        df["round"] = round_num
    return df


def fetch_all_games(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year in tqdm(range(start_year, end_year + 1), desc="Squiggle games"):
        df = fetch_games(year)
        if not df.empty:
            frames.append(df)
            log.info(f"Squiggle games {year}: {len(df)} matches")
    if not frames:
        return pd.DataFrame()
    all_games = pd.concat(frames, ignore_index=True)
    all_games.to_csv(DATA_RAW / "squiggle_games.csv", index=False)
    log.info(f"Saved {len(all_games)} total games to squiggle_games.csv")

    # Clean version with normalized team names
    clean = all_games.copy()
    clean["home_team"] = clean["home_team"].apply(normalize_team)
    clean["away_team"] = clean["away_team"].apply(normalize_team)
    clean["winner"] = clean["winner"].apply(lambda x: normalize_team(x) if pd.notna(x) and x else x)
    clean["date"] = pd.to_datetime(clean["date"])
    clean.to_csv(DATA_PROCESSED / "squiggle_games_clean.csv", index=False)
    log.info(f"Saved cleaned games to squiggle_games_clean.csv")
    return clean


def fetch_all_standings(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch end-of-round standings for every (year, round) combination.

    Caches to data/processed/squiggle_standings.csv.  Skips rounds that
    are already present in the cache so it can be re-run cheaply.
    """
    csv_path = DATA_PROCESSED / "squiggle_standings.csv"

    existing = pd.DataFrame()
    existing_keys: set[tuple[int, int]] = set()
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if {"year", "round"}.issubset(existing.columns):
            existing_keys = set(zip(existing["year"], existing["round"]))
            log.info(f"Standings cache: {len(existing_keys)} (year, round) entries")

    frames = [existing] if not existing.empty else []

    for year in tqdm(range(start_year, end_year + 1), desc="Standings"):
        # Discover which rounds have completed games
        games_data = _squiggle_get({"q": "games", "year": str(year), "complete": "100"})
        rounds_with_games = sorted({
            int(g["round"]) for g in games_data.get("games", [])
            if g.get("round") is not None
        })

        for rnd in rounds_with_games:
            if (year, rnd) in existing_keys:
                continue
            try:
                df = fetch_standings(year, rnd)
                if not df.empty:
                    frames.append(df)
                    existing_keys.add((year, rnd))
            except Exception as e:
                log.error(f"Failed standings {year} R{rnd}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    if "name" in result.columns:
        result["team"] = result["name"].apply(
            lambda x: normalize_team(x) if pd.notna(x) else x
        )
    result.to_csv(csv_path, index=False)
    log.info(f"Saved standings: {len(result)} rows to {csv_path}")
    return result


def fetch_all_tips(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year in tqdm(range(start_year, end_year + 1), desc="Squiggle tips"):
        df = fetch_tips(year)
        if not df.empty:
            frames.append(df)
            log.info(f"Squiggle tips {year}: {len(df)} tips")
    if not frames:
        return pd.DataFrame()
    all_tips = pd.concat(frames, ignore_index=True)
    all_tips.to_csv(DATA_RAW / "squiggle_tips.csv", index=False)

    # Aggregate: mean confidence and margin per game
    all_tips["confidence"] = pd.to_numeric(all_tips["confidence"], errors="coerce")
    all_tips["hconfidence"] = pd.to_numeric(all_tips["hconfidence"], errors="coerce")
    all_tips["margin"] = pd.to_numeric(all_tips["margin"], errors="coerce")
    all_tips["hmargin"] = pd.to_numeric(all_tips["hmargin"], errors="coerce")

    agg = all_tips.groupby("game_id").agg(
        mean_confidence=("confidence", "mean"),
        mean_hconfidence=("hconfidence", "mean"),
        mean_margin=("margin", "mean"),
        mean_hmargin=("hmargin", "mean"),
        n_models=("source", "count"),
        n_correct=("correct", "sum"),
    ).reset_index()
    agg.to_csv(DATA_PROCESSED / "squiggle_tips_agg.csv", index=False)
    log.info(f"Saved aggregated tips ({len(agg)} games) to squiggle_tips_agg.csv")
    return agg
