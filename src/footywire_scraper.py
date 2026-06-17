import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path

from src.config import FOOTYWIRE_BASE_URL, DATA_RAW, DATA_PROCESSED
from src.utils import fetch_url, setup_logging
from src.team_mapping import normalize_team

log = setup_logging()

STAT_HEADERS = ["Player", "K", "HB", "D", "M", "G", "B", "T", "HO", "GA",
                "I50", "CL", "CG", "R50", "FF", "FA", "AF", "SC"]


def scrape_match_list(year: int) -> list[dict]:
    """Get all match IDs for a season from Footywire."""
    url = f"{FOOTYWIRE_BASE_URL}ft_match_list?year={year}"
    resp = fetch_url(url)
    soup = BeautifulSoup(resp.text, "lxml")

    seen = set()
    matches = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "ft_match_statistics" in href:
            mid_match = re.search(r"mid=(\d+)", href)
            if mid_match:
                mid = int(mid_match.group(1))
                if mid not in seen:
                    seen.add(mid)
                    matches.append({"mid": mid, "year": year})

    return matches


def scrape_match_stats(mid: int) -> pd.DataFrame:
    """Scrape player-level stats for a single match from Footywire."""
    url = f"{FOOTYWIRE_BASE_URL}ft_match_statistics?mid={mid}"
    resp = fetch_url(url)
    soup = BeautifulSoup(resp.text, "lxml")

    # Find the team name headers: "X Match Statistics (Sorted by Disposals)"
    team_names = []
    for tag in soup.find_all(string=re.compile(r"Match Statistics \(Sorted by Disposals\)")):
        m = re.match(r"(.+?)\s+Match Statistics", tag.strip())
        if m:
            team_names.append(m.group(1).strip())

    # Find player stats tables: tables where first header row is Player, K, HB, D...
    stats_tables = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue
        first_row_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        if len(first_row_cells) >= 10 and first_row_cells[0] == "Player" and first_row_cells[1] == "K":
            stats_tables.append(table)

    if len(stats_tables) < 2 or len(team_names) < 2:
        log.warning(f"mid={mid}: found {len(stats_tables)} stat tables, {len(team_names)} team names")
        return pd.DataFrame()

    all_rows = []
    for idx, table in enumerate(stats_tables[:2]):
        team = team_names[idx] if idx < len(team_names) else f"Team{idx}"
        rows = table.find_all("tr")
        headers = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) != len(headers):
                continue
            values = [c.get_text(strip=True) for c in cells]
            # Skip empty or summary rows
            if not values[0] or values[0].upper() in ("TOTALS", "TOTAL"):
                continue
            # Clean player name (remove sub arrows)
            values[0] = values[0].replace("\u2197", "").replace("\u2199", "").strip()
            row_dict = dict(zip(headers, values))
            row_dict["mid"] = mid
            row_dict["team"] = team
            all_rows.append(row_dict)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def aggregate_to_team_level(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level stats to team totals per match."""
    if player_df.empty:
        return pd.DataFrame()

    sum_cols = ["K", "HB", "D", "M", "G", "B", "T", "HO", "GA", "I50",
                "CL", "CG", "R50", "FF", "FA", "AF", "SC"]
    available = [c for c in sum_cols if c in player_df.columns]

    for col in available:
        player_df[col] = pd.to_numeric(player_df[col], errors="coerce")

    group_cols = ["mid", "team"]
    if "year" in player_df.columns:
        group_cols.append("year")
    agg = player_df.groupby(group_cols)[available].sum().reset_index()
    return agg


def scrape_all_seasons(start_year: int, end_year: int) -> pd.DataFrame:
    """Scrape all Footywire match stats for a range of years."""
    player_csv = DATA_RAW / "footywire_player_stats.csv"

    # Load existing data for resume support
    existing_mids = set()
    if player_csv.exists():
        existing = pd.read_csv(player_csv)
        existing_mids = set(existing["mid"].unique())
        log.info(f"Resuming: {len(existing_mids)} matches already scraped")

    # Collect all match IDs
    all_match_info = []
    for year in tqdm(range(start_year, end_year + 1), desc="Footywire match lists"):
        try:
            matches = scrape_match_list(year)
            all_match_info.extend(matches)
            log.info(f"Footywire {year}: {len(matches)} matches found")
        except Exception as e:
            log.error(f"Failed to get Footywire match list {year}: {e}")

    # Filter out already-scraped matches
    to_scrape = [m for m in all_match_info if m["mid"] not in existing_mids]
    log.info(f"Footywire: {len(to_scrape)} matches to scrape ({len(existing_mids)} already done)")

    # Scrape each match
    new_rows = []
    for match in tqdm(to_scrape, desc="Footywire match stats"):
        try:
            df = scrape_match_stats(match["mid"])
            if not df.empty:
                df["year"] = match["year"]
                new_rows.append(df)
                if len(new_rows) % 50 == 0:
                    _save_incremental(new_rows, player_csv, existing_mids)
                    new_rows = []
        except Exception as e:
            log.error(f"Failed to scrape Footywire mid={match['mid']}: {e}")

    if new_rows:
        _save_incremental(new_rows, player_csv, existing_mids)

    # Load full dataset and aggregate
    if player_csv.exists():
        full = pd.read_csv(player_csv)
        team_stats = aggregate_to_team_level(full)
        if "team" in team_stats.columns:
            team_stats["team"] = team_stats["team"].apply(
                lambda x: normalize_team(x) if pd.notna(x) else x
            )
        team_stats.to_csv(DATA_PROCESSED / "footywire_team_stats.csv", index=False)
        log.info(f"Saved {len(team_stats)} team-match stat rows")
        return team_stats

    return pd.DataFrame()


def _save_incremental(new_rows: list[pd.DataFrame], csv_path: Path, existing_mids: set):
    """Append new data to CSV file."""
    new_df = pd.concat(new_rows, ignore_index=True)
    if csv_path.exists():
        new_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(csv_path, index=False)
    new_mids = new_df["mid"].unique()
    existing_mids.update(new_mids)
    log.info(f"Saved {len(new_mids)} new matches incrementally")
