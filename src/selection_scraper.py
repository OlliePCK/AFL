"""Scrape pre-match team selections from Footywire."""

import re
import pandas as pd
from bs4 import BeautifulSoup, Tag

from src.config import FOOTYWIRE_BASE_URL, DATA_RAW
from src.team_mapping import normalize_team
from src.utils import fetch_url, setup_logging

log = setup_logging()

# Footywire team slug -> canonical name mapping
_SLUG_TO_TEAM = {
    "adelaide-crows": "Adelaide",
    "brisbane-lions": "Brisbane Lions",
    "carlton-blues": "Carlton",
    "collingwood-magpies": "Collingwood",
    "essendon-bombers": "Essendon",
    "fremantle-dockers": "Fremantle",
    "geelong-cats": "Geelong",
    "gold-coast-suns": "Gold Coast",
    "greater-western-sydney-giants": "Greater Western Sydney",
    "gws-giants": "Greater Western Sydney",
    "hawthorn-hawks": "Hawthorn",
    "melbourne-demons": "Melbourne",
    "north-melbourne-kangaroos": "North Melbourne",
    "port-adelaide-power": "Port Adelaide",
    "richmond-tigers": "Richmond",
    "st-kilda-saints": "St Kilda",
    "sydney-swans": "Sydney",
    "west-coast-eagles": "West Coast",
    "western-bulldogs": "Western Bulldogs",
}

# Position line labels to role mapping
_POSITION_ROLES = {
    "FB": "defender", "HB": "defender",
    "C": "midfielder",
    "HF": "forward", "FF": "forward",
    "Fol": "ruck",  # ruck + rovers
}


def _extract_player_name(a_tag: Tag) -> str | None:
    """Extract player display name from an <a> tag."""
    text = a_tag.get_text(strip=True)
    if not text:
        return None
    # Clean sub arrows and whitespace
    text = text.replace("\u2197", "").replace("\u2199", "").strip()
    return text if text else None


def _extract_team_from_href(a_tag: Tag) -> str | None:
    """Extract canonical team name from player href like 'pp-geelong-cats--name'."""
    href = a_tag.get("href", "")
    m = re.match(r"pp-(.+?)--", href)
    if m:
        slug = m.group(1)
        return _SLUG_TO_TEAM.get(slug)
    return None


def _parse_side_column(td: Tag) -> dict:
    """Parse a side column (left=home, right=away) for I/C, emergencies, ins, outs."""
    result = {"interchange": [], "emergencies": [], "ins": [], "outs": []}
    current_section = None

    table = td.find("table")
    if not table:
        return result

    for tr in table.find_all("tr"):
        # Check for section header
        b_tag = tr.find("b")
        if b_tag:
            header = b_tag.get_text(strip=True).lower()
            if "interchange" in header:
                current_section = "interchange"
            elif "emergenc" in header:
                current_section = "emergencies"
            elif header == "ins":
                current_section = "ins"
            elif header == "outs":
                current_section = "outs"
            continue

        # Check for player link
        a_tag = tr.find("a")
        if a_tag and current_section:
            name = _extract_player_name(a_tag)
            if name:
                result[current_section].append(name)

    return result


def _parse_position_grid(td: Tag) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Parse the centre column position grid.

    Returns (home_positions, away_positions) where each is
    {role: [player_names]} mapping.
    """
    home_positions: dict[str, list[str]] = {}
    away_positions: dict[str, list[str]] = {}

    div = td.find("div", class_="divseparator")
    if not div:
        return home_positions, away_positions

    table = div.find("table")
    if not table:
        return home_positions, away_positions

    # Home rows are lightcolor, away rows are darkcolor
    # Position mirroring: home FB pairs with away FF, home HB with away HF, etc.
    mirror = {"FB": "FF", "HB": "HF", "C": "C", "HF": "HB", "FF": "FB", "Fol": "Fol"}

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        # First cell has the position label
        pos_text = tds[0].get_text(strip=True)
        if not pos_text or pos_text not in ("FB", "HB", "C", "HF", "FF", "Fol"):
            continue

        is_home = "lightcolor" in (tr.get("class") or [])
        players = []
        for cell in tds[1:]:
            a_tag = cell.find("a")
            if a_tag:
                name = _extract_player_name(a_tag)
                if name:
                    players.append(name)

        role = _POSITION_ROLES.get(pos_text, "other")

        if is_home:
            home_positions.setdefault(role, []).extend(players)
        else:
            # Away position label is the mirror of what's shown
            away_positions.setdefault(role, []).extend(players)

    return home_positions, away_positions


def scrape_team_selections() -> list[dict]:
    """Scrape current round's team selections from Footywire.

    Returns a list of match dicts, each containing:
    - round, year
    - home_team, away_team, venue
    - home_players: set of all selected player names (22 + interchange)
    - away_players: set of all selected player names
    - home_positions: {role: [players]} for position grid players
    - away_positions: {role: [players]}
    - home_ins, home_outs, away_ins, away_outs: lists of player names
    """
    url = f"{FOOTYWIRE_BASE_URL}afl_team_selections"
    resp = fetch_url(url)
    soup = BeautifulSoup(resp.text, "lxml")

    # Extract round/year from h1
    h1 = soup.find("h1", class_="centertitle")
    year, round_num = None, None
    if h1:
        m = re.search(r"AFL\s+(\d{4})\s+Round\s+(\d+)", h1.get_text())
        if m:
            year = int(m.group(1))
            round_num = int(m.group(2))

    log.info(f"Scraping team selections: {year} Round {round_num}")

    # Find all match headers (td with class tbtitle containing an <a name="...">)
    matches = []
    match_headers = soup.find_all("td", class_="tbtitle", attrs={"height": "30"})

    for header_td in match_headers:
        a_tag = header_td.find("a", attrs={"name": True})
        if not a_tag:
            continue

        mid = a_tag.get("name")
        header_text = header_td.get_text(strip=True)

        # Parse "Team1 v Team2 (Venue)"
        m = re.match(r"(.+?)\s+v\s+(.+?)\s*\((.+?)\)", header_text)
        if not m:
            continue

        raw_home, raw_away, venue = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

        try:
            home_team = normalize_team(raw_home)
            away_team = normalize_team(raw_away)
        except ValueError:
            log.warning(f"Could not normalize teams: {raw_home} v {raw_away}")
            continue

        # Navigate to the match content row: the TR containing three TDs
        # Go up to the header's TR, then find the next TR with 3 TDs
        header_tr = header_td.find_parent("tr")
        if not header_tr:
            continue

        content_tr = None
        for sibling in header_tr.find_next_siblings("tr"):
            tds = sibling.find_all("td", recursive=False)
            if len(tds) == 3:
                content_tr = sibling
                break

        if not content_tr:
            log.warning(f"No content row found for {home_team} v {away_team}")
            continue

        left_td, centre_td, right_td = content_tr.find_all("td", recursive=False)

        # Parse side columns
        home_side = _parse_side_column(left_td)
        away_side = _parse_side_column(right_td)

        # Parse position grid
        home_positions, away_positions = _parse_position_grid(centre_td)

        # Build full player sets (position grid + interchange)
        home_players = set()
        for players in home_positions.values():
            home_players.update(players)
        home_players.update(home_side["interchange"])

        away_players = set()
        for players in away_positions.values():
            away_players.update(players)
        away_players.update(away_side["interchange"])

        match_data = {
            "mid": mid,
            "year": year,
            "round": round_num,
            "venue": venue,
            "home_team": home_team,
            "away_team": away_team,
            "home_players": home_players,
            "away_players": away_players,
            "home_positions": home_positions,
            "away_positions": away_positions,
            "home_ins": home_side["ins"],
            "home_outs": home_side["outs"],
            "away_ins": away_side["ins"],
            "away_outs": away_side["outs"],
            "home_emergencies": home_side["emergencies"],
            "away_emergencies": away_side["emergencies"],
        }
        matches.append(match_data)

    log.info(f"Scraped {len(matches)} match selections")

    # Save to CSV for reference
    if matches:
        rows = []
        for match in matches:
            for team_prefix in ["home", "away"]:
                team = match[f"{team_prefix}_team"]
                for player in sorted(match[f"{team_prefix}_players"]):
                    rows.append({
                        "year": match["year"],
                        "round": match["round"],
                        "team": team,
                        "player": player,
                        "type": "selected",
                    })
                for player in match[f"{team_prefix}_emergencies"]:
                    rows.append({
                        "year": match["year"],
                        "round": match["round"],
                        "team": team,
                        "player": player,
                        "type": "emergency",
                    })
        sel_df = pd.DataFrame(rows)
        sel_path = DATA_RAW / "team_selections.csv"
        sel_df.to_csv(sel_path, index=False)
        log.info(f"Saved {len(sel_df)} selection rows to {sel_path}")

    return matches
