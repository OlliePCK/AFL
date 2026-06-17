import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import AFLTABLES_BASE_URL, DATA_RAW, DATA_PROCESSED
from src.utils import fetch_url, setup_logging
from src.team_mapping import normalize_team

log = setup_logging()

# Pattern: "Team Name" followed by quarter scores like "3.3  4.3  7.10 12.14  86"
_SCORE_LINE = re.compile(
    r"^(.+?)\s*"                       # Team name (greedy but stops at first unicode char or digit pattern)
    r"[\s\xa0]*(\d+)\.(\d+)"           # Q1: goals.behinds
    r"[\s\xa0]+(\d+)\.(\d+)"           # Q2
    r"[\s\xa0]+(\d+)\.(\d+)"           # Q3
    r"[\s\xa0]+(\d+)\.(\d+)"           # Q4
    r"[\s\xa0]+(\d+)"                  # Final score
)


def _parse_score_line(text: str) -> dict | None:
    """Parse a line like 'Sydney  3.3  4.3  7.10 12.14  86Thu 07-Mar...' """
    # Clean up unicode spaces
    text = text.replace("\xa0", " ").replace("\u2002", " ").strip()

    m = _SCORE_LINE.match(text)
    if not m:
        return None

    return {
        "team": m.group(1).strip().rstrip("\u2002\xa0 "),
        "q1_goals": int(m.group(2)), "q1_behinds": int(m.group(3)),
        "q2_goals": int(m.group(4)), "q2_behinds": int(m.group(5)),
        "q3_goals": int(m.group(6)), "q3_behinds": int(m.group(7)),
        "q4_goals": int(m.group(8)), "q4_behinds": int(m.group(9)),
        "final_score": int(m.group(10)),
    }


def scrape_season_page(year: int) -> list[dict]:
    """Scrape a single season's match results from AFL Tables."""
    url = f"{AFLTABLES_BASE_URL}seas/{year}.html"
    resp = fetch_url(url)
    soup = BeautifulSoup(resp.text, "lxml")

    matches = []
    current_round = None
    tables = soup.find_all("table")

    for table in tables:
        text = table.get_text()

        # Detect round header tables (short text with "Round X")
        round_match = re.search(
            r"(Round\s+\d+|Qualifying Final|Elimination Final|"
            r"Semi Final|Preliminary Final|Grand Final|Finals Week \d+)",
            text
        )
        if round_match and len(text.strip()) < 200 and "Venue:" not in text:
            current_round = round_match.group(1).strip()
            continue

        # Try to parse as a match table (two score lines)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            continue

        # Join all text and try to find two score patterns
        full_text = table.get_text()

        # Extract attendance
        att_match = re.search(r"Att:\s*([\d,]+)", full_text)
        attendance = int(att_match.group(1).replace(",", "")) if att_match else None

        # Extract venue
        venue = None
        venue_match = re.search(r"Venue:\s*(.+?)(?:\n|$)", full_text)
        if venue_match:
            venue = venue_match.group(1).strip()

        # Extract date
        date_str = None
        date_match = re.search(r"(\w{3}\s+\d{2}-\w{3}-\d{4})", full_text)
        if date_match:
            date_str = date_match.group(1)

        # Parse the two team lines from table rows
        rows = table.find_all("tr")
        team_scores = []
        for row in rows:
            row_text = row.get_text()
            parsed = _parse_score_line(row_text)
            if parsed:
                team_scores.append(parsed)

        if len(team_scores) == 2:
            home = team_scores[0]
            away = team_scores[1]
            match = {
                "year": year,
                "round": current_round,
                "date": date_str,
                "venue": venue,
                "attendance": attendance,
                "home_team": home["team"],
                "away_team": away["team"],
                "home_q1": home["q1_goals"] * 6 + home["q1_behinds"],
                "home_q2": home["q2_goals"] * 6 + home["q2_behinds"],
                "home_q3": home["q3_goals"] * 6 + home["q3_behinds"],
                "home_q4": home["q4_goals"] * 6 + home["q4_behinds"],
                "home_score": home["final_score"],
                "away_q1": away["q1_goals"] * 6 + away["q1_behinds"],
                "away_q2": away["q2_goals"] * 6 + away["q2_behinds"],
                "away_q3": away["q3_goals"] * 6 + away["q3_behinds"],
                "away_q4": away["q4_goals"] * 6 + away["q4_behinds"],
                "away_score": away["final_score"],
            }
            matches.append(match)

    return matches


def scrape_all_seasons(start_year: int, end_year: int) -> pd.DataFrame:
    """Scrape match results for a range of seasons."""
    all_matches = []
    for year in tqdm(range(start_year, end_year + 1), desc="AFL Tables seasons"):
        try:
            matches = scrape_season_page(year)
            all_matches.extend(matches)
            log.info(f"AFL Tables {year}: {len(matches)} matches")
        except Exception as e:
            log.error(f"Failed to scrape AFL Tables {year}: {e}")

    if not all_matches:
        return pd.DataFrame()

    df = pd.DataFrame(all_matches)
    df.to_csv(DATA_RAW / "afltables_matches.csv", index=False)

    # Normalize team names
    clean = df.copy()
    for col in ["home_team", "away_team"]:
        clean[col] = clean[col].apply(lambda x: normalize_team(x) if pd.notna(x) else x)
    clean.to_csv(DATA_PROCESSED / "afltables_matches_clean.csv", index=False)
    log.info(f"Saved {len(clean)} AFL Tables matches")
    return clean
