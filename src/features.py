"""
Feature engineering for AFL match prediction.

Every feature is computed using ONLY data available BEFORE the match (no leakage).
Features are added to the master dataset as new columns.
"""
import numpy as np
import pandas as pd

from src.players import load_player_data, compute_player_features
from src.utils import setup_logging

log = setup_logging()

# ──────────────────────────────────────────────────────────────────────
# Team home states (for venue-specific features)
# ──────────────────────────────────────────────────────────────────────
TEAM_HOME_STATE = {
    "Adelaide": "SA", "Port Adelaide": "SA",
    "Brisbane Lions": "QLD", "Gold Coast": "QLD",
    "Carlton": "VIC", "Collingwood": "VIC", "Essendon": "VIC",
    "Geelong": "VIC", "Hawthorn": "VIC", "Melbourne": "VIC",
    "North Melbourne": "VIC", "Richmond": "VIC", "St Kilda": "VIC",
    "Western Bulldogs": "VIC",
    "Fremantle": "WA", "West Coast": "WA",
    "Sydney": "NSW", "Greater Western Sydney": "NSW",
}

VENUE_STATE = {
    "M.C.G.": "VIC", "Docklands": "VIC", "Marvel Stadium": "VIC",
    "Kardinia Park": "VIC", "GMHBA Stadium": "VIC",
    "Adelaide Oval": "SA", "Football Park": "SA",
    "Gabba": "QLD", "Carrara": "QLD",
    "Perth Stadium": "WA", "Subiaco": "WA",
    "S.C.G.": "NSW", "Sydney Showground": "NSW", "Stadium Australia": "NSW",
    "Bellerive Oval": "TAS", "York Park": "TAS", "Blundstone Arena": "TAS",
    "University of Tasmania Stadium": "TAS",
    "TIO Stadium": "NT", "Marrara Oval": "NT",
    "Manuka Oval": "ACT", "Canberra Oval": "ACT",
    "Cazalys Stadium": "QLD", "Riverway Stadium": "QLD",
    "Jiangwan Stadium": "INTL", "Wellington": "INTL",
    "Traeger Park": "NT", "TIO Traeger Park": "NT",
    "Eureka Stadium": "VIC", "Mars Stadium": "VIC",
    "Norwood Oval": "SA",
}

# Home ground mapping: team -> list of primary home venues
TEAM_HOME_VENUES = {
    "Adelaide": ["Adelaide Oval"],
    "Brisbane Lions": ["Gabba"],
    "Carlton": ["M.C.G.", "Docklands", "Marvel Stadium"],
    "Collingwood": ["M.C.G.", "Docklands", "Marvel Stadium"],
    "Essendon": ["M.C.G.", "Docklands", "Marvel Stadium"],
    "Fremantle": ["Perth Stadium", "Subiaco"],
    "Geelong": ["Kardinia Park", "GMHBA Stadium", "M.C.G."],
    "Gold Coast": ["Carrara"],
    "Greater Western Sydney": ["Sydney Showground"],
    "Hawthorn": ["M.C.G.", "Docklands", "Marvel Stadium", "York Park", "University of Tasmania Stadium"],
    "Melbourne": ["M.C.G.", "Docklands", "Marvel Stadium"],
    "North Melbourne": ["M.C.G.", "Docklands", "Marvel Stadium", "Bellerive Oval", "Blundstone Arena"],
    "Port Adelaide": ["Adelaide Oval"],
    "Richmond": ["M.C.G.", "Docklands", "Marvel Stadium"],
    "St Kilda": ["M.C.G.", "Docklands", "Marvel Stadium"],
    "Sydney": ["S.C.G."],
    "West Coast": ["Perth Stadium", "Subiaco"],
    "Western Bulldogs": ["M.C.G.", "Docklands", "Marvel Stadium"],
}


# ──────────────────────────────────────────────────────────────────────
# Team home-city coordinates (for travel distance features)
# ──────────────────────────────────────────────────────────────────────
TEAM_HOME_COORDS = {
    "Adelaide":              (-34.9285, 138.6007),  # Adelaide CBD
    "Port Adelaide":         (-34.9285, 138.6007),
    "Brisbane Lions":        (-27.4698, 153.0251),  # Brisbane CBD
    "Gold Coast":            (-28.0167, 153.4000),
    "Carlton":               (-37.8136, 144.9631),  # Melbourne CBD
    "Collingwood":           (-37.8136, 144.9631),
    "Essendon":              (-37.8136, 144.9631),
    "Geelong":               (-38.1499, 144.3617),  # Geelong
    "Hawthorn":              (-37.8136, 144.9631),
    "Melbourne":             (-37.8136, 144.9631),
    "North Melbourne":       (-37.8136, 144.9631),
    "Richmond":              (-37.8136, 144.9631),
    "St Kilda":              (-37.8136, 144.9631),
    "Western Bulldogs":      (-37.8136, 144.9631),
    "Fremantle":             (-31.9505, 115.8605),  # Perth CBD
    "West Coast":            (-31.9505, 115.8605),
    "Sydney":                (-33.8688, 151.2093),  # Sydney CBD
    "Greater Western Sydney": (-33.8688, 151.2093),
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth (km)."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_features(df: pd.DataFrame, selection_lineups: dict | None = None) -> pd.DataFrame:
    """Add all features to the master dataset. df must be sorted by date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log.info("Building features...")

    # Elo ratings
    df = _add_elo_ratings(df)

    # Elo velocity (momentum over last 5 games)
    df = _add_elo_velocity(df, window=5)

    # Rolling form features
    df = _add_rolling_form(df, windows=[5, 10])

    # Form consistency (margin volatility + streak)
    df = _add_form_consistency(df, window=5)

    # Head-to-head record
    df = _add_head_to_head(df, n_meetings=10)

    # Venue features
    df = _add_venue_features(df)

    # Travel distance (continuous km, replaces binary interstate)
    df = _add_travel_distance(df)

    # Rest days
    df = _add_rest_days(df)

    # Season context
    df = _add_season_context(df)

    # Ladder features (from Squiggle standings)
    df = _add_ladder_features(df)

    # Rolling detailed stats (Footywire) — only if data is available
    df = _add_rolling_detailed_stats(df, window=5)

    # Scoring efficiency (I50 conversion, kick accuracy)
    df = _add_scoring_efficiency(df, window=5)

    # Player availability / lineup features
    df = _add_player_features(df, selection_lineups=selection_lineups)

    # Weather features (only if cache exists — fetching is a separate step)
    df = _add_weather_features(df)

    log.info(f"Features complete. Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────
# ELO RATINGS
# ──────────────────────────────────────────────────────────────────────
def _add_elo_ratings(df: pd.DataFrame, k: float = 30.0, home_adv: float = 30.0,
                     mean_revert: float = 0.1) -> pd.DataFrame:
    """Maintain running Elo ratings for every team, updated after each game.

    - K-factor of 30 (responsive to results)
    - 30-point home advantage
    - 10% mean reversion at season start
    - Margin-of-victory multiplier to weight blowouts less
    """
    elo = {}  # team -> current elo
    BASE_ELO = 1500.0

    home_elos = []
    away_elos = []
    prev_year = None

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]

        # Season start: mean-revert
        if row["year"] != prev_year:
            for team in elo:
                elo[team] = elo[team] * (1 - mean_revert) + BASE_ELO * mean_revert
            prev_year = row["year"]

        h_elo = elo.get(ht, BASE_ELO)
        a_elo = elo.get(at, BASE_ELO)

        # Record pre-match Elo (this is what the model sees)
        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # Update after match
        h_score = row["home_score"]
        a_score = row["away_score"]
        if pd.isna(h_score) or pd.isna(a_score):
            continue

        # Expected score (with home advantage)
        exp_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo - home_adv) / 400.0))
        actual_h = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)

        # Margin-of-victory multiplier (dampens effect of blowouts)
        margin = abs(h_score - a_score)
        mov_mult = np.log(max(margin, 1) + 1) / np.log(10 + 1)  # normalised

        # Update
        shift = k * mov_mult * (actual_h - exp_h)
        elo[ht] = h_elo + shift
        elo[at] = a_elo - shift

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["elo_expected"] = 1.0 / (1.0 + 10.0 ** ((df["away_elo"] - df["home_elo"] - home_adv) / 400.0))

    log.info(f"Elo ratings computed. Range: {df['home_elo'].min():.0f} - {df['home_elo'].max():.0f}")
    return df


# ──────────────────────────────────────────────────────────────────────
# ELO VELOCITY (momentum)
# ──────────────────────────────────────────────────────────────────────
def _add_elo_velocity(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Elo change over the last N completed games for each team.

    Reads the home_elo / away_elo columns already computed by _add_elo_ratings.
    For each team, tracks the Elo value at the time of each match, then
    computes velocity = current_elo - elo_N_games_ago.
    """
    from collections import deque

    # Reconstruct per-team Elo history by iterating chronologically
    team_elo_history: dict[str, deque] = {}  # team -> deque of pre-match Elo values

    home_vel = np.full(len(df), np.nan)
    away_vel = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        h_elo = row.get("home_elo", np.nan)
        a_elo = row.get("away_elo", np.nan)

        # Compute velocity BEFORE appending current Elo
        if ht in team_elo_history and len(team_elo_history[ht]) >= window:
            home_vel[i] = h_elo - team_elo_history[ht][0]
        if at in team_elo_history and len(team_elo_history[at]) >= window:
            away_vel[i] = a_elo - team_elo_history[at][0]

        # Only record history for completed matches (where Elo actually updated)
        if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
            if ht not in team_elo_history:
                team_elo_history[ht] = deque(maxlen=window)
            team_elo_history[ht].append(h_elo)

            if at not in team_elo_history:
                team_elo_history[at] = deque(maxlen=window)
            team_elo_history[at].append(a_elo)

    df["home_elo_velocity"] = home_vel
    df["away_elo_velocity"] = away_vel
    df["elo_velocity_diff"] = home_vel - away_vel

    non_null = np.count_nonzero(~np.isnan(home_vel))
    log.info(f"Elo velocity added (window={window}, {non_null}/{len(df)} populated)")
    return df


# ──────────────────────────────────────────────────────────────────────
# ROLLING FORM
# ──────────────────────────────────────────────────────────────────────
def _add_rolling_form(df: pd.DataFrame, windows: list[int] = [5, 10]) -> pd.DataFrame:
    """Rolling averages of scoring margin, win rate, and points for/against.

    For each team, computed over their last N matches (not just home or away).
    """
    # Build a flat list of (date, team, score_for, score_against, win) for every team-match
    # Skip rows with NaN scores (future/unplayed matches) to avoid contamination
    records = []
    for _, row in df.iterrows():
        date = row["date"]
        h_score, a_score = row["home_score"], row["away_score"]
        if pd.isna(h_score) or pd.isna(a_score):
            continue
        records.append((date, row["home_team"], h_score, a_score,
                        1 if h_score > a_score else 0))
        records.append((date, row["away_team"], a_score, h_score,
                        1 if a_score > h_score else 0))

    team_history: dict[str, list] = {}
    for date, team, sf, sa, win in records:
        if team not in team_history:
            team_history[team] = []
        team_history[team].append({"date": date, "score_for": sf, "score_against": sa,
                                    "win": win, "margin": sf - sa})

    # Now compute rolling features for each match
    # We need to track how many games each team has played before this match
    team_game_idx: dict[str, int] = {}

    for w in windows:
        df[f"home_win_rate_{w}"] = np.nan
        df[f"away_win_rate_{w}"] = np.nan
        df[f"home_avg_margin_{w}"] = np.nan
        df[f"away_avg_margin_{w}"] = np.nan
        df[f"home_avg_score_for_{w}"] = np.nan
        df[f"away_avg_score_for_{w}"] = np.nan
        df[f"home_avg_score_against_{w}"] = np.nan
        df[f"away_avg_score_against_{w}"] = np.nan

    # Reset game counters
    team_game_idx = {t: 0 for t in team_history}

    for idx, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        h_idx = team_game_idx.get(ht, 0)
        a_idx = team_game_idx.get(at, 0)

        for w in windows:
            # Home team rolling stats (last w games BEFORE this one)
            if h_idx >= 1:
                h_hist = team_history[ht][max(0, h_idx - w):h_idx]
                df.at[idx, f"home_win_rate_{w}"] = np.mean([g["win"] for g in h_hist])
                df.at[idx, f"home_avg_margin_{w}"] = np.mean([g["margin"] for g in h_hist])
                df.at[idx, f"home_avg_score_for_{w}"] = np.mean([g["score_for"] for g in h_hist])
                df.at[idx, f"home_avg_score_against_{w}"] = np.mean([g["score_against"] for g in h_hist])

            # Away team rolling stats
            if a_idx >= 1:
                a_hist = team_history[at][max(0, a_idx - w):a_idx]
                df.at[idx, f"away_win_rate_{w}"] = np.mean([g["win"] for g in a_hist])
                df.at[idx, f"away_avg_margin_{w}"] = np.mean([g["margin"] for g in a_hist])
                df.at[idx, f"away_avg_score_for_{w}"] = np.mean([g["score_for"] for g in a_hist])
                df.at[idx, f"away_avg_score_against_{w}"] = np.mean([g["score_against"] for g in a_hist])

        # Advance game indices only for completed matches
        if not (pd.isna(row["home_score"]) or pd.isna(row["away_score"])):
            team_game_idx[ht] = h_idx + 1
            team_game_idx[at] = a_idx + 1

    # Compute differentials
    for w in windows:
        df[f"win_rate_diff_{w}"] = df[f"home_win_rate_{w}"] - df[f"away_win_rate_{w}"]
        df[f"avg_margin_diff_{w}"] = df[f"home_avg_margin_{w}"] - df[f"away_avg_margin_{w}"]
        df[f"score_for_diff_{w}"] = df[f"home_avg_score_for_{w}"] - df[f"away_avg_score_for_{w}"]
        df[f"score_against_diff_{w}"] = df[f"home_avg_score_against_{w}"] - df[f"away_avg_score_against_{w}"]

    log.info("Rolling form features added")
    return df


# ──────────────────────────────────────────────────────────────────────
# FORM CONSISTENCY (volatility + streak)
# ──────────────────────────────────────────────────────────────────────
def _add_form_consistency(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Margin volatility (std over last N games) and current win/loss streak.

    Streak is positive for consecutive wins, negative for consecutive losses.
    A draw (margin == 0) resets the streak to 0.
    """
    team_margins: dict[str, list[float]] = {}   # team -> list of past margins
    team_outcomes: dict[str, list[int]] = {}    # team -> list of past +1/-1/0

    home_vol = np.full(len(df), np.nan)
    away_vol = np.full(len(df), np.nan)
    home_streak = np.full(len(df), np.nan)
    away_streak = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]

        # Read BEFORE updating
        if ht in team_margins and len(team_margins[ht]) >= 2:
            recent = team_margins[ht][-window:]
            home_vol[i] = float(np.std(recent, ddof=1))
        if at in team_margins and len(team_margins[at]) >= 2:
            recent = team_margins[at][-window:]
            away_vol[i] = float(np.std(recent, ddof=1))

        # Streak: walk backward from most recent outcome
        if ht in team_outcomes and team_outcomes[ht]:
            streak = 0
            outcomes = team_outcomes[ht]
            last = outcomes[-1]
            if last != 0:
                for o in reversed(outcomes):
                    if o == last:
                        streak += 1
                    else:
                        break
                streak *= last  # negative for losses
            home_streak[i] = streak

        if at in team_outcomes and team_outcomes[at]:
            streak = 0
            outcomes = team_outcomes[at]
            last = outcomes[-1]
            if last != 0:
                for o in reversed(outcomes):
                    if o == last:
                        streak += 1
                    else:
                        break
                streak *= last
            away_streak[i] = streak

        # Record this match (only completed)
        h_score, a_score = row["home_score"], row["away_score"]
        if pd.notna(h_score) and pd.notna(a_score):
            h_margin = float(h_score - a_score)
            team_margins.setdefault(ht, []).append(h_margin)
            team_margins.setdefault(at, []).append(-h_margin)

            if h_margin > 0:
                team_outcomes.setdefault(ht, []).append(1)
                team_outcomes.setdefault(at, []).append(-1)
            elif h_margin < 0:
                team_outcomes.setdefault(ht, []).append(-1)
                team_outcomes.setdefault(at, []).append(1)
            else:
                team_outcomes.setdefault(ht, []).append(0)
                team_outcomes.setdefault(at, []).append(0)

    df["home_margin_volatility"] = home_vol
    df["away_margin_volatility"] = away_vol
    df["margin_volatility_diff_5"] = home_vol - away_vol
    df["home_form_streak"] = home_streak
    df["away_form_streak"] = away_streak
    df["form_streak_diff"] = home_streak - away_streak

    non_null = np.count_nonzero(~np.isnan(home_vol))
    log.info(f"Form consistency added (window={window}, {non_null}/{len(df)} volatility populated)")
    return df


# ──────────────────────────────────────────────────────────────────────
# HEAD TO HEAD
# ──────────────────────────────────────────────────────────────────────
def _add_head_to_head(df: pd.DataFrame, n_meetings: int = 10) -> pd.DataFrame:
    """Head-to-head record between the two teams over their last N meetings."""
    h2h_history: dict[tuple, list] = {}  # (teamA, teamB) -> list of (date, winner)

    h2h_home_wins = []
    h2h_total = []

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        key = tuple(sorted([ht, at]))

        past = h2h_history.get(key, [])
        recent = past[-n_meetings:] if past else []

        if recent:
            home_wins = sum(1 for _, w in recent if w == ht)
            h2h_home_wins.append(home_wins / len(recent))
            h2h_total.append(len(recent))
        else:
            h2h_home_wins.append(np.nan)
            h2h_total.append(0)

        # Record this match result (skip future/unplayed matches)
        winner = row["winner"] if pd.notna(row.get("winner")) else None
        if winner is not None:
            h2h_history.setdefault(key, []).append((row["date"], winner))

    df["h2h_home_win_rate"] = h2h_home_wins
    df["h2h_meetings"] = h2h_total

    log.info("Head-to-head features added")
    return df


# ──────────────────────────────────────────────────────────────────────
# VENUE FEATURES
# ──────────────────────────────────────────────────────────────────────
def _add_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Venue-related features: interstate travel, home ground advantage, venue familiarity."""

    home_is_interstate = []
    away_is_interstate = []
    home_at_home_ground = []
    away_at_home_ground = []
    home_venue_win_rate = []
    away_venue_win_rate = []

    # Track team-venue records
    venue_records: dict[tuple, list] = {}  # (team, venue) -> list of wins (1/0)

    for _, row in df.iterrows():
        ht, at, venue = row["home_team"], row["away_team"], row["venue"]
        venue_state = VENUE_STATE.get(venue, "UNK")
        h_state = TEAM_HOME_STATE.get(ht, "UNK")
        a_state = TEAM_HOME_STATE.get(at, "UNK")

        # Interstate flags
        home_is_interstate.append(1 if h_state != venue_state and venue_state != "UNK" else 0)
        away_is_interstate.append(1 if a_state != venue_state and venue_state != "UNK" else 0)

        # Home ground flags
        h_homes = TEAM_HOME_VENUES.get(ht, [])
        a_homes = TEAM_HOME_VENUES.get(at, [])
        home_at_home_ground.append(1 if venue in h_homes else 0)
        away_at_home_ground.append(1 if venue in a_homes else 0)

        # Venue win rate (pre-match)
        h_venue_key = (ht, venue)
        a_venue_key = (at, venue)
        h_vrec = venue_records.get(h_venue_key, [])
        a_vrec = venue_records.get(a_venue_key, [])
        home_venue_win_rate.append(np.mean(h_vrec) if h_vrec else np.nan)
        away_venue_win_rate.append(np.mean(a_vrec) if a_vrec else np.nan)

        # Update venue records only for completed matches
        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            h_win = 1 if row["home_score"] > row["away_score"] else 0
            venue_records.setdefault(h_venue_key, []).append(h_win)
            venue_records.setdefault(a_venue_key, []).append(1 - h_win)

    df["home_interstate"] = home_is_interstate
    df["away_interstate"] = away_is_interstate
    df["interstate_diff"] = df["away_interstate"] - df["home_interstate"]  # positive = away traveling more
    df["home_at_home_ground"] = home_at_home_ground
    df["away_at_home_ground"] = away_at_home_ground
    df["home_venue_win_rate"] = home_venue_win_rate
    df["away_venue_win_rate"] = away_venue_win_rate
    df["venue_win_rate_diff"] = df["home_venue_win_rate"] - df["away_venue_win_rate"]

    log.info("Venue features added")
    return df


# ──────────────────────────────────────────────────────────────────────
# TRAVEL DISTANCE
# ──────────────────────────────────────────────────────────────────────
def _add_travel_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Haversine distance (km) from each team's home city to the match venue.

    Uses VENUE_COORDINATES from src.weather and TEAM_HOME_COORDS defined above.
    Vectorised — no row iteration needed.
    """
    from src.weather import VENUE_COORDINATES

    home_km = np.full(len(df), np.nan)
    away_km = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        venue = row["venue"]
        ht, at = row["home_team"], row["away_team"]

        v_coords = VENUE_COORDINATES.get(venue)
        h_coords = TEAM_HOME_COORDS.get(ht)
        a_coords = TEAM_HOME_COORDS.get(at)

        if v_coords is not None and h_coords is not None:
            home_km[i] = _haversine_km(h_coords[0], h_coords[1],
                                        v_coords[0], v_coords[1])
        if v_coords is not None and a_coords is not None:
            away_km[i] = _haversine_km(a_coords[0], a_coords[1],
                                        v_coords[0], v_coords[1])

    df["home_travel_km"] = home_km
    df["away_travel_km"] = away_km
    df["travel_distance_diff"] = home_km - away_km

    non_null = np.count_nonzero(~np.isnan(home_km))
    log.info(f"Travel distance added ({non_null}/{len(df)} populated)")
    return df


# ──────────────────────────────────────────────────────────────────────
# REST DAYS
# ──────────────────────────────────────────────────────────────────────
def _add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's last match."""
    last_game: dict[str, pd.Timestamp] = {}

    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        date = row["date"]

        h_last = last_game.get(ht)
        a_last = last_game.get(at)

        home_rest.append((date - h_last).days if h_last else np.nan)
        away_rest.append((date - a_last).days if a_last else np.nan)

        # Only update last_game for completed matches
        if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
            last_game[ht] = date
            last_game[at] = date

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    # Cap rest days at 30 (off-season gaps are not meaningful)
    df["home_rest_days"] = df["home_rest_days"].clip(upper=30)
    df["away_rest_days"] = df["away_rest_days"].clip(upper=30)
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

    # Bye detection: rest > 11 days during the season (not off-season gaps)
    df["home_had_bye"] = ((df["home_rest_days"] > 11) & (df["home_rest_days"] <= 20)).astype(int)
    df["away_had_bye"] = ((df["away_rest_days"] > 11) & (df["away_rest_days"] <= 20)).astype(int)
    df["bye_advantage"] = df["home_had_bye"] - df["away_had_bye"]

    log.info("Rest day features added (with bye detection)")
    return df


# ──────────────────────────────────────────────────────────────────────
# SEASON CONTEXT
# ──────────────────────────────────────────────────────────────────────
def _add_season_context(df: pd.DataFrame) -> pd.DataFrame:
    """Round number, finals flag, season progress."""
    # Season progress (0 = start, 1 = end of home-and-away)
    max_round = df.groupby("year")["round"].transform("max")
    df["season_progress"] = df["round"] / max_round
    df["season_progress"] = df["season_progress"].clip(0, 1)

    log.info("Season context features added")
    return df


# ──────────────────────────────────────────────────────────────────────
# LADDER / STANDINGS FEATURES
# ──────────────────────────────────────────────────────────────────────
def _add_ladder_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-match ladder position from Squiggle standings.

    For each match in round R, uses standings from round R-1 to avoid leakage.
    For Round 1, uses end-of-previous-season standings.
    """
    from src.config import DATA_PROCESSED

    standings_path = DATA_PROCESSED / "squiggle_standings.csv"
    if not standings_path.exists():
        log.info("No standings data — skipping ladder features")
        return df

    standings = pd.read_csv(standings_path)
    if "team" not in standings.columns or "round" not in standings.columns:
        log.info("Standings missing required columns — skipping")
        return df

    # Build lookup: (year, round, team) -> {rank, percentage, wins, losses, played, pts}
    lookup: dict[tuple, dict] = {}
    for _, row in standings.iterrows():
        key = (int(row["year"]), int(row["round"]), row["team"])
        lookup[key] = {
            "rank": row.get("rank"),
            "percentage": row.get("percentage"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "played": row.get("played"),
            "pts": row.get("pts"),
            "for": row.get("for"),
            "against": row.get("against"),
        }

    # Rounds available per season (sorted) so we can forward-fill from the
    # latest completed round when R-1 hasn't been published yet (e.g. a live
    # upcoming match in R+N where only R-k standings exist).
    rounds_by_year: dict[int, list[int]] = {}
    for (yr, rnd, _) in lookup:
        rounds_by_year.setdefault(yr, []).append(rnd)
    for yr in rounds_by_year:
        rounds_by_year[yr] = sorted(set(rounds_by_year[yr]))

    last_round_by_year: dict[int, int] = {
        yr: rs[-1] for yr, rs in rounds_by_year.items() if rs
    }

    def _latest_round_at_or_before(year: int, target: int) -> int | None:
        """Largest round in `year` with standings where round <= target."""
        rounds = rounds_by_year.get(year)
        if not rounds:
            return None
        candidate = None
        for r in rounds:
            if r <= target:
                candidate = r
            else:
                break
        return candidate

    home_rank, away_rank = [], []
    home_pct, away_pct = [], []

    for _, row in df.iterrows():
        year = int(row["year"])
        rnd = int(row["round"]) if pd.notna(row.get("round")) else 0
        ht, at = row["home_team"], row["away_team"]

        # Prefer standings from round R-1 in the same season. If that round
        # hasn't been published yet (upcoming match with no R-1 standings),
        # walk back to the latest available round — a team's rank after
        # round N-k is still the correct prior for round N.
        prev_year = prev_round = None
        if rnd > 1:
            found = _latest_round_at_or_before(year, rnd - 1)
            if found is not None:
                prev_year, prev_round = year, found
        if prev_round is None:
            # Round 1 (or no in-season standings at all): use end of prior season
            prev_year = year - 1
            prev_round = last_round_by_year.get(prev_year, 0)

        h_data = lookup.get((prev_year, prev_round, ht), {})
        a_data = lookup.get((prev_year, prev_round, at), {})

        home_rank.append(h_data.get("rank"))
        away_rank.append(a_data.get("rank"))
        home_pct.append(h_data.get("percentage"))
        away_pct.append(a_data.get("percentage"))

    df["home_ladder_rank"] = pd.array(home_rank, dtype=pd.Int64Dtype())
    df["away_ladder_rank"] = pd.array(away_rank, dtype=pd.Int64Dtype())
    df["ladder_rank_diff"] = df["away_ladder_rank"] - df["home_ladder_rank"]  # positive = home ranked higher
    df["home_percentage"] = pd.array(home_pct, dtype="Float64")
    df["away_percentage"] = pd.array(away_pct, dtype="Float64")
    df["percentage_diff"] = df["home_percentage"] - df["away_percentage"]
    df["home_top4"] = (df["home_ladder_rank"] <= 4).astype("Int64")
    df["away_top4"] = (df["away_ladder_rank"] <= 4).astype("Int64")
    df["home_top8"] = (df["home_ladder_rank"] <= 8).astype("Int64")
    df["away_top8"] = (df["away_ladder_rank"] <= 8).astype("Int64")

    matched = df["home_ladder_rank"].notna().sum()
    log.info(f"Ladder features added ({matched}/{len(df)} matches with standings)")
    return df


# ──────────────────────────────────────────────────────────────────────
# ROLLING DETAILED STATS (from Footywire)
# ──────────────────────────────────────────────────────────────────────
DETAILED_STAT_COLS = ["D", "I50", "CL", "T", "HO", "CG", "R50", "M", "FF", "FA"]


def _add_rolling_detailed_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling averages of detailed team stats (disposals, I50, clearances, etc.).

    Only adds features if Footywire stat columns exist in the dataset.
    Computes rolling averages and differentials for each stat.
    """
    # Check which detailed stats are available
    available = []
    for stat in DETAILED_STAT_COLS:
        if f"home_{stat}" in df.columns and f"away_{stat}" in df.columns:
            available.append(stat)

    if not available:
        log.info("No Footywire stats available — skipping rolling detailed stats")
        return df

    log.info(f"Building rolling stats for: {available}")

    # Build per-team history of each stat
    team_stat_history: dict[str, dict[str, list]] = {}  # team -> stat -> [values]

    # Pre-init columns
    for stat in available:
        for prefix in ["home", "away"]:
            df[f"{prefix}_avg_{stat}_{window}"] = np.nan
        df[f"avg_{stat}_diff_{window}"] = np.nan

    team_game_count: dict[str, int] = {}

    for idx, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]

        # Read rolling averages BEFORE updating with this game's stats
        h_count = team_game_count.get(ht, 0)
        a_count = team_game_count.get(at, 0)

        for stat in available:
            if h_count >= 1 and ht in team_stat_history:
                h_vals = team_stat_history[ht].get(stat, [])
                recent = h_vals[max(0, len(h_vals) - window):]
                if recent:
                    df.at[idx, f"home_avg_{stat}_{window}"] = np.mean(recent)

            if a_count >= 1 and at in team_stat_history:
                a_vals = team_stat_history[at].get(stat, [])
                recent = a_vals[max(0, len(a_vals) - window):]
                if recent:
                    df.at[idx, f"away_avg_{stat}_{window}"] = np.mean(recent)

        # Now record this game's stats for future rolling calculations
        for stat in available:
            h_val = row.get(f"home_{stat}")
            a_val = row.get(f"away_{stat}")

            if pd.notna(h_val):
                team_stat_history.setdefault(ht, {}).setdefault(stat, []).append(float(h_val))
            if pd.notna(a_val):
                team_stat_history.setdefault(at, {}).setdefault(stat, []).append(float(a_val))

        team_game_count[ht] = h_count + 1
        team_game_count[at] = a_count + 1

    # Compute differentials
    for stat in available:
        df[f"avg_{stat}_diff_{window}"] = (
            df[f"home_avg_{stat}_{window}"] - df[f"away_avg_{stat}_{window}"]
        )

    log.info(f"Rolling detailed stats added ({len(available)} stats, window={window})")
    return df


# ──────────────────────────────────────────────────────────────────────
# SCORING EFFICIENCY (conversion rates)
# ──────────────────────────────────────────────────────────────────────
def _add_scoring_efficiency(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling I50 conversion rate and kicking accuracy over the last N games.

    - I50 conversion = Goals / Inside 50s (per game, then averaged)
    - Kick accuracy  = Goals / (Goals + Behinds) (per game, then averaged)

    Requires Footywire columns: home_G, home_B, home_I50 (and away_ equivalents).
    """
    if "home_G" not in df.columns or "home_I50" not in df.columns:
        log.info("Footywire goal/I50 columns not available — skipping scoring efficiency")
        return df

    # Per-team history of per-game ratios
    team_i50_conv: dict[str, list[float]] = {}
    team_kick_acc: dict[str, list[float]] = {}

    home_i50 = np.full(len(df), np.nan)
    away_i50 = np.full(len(df), np.nan)
    home_acc = np.full(len(df), np.nan)
    away_acc = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]

        # Read rolling averages BEFORE updating
        if ht in team_i50_conv and team_i50_conv[ht]:
            recent = team_i50_conv[ht][-window:]
            home_i50[i] = float(np.nanmean(recent))
        if at in team_i50_conv and team_i50_conv[at]:
            recent = team_i50_conv[at][-window:]
            away_i50[i] = float(np.nanmean(recent))

        if ht in team_kick_acc and team_kick_acc[ht]:
            recent = team_kick_acc[ht][-window:]
            home_acc[i] = float(np.nanmean(recent))
        if at in team_kick_acc and team_kick_acc[at]:
            recent = team_kick_acc[at][-window:]
            away_acc[i] = float(np.nanmean(recent))

        # Record this game's per-team ratios (only completed matches with stats)
        for prefix, team in [("home", ht), ("away", at)]:
            g = row.get(f"{prefix}_G")
            b = row.get(f"{prefix}_B")
            i50 = row.get(f"{prefix}_I50")

            if pd.notna(g) and pd.notna(i50) and i50 > 0:
                team_i50_conv.setdefault(team, []).append(float(g) / float(i50))
            elif pd.notna(g):
                team_i50_conv.setdefault(team, []).append(np.nan)

            if pd.notna(g) and pd.notna(b) and (g + b) > 0:
                team_kick_acc.setdefault(team, []).append(float(g) / float(g + b))
            elif pd.notna(g):
                team_kick_acc.setdefault(team, []).append(np.nan)

    df["home_i50_conversion"] = home_i50
    df["away_i50_conversion"] = away_i50
    df["i50_conversion_diff_5"] = home_i50 - away_i50
    df["home_kick_accuracy"] = home_acc
    df["away_kick_accuracy"] = away_acc
    df["kick_accuracy_diff_5"] = home_acc - away_acc

    non_null = np.count_nonzero(~np.isnan(home_i50))
    log.info(f"Scoring efficiency added (window={window}, {non_null}/{len(df)} populated)")
    return df


# ──────────────────────────────────────────────────────────────────────
# PLAYER AVAILABILITY / LINEUP FEATURES
# ──────────────────────────────────────────────────────────────────────
def _add_player_features(
    df: pd.DataFrame,
    selection_lineups: dict | None = None,
) -> pd.DataFrame:
    """Add player availability, lineup disruption, and quality features."""
    player_df = load_player_data()
    if player_df.empty:
        log.warning("No player data -- skipping player features")
        return df
    return compute_player_features(df, player_df, selection_lineups=selection_lineups)


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather data and add derived features.

    Raw: temp_c, rain_mm, wind_kmh, wind_gust_kmh, humidity
    Derived: is_wet (rain>1mm, not roofed), wind_strong (>25kmh), is_roofed
    CatBoost handles NaN natively so missing weather is fine.
    """
    # If weather columns already exist (from dataset_builder), skip re-merge
    if "rain_mm" in df.columns and df["rain_mm"].notna().any():
        log.info(f"Weather columns already present ({df['rain_mm'].notna().sum()} matches)")
        return df

    from src.weather import merge_weather, load_weather_cache

    cache = load_weather_cache()
    if cache.empty:
        log.info("No weather cache — skipping weather features")
        return df

    df = merge_weather(df)
    return df
