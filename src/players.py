"""Player-level feature engineering: ratings, role inference, lineup features."""

import logging
import numpy as np
import pandas as pd

from src.config import DATA_RAW
from src.team_mapping import normalize_team

log = logging.getLogger(__name__)

# --- Constants ---

RATING_ALPHA = 0.3  # EMA decay: ~5-game effective window

# Composite player score weights (approximates SuperCoach-style scoring)
SCORE_WEIGHTS = {
    "D": 1.0, "G": 6.0, "M": 1.5, "T": 2.0, "HO": 1.5,
    "I50": 2.0, "CL": 2.0, "CG": 1.5, "R50": 1.5, "FF": 1.0, "FA": -1.0,
}

# Role inference thresholds (applied to EMA-smoothed per-stat averages)
ROLE_THRESHOLDS = {
    "ruck": {"HO": 15},
    "key_forward": {"G": 1.5, "I50": 3},
    "midfielder": {"CL": 3, "D": 25},  # CL >= 3 OR D >= 25
    "defender": {"R50": 3},
}


def load_player_data() -> pd.DataFrame:
    """Load and normalize the raw Footywire player stats."""
    path = DATA_RAW / "footywire_player_stats.csv"
    if not path.exists():
        log.warning("No player stats file found at %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["team"] = df["team"].apply(normalize_team)
    return df


def build_match_lineups(player_df: pd.DataFrame) -> dict[int, dict[str, set[str]]]:
    """Build {mid: {team: set(player_names)}} from player stats."""
    lineups: dict[int, dict[str, set[str]]] = {}
    for (mid, team), group in player_df.groupby(["mid", "team"]):
        lineups.setdefault(int(mid), {})[team] = set(group["Player"].tolist())
    return lineups


def build_abbrev_to_full_map(player_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Build mapping from (abbreviated_name, team) -> full_name.

    Abbreviated names use 'F Surname' format (from Footywire selections).
    Full names use 'First Surname' format (from Footywire match stats).
    Maps using first initial + last name within each team.
    """
    name_map: dict[tuple[str, str], str] = {}

    # Use most recent appearance per (player, team) for recency
    latest = player_df.sort_values("mid").drop_duplicates(
        subset=["Player", "team"], keep="last"
    )

    for _, row in latest.iterrows():
        full_name = row["Player"]
        team = row["team"]
        parts = full_name.split()
        if len(parts) < 2:
            continue
        first_initial = parts[0][0]
        surname = " ".join(parts[1:])  # Handle multi-word surnames
        abbrev = f"{first_initial} {surname}"
        # On collision, keep the most recent (already sorted by mid)
        name_map[(abbrev, team)] = full_name

    return name_map


def build_latest_team_lineups(player_df: pd.DataFrame) -> dict[str, set[str]]:
    """Build {team: latest known lineup} from the most recent player-stats match."""
    latest_lineups: dict[str, set[str]] = {}
    for team, group in player_df.groupby("team"):
        mids = pd.to_numeric(group["mid"], errors="coerce").dropna()
        if mids.empty:
            continue
        last_mid = int(mids.max())
        latest_players = group[group["mid"] == last_mid]["Player"].dropna().tolist()
        if latest_players:
            latest_lineups[team] = set(latest_players)
    return latest_lineups


def resolve_selection_names(
    selection_lineups: dict[tuple, dict[str, set[str]]],
    player_df: pd.DataFrame,
) -> dict[tuple, dict[str, set[str]]]:
    """Convert abbreviated selection names to full names matching historical data.

    Returns a new dict with the same structure but full player names where possible.
    Unmatched names are kept as-is (they'll just won't match historical ratings).
    """
    abbrev_map = build_abbrev_to_full_map(player_df)
    resolved: dict[tuple, dict[str, set[str]]] = {}

    for match_key, team_lineups in selection_lineups.items():
        resolved_match: dict[str, set[str]] = {}
        for team, players in team_lineups.items():
            resolved_players = set()
            for abbrev_name in players:
                full = abbrev_map.get((abbrev_name, team))
                if full:
                    resolved_players.add(full)
                else:
                    resolved_players.add(abbrev_name)
            resolved_match[team] = resolved_players
        resolved[match_key] = resolved_match

    return resolved


def _compute_player_score(row: pd.Series) -> float:
    """Compute composite score for a single player-match row."""
    score = 0.0
    for stat, weight in SCORE_WEIGHTS.items():
        val = row.get(stat, 0)
        if pd.notna(val):
            score += float(val) * weight
    return score


def _classify_role(stat_avgs: dict[str, float]) -> str:
    """Infer player role from EMA-smoothed stat averages."""
    if stat_avgs.get("HO", 0) >= ROLE_THRESHOLDS["ruck"]["HO"]:
        return "ruck"
    if (stat_avgs.get("G", 0) >= ROLE_THRESHOLDS["key_forward"]["G"]
            and stat_avgs.get("I50", 0) >= ROLE_THRESHOLDS["key_forward"]["I50"]):
        return "forward"
    if (stat_avgs.get("CL", 0) >= ROLE_THRESHOLDS["midfielder"]["CL"]
            or stat_avgs.get("D", 0) >= ROLE_THRESHOLDS["midfielder"]["D"]):
        return "midfielder"
    if stat_avgs.get("R50", 0) >= ROLE_THRESHOLDS["defender"]["R50"]:
        return "defender"
    # Check for general forward (high I50 but not midfielder)
    if stat_avgs.get("I50", 0) >= 3:
        return "forward"
    return "other"


def compute_player_features(
    df: pd.DataFrame,
    player_df: pd.DataFrame,
    selection_lineups: dict[tuple, dict[str, set[str]]] | None = None,
) -> pd.DataFrame:
    """Add player availability and lineup features to the match DataFrame.

    Requires 'mid' column in df to link matches to player data.
    Single chronological pass — no future leakage.

    selection_lineups: optional {key: {team: set(players)}} for upcoming matches
        without mid/player stats. Keys may be either
        (year, round, home_team, away_team) or the legacy
        (home_team, away_team) tuple.
    """
    df = df.copy()

    if "mid" not in df.columns:
        log.warning("No 'mid' column in dataset -- skipping player features")
        return df

    if player_df.empty:
        log.warning("No player data available -- skipping player features")
        return df

    if selection_lineups is None:
        selection_lineups = {}

    # Build lineups lookup
    lineups = build_match_lineups(player_df)

    # Pre-compute per-player per-match scores
    player_df = player_df.copy()
    player_df["_score"] = player_df.apply(_compute_player_score, axis=1)

    # Build {mid: {team: {player: score}}} for quick lookup
    match_player_scores: dict[int, dict[str, dict[str, float]]] = {}
    for (mid, team), group in player_df.groupby(["mid", "team"]):
        scores = dict(zip(group["Player"], group["_score"]))
        match_player_scores.setdefault(int(mid), {})[team] = scores

    # Pre-compute per-player per-match stat vectors (for role inference)
    role_stats = ["D", "G", "M", "T", "HO", "I50", "CL", "CG", "R50"]
    match_player_stats: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    for (mid, team), group in player_df.groupby(["mid", "team"]):
        pstats: dict[str, dict[str, float]] = {}
        for _, row in group.iterrows():
            pstats[row["Player"]] = {s: float(row[s]) if pd.notna(row.get(s)) else 0.0
                                     for s in role_stats}
        match_player_stats.setdefault(int(mid), {})[team] = pstats

    # Initialize all output columns
    feature_cols = [
        # Stage 1: lineup disruption
        "home_lineup_changes", "away_lineup_changes",
        "lineup_changes_diff", "lineup_continuity_diff",
        "home_ruck_missing", "away_ruck_missing",
        # Stage 2: quality-weighted absences
        "missing_rating_diff", "net_quality_change_diff",
        "missing_mid_rating_diff", "missing_fwd_rating_diff",
        "missing_def_rating_diff",
        # Stage 3: selection matchup
        "selected_total_rating_diff",
        "selected_mid_rating_diff", "selected_fwd_rating_diff",
    ]
    for col in feature_cols:
        df[col] = np.nan

    # Rolling state (maintained across the chronological pass)
    last_lineup: dict[str, set[str]] = {}      # team -> set of players
    player_ratings: dict[tuple[str, str], float] = {}  # (player, team) -> EMA rating
    player_stat_emas: dict[tuple[str, str], dict[str, float]] = {}  # (player, team) -> EMA stats

    log.info("Computing player features (single-pass)...")

    for idx, row in df.iterrows():
        mid = row.get("mid")
        ht, at = row["home_team"], row["away_team"]
        h_score, a_score = row.get("home_score"), row.get("away_score")
        is_completed = pd.notna(h_score) and pd.notna(a_score)

        # Try mid-based lookup first, then fall back to selection_lineups
        h_lineup, a_lineup = set(), set()
        if pd.notna(mid):
            mid = int(mid)
            if mid in lineups:
                h_lineup = lineups[mid].get(ht, set())
                a_lineup = lineups[mid].get(at, set())

        if not h_lineup or not a_lineup:
            # Check selection_lineups (for upcoming matches)
            lookup_keys = []
            year = pd.to_numeric(pd.Series([row.get("year")]), errors="coerce").iloc[0]
            round_num = pd.to_numeric(pd.Series([row.get("round")]), errors="coerce").iloc[0]
            if pd.notna(year) and pd.notna(round_num):
                lookup_keys.append((int(year), int(round_num), ht, at))
            lookup_keys.append((ht, at))

            for key in lookup_keys:
                sel = selection_lineups.get(key)
                if sel:
                    h_lineup = sel.get(ht, set())
                    a_lineup = sel.get(at, set())
                    if h_lineup and a_lineup:
                        break

        if not h_lineup or not a_lineup:
            continue

        # --- Stage 1: Lineup disruption ---
        h_prev = last_lineup.get(ht)
        a_prev = last_lineup.get(at)

        if h_prev is not None and a_prev is not None:
            h_outs = len(h_prev - h_lineup)
            h_ins = len(h_lineup - h_prev)
            h_cont = len(h_prev & h_lineup)
            a_outs = len(a_prev - a_lineup)
            a_ins = len(a_lineup - a_prev)
            a_cont = len(a_prev & a_lineup)

            df.at[idx, "home_lineup_changes"] = h_outs
            df.at[idx, "away_lineup_changes"] = a_outs
            df.at[idx, "lineup_changes_diff"] = h_outs - a_outs
            df.at[idx, "lineup_continuity_diff"] = h_cont - a_cont

            # Ruck missing: check if the player with highest HO EMA from
            # last lineup is absent from current lineup
            for prefix, team, prev_lu, cur_lu in [
                ("home", ht, h_prev, h_lineup),
                ("away", at, a_prev, a_lineup),
            ]:
                # Find the primary ruck from previous lineup
                best_ruck = None
                best_ho = 0.0
                for p in prev_lu:
                    ema = player_stat_emas.get((p, team), {})
                    ho = ema.get("HO", 0.0)
                    if ho > best_ho:
                        best_ho = ho
                        best_ruck = p
                if best_ruck and best_ho >= 10 and best_ruck not in cur_lu:
                    df.at[idx, f"{prefix}_ruck_missing"] = 1
                else:
                    df.at[idx, f"{prefix}_ruck_missing"] = 0

        # --- Stage 2: Quality-weighted absences ---
        if h_prev is not None and a_prev is not None:
            h_missing_players = h_prev - h_lineup
            h_gained_players = h_lineup - h_prev
            a_missing_players = a_prev - a_lineup
            a_gained_players = a_lineup - a_prev

            h_missing_rating = sum(player_ratings.get((p, ht), 0) for p in h_missing_players)
            h_gained_rating = sum(player_ratings.get((p, ht), 0) for p in h_gained_players)
            a_missing_rating = sum(player_ratings.get((p, at), 0) for p in a_missing_players)
            a_gained_rating = sum(player_ratings.get((p, at), 0) for p in a_gained_players)

            df.at[idx, "missing_rating_diff"] = h_missing_rating - a_missing_rating
            df.at[idx, "net_quality_change_diff"] = (
                (h_gained_rating - h_missing_rating) - (a_gained_rating - a_missing_rating)
            )

            # Role-split missing ratings
            for role_name, feat_suffix in [
                ("midfielder", "mid"), ("forward", "fwd"), ("defender", "def")
            ]:
                h_role_miss = sum(
                    player_ratings.get((p, ht), 0) for p in h_missing_players
                    if _classify_role(player_stat_emas.get((p, ht), {})) == role_name
                )
                a_role_miss = sum(
                    player_ratings.get((p, at), 0) for p in a_missing_players
                    if _classify_role(player_stat_emas.get((p, at), {})) == role_name
                )
                df.at[idx, f"missing_{feat_suffix}_rating_diff"] = h_role_miss - a_role_miss

        # --- Stage 3: Selection matchup ---
        h_total = sum(player_ratings.get((p, ht), 0) for p in h_lineup)
        a_total = sum(player_ratings.get((p, at), 0) for p in a_lineup)
        df.at[idx, "selected_total_rating_diff"] = h_total - a_total

        for role_name, feat_suffix in [("midfielder", "mid"), ("forward", "fwd")]:
            h_role_total = sum(
                player_ratings.get((p, ht), 0) for p in h_lineup
                if _classify_role(player_stat_emas.get((p, ht), {})) == role_name
            )
            a_role_total = sum(
                player_ratings.get((p, at), 0) for p in a_lineup
                if _classify_role(player_stat_emas.get((p, at), {})) == role_name
            )
            df.at[idx, f"selected_{feat_suffix}_rating_diff"] = h_role_total - a_role_total

        # --- Update rolling state (only for completed matches) ---
        if is_completed:
            last_lineup[ht] = h_lineup
            last_lineup[at] = a_lineup

            # Update player ratings and stat EMAs for all players in this match
            for team, lineup in [(ht, h_lineup), (at, a_lineup)]:
                scores = match_player_scores.get(mid, {}).get(team, {})
                stats = match_player_stats.get(mid, {}).get(team, {})
                for player in lineup:
                    key = (player, team)
                    # Update composite rating EMA
                    new_score = scores.get(player, 0)
                    old_rating = player_ratings.get(key)
                    if old_rating is None:
                        player_ratings[key] = new_score
                    else:
                        player_ratings[key] = (
                            RATING_ALPHA * new_score + (1 - RATING_ALPHA) * old_rating
                        )
                    # Update per-stat EMAs (for role inference)
                    new_stats = stats.get(player, {})
                    old_emas = player_stat_emas.get(key)
                    if old_emas is None:
                        player_stat_emas[key] = dict(new_stats)
                    else:
                        updated = {}
                        for s in role_stats:
                            old_v = old_emas.get(s, 0)
                            new_v = new_stats.get(s, 0)
                            updated[s] = RATING_ALPHA * new_v + (1 - RATING_ALPHA) * old_v
                        player_stat_emas[key] = updated

    # Summary stats
    non_null = df[feature_cols].notna().sum()
    log.info("Player features computed. Non-null counts:")
    for col in feature_cols:
        log.info("  %s: %d", col, non_null[col])

    return df
