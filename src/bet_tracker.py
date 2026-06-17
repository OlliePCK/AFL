"""
Bet tracking — log value bets, reconcile results, track P&L.

Stores bet history in data/betting/bet_log.csv.
"""
from datetime import datetime

import pandas as pd
import numpy as np

from src.config import PROJECT_ROOT
from src.utils import setup_logging

log = setup_logging()

BET_LOG_DIR = PROJECT_ROOT / "data" / "betting"
BET_LOG_PATH = BET_LOG_DIR / "bet_log.csv"

BET_LOG_COLUMNS = [
    "bet_id", "date_placed", "round", "home_team", "away_team",
    "bet_team", "bet_side", "odds", "model_prob", "edge",
    "kelly_pct", "bet_amount", "status", "profit_loss", "reconciled_at",
]


def _load_bet_log() -> pd.DataFrame:
    if BET_LOG_PATH.exists():
        return pd.read_csv(BET_LOG_PATH)
    return pd.DataFrame(columns=BET_LOG_COLUMNS)


def _save_bet_log(df: pd.DataFrame):
    BET_LOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BET_LOG_PATH, index=False)


def log_bets(value_bets: list[dict], round_name: str = "") -> int:
    """Append detected value bets to the bet log. Returns count of new bets logged."""
    if not value_bets:
        return 0

    existing = _load_bet_log()
    existing_ids = set(existing["bet_id"]) if not existing.empty else set()

    new_rows = []
    now = datetime.now().isoformat()

    for bet in value_bets:
        # Parse home/away from the "X vs Y" match string
        match_str = bet.get("match", "")
        if " vs " in match_str:
            home, away = match_str.split(" vs ", 1)
            home, away = home.strip(), away.strip()
        else:
            home, away = bet.get("home_team", ""), bet.get("away_team", "")

        side = bet.get("bet_side", "")
        # bet['date'] may be a pandas Timestamp, datetime, or string; str()
        # gives a YYYY-MM-DD ... prefix in every case, so slicing [:10] is safe.
        raw_date = bet.get("date") or now
        date_str = str(raw_date)[:10]
        bet_id = f"{date_str}_{home}_{away}_{side}"

        if bet_id in existing_ids:
            continue

        new_rows.append({
            "bet_id": bet_id,
            "date_placed": now,
            "round": round_name,
            "home_team": home,
            "away_team": away,
            "bet_team": bet.get("bet_team", ""),
            "bet_side": side,
            "odds": bet.get("odds", 0),
            "model_prob": bet.get("model_prob", 0),
            "edge": bet.get("edge", 0),
            "kelly_pct": bet.get("kelly_pct", 0),
            "bet_amount": bet.get("bet_amount", 0),
            "status": "pending",
            "profit_loss": None,
            "reconciled_at": None,
        })

    if not new_rows:
        log.info("No new bets to log (all already recorded)")
        return 0

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    _save_bet_log(combined)
    log.info(f"Logged {len(new_rows)} new bet(s) to {BET_LOG_PATH}")
    return len(new_rows)


def reconcile_results(year: int = 2026) -> int:
    """Match pending bets against completed results. Returns count reconciled."""
    from src.squiggle_client import fetch_games
    from src.team_mapping import normalize_team

    bet_log = _load_bet_log()
    pending = bet_log[bet_log["status"] == "pending"]

    if pending.empty:
        log.info("No pending bets to reconcile")
        return 0

    # Ensure object dtype on columns we'll write strings into
    # (CSV round-trips can leave all-None columns as float64)
    for col in ("status", "reconciled_at"):
        if col in bet_log.columns and bet_log[col].dtype != object:
            bet_log[col] = bet_log[col].astype(object)

    # Fetch completed games
    results = fetch_games(year)
    if results.empty:
        log.info(f"No completed games for {year}")
        return 0

    # Normalize team names in results
    for col in ["home_team", "away_team"]:
        results[col] = results[col].apply(
            lambda t: normalize_team(t) if pd.notna(t) else t
        )

    # Build lookup: (home_team, away_team) -> result row
    result_lookup = {}
    for _, row in results.iterrows():
        key = (row["home_team"], row["away_team"])
        result_lookup[key] = row

    now = datetime.now().isoformat()
    reconciled = 0

    for idx in pending.index:
        ht = bet_log.at[idx, "home_team"]
        at = bet_log.at[idx, "away_team"]
        result = result_lookup.get((ht, at))

        if result is None:
            continue

        bet_team = bet_log.at[idx, "bet_team"]
        bet_amount = float(bet_log.at[idx, "bet_amount"])
        odds = float(bet_log.at[idx, "odds"])

        # Determine winner
        home_score = result.get("home_score", 0)
        away_score = result.get("away_score", 0)
        if home_score > away_score:
            winner = ht
        elif away_score > home_score:
            winner = at
        else:
            winner = "draw"

        if bet_team == winner:
            bet_log.at[idx, "status"] = "won"
            bet_log.at[idx, "profit_loss"] = bet_amount * (odds - 1)
        elif winner == "draw":
            bet_log.at[idx, "status"] = "void"
            bet_log.at[idx, "profit_loss"] = 0
        else:
            bet_log.at[idx, "status"] = "lost"
            bet_log.at[idx, "profit_loss"] = -bet_amount

        bet_log.at[idx, "reconciled_at"] = now
        reconciled += 1

    _save_bet_log(bet_log)
    log.info(f"Reconciled {reconciled} bet(s)")
    return reconciled


def get_performance():
    """Print betting performance summary."""
    bet_log = _load_bet_log()

    if bet_log.empty:
        log.info("No bets recorded yet")
        return

    total = len(bet_log)
    pending = bet_log[bet_log["status"] == "pending"]
    settled = bet_log[bet_log["status"].isin(["won", "lost"])]
    won = bet_log[bet_log["status"] == "won"]
    lost = bet_log[bet_log["status"] == "lost"]

    log.info("\n" + "=" * 60)
    log.info("BETTING PERFORMANCE")
    log.info("=" * 60)

    log.info(f"  Total bets:     {total}")
    log.info(f"  Pending:        {len(pending)}")
    log.info(f"  Settled:        {len(settled)}")

    if not settled.empty:
        record = f"{len(won)}W - {len(lost)}L"
        win_rate = len(won) / len(settled)
        total_staked = settled["bet_amount"].astype(float).sum()
        total_pnl = settled["profit_loss"].astype(float).sum()
        roi = total_pnl / total_staked if total_staked > 0 else 0

        log.info(f"  Record:         {record}")
        log.info(f"  Win rate:       {win_rate:.1%}")
        log.info(f"  Total staked:   ${total_staked:,.0f}")
        log.info(f"  Total P&L:      ${total_pnl:+,.0f}")
        log.info(f"  ROI:            {roi:+.1%}")

        # Per-round breakdown if rounds are logged
        if "round" in settled.columns and settled["round"].notna().any():
            log.info(f"\n  {'Round':<15} {'Bets':>5} {'W':>4} {'L':>4} {'P&L':>10} {'ROI':>8}")
            log.info(f"  {'-'*48}")
            for rnd, group in settled.groupby("round"):
                if pd.isna(rnd) or rnd == "":
                    continue
                rnd_won = (group["status"] == "won").sum()
                rnd_lost = (group["status"] == "lost").sum()
                rnd_staked = group["bet_amount"].astype(float).sum()
                rnd_pnl = group["profit_loss"].astype(float).sum()
                rnd_roi = rnd_pnl / rnd_staked if rnd_staked > 0 else 0
                log.info(f"  {rnd:<15} {len(group):>5} {rnd_won:>4} {rnd_lost:>4} "
                         f"${rnd_pnl:>+8,.0f} {rnd_roi:>+7.1%}")
    else:
        log.info("  No settled bets yet")

    if not pending.empty:
        pending_exposure = pending["bet_amount"].astype(float).sum()
        log.info(f"\n  Pending exposure: ${pending_exposure:,.0f} across {len(pending)} bet(s)")

    log.info("=" * 60)
