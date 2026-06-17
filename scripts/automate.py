"""
AFL model automation entrypoint for cron-driven container tasks.

Usage:
    python scripts/automate.py odds
    python scripts/automate.py update
    python scripts/automate.py predict
    python scripts/automate.py predict-cached
    python scripts/automate.py sync-results
    python scripts/automate.py refresh-round
    python scripts/automate.py retrain
    python scripts/automate.py export
    python scripts/automate.py reconcile
    python scripts/automate.py drift-check
    python scripts/automate.py sanity-check
    python scripts/automate.py full-cycle
"""
import importlib.util
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def log(msg: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run_python(args: list[str]) -> bool:
    """Run a Python command, return True on success."""
    cmd = [sys.executable] + args
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        log(f"FAILED (exit code {result.returncode})")
        return False
    return True


def _load_export_module():
    """Load scripts/export_dashboard_data.py as a module."""
    script_path = PROJECT_ROOT / "scripts" / "export_dashboard_data.py"
    spec = importlib.util.spec_from_file_location("export_dashboard_data", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load export module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_odds():
    """Fetch live odds from The Odds API."""
    log("=== FETCHING LIVE ODDS ===")
    from src.odds_monitor import fetch_and_save

    odds = fetch_and_save()
    if odds.empty:
        log("No odds returned (API key missing or no upcoming matches)")
        return

    log(f"Fetched odds for {len(odds)} matches")
    for _, row in odds.iterrows():
        log(
            f"  {row['home_team']:>25} vs {row['away_team']:<25} "
            f"H:{row['home_odds']:.2f} A:{row['away_odds']:.2f}"
        )


def cmd_update():
    """Update the master dataset with newly completed matches."""
    log("=== UPDATING DATASET ===")
    from src.dataset_builder import update_master_dataset

    n_new = update_master_dataset()
    log(f"Added {n_new} new matches to dataset")
    return n_new


def cmd_predict():
    """Run predictions and fetch fresh odds if available."""
    log("=== RUNNING PREDICTIONS ===")
    run_python(["run_predictions.py"])


def cmd_predict_cached():
    """Run predictions using cached odds only."""
    log("=== RUNNING PREDICTIONS (CACHED ODDS) ===")
    run_python(["run_predictions.py", "--no-odds"])


def cmd_retrain():
    """Retrain the production ensembles (classifier + margin + analytical) on all completed data."""
    log("=== RETRAINING ENSEMBLES ===")
    from src.model import retrain_production_ensembles

    result = retrain_production_ensembles()
    if result is None:
        log("Ensemble retrain FAILED -- production is running on stale models")
        return
    log("Retrain complete")

    # Analytical ensemble (V3: 3 odds-only features, for tipping accuracy).
    # Superseded train_analytical_ensemble.py (47-feature V1) on 2026-04-17 —
    # see data/analysis/analytical_model_decision_memo.md for rationale.
    log("=== RETRAINING ANALYTICAL ENSEMBLE (V3 odds-only) ===")
    ok = run_python([str(PROJECT_ROOT / "scripts" / "train_analytical_odds_only.py")])
    if not ok:
        log("Analytical ensemble retrain FAILED (non-critical)")
    else:
        log("Analytical ensemble retrain complete")


def cmd_export():
    """Export dashboard data artifacts written by the app."""
    log("=== EXPORTING DASHBOARD DATA ===")
    run_python([str(PROJECT_ROOT / "scripts" / "export_dashboard_data.py")])


def cmd_export_current_results():
    """Export just the current-season results CSV used by the dashboard."""
    log("=== EXPORTING CURRENT SEASON RESULTS ===")
    module = _load_export_module()
    module.export_current_season_results()


def cmd_reconcile():
    """Reconcile pending bets against completed results."""
    log("=== RECONCILING BETS ===")
    from src.bet_tracker import get_performance, reconcile_results

    reconcile_results()
    get_performance()


def cmd_drift_check():
    """Check calibration drift of the production ensemble on the current season."""
    log("=== CHECKING CALIBRATION DRIFT ===")
    ok = run_python([str(PROJECT_ROOT / "scripts" / "calibration_drift.py")])
    if not ok:
        log("Drift check failed to run")
        return
    # Read back the report to surface any drift at the automate log level
    import json
    report_path = PROJECT_ROOT / "data" / "model" / "calibration_drift.json"
    if not report_path.exists():
        log("No drift report written")
        return
    with open(report_path) as f:
        report = json.load(f)
    if report.get("is_drifting"):
        bins = ", ".join(b["bin"] for b in report.get("drifting_bins", []))
        log(f"ALERT: calibration drift detected in bins: {bins} -- consider retrain")
    else:
        status = report.get("status", "unknown")
        n = report.get("n_samples", 0)
        log(f"Calibration healthy (status={status}, n={n})")


def cmd_sanity_check():
    """Run guardrail checks over the latest upcoming predictions."""
    log("=== SANITY CHECK (predictions) ===")
    ok = run_python([str(PROJECT_ROOT / "scripts" / "sanity_check_predictions.py")])
    if not ok:
        # Non-zero exit means at least one WARN/FAIL — surface it in the cron
        # log but do not halt the wider pipeline (predictions are already
        # saved; the check is diagnostic, not blocking).
        log("Sanity check reported WARN/FAIL — review data/model/prediction_sanity.json")
    else:
        log("Sanity check clean")


def cmd_sync_results():
    """Refresh in-season results, standings, stats, and dashboard result files."""
    log("=== SYNCING IN-SEASON RESULTS ===")

    import os
    import pandas as pd

    from src.config import CURRENT_YEAR
    from src.footywire_scraper import scrape_all_seasons
    from src.squiggle_client import fetch_all_standings

    cmd_update()

    log(f"=== REFRESHING SQUIGGLE STANDINGS ({CURRENT_YEAR}) ===")
    standings = fetch_all_standings(CURRENT_YEAR, CURRENT_YEAR)
    if standings.empty:
        log("No standings rows refreshed")
    else:
        current = standings[standings["year"] == CURRENT_YEAR]
        rounds = sorted(current["round"].dropna().astype(int).unique()) if not current.empty else []
        latest_round = rounds[-1] if rounds else "n/a"
        log(f"Standings refreshed through round {latest_round}")

    log(f"=== REFRESHING FOOTYWIRE MATCH STATS ({CURRENT_YEAR}) ===")
    team_stats = scrape_all_seasons(CURRENT_YEAR, CURRENT_YEAR)
    if team_stats.empty:
        log("No Footywire team stats refreshed")
    else:
        latest_mid = int(team_stats["mid"].max()) if "mid" in team_stats.columns else "n/a"
        log(f"Footywire team stats refreshed ({len(team_stats)} team rows, max mid {latest_mid})")

    if os.environ.get("VISUAL_CROSSING_API_KEY"):
        from src.weather import fetch_all_weather

        master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
        if master_path.exists():
            season = pd.read_csv(master_path, parse_dates=["date"])
            season = season[season["year"] == CURRENT_YEAR].copy()
            if not season.empty:
                log(f"=== REFRESHING WEATHER CACHE ({CURRENT_YEAR}) ===")
                fetch_all_weather(season)

    cmd_export_current_results()

    from src.predict import backfill_prediction_explanations_history

    log(f"=== BACKFILLING HISTORICAL EXPLANATIONS ({CURRENT_YEAR}) ===")
    n_backfilled = backfill_prediction_explanations_history(CURRENT_YEAR)
    if n_backfilled == 0:
        log("Historical explanation snapshots already current")

    cmd_reconcile()
    cmd_drift_check()


def cmd_refresh_round():
    """Fetch fresh odds, then refresh predictions using the saved snapshot."""
    log("=== REFRESHING ROUND DATA ===")
    cmd_odds()
    cmd_predict_cached()
    cmd_sanity_check()


def cmd_full_cycle():
    """Full weekly cycle: sync -> retrain -> export -> predict."""
    log("========================================")
    log("  FULL AUTOMATION CYCLE")
    log("========================================")

    steps = [
        ("Sync in-season data", cmd_sync_results),
        ("Retrain model", cmd_retrain),
        ("Export dashboard data", cmd_export),
        ("Run predictions (cached odds)", cmd_predict_cached),
        ("Sanity-check predictions", cmd_sanity_check),
    ]

    for name, func in steps:
        try:
            func()
        except Exception:
            log(f"ERROR in '{name}':\n{traceback.format_exc()}")
            log("Continuing to next step...")

    log("========================================")
    log("  FULL CYCLE COMPLETE")
    log("========================================")


COMMANDS = {
    "odds": cmd_odds,
    "update": cmd_update,
    "predict": cmd_predict,
    "predict-cached": cmd_predict_cached,
    "sync-results": cmd_sync_results,
    "refresh-round": cmd_refresh_round,
    "retrain": cmd_retrain,
    "export": cmd_export,
    "reconcile": cmd_reconcile,
    "drift-check": cmd_drift_check,
    "sanity-check": cmd_sanity_check,
    "full-cycle": cmd_full_cycle,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python scripts/automate.py <command>")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    command = sys.argv[1]
    log(f"Starting command: {command}")

    try:
        COMMANDS[command]()
    except Exception:
        log(f"FATAL ERROR:\n{traceback.format_exc()}")
        sys.exit(1)

    log(f"Command '{command}' finished")
