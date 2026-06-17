"""Retroactively predict already-played 2026 rounds and append to history.

Why this exists: predictions_history.csv only contains rows from snapshots
taken on or after deployment. Earlier rounds (R1-R4 2026) were played before
the dashboard tracked snapshots, so the round-history view shows results
without any predicted-winner overlay.

This script fills that gap by running the current production model against
the master dataset (features are causal, computed from prior matches only)
and writing one row per match into predictions_history.csv with a snapshot
date set to the day before the match. Existing rows for the same
(game_id, snapshot_date) are overwritten.

Note: the production model uses 2025-2026 as its early-stopping val window,
so backfill probabilities here are mildly in-sample. They're meant to give
the dashboard a "what would the model have said" overlay, not to claim a
true out-of-sample track record.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.config import PROJECT_ROOT
from src.features import build_features
from src.model import get_feature_cols, load_calibrator


def backfill(year: int = 2026, rounds: list[str] | None = None) -> None:
    if rounds is None:
        rounds = [f"Round {n}" for n in range(1, 5)]

    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    df = build_features(df)

    target = df[
        (df["year"] == year)
        & (df["roundname"].isin(rounds))
        & df["home_score"].notna()
    ].copy()
    if target.empty:
        print(f"No completed matches found for {year} rounds {rounds}")
        return

    feature_cols = [c for c in get_feature_cols() if c in target.columns]
    X = target[feature_cols]

    clf = CatBoostClassifier()
    clf.load_model(str(PROJECT_ROOT / "data" / "model.cbm"))
    raw = clf.predict_proba(X)[:, 1]

    calibrator = load_calibrator()
    if calibrator is not None:
        probs = np.clip(calibrator.predict(raw), 0.02, 0.98)
    else:
        probs = raw

    margin_path = PROJECT_ROOT / "data" / "margin_model.cbm"
    if margin_path.exists():
        margin_model = CatBoostRegressor()
        margin_model.load_model(str(margin_path))
        margins = margin_model.predict(X)
    else:
        margins = np.full(len(target), np.nan)

    snapshot = pd.DataFrame({
        "game_id": target["game_id"].astype(int).values,
        "date": target["date"].dt.strftime("%Y-%m-%d %H:%M:%S").values,
        "roundname": target["roundname"].values,
        "venue": target["venue"].values,
        "home_team": target["home_team"].values,
        "away_team": target["away_team"].values,
        "home_win_prob": probs,
        "away_win_prob": 1 - probs,
        "predicted_winner": np.where(
            np.isclose(probs, 0.5),
            "",
            np.where(probs > 0.5, target["home_team"].values, target["away_team"].values),
        ),
        "confidence": np.maximum(probs, 1 - probs),
        "predicted_margin": margins,
        "home_elo": target.get("home_elo", pd.Series([np.nan] * len(target))).values,
        "away_elo": target.get("away_elo", pd.Series([np.nan] * len(target))).values,
        "elo_diff": target.get("elo_diff", pd.Series([np.nan] * len(target))).values,
        "home_odds": np.nan,
        "away_odds": np.nan,
        # Snapshot date = day of the match (proxy for "made on the morning")
        "snapshot_date": target["date"].dt.strftime("%Y-%m-%d").values,
    })

    history_path = PROJECT_ROOT / "data" / "master" / "predictions_history.csv"
    if history_path.exists():
        existing = pd.read_csv(history_path)
        # Drop any rows that overlap on (game_id, snapshot_date)
        keys = set(zip(snapshot["game_id"], snapshot["snapshot_date"]))
        if "game_id" in existing.columns and "snapshot_date" in existing.columns:
            existing = existing[~existing.apply(
                lambda r: (r["game_id"], r["snapshot_date"]) in keys, axis=1
            )]
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        combined = snapshot

    combined.to_csv(history_path, index=False)
    print(f"Backfilled {len(snapshot)} predictions to {history_path}")
    for _, r in snapshot.iterrows():
        result_team = r["predicted_winner"] or "Toss-up"
        print(
            f"  {r['roundname']:>9s}  {r['home_team']:>22s} vs {r['away_team']:<22s}"
            f" | {result_team} ({r['confidence']:.0%})"
        )


if __name__ == "__main__":
    backfill()
