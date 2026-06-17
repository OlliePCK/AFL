"""Export model metrics and feature importances to JSON for the dashboard."""
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from pathlib import Path
from catboost import CatBoostClassifier
from src.config import PROJECT_ROOT
from src.model import BETTING_FEATURE_COLS

MODEL_DIR = PROJECT_ROOT / "data" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def export_feature_importance():
    model = CatBoostClassifier()
    model.load_model(str(PROJECT_ROOT / "data" / "model.cbm"))
    names = model.feature_names_
    importances = model.get_feature_importance()
    data = sorted(
        [{"feature": n, "importance": round(float(i), 2)} for n, i in zip(names, importances)],
        key=lambda x: x["importance"], reverse=True
    )
    path = MODEL_DIR / "feature_importance.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported {len(data)} features to {path}")


def export_model_metrics():
    path = MODEL_DIR / "model_metrics.json"
    if path.exists():
        print(f"Model metrics already saved at {path} (written by retrain)")
        return
    # Fallback: hardcoded values from initial training
    metrics = {
        "new_model": {
            "accuracy": 0.6991, "log_loss": 0.5256,
            "auc_roc": 0.8001, "brier_score": 0.1792,
        },
        "old_model": {
            "accuracy": 0.7083, "log_loss": 0.5330,
            "auc_roc": 0.7962, "brier_score": 0.1809,
        },
        "train_period": "2012-2024",
        "val_period": "2025",
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Exported model metrics to {path}")


def export_current_season_results():
    """Write a small CSV of completed matches for the current season,
    joined with the most recent prediction snapshot per game (if any).

    Output: data/master/current_season_results.csv
    Used by the dashboard to show round history.
    """
    import pandas as pd

    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    if not master_path.exists():
        print(f"No master dataset at {master_path}, skipping")
        return

    df = pd.read_csv(master_path, parse_dates=["date"])
    current_year = int(df["year"].max())
    season = df[df["year"] == current_year].copy()
    completed = season.dropna(subset=["home_score", "away_score"]).copy()

    if completed.empty:
        print(f"No completed matches for {current_year} yet")
        return

    cols = ["game_id", "year", "round", "roundname", "date", "venue",
            "home_team", "away_team", "home_score", "away_score", "winner"]
    out = completed[[c for c in cols if c in completed.columns]].copy()
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)

    # Join with most recent prediction snapshot per game_id (if available)
    history_path = PROJECT_ROOT / "data" / "master" / "predictions_history.csv"
    if history_path.exists():
        hist = pd.read_csv(history_path)
        if not hist.empty and "snapshot_date" in hist.columns:
            hist = hist.sort_values("snapshot_date").drop_duplicates("game_id", keep="last")
            snap_cols = ["game_id", "predicted_winner", "home_win_prob",
                         "away_win_prob", "predicted_margin", "snapshot_date"]
            snap = hist[[c for c in snap_cols if c in hist.columns]]
            out = out.merge(snap, on="game_id", how="left")

    out = out.sort_values("date")
    out_path = PROJECT_ROOT / "data" / "master" / "current_season_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Exported {len(out)} {current_year} results to {out_path}")


def export_calibration_curve():
    """Compute reliability diagram bins for the current model on the val set.

    Writes data/model/calibration_curve.json with raw and calibrated bins so
    the dashboard can plot predicted probability vs actual win rate.
    """
    import pandas as pd
    import numpy as np
    from src.model import load_calibrator, get_feature_cols, TARGET
    from src.features import build_features

    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    model_path = PROJECT_ROOT / "data" / "model.cbm"
    if not master_path.exists() or not model_path.exists():
        print("Missing master dataset or model; skipping calibration export")
        return

    df = pd.read_csv(master_path, parse_dates=["date"])
    df = build_features(df)
    df = df.dropna(subset=[TARGET]).sort_values("date").reset_index(drop=True)

    # Same val slice used in training (last 207 completed matches)
    val = df.iloc[-207:].copy()

    feature_cols = [c for c in get_feature_cols() if c in val.columns]
    X = val[feature_cols]
    y = val[TARGET].astype(int).values

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    raw_probs = model.predict_proba(X)[:, 1]

    calibrator = load_calibrator()
    cal_probs = calibrator.predict(raw_probs) if calibrator is not None else None

    def bin_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
        bins = []
        for b in range(n_bins):
            mask = idx == b
            count = int(mask.sum())
            if count == 0:
                continue
            bins.append({
                "bin_mid": round(float((edges[b] + edges[b + 1]) / 2), 3),
                "predicted": round(float(probs[mask].mean()), 4),
                "observed": round(float(labels[mask].mean()), 4),
                "count": count,
            })
        return bins

    data = {
        "val_period": f"{int(val['year'].min())}-{int(val['year'].max())}",
        "n_samples": int(len(val)),
        "raw": bin_curve(raw_probs, y),
    }
    if cal_probs is not None:
        data["calibrated"] = bin_curve(cal_probs, y)

    path = MODEL_DIR / "calibration_curve.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported calibration curve ({data['n_samples']} val samples) to {path}")


def export_walk_forward():
    import pandas as pd
    bets_path = PROJECT_ROOT / "data" / "master" / "wf_best_config_bets.csv"
    if not bets_path.exists():
        print(f"No walk-forward bets file at {bets_path}, skipping")
        return

    df = pd.read_csv(bets_path)
    yearly = []
    for year, group in df.groupby("test_year"):
        profit_col = "profit" if "profit" in group.columns else "profit_loss"
        won_col = "won" if "won" in group.columns else "correct"
        yearly.append({
            "year": int(year),
            "n_bets": len(group),
            "win_rate": round(float(group[won_col].mean()), 3) if won_col in group.columns else 0,
            "roi": round(float(group[profit_col].sum() / group["bet_amount"].sum() * 100), 1) if "bet_amount" in group.columns else 0,
            "total_profit": round(float(group[profit_col].sum()), 2) if profit_col in group.columns else 0,
        })

    total_bets = sum(y["n_bets"] for y in yearly)
    total_profit = sum(y["total_profit"] for y in yearly)
    profitable = sum(1 for y in yearly if y["total_profit"] > 0)

    data = {
        "yearly": yearly,
        "summary": {
            "total_bets": total_bets,
            "overall_roi": round(total_profit / max(sum(y["n_bets"] * 100 for y in yearly), 1) * 100, 1),
            "profitable_years": profitable,
            "total_years": len(yearly),
        }
    }
    path = MODEL_DIR / "walk_forward_results.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported walk-forward results ({len(yearly)} years) to {path}")


if __name__ == "__main__":
    export_feature_importance()
    export_model_metrics()
    export_current_season_results()
    export_walk_forward()
    export_calibration_curve()
    print("Done!")
