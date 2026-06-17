"""
Experiment: compare margin model loss functions.

Tests MAE (current), Huber (robust to outliers), and Quantile-0.5 (median)
using 5-seed walk-forward CV. Same features, same seeds, same splits —
only the loss function changes.

Writes: data/analysis/margin_loss_experiment.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features  # noqa: E402
from src.model import TARGET, load_optimization_results  # noqa: E402

SEEDS = [42, 123, 256, 789, 1337]
MARGIN_OPT_PATH = PROJECT_ROOT / "data" / "margin_optimization_results.json"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "margin_loss_experiment.json"


def _normalize_params(p: dict) -> dict:
    out = dict(p)
    bt = out.get("bootstrap_type", "Bayesian")
    if bt == "Bayesian":
        out.pop("subsample", None)
    else:
        out.pop("bagging_temperature", None)
    return out


def _base_params() -> dict:
    with open(MARGIN_OPT_PATH) as f:
        opt = json.load(f)
    best = opt["tuned"]["best_params"]
    params = {
        "iterations": 2000,
        "early_stopping_rounds": 100,
        "use_best_model": True,
        "verbose": 0,
        **best,
    }
    return _normalize_params(params)


LOSS_CONFIGS = {
    "MAE": {"loss_function": "MAE", "eval_metric": "MAE"},
    "Huber:10": {"loss_function": "Huber:delta=10", "eval_metric": "MAE"},
    "Huber:20": {"loss_function": "Huber:delta=20", "eval_metric": "MAE"},
    "Huber:30": {"loss_function": "Huber:delta=30", "eval_metric": "MAE"},
    "Quantile:0.5": {"loss_function": "Quantile:alpha=0.5", "eval_metric": "MAE"},
}


def run_wf_margin_ensemble(df: pd.DataFrame, features: list[str],
                            base_params: dict, loss_cfg: dict,
                            wf_years: list[int]) -> dict:
    """5-seed walk-forward margin regression."""
    per_seed_preds: dict[int, dict[int, np.ndarray]] = {s: {} for s in SEEDS}
    fold_true: dict[int, np.ndarray] = {}

    params = {**base_params, **loss_cfg}

    for vy in wf_years:
        train_df = df[df["year"] < vy].dropna(subset=["margin"])
        val_df = df[df["year"] == vy].dropna(subset=["margin"])
        if train_df.empty or val_df.empty:
            continue
        fold_true[vy] = val_df["margin"].values.astype(float)

        for seed in SEEDS:
            p = _normalize_params({**params, "random_seed": seed})
            m = CatBoostRegressor(**p)
            m.fit(
                Pool(train_df[features], train_df["margin"]),
                eval_set=Pool(val_df[features], val_df["margin"]),
            )
            per_seed_preds[seed][vy] = m.predict(val_df[features])

    # Ensemble average
    ensemble_preds: dict[int, np.ndarray] = {}
    for vy in wf_years:
        stacks = [per_seed_preds[s][vy] for s in SEEDS if vy in per_seed_preds[s]]
        if stacks:
            ensemble_preds[vy] = np.mean(stacks, axis=0)

    # Aggregate
    all_preds = np.concatenate([ensemble_preds[y] for y in wf_years if y in ensemble_preds])
    all_true = np.concatenate([fold_true[y] for y in wf_years if y in fold_true])

    per_fold = []
    for vy in wf_years:
        if vy not in ensemble_preds:
            continue
        p = ensemble_preds[vy]
        t = fold_true[vy]
        per_fold.append({
            "year": int(vy),
            "n": int(len(t)),
            "mae": float(np.mean(np.abs(t - p))),
            "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
            "direction_acc": float(((p > 0) == (t > 0)).mean()),
        })

    return {
        "n": int(len(all_true)),
        "mae": float(np.mean(np.abs(all_true - all_preds))),
        "rmse": float(np.sqrt(np.mean((all_true - all_preds) ** 2))),
        "direction_acc": float(((all_preds > 0) == (all_true > 0)).mean()),
        "per_fold": per_fold,
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("=" * 80)
    print("MARGIN LOSS FUNCTION EXPERIMENT")
    print("=" * 80)

    print("\nLoading + featuring...")
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    df = build_features(df)
    df_complete = df.dropna(subset=["margin", TARGET]).copy().sort_values("date").reset_index(drop=True)
    min_year = int(df_complete["year"].min())
    max_year = int(df_complete["year"].max())
    print(f"  {len(df_complete)} rows, {min_year}-{max_year}")

    opt = load_optimization_results()
    features = [c for c in opt["best_features"] if c in df_complete.columns]
    base_params = _base_params()
    print(f"  {len(features)} features, base params loaded")

    wf_years = list(range(max(2019, min_year + 5), max_year + 1))
    print(f"  WF years: {wf_years}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Loss configs: {list(LOSS_CONFIGS.keys())}")

    results = {}
    for name, loss_cfg in LOSS_CONFIGS.items():
        t0 = time.time()
        print(f"\n==> {name}")
        r = run_wf_margin_ensemble(df_complete, features, base_params, loss_cfg, wf_years)
        results[name] = r
        print(f"   MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
              f"dir={r['direction_acc']:.4f}  ({time.time() - t0:.0f}s)")

    # Comparison
    baseline = results["MAE"]
    print("\n" + "=" * 80)
    print(f"{'config':<18s} {'MAE':>8s} {'d_MAE':>8s} {'RMSE':>8s} {'dir':>8s} {'d_dir':>8s}")
    print("  " + "-" * 58)
    for name, r in results.items():
        d_mae = r["mae"] - baseline["mae"]
        d_dir = r["direction_acc"] - baseline["direction_acc"]
        print(f"  {name:<18s} {r['mae']:>8.3f} {d_mae:>+8.3f} "
              f"{r['rmse']:>8.3f} {r['direction_acc']:>8.4f} {d_dir:>+8.4f}")

    print("\n  Per-fold direction accuracy:")
    print(f"  {'year':>6s}", end="")
    for name in LOSS_CONFIGS:
        print(f" {name:>12s}", end="")
    print()
    for fold_i in range(len(wf_years)):
        yr = wf_years[fold_i]
        print(f"  {yr:>6d}", end="")
        for name in LOSS_CONFIGS:
            folds = results[name]["per_fold"]
            if fold_i < len(folds):
                print(f" {folds[fold_i]['direction_acc']:>12.3f}", end="")
        print()

    report = {
        "seeds": SEEDS,
        "wf_years": wf_years,
        "n_features": len(features),
        "results": results,
        "runtime_sec": time.time() - t_start,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {REPORT_PATH}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
