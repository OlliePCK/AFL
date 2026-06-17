"""
Item B -- train a 5-seed CatBoost margin-regression ensemble.

Mirrors scripts/train_ensemble.py but for the margin head:
  1. Load Optuna-winning hyperparameters from data/margin_optimization_results.json
  2. Walk-forward CV for each seed to measure per-seed + ensemble MAE
  3. Train 5 production margin regressors on the full train/val split
  4. Save to data/ensemble/margin_model_{0..4}.cbm
  5. Write report to data/analysis/margin_ensemble_report.json

predict.py will auto-detect the ensemble via load_margin_ensemble() on next run.
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
ENSEMBLE_DIR = PROJECT_ROOT / "data" / "ensemble"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "margin_ensemble_report.json"
MARGIN_OPT_PATH = PROJECT_ROOT / "data" / "margin_optimization_results.json"


def _normalize_params(p: dict) -> dict:
    """Keep only the bootstrap param that matches the chosen bootstrap_type."""
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
        "eval_metric": "MAE",
        "loss_function": "Huber:delta=30",
        "use_best_model": True,
        "verbose": 0,
        **best,
    }
    return _normalize_params(params)


def walk_forward_oof_margin(df: pd.DataFrame, features: list[str],
                             params: dict, wf_years: list[int],
                             seed: int) -> dict:
    """Walk-forward CV for margin regression -- one seed."""
    p = _normalize_params({**params, "random_seed": seed})
    fold_preds: dict[int, np.ndarray] = {}
    fold_true: dict[int, np.ndarray] = {}

    for vy in wf_years:
        train_df = df[df["year"] < vy].dropna(subset=["margin"])
        val_df = df[df["year"] == vy].dropna(subset=["margin"])
        if train_df.empty or val_df.empty:
            continue
        m = CatBoostRegressor(**p)
        m.fit(
            Pool(train_df[features], train_df["margin"]),
            eval_set=Pool(val_df[features], val_df["margin"]),
        )
        fold_preds[vy] = m.predict(val_df[features])
        fold_true[vy] = val_df["margin"].values.astype(float)

    return {"preds_by_fold": fold_preds, "true_by_fold": fold_true}


def summarize(name: str, preds: np.ndarray, true: np.ndarray) -> dict:
    return {
        "config": name,
        "mae": float(np.mean(np.abs(true - preds))),
        "rmse": float(np.sqrt(np.mean((true - preds) ** 2))),
        "direction_acc": float(((preds > 0) == (true > 0)).mean()),
        "n": int(len(true)),
    }


def main() -> None:
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 80)
    print("ITEM B -- 5-seed CatBoost margin ensemble")
    print("=" * 80)

    # -------------------- load data + features + params --------------------
    print("\n[1/6] Loading master dataset + building features...")
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    print(f"  loaded {len(df)} raw matches")
    df = build_features(df)
    df_complete = df.dropna(subset=["margin"]).copy().sort_values("date").reset_index(drop=True)
    min_year = int(df_complete["year"].min())
    max_year = int(df_complete["year"].max())
    print(f"  featured: {len(df_complete)} completed matches, {min_year}-{max_year}")

    opt = load_optimization_results()
    if opt is None:
        print("  FATAL: optimization_results.json (classifier) missing")
        sys.exit(1)
    features = [c for c in opt["best_features"] if c in df_complete.columns]

    if not MARGIN_OPT_PATH.exists():
        print(f"  FATAL: {MARGIN_OPT_PATH} not found. Run scripts/tune_margin.py first.")
        sys.exit(1)
    params = _base_params()
    print(f"  using {len(features)} features (same as classifier)")
    print(f"  using margin-tuned params: lr={params.get('learning_rate'):.4f} "
          f"depth={params.get('depth')} l2={params.get('l2_leaf_reg'):.2f} "
          f"bootstrap={params.get('bootstrap_type')}")

    # -------------------- walk-forward CV per seed --------------------
    wf_start = max(2019, min_year + 5)
    wf_years = list(range(wf_start, max_year + 1))
    print(f"\n[2/6] Walk-forward CV -- {len(SEEDS)} seeds x {len(wf_years)} folds "
          f"({wf_years[0]}-{wf_years[-1]}) = {len(SEEDS) * len(wf_years)} fits")
    t0 = time.time()

    per_seed_oof: dict[int, dict] = {}
    for i, seed in enumerate(SEEDS, 1):
        ts = time.time()
        print(f"  seed {seed} ({i}/{len(SEEDS)})...", end="", flush=True)
        per_seed_oof[seed] = walk_forward_oof_margin(df_complete, features, params, wf_years, seed)
        print(f" done ({time.time() - ts:.0f}s)")
    print(f"  [WF CV total: {time.time() - t0:.0f}s]")

    # -------------------- aggregate metrics --------------------
    print(f"\n[3/6] Aggregating OOF predictions...")

    per_seed_pooled = {}
    for seed, res in per_seed_oof.items():
        preds = np.concatenate([res["preds_by_fold"][y] for y in wf_years if y in res["preds_by_fold"]])
        true = np.concatenate([res["true_by_fold"][y] for y in wf_years if y in res["true_by_fold"]])
        per_seed_pooled[seed] = {"preds": preds, "true": true,
                                  "metrics": summarize(f"seed_{seed}", preds, true)}

    reference_true = per_seed_pooled[SEEDS[0]]["true"]
    for seed in SEEDS[1:]:
        assert np.array_equal(per_seed_pooled[seed]["true"], reference_true), \
            f"Seed {seed} has different label order"

    # Ensemble = mean of predictions across seeds
    ensemble_preds = np.mean(
        [per_seed_pooled[seed]["preds"] for seed in SEEDS], axis=0
    )
    ensemble_metrics = summarize("ensemble_mean", ensemble_preds, reference_true)

    seed_maes = [per_seed_pooled[seed]["metrics"]["mae"] for seed in SEEDS]
    seed_mean = float(np.mean(seed_maes))
    seed_std = float(np.std(seed_maes))
    seed_range = float(max(seed_maes) - min(seed_maes))

    # Per-fold MAE with and without ensemble
    per_fold_rows = []
    for vy in wf_years:
        y = per_seed_oof[SEEDS[0]]["true_by_fold"][vy]
        seed_fold_maes = []
        fold_preds_by_seed = []
        for seed in SEEDS:
            p = per_seed_oof[seed]["preds_by_fold"][vy]
            seed_fold_maes.append(float(np.mean(np.abs(y - p))))
            fold_preds_by_seed.append(p)
        ensemble_fold_preds = np.mean(fold_preds_by_seed, axis=0)
        ensemble_fold_mae = float(np.mean(np.abs(y - ensemble_fold_preds)))
        per_fold_rows.append({
            "year": int(vy),
            "n": int(len(y)),
            "seed_mean_mae": float(np.mean(seed_fold_maes)),
            "seed_std_mae": float(np.std(seed_fold_maes)),
            "ensemble_mae": ensemble_fold_mae,
        })

    print(f"  single-seed pooled MAE: mean={seed_mean:.4f}  std={seed_std:.4f}  range={seed_range:.4f}")
    print(f"  ensemble pooled MAE:    {ensemble_metrics['mae']:.4f}")
    print(f"  ensemble delta vs seed-mean: {ensemble_metrics['mae'] - seed_mean:+.4f}")
    print(f"  ensemble direction acc: {ensemble_metrics['direction_acc']:.3f}")
    for row in per_fold_rows:
        print(
            f"    {row['year']}  n={row['n']:3d}  "
            f"seed_mean={row['seed_mean_mae']:.3f} (+/-{row['seed_std_mae']:.3f})  "
            f"ensemble={row['ensemble_mae']:.3f}  "
            f"delta={row['ensemble_mae'] - row['seed_mean_mae']:+.3f}"
        )

    # -------------------- final production models --------------------
    val_size = 207
    train_split = df_complete.iloc[:-val_size]
    val_split = df_complete.iloc[-val_size:]
    train_years = f"{min_year}-{int(train_split['year'].max())}"
    val_years_range = f"{int(val_split['year'].min())}-{int(val_split['year'].max())}"
    print(f"\n[4/6] Training 5 production margin models on train ({train_years}), val ({val_years_range})")
    print(f"  train: {len(train_split)} rows | val: {len(val_split)} rows")

    X_train = train_split[features]
    y_train_margin = train_split["margin"]
    X_val = val_split[features]
    y_val_margin = val_split["margin"]

    final_val_preds_per_seed: list[np.ndarray] = []
    saved_paths: list[str] = []
    for i, seed in enumerate(SEEDS):
        p = _normalize_params({**params, "random_seed": seed})
        ts = time.time()
        m = CatBoostRegressor(**p)
        m.fit(Pool(X_train, y_train_margin), eval_set=Pool(X_val, y_val_margin))
        model_path = ENSEMBLE_DIR / f"margin_model_{i}.cbm"
        m.save_model(str(model_path))
        saved_paths.append(str(model_path))
        val_preds = m.predict(X_val)
        final_val_preds_per_seed.append(val_preds)
        mae_seed = float(np.mean(np.abs(y_val_margin.values - val_preds)))
        dir_seed = float(((val_preds > 0) == (y_val_margin.values > 0)).mean())
        print(f"  margin {i} (seed={seed}): val_mae={mae_seed:.3f} dir={dir_seed:.3f} "
              f"iters={m.best_iteration_} ({time.time() - ts:.0f}s)")

    ensemble_val_preds = np.mean(final_val_preds_per_seed, axis=0)
    y_val_arr = y_val_margin.values
    final_val = {
        "per_seed_mae": [float(np.mean(np.abs(y_val_arr - p))) for p in final_val_preds_per_seed],
        "per_seed_mean_mae": float(np.mean([np.mean(np.abs(y_val_arr - p)) for p in final_val_preds_per_seed])),
        "ensemble_mae": float(np.mean(np.abs(y_val_arr - ensemble_val_preds))),
        "ensemble_rmse": float(np.sqrt(np.mean((y_val_arr - ensemble_val_preds) ** 2))),
        "ensemble_direction_acc": float(((ensemble_val_preds > 0) == (y_val_arr > 0)).mean()),
        "val_period": val_years_range,
        "val_n": int(len(y_val_arr)),
    }
    print(f"\n  Final holdout ({val_years_range}, n={len(y_val_arr)}):")
    print(f"    per-seed mean MAE:      {final_val['per_seed_mean_mae']:.3f}")
    print(f"    ensemble MAE:           {final_val['ensemble_mae']:.3f}")
    print(f"    ensemble RMSE:          {final_val['ensemble_rmse']:.3f}")
    print(f"    ensemble direction acc: {final_val['ensemble_direction_acc']:.3f}")

    # -------------------- report --------------------
    print(f"\n[5/6] Writing report -> {REPORT_PATH}")
    report = {
        "seeds": SEEDS,
        "n_features": len(features),
        "features": features,
        "hyperparams": params,
        "wf_years": wf_years,
        "wf_cv": {
            "per_seed": {str(seed): per_seed_pooled[seed]["metrics"] for seed in SEEDS},
            "ensemble": ensemble_metrics,
            "seed_mean_mae": seed_mean,
            "seed_std_mae": seed_std,
            "seed_range_mae": seed_range,
            "ensemble_delta_vs_seed_mean": ensemble_metrics["mae"] - seed_mean,
            "per_fold": per_fold_rows,
        },
        "final_holdout": final_val,
        "saved_models": saved_paths,
        "training_time_sec": time.time() - t_start,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote {REPORT_PATH}")

    print(f"\n[6/6] Total time: {time.time() - t_start:.0f}s")
    print("=" * 80)
    print("DONE -- margin ensemble artifacts written to data/ensemble/margin_model_*.cbm")
    print("Next: wire load_margin_ensemble() into predict.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
