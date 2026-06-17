"""
Item C — train a 5-seed CatBoost classifier ensemble.

Mirrors the logic of `src.model.retrain_production_model()` but:
  1. Trains 5 CatBoost classifiers with different random seeds (same
     Optuna-winning hyperparams and feature set).
  2. Runs walk-forward CV for EACH seed to collect per-seed OOF probs.
  3. Averages the OOF probs across seeds, then refits the isotonic
     calibrator on the AVERAGED ensemble OOF probs (critical — the
     calibrator must be fit on the same distribution it will be applied
     to at inference time).
  4. Trains 5 production models on the full train/val split and saves
     them to data/ensemble/betting_model_{i}.cbm.
  5. Writes a comparison report: single-seed vs ensemble log loss,
     round-level variance, Brier, accuracy, and the effective noise floor.

Output artifacts:
  data/ensemble/betting_model_0.cbm       (seed 42, same as current)
  data/ensemble/betting_model_1.cbm       (seed 123)
  data/ensemble/betting_model_2.cbm       (seed 256)
  data/ensemble/betting_model_3.cbm       (seed 789)
  data/ensemble/betting_model_4.cbm       (seed 1337)
  data/calibrator.pkl                     (refit on ensemble-averaged OOF)
  data/feature_schema.json                (unchanged, shared with single model)
  data/analysis/ensemble_report.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TARGET, load_optimization_results  # noqa: E402
from src.features import build_features  # noqa: E402

SEEDS = [42, 123, 256, 789, 1337]
ENSEMBLE_DIR = PROJECT_ROOT / "data" / "ensemble"
CAL_PATH = PROJECT_ROOT / "data" / "calibrator.pkl"
SCHEMA_PATH = PROJECT_ROOT / "data" / "feature_schema.json"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "ensemble_report.json"


def _base_params(opt_params: dict) -> dict:
    params = dict(opt_params)
    params.update({
        "iterations": 2000,
        "early_stopping_rounds": 100,
        "verbose": 0,
        "eval_metric": "Logloss",
        "use_best_model": True,
    })
    if params.get("subsample", 1.0) < 1.0:
        params["bootstrap_type"] = "Bernoulli"
    return params


def walk_forward_oof(df: pd.DataFrame, features: list[str], params: dict,
                     wf_years: list[int], seed: int) -> dict:
    """Walk-forward CV for a single seed. Returns per-fold metrics + OOF probs."""
    p = {**params, "random_seed": seed}
    fold_probs: dict[int, np.ndarray] = {}
    fold_labels: dict[int, np.ndarray] = {}
    fold_index: dict[int, np.ndarray] = {}  # preserves row identity across seeds

    for vy in wf_years:
        train_df = df[df["year"] < vy].dropna(subset=[TARGET])
        val_df = df[df["year"] == vy].dropna(subset=[TARGET])
        if train_df.empty or val_df.empty:
            continue
        m = CatBoostClassifier(**p)
        m.fit(
            Pool(train_df[features], train_df[TARGET]),
            eval_set=Pool(val_df[features], val_df[TARGET]),
        )
        probs = m.predict_proba(val_df[features])[:, 1]
        fold_probs[vy] = probs
        fold_labels[vy] = val_df[TARGET].values.astype(float)
        fold_index[vy] = val_df.index.values

    return {"probs_by_fold": fold_probs,
            "labels_by_fold": fold_labels,
            "index_by_fold": fold_index}


def summarize(name: str, probs: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "config": name,
        "log_loss": float(log_loss(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "accuracy": float(((probs >= 0.5).astype(int) == labels).mean()),
        "n": int(len(labels)),
    }


def fold_log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(log_loss(labels, probs))


def main() -> None:
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 80)
    print("ITEM C -- 5-seed CatBoost ensemble")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load data + features + optuna-winning config
    # ------------------------------------------------------------------
    print("\n[1/6] Loading master dataset + building features...")
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    print(f"  loaded {len(df)} raw matches")
    df = build_features(df)
    df_complete = df.dropna(subset=[TARGET]).copy().sort_values("date").reset_index(drop=True)
    min_year = int(df_complete["year"].min())
    max_year = int(df_complete["year"].max())
    print(f"  featured: {len(df_complete)} completed matches, {min_year}-{max_year}")

    opt = load_optimization_results()
    if opt is None:
        print("  FATAL: optimization_results.json not found. Run Optuna first.")
        sys.exit(1)
    features = [c for c in opt["best_features"] if c in df_complete.columns]
    params = _base_params(opt["best_params"])
    print(f"  using {len(features)} optuna-winning features")
    print(f"  using hyperparams from Optuna winner")

    # Persist schema (same as single model)
    with open(SCHEMA_PATH, "w") as f:
        json.dump({"version": 1, "features": features, "target": TARGET}, f, indent=2)

    # ------------------------------------------------------------------
    # Walk-forward CV for each seed
    # ------------------------------------------------------------------
    wf_start = max(2019, min_year + 5)
    wf_years = list(range(wf_start, max_year + 1))
    print(f"\n[2/6] Walk-forward CV — {len(SEEDS)} seeds × {len(wf_years)} folds "
          f"({wf_years[0]}-{wf_years[-1]}) = {len(SEEDS) * len(wf_years)} fits")
    t0 = time.time()

    per_seed_oof: dict[int, dict] = {}
    for i, seed in enumerate(SEEDS, 1):
        ts = time.time()
        print(f"  seed {seed} ({i}/{len(SEEDS)})...", end="", flush=True)
        per_seed_oof[seed] = walk_forward_oof(df_complete, features, params, wf_years, seed)
        print(f" done ({time.time() - ts:.0f}s)")
    print(f"  [WF CV total: {time.time() - t0:.0f}s]")

    # ------------------------------------------------------------------
    # Per-seed + ensemble pooled metrics
    # ------------------------------------------------------------------
    print(f"\n[3/6] Aggregating OOF predictions...")

    # Pool within each seed (concat across folds)
    per_seed_pooled = {}
    for seed, res in per_seed_oof.items():
        probs = np.concatenate([res["probs_by_fold"][y] for y in wf_years if y in res["probs_by_fold"]])
        labels = np.concatenate([res["labels_by_fold"][y] for y in wf_years if y in res["labels_by_fold"]])
        per_seed_pooled[seed] = {"probs": probs, "labels": labels,
                                 "metrics": summarize(f"seed_{seed}", probs, labels)}

    # Sanity: all seeds should have identical labels in identical order
    reference_labels = per_seed_pooled[SEEDS[0]]["labels"]
    for seed in SEEDS[1:]:
        assert np.array_equal(per_seed_pooled[seed]["labels"], reference_labels), \
            f"Seed {seed} has different label order"

    # Ensemble = mean of probabilities across seeds
    ensemble_probs = np.mean(
        [per_seed_pooled[seed]["probs"] for seed in SEEDS], axis=0
    )
    ensemble_metrics = summarize("ensemble_mean", ensemble_probs, reference_labels)

    # Per-seed log loss for variance calculation
    seed_losses = [per_seed_pooled[seed]["metrics"]["log_loss"] for seed in SEEDS]
    seed_mean = float(np.mean(seed_losses))
    seed_std = float(np.std(seed_losses))
    seed_range = float(max(seed_losses) - min(seed_losses))

    # Per-fold log loss with and without ensemble (to show variance reduction)
    per_fold_rows = []
    for vy in wf_years:
        y = per_seed_oof[SEEDS[0]]["labels_by_fold"][vy]
        seed_fold_lls = []
        fold_probs_by_seed = []
        for seed in SEEDS:
            p = per_seed_oof[seed]["probs_by_fold"][vy]
            seed_fold_lls.append(fold_log_loss(p, y))
            fold_probs_by_seed.append(p)
        ensemble_fold_probs = np.mean(fold_probs_by_seed, axis=0)
        ensemble_fold_ll = fold_log_loss(ensemble_fold_probs, y)
        per_fold_rows.append({
            "year": int(vy),
            "n": int(len(y)),
            "seed_mean_log_loss": float(np.mean(seed_fold_lls)),
            "seed_std_log_loss": float(np.std(seed_fold_lls)),
            "ensemble_log_loss": float(ensemble_fold_ll),
        })

    print(f"  single-seed pooled LL: mean={seed_mean:.5f}  std={seed_std:.5f}  range={seed_range:.5f}")
    print(f"  ensemble pooled LL:    {ensemble_metrics['log_loss']:.5f}")
    print(f"  ensemble delta vs seed-mean: {ensemble_metrics['log_loss'] - seed_mean:+.5f}")
    for row in per_fold_rows:
        print(
            f"    {row['year']}  n={row['n']:3d}  "
            f"seed_mean={row['seed_mean_log_loss']:.4f} (+/-{row['seed_std_log_loss']:.4f})  "
            f"ensemble={row['ensemble_log_loss']:.4f}  "
            f"delta={row['ensemble_log_loss'] - row['seed_mean_log_loss']:+.4f}"
        )

    # ------------------------------------------------------------------
    # Fit calibrator on ensemble-averaged OOF probs
    # ------------------------------------------------------------------
    print(f"\n[4/6] Fitting isotonic calibrator on ensemble-averaged OOF...")
    raw_brier = brier_score_loss(reference_labels, ensemble_probs)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(ensemble_probs, reference_labels)
    cal_probs = calibrator.predict(ensemble_probs)
    cal_brier = brier_score_loss(reference_labels, cal_probs)
    cal_ll = log_loss(reference_labels, cal_probs)
    print(f"  Brier: {raw_brier:.5f} -> {cal_brier:.5f} (delta {cal_brier - raw_brier:+.5f})")
    print(f"  Log loss (calibrated): {cal_ll:.5f} (raw ensemble: {ensemble_metrics['log_loss']:.5f})")
    with open(CAL_PATH, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"  saved calibrator -> {CAL_PATH}")

    # ------------------------------------------------------------------
    # Train 5 final production models (train = all but last 207, val = last 207)
    # ------------------------------------------------------------------
    val_size = 207
    train_split = df_complete.iloc[:-val_size]
    val_split = df_complete.iloc[-val_size:]
    train_years = f"{min_year}-{int(train_split['year'].max())}"
    val_years_range = f"{int(val_split['year'].min())}-{int(val_split['year'].max())}"
    print(f"\n[5/6] Training 5 production models on train ({train_years}), val ({val_years_range})")
    print(f"  train: {len(train_split)} rows | val: {len(val_split)} rows")

    X_train = train_split[features]
    y_train = train_split[TARGET]
    X_val = val_split[features]
    y_val = val_split[TARGET]

    final_val_preds_per_seed: list[np.ndarray] = []
    saved_paths: list[str] = []
    for i, seed in enumerate(SEEDS):
        p = {**params, "random_seed": seed}
        ts = time.time()
        m = CatBoostClassifier(**p)
        m.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val))
        model_path = ENSEMBLE_DIR / f"betting_model_{i}.cbm"
        m.save_model(str(model_path))
        saved_paths.append(str(model_path))
        val_probs = m.predict_proba(X_val)[:, 1]
        final_val_preds_per_seed.append(val_probs)
        ll_seed = float(log_loss(y_val, val_probs))
        print(f"  model {i} (seed={seed}): val_log_loss={ll_seed:.5f} "
              f"iters={m.best_iteration_} ({time.time() - ts:.0f}s)")

    ensemble_val_probs = np.mean(final_val_preds_per_seed, axis=0)
    ensemble_val_cal = np.clip(calibrator.predict(ensemble_val_probs), 0.02, 0.98)
    final_val = {
        "per_seed_log_loss": [float(log_loss(y_val, p)) for p in final_val_preds_per_seed],
        "per_seed_mean": float(np.mean([log_loss(y_val, p) for p in final_val_preds_per_seed])),
        "ensemble_log_loss_raw": float(log_loss(y_val, ensemble_val_probs)),
        "ensemble_log_loss_calibrated": float(log_loss(y_val, ensemble_val_cal)),
        "ensemble_brier_raw": float(brier_score_loss(y_val, ensemble_val_probs)),
        "ensemble_brier_calibrated": float(brier_score_loss(y_val, ensemble_val_cal)),
        "ensemble_accuracy": float(((ensemble_val_cal >= 0.5).astype(int) == y_val.values).mean()),
        "val_period": val_years_range,
        "val_n": int(len(y_val)),
    }
    print(f"\n  Final holdout ({val_years_range}, n={len(y_val)}):")
    print(f"    per-seed mean LL: {final_val['per_seed_mean']:.5f}")
    print(f"    ensemble raw LL:  {final_val['ensemble_log_loss_raw']:.5f}")
    print(f"    ensemble cal LL:  {final_val['ensemble_log_loss_calibrated']:.5f}")
    print(f"    ensemble cal Brier: {final_val['ensemble_brier_calibrated']:.5f}")
    print(f"    ensemble accuracy: {final_val['ensemble_accuracy']:.3f}")

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    print(f"\n[6/6] Writing report -> {REPORT_PATH}")
    report = {
        "seeds": SEEDS,
        "n_features": len(features),
        "features": features,
        "hyperparams": params,
        "wf_years": wf_years,
        "wf_cv": {
            "per_seed": {str(seed): per_seed_pooled[seed]["metrics"] for seed in SEEDS},
            "ensemble": ensemble_metrics,
            "ensemble_calibrated": {
                "log_loss": float(cal_ll),
                "brier": float(cal_brier),
                "n": int(len(reference_labels)),
            },
            "seed_mean_log_loss": seed_mean,
            "seed_std_log_loss": seed_std,
            "seed_range_log_loss": seed_range,
            "ensemble_delta_vs_seed_mean": ensemble_metrics["log_loss"] - seed_mean,
            "per_fold": per_fold_rows,
        },
        "final_holdout": final_val,
        "saved_models": saved_paths,
        "calibrator_path": str(CAL_PATH),
        "training_time_sec": time.time() - t_start,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote {REPORT_PATH}")

    print(f"\nTotal time: {time.time() - t_start:.0f}s")
    print("=" * 80)
    print("DONE — ensemble artifacts written to data/ensemble/, calibrator refit.")
    print("predict.py will auto-detect the ensemble on next run.")
    print("=" * 80)


if __name__ == "__main__":
    main()
