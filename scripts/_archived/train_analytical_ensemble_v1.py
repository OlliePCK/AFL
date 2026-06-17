"""
ARCHIVED 2026-04-17 — superseded by scripts/train_analytical_odds_only.py (V3).

Why archived: this 47-feature trainer used implied_home_close, overround_close,
and home_line_close — all of which are NaN for upcoming matches because "close"
values are only known after a game starts. In production, CatBoost silently
fell back on the global prior and produced a near-constant ~0.58 for every
live prediction for months. The odds-only V3 model (3 features, open-line
proxies) matches V1's walk-forward LL (0.5853 vs 0.5853) and actually works
for live inference.

See data/analysis/analytical_model_decision_memo.md for the full decision
trail. Do NOT re-run this script against production artifact paths — V3 is
the current production model.

----

Train a 5-seed CatBoost classifier ensemble with odds features (analytical model).

Mirrors train_ensemble.py but adds the "odds" feature group:
  implied_home_close, overround_close, home_line_close

This model is NOT for value betting (it uses market data as input).
It is for tipping competitions, match analysis, and calibration reference.

Output artifacts:
  data/ensemble/analytical_model_0..4.cbm
  data/analytical_calibrator.pkl
  data/analytical_feature_schema.json
  data/analysis/analytical_ensemble_report.json
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

from src.model import TARGET, load_optimization_results, FEATURE_GROUPS  # noqa: E402
from src.features import build_features  # noqa: E402

SEEDS = [42, 123, 256, 789, 1337]
ENSEMBLE_DIR = PROJECT_ROOT / "data" / "ensemble"
CAL_PATH = PROJECT_ROOT / "data" / "analytical_calibrator.pkl"
SCHEMA_PATH = PROJECT_ROOT / "data" / "analytical_feature_schema.json"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "analytical_ensemble_report.json"


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
    p = {**params, "random_seed": seed}
    fold_probs: dict[int, np.ndarray] = {}
    fold_labels: dict[int, np.ndarray] = {}

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
        fold_probs[vy] = m.predict_proba(val_df[features])[:, 1]
        fold_labels[vy] = val_df[TARGET].values.astype(float)

    return {"probs_by_fold": fold_probs, "labels_by_fold": fold_labels}


def summarize(name: str, probs: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "config": name,
        "log_loss": float(log_loss(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "accuracy": float(((probs >= 0.5).astype(int) == labels).mean()),
        "n": int(len(labels)),
    }


def main() -> None:
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 80)
    print("ANALYTICAL ENSEMBLE -- 5-seed CatBoost with odds features")
    print("=" * 80)

    # Load data + features
    print("\n[1/6] Loading master dataset + building features...")
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    print(f"  loaded {len(df)} raw matches")
    df = build_features(df)
    df_complete = df.dropna(subset=[TARGET]).copy().sort_values("date").reset_index(drop=True)
    min_year = int(df_complete["year"].min())
    max_year = int(df_complete["year"].max())
    print(f"  featured: {len(df_complete)} completed matches, {min_year}-{max_year}")

    # Build analytical feature set: betting features + Optuna groups + odds
    opt = load_optimization_results()
    if opt is None:
        print("  FATAL: optimization_results.json not found.")
        sys.exit(1)

    # Start from the Optuna-winning feature set and add odds
    base_features = list(opt["best_features"])
    odds_features = FEATURE_GROUPS["odds"]
    features = base_features + [f for f in odds_features if f not in base_features]
    features = [c for c in features if c in df_complete.columns]
    params = _base_params(opt["best_params"])

    print(f"  base features: {len(base_features)} (Optuna winner)")
    print(f"  + odds features: {odds_features}")
    print(f"  total: {len(features)} features")

    # Save schema
    with open(SCHEMA_PATH, "w") as f:
        json.dump({"version": 1, "features": features, "target": TARGET,
                    "note": "analytical model with odds — NOT for value betting"}, f, indent=2)

    # Walk-forward CV
    wf_start = max(2019, min_year + 5)
    wf_years = list(range(wf_start, max_year + 1))
    print(f"\n[2/6] Walk-forward CV -- {len(SEEDS)} seeds x {len(wf_years)} folds")
    t0 = time.time()

    per_seed_oof: dict[int, dict] = {}
    for i, seed in enumerate(SEEDS, 1):
        ts = time.time()
        print(f"  seed {seed} ({i}/{len(SEEDS)})...", end="", flush=True)
        per_seed_oof[seed] = walk_forward_oof(df_complete, features, params, wf_years, seed)
        print(f" done ({time.time() - ts:.0f}s)")
    print(f"  [WF CV total: {time.time() - t0:.0f}s]")

    # Aggregate OOF
    print(f"\n[3/6] Aggregating OOF predictions...")
    per_seed_pooled = {}
    for seed, res in per_seed_oof.items():
        probs = np.concatenate([res["probs_by_fold"][y] for y in wf_years if y in res["probs_by_fold"]])
        labels = np.concatenate([res["labels_by_fold"][y] for y in wf_years if y in res["labels_by_fold"]])
        per_seed_pooled[seed] = {"probs": probs, "labels": labels,
                                 "metrics": summarize(f"seed_{seed}", probs, labels)}

    reference_labels = per_seed_pooled[SEEDS[0]]["labels"]
    ensemble_probs = np.mean([per_seed_pooled[s]["probs"] for s in SEEDS], axis=0)
    ensemble_metrics = summarize("ensemble_mean", ensemble_probs, reference_labels)

    seed_losses = [per_seed_pooled[s]["metrics"]["log_loss"] for s in SEEDS]
    seed_mean = float(np.mean(seed_losses))
    seed_std = float(np.std(seed_losses))

    per_fold_rows = []
    for vy in wf_years:
        y = per_seed_oof[SEEDS[0]]["labels_by_fold"][vy]
        fold_probs_by_seed = [per_seed_oof[s]["probs_by_fold"][vy] for s in SEEDS]
        ensemble_fold_probs = np.mean(fold_probs_by_seed, axis=0)
        per_fold_rows.append({
            "year": int(vy),
            "n": int(len(y)),
            "ensemble_log_loss": float(log_loss(y, ensemble_fold_probs)),
        })

    print(f"  single-seed pooled LL: mean={seed_mean:.5f}  std={seed_std:.5f}")
    print(f"  ensemble pooled LL:    {ensemble_metrics['log_loss']:.5f}")
    for row in per_fold_rows:
        print(f"    {row['year']}  n={row['n']:3d}  ensemble={row['ensemble_log_loss']:.4f}")

    # Calibrator
    print(f"\n[4/6] Fitting isotonic calibrator on ensemble-averaged OOF...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(ensemble_probs, reference_labels)
    cal_probs = calibrator.predict(ensemble_probs)
    cal_ll = log_loss(reference_labels, cal_probs)
    cal_brier = brier_score_loss(reference_labels, cal_probs)
    print(f"  calibrated LL: {cal_ll:.5f}  Brier: {cal_brier:.5f}")
    with open(CAL_PATH, "wb") as f:
        pickle.dump(calibrator, f)

    # Train production models
    val_size = 207
    train_split = df_complete.iloc[:-val_size]
    val_split = df_complete.iloc[-val_size:]
    print(f"\n[5/6] Training 5 production models...")
    print(f"  train: {len(train_split)} rows | val: {len(val_split)} rows")

    final_val_preds: list[np.ndarray] = []
    for i, seed in enumerate(SEEDS):
        p = {**params, "random_seed": seed}
        ts = time.time()
        m = CatBoostClassifier(**p)
        m.fit(Pool(train_split[features], train_split[TARGET]),
              eval_set=Pool(val_split[features], val_split[TARGET]))
        model_path = ENSEMBLE_DIR / f"analytical_model_{i}.cbm"
        m.save_model(str(model_path))
        val_probs = m.predict_proba(val_split[features])[:, 1]
        final_val_preds.append(val_probs)
        ll = float(log_loss(val_split[TARGET], val_probs))
        print(f"  model {i} (seed={seed}): val_LL={ll:.5f} iters={m.best_iteration_} ({time.time() - ts:.0f}s)")

    y_val = val_split[TARGET].values.astype(float)
    ens_val = np.mean(final_val_preds, axis=0)
    ens_val_cal = np.clip(calibrator.predict(ens_val), 0.02, 0.98)
    final_val = {
        "ensemble_log_loss_raw": float(log_loss(y_val, ens_val)),
        "ensemble_log_loss_calibrated": float(log_loss(y_val, ens_val_cal)),
        "ensemble_brier_calibrated": float(brier_score_loss(y_val, ens_val_cal)),
        "ensemble_accuracy": float(((ens_val_cal >= 0.5).astype(int) == y_val).mean()),
        "val_n": int(len(y_val)),
    }
    print(f"\n  Holdout (n={len(y_val)}):")
    print(f"    ensemble raw LL:  {final_val['ensemble_log_loss_raw']:.5f}")
    print(f"    ensemble cal LL:  {final_val['ensemble_log_loss_calibrated']:.5f}")
    print(f"    ensemble accuracy: {final_val['ensemble_accuracy']:.3f}")

    # Report
    print(f"\n[6/6] Writing report -> {REPORT_PATH}")
    report = {
        "model_type": "analytical (with odds)",
        "seeds": SEEDS,
        "n_features": len(features),
        "features": features,
        "odds_features": odds_features,
        "wf_cv": {
            "per_seed": {str(s): per_seed_pooled[s]["metrics"] for s in SEEDS},
            "ensemble": ensemble_metrics,
            "calibrated_ll": float(cal_ll),
            "seed_mean_ll": seed_mean,
            "seed_std_ll": seed_std,
            "per_fold": per_fold_rows,
        },
        "final_holdout": final_val,
        "training_time_sec": time.time() - t_start,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTotal time: {time.time() - t_start:.0f}s")
    print("=" * 80)
    print("DONE -- analytical ensemble saved to data/ensemble/analytical_model_*.cbm")
    print("=" * 80)


if __name__ == "__main__":
    main()
