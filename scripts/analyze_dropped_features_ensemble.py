"""
Item A re-run with 5-seed ensemble averaging.

Same A/B test as analyze_dropped_features.py, but for each (config, fold)
we train 5 CatBoost classifiers with different random seeds and average
their OOF probabilities before computing log loss. This mirrors how the
production ensemble scores matches, so the observed deltas are now
directly comparable to the ~0.001 effective noise floor instead of the
~0.0025 single-seed noise floor that swamped the first analysis.

Configs tested (all use the Optuna-winning hyperparams):
  1. baseline       -- Optuna winner (no weather, no h2h)
  2. baseline+weather
  3. baseline+h2h
  4. baseline+both

Writes: data/analysis/dropped_features_ensemble_report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import BETTING_FEATURE_COLS, FEATURE_GROUPS, TARGET  # noqa: E402

VAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEEDS = [42, 123, 256, 789, 1337]


def load_best_params() -> tuple[dict, list[str]]:
    with open(PROJECT_ROOT / "data" / "optimization_results.json") as f:
        opt = json.load(f)
    params = dict(opt["best_params"])
    params.update({
        "iterations": 1500,
        "early_stopping_rounds": 80,
        "verbose": 0,
        "eval_metric": "Logloss",
        "use_best_model": True,
    })
    if params.get("subsample", 1.0) < 1.0:
        params["bootstrap_type"] = "Bernoulli"
    return params, list(opt["best_feature_groups"])


def features_for_groups(groups: list[str]) -> list[str]:
    cols = list(BETTING_FEATURE_COLS)
    for g in groups:
        cols.extend(FEATURE_GROUPS[g])
    return cols


def run_wf_cv_ensemble(df: pd.DataFrame, feature_cols: list[str],
                        base_params: dict) -> dict:
    """Walk-forward CV with 5-seed ensemble averaging per fold."""
    available = [c for c in feature_cols if c in df.columns]
    fold_rows = []
    pooled_ensemble_probs, pooled_labels = [], []
    pooled_single_seed_probs: dict[int, list] = {s: [] for s in SEEDS}

    for val_year in VAL_YEARS:
        train_df = df[df["year"] < val_year].dropna(subset=[TARGET])
        val_df = df[df["year"] == val_year].dropna(subset=[TARGET])
        if train_df.empty or val_df.empty:
            continue

        y = val_df[TARGET].values.astype(float)
        seed_probs: list[np.ndarray] = []

        for seed in SEEDS:
            p = {**base_params, "random_seed": seed}
            m = CatBoostClassifier(**p)
            m.fit(
                Pool(train_df[available], train_df[TARGET]),
                eval_set=Pool(val_df[available], val_df[TARGET]),
            )
            probs = m.predict_proba(val_df[available])[:, 1]
            seed_probs.append(probs)
            pooled_single_seed_probs[seed].extend(probs.tolist())

        ensemble_probs = np.mean(seed_probs, axis=0)
        fold_seed_lls = [float(log_loss(y, p)) for p in seed_probs]
        fold_ensemble_ll = float(log_loss(y, ensemble_probs))

        fold_rows.append({
            "year": int(val_year),
            "n": int(len(val_df)),
            "seed_mean_ll": float(np.mean(fold_seed_lls)),
            "seed_std_ll": float(np.std(fold_seed_lls)),
            "ensemble_ll": fold_ensemble_ll,
        })

        pooled_ensemble_probs.extend(ensemble_probs.tolist())
        pooled_labels.extend(y.tolist())

    pooled_ensemble_probs = np.asarray(pooled_ensemble_probs)
    pooled_labels = np.asarray(pooled_labels)

    # Per-seed pooled LL (so we can see seed variance across the whole CV run)
    per_seed_pooled = {}
    for s in SEEDS:
        if pooled_single_seed_probs[s]:
            per_seed_pooled[s] = float(log_loss(
                pooled_labels, np.asarray(pooled_single_seed_probs[s])
            ))

    seed_mean = float(np.mean(list(per_seed_pooled.values())))
    seed_std = float(np.std(list(per_seed_pooled.values())))

    return {
        "n_features": len(available),
        "folds": fold_rows,
        "pooled_ensemble_log_loss": float(log_loss(pooled_labels, pooled_ensemble_probs)),
        "pooled_ensemble_brier": float(brier_score_loss(pooled_labels, pooled_ensemble_probs)),
        "pooled_ensemble_accuracy": float(((pooled_ensemble_probs >= 0.5).astype(int) == pooled_labels).mean()),
        "per_seed_pooled_log_loss": per_seed_pooled,
        "seed_mean_pooled_log_loss": seed_mean,
        "seed_std_pooled_log_loss": seed_std,
    }


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "master" / "afl_featured_dataset.csv"
    out_path = PROJECT_ROOT / "data" / "analysis" / "dropped_features_ensemble_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 80)
    print("ITEM A (ensemble) -- re-test weather + H2H using 5-seed averaging")
    print("=" * 80)

    print(f"\nLoading {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"], low_memory=False)
    print(f"  rows={len(df)}, years={df['year'].min()}-{df['year'].max()}")

    base_params, best_groups = load_best_params()
    print(f"Best feature groups from Optuna: {best_groups}")
    print(f"Seeds: {SEEDS}")
    print(f"Val years: {VAL_YEARS}")

    baseline_groups = list(best_groups)
    configs = {
        "baseline":         baseline_groups,
        "baseline+weather": baseline_groups + ["weather"],
        "baseline+h2h":     baseline_groups + ["h2h"],
        "baseline+both":    baseline_groups + ["weather", "h2h"],
    }

    results = {}
    for name, groups in configs.items():
        t0 = time.time()
        print(f"\n==> {name}  groups={groups}")
        cols = features_for_groups(groups)
        cv = run_wf_cv_ensemble(df, cols, base_params)
        results[name] = {"groups": groups, **cv}
        print(
            f"   ensemble pooled LL: {cv['pooled_ensemble_log_loss']:.5f}  "
            f"Brier: {cv['pooled_ensemble_brier']:.5f}  "
            f"acc: {cv['pooled_ensemble_accuracy']:.3f}  "
            f"({cv['n_features']} feats, {time.time() - t0:.0f}s)"
        )
        print(
            f"   seed pooled LL: mean={cv['seed_mean_pooled_log_loss']:.5f}  "
            f"std={cv['seed_std_pooled_log_loss']:.5f}"
        )
        per_seed_str = ", ".join(
            f"{s}:{v:.5f}" for s, v in cv["per_seed_pooled_log_loss"].items()
        )
        print(f"   per-seed: {per_seed_str}")

    baseline_ll = results["baseline"]["pooled_ensemble_log_loss"]
    noise_floor = results["baseline"]["seed_std_pooled_log_loss"]
    print("\n" + "=" * 80)
    print("Deltas vs baseline (ENSEMBLE log loss — NEGATIVE = better)")
    print(f"Noise floor (baseline seed std): +/- {noise_floor:.5f}")
    print("=" * 80)
    for name, r in results.items():
        delta = r["pooled_ensemble_log_loss"] - baseline_ll
        if abs(delta) < noise_floor:
            marker = "NOISE"
        elif delta < 0:
            marker = "BETTER"
        else:
            marker = "WORSE"
        print(f"   {name:22s} {delta:+.5f}  [{marker}]")

    report = {
        "seeds": SEEDS,
        "baseline_groups": baseline_groups,
        "best_params": base_params,
        "val_years": VAL_YEARS,
        "configs": results,
        "deltas_vs_baseline": {
            name: r["pooled_ensemble_log_loss"] - baseline_ll for name, r in results.items()
        },
        "noise_floor_seed_std": noise_floor,
        "runtime_sec": time.time() - t_start,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")
    print(f"Total runtime: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
