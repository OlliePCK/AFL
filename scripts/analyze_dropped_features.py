"""
Item A analysis — test whether weather + H2H were correctly dropped by Optuna.

Reproduces the walk-forward CV used during tuning with 4 configurations:
  1. BASELINE  : Optuna's winning feature set (weather + h2h both off)
  2. +WEATHER  : baseline + weather group re-enabled
  3. +H2H      : baseline + h2h group re-enabled
  4. +BOTH     : baseline + both re-enabled

Uses the exact same hyperparameters as the Optuna winner (from
data/optimization_results.json) and the same walk-forward years, so any
delta we observe is attributable purely to the dropped feature groups.

Also trains one "all features" model and dumps its top-40 SHAP importances
so we can see where weather + H2H features actually rank.

Read-only — writes analysis output to data/analysis/dropped_features_report.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import BETTING_FEATURE_COLS, FEATURE_GROUPS, TARGET  # noqa: E402

VAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def load_best_params() -> tuple[dict, list[str]]:
    with open(PROJECT_ROOT / "data" / "optimization_results.json") as f:
        opt = json.load(f)
    params = dict(opt["best_params"])
    params.update({
        "iterations": 1500,
        "early_stopping_rounds": 80,
        "random_seed": 42,
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


def run_wf_cv(df: pd.DataFrame, feature_cols: list[str], params: dict) -> dict:
    """Walk-forward CV identical to the Optuna objective — returns per-fold + pooled."""
    available = [c for c in feature_cols if c in df.columns]
    fold_rows, all_probs, all_labels = [], [], []

    for val_year in VAL_YEARS:
        train_df = df[df["year"] < val_year].dropna(subset=[TARGET])
        val_df = df[df["year"] == val_year].dropna(subset=[TARGET])
        if train_df.empty or val_df.empty:
            continue

        model = CatBoostClassifier(**params)
        model.fit(
            Pool(train_df[available], train_df[TARGET]),
            eval_set=Pool(val_df[available], val_df[TARGET]),
        )

        probs = model.predict_proba(val_df[available])[:, 1]
        y = val_df[TARGET].values.astype(float)

        fold_rows.append({
            "year": int(val_year),
            "n": len(val_df),
            "log_loss": float(log_loss(y, probs)),
            "brier": float(brier_score_loss(y, probs)),
            "accuracy": float(((probs >= 0.5).astype(int) == y).mean()),
        })
        all_probs.extend(probs.tolist())
        all_labels.extend(y.tolist())

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)
    return {
        "pooled_log_loss": float(log_loss(all_labels, all_probs)),
        "pooled_brier": float(brier_score_loss(all_labels, all_probs)),
        "pooled_accuracy": float(((all_probs >= 0.5).astype(int) == all_labels).mean()),
        "n_features": len(available),
        "folds": fold_rows,
    }


def run_importance_probe(df: pd.DataFrame, params: dict) -> list[dict]:
    """Train one CatBoost on ALL feature groups, dump top-40 importances."""
    all_groups = list(FEATURE_GROUPS.keys())
    cols = features_for_groups(all_groups)
    available = [c for c in cols if c in df.columns]

    # Train/val split — use 2019-2022 as train, 2023 as val (what Optuna trial 0 would see)
    train_df = df[df["year"] <= 2022].dropna(subset=[TARGET])
    val_df = df[df["year"] == 2023].dropna(subset=[TARGET])

    p = dict(params)
    p["verbose"] = 0
    model = CatBoostClassifier(**p)
    model.fit(
        Pool(train_df[available], train_df[TARGET]),
        eval_set=Pool(val_df[available], val_df[TARGET]),
    )

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    top = []
    for rank, idx in enumerate(order[:40], start=1):
        name = available[idx]
        # Tag which group this feature belongs to, if any
        group = "base"
        for g, members in FEATURE_GROUPS.items():
            if name in members:
                group = g
                break
        if name in BETTING_FEATURE_COLS:
            group = "base"
        top.append({
            "rank": rank,
            "feature": name,
            "importance": float(importances[idx]),
            "group": group,
        })
    return top


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "master" / "afl_featured_dataset.csv"
    out_path = PROJECT_ROOT / "data" / "analysis" / "dropped_features_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"], low_memory=False)
    print(f"  rows={len(df)}, years={df['year'].min()}-{df['year'].max()}")

    best_params, best_groups = load_best_params()
    print(f"Best feature groups from Optuna: {best_groups}")
    print(f"Best params: {best_params}")

    baseline_groups = list(best_groups)  # [form_10g, rest, ladder, detailed_stats, season_context]
    configs = {
        "baseline":          baseline_groups,
        "baseline+weather":  baseline_groups + ["weather"],
        "baseline+h2h":      baseline_groups + ["h2h"],
        "baseline+both":     baseline_groups + ["weather", "h2h"],
    }

    results = {}
    for name, groups in configs.items():
        print(f"\n==> {name}  groups={groups}")
        cols = features_for_groups(groups)
        cv = run_wf_cv(df, cols, best_params)
        results[name] = {"groups": groups, **cv}
        print(
            f"   log_loss={cv['pooled_log_loss']:.5f} "
            f"brier={cv['pooled_brier']:.5f} "
            f"acc={cv['pooled_accuracy']:.3f} "
            f"({cv['n_features']} features)"
        )

    baseline_ll = results["baseline"]["pooled_log_loss"]
    print("\n==> Deltas vs baseline (NEGATIVE = better, POSITIVE = worse)")
    for name, r in results.items():
        delta = r["pooled_log_loss"] - baseline_ll
        marker = "BETTER" if delta < -0.0005 else ("WORSE" if delta > 0.0005 else "=")
        print(f"   {name:22s} {delta:+.5f}  [{marker}]")

    print("\n==> Training importance probe (all feature groups enabled)")
    importances = run_importance_probe(df, best_params)
    weather_cols = set(FEATURE_GROUPS["weather"])
    h2h_cols = set(FEATURE_GROUPS["h2h"])
    print("   top-40 features:")
    for row in importances:
        tag = ""
        if row["feature"] in weather_cols:
            tag = " <- WEATHER"
        elif row["feature"] in h2h_cols:
            tag = " <- H2H"
        print(f"     {row['rank']:3d}. {row['feature']:32s} {row['importance']:8.3f}{tag}")

    report = {
        "baseline_groups": baseline_groups,
        "best_params": best_params,
        "val_years": VAL_YEARS,
        "configs": results,
        "deltas_vs_baseline": {
            name: r["pooled_log_loss"] - baseline_ll for name, r in results.items()
        },
        "top_40_importances": importances,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
