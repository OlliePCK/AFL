"""
Experiment: two-stage margin prediction.

Instead of one regressor that predicts signed margin directly, split into:
  Stage 1: classifier ensemble (existing) predicts P(home_win)
  Stage 2: two regressors predict |margin| — one trained only on home-win
            rows, one only on away-win rows

At WF-test time:
  if clf_prob >= 0.5 → predict +|margin_hw_model|
  else               → predict -|margin_aw_model|

This forces the margin's sign to align with the classifier's call (0.725 acc)
and lets each regressor specialize in its regime.

Also tests:  two-stage with Huber:30 (best from loss experiment)

Writes: data/analysis/two_stage_margin_experiment.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features  # noqa: E402
from src.model import TARGET, load_optimization_results  # noqa: E402

SEEDS = [42, 123, 256, 789, 1337]
MARGIN_OPT_PATH = PROJECT_ROOT / "data" / "margin_optimization_results.json"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "two_stage_margin_experiment.json"


def _normalize_params(p: dict) -> dict:
    out = dict(p)
    bt = out.get("bootstrap_type", "Bayesian")
    if bt == "Bayesian":
        out.pop("subsample", None)
    else:
        out.pop("bagging_temperature", None)
    return out


def _clf_base_params(opt_params: dict) -> dict:
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


def _margin_base_params(loss_override: dict | None = None) -> dict:
    with open(MARGIN_OPT_PATH) as f:
        opt = json.load(f)
    best = opt["tuned"]["best_params"]
    params = {
        "iterations": 2000,
        "early_stopping_rounds": 100,
        "eval_metric": "MAE",
        "loss_function": "MAE",
        "use_best_model": True,
        "verbose": 0,
        **best,
    }
    if loss_override:
        params.update(loss_override)
    return _normalize_params(params)


def run_baseline(df: pd.DataFrame, features: list[str], margin_params: dict,
                 wf_years: list[int]) -> dict:
    """Standard single-regressor margin (for comparison)."""
    per_seed_preds = {s: {} for s in SEEDS}
    fold_true = {}

    for vy in wf_years:
        train = df[df["year"] < vy].dropna(subset=["margin"])
        val = df[df["year"] == vy].dropna(subset=["margin"])
        if train.empty or val.empty:
            continue
        fold_true[vy] = val["margin"].values.astype(float)
        for seed in SEEDS:
            p = _normalize_params({**margin_params, "random_seed": seed})
            m = CatBoostRegressor(**p)
            m.fit(Pool(train[features], train["margin"]),
                  eval_set=Pool(val[features], val["margin"]))
            per_seed_preds[seed][vy] = m.predict(val[features])

    return _aggregate(per_seed_preds, fold_true, wf_years)


def run_two_stage(df: pd.DataFrame, features: list[str],
                  clf_params: dict, margin_params: dict,
                  wf_years: list[int]) -> dict:
    """Two-stage: classifier sign + conditional |margin| regressors."""
    fold_true = {}
    ensemble_preds = {}

    for vy in wf_years:
        train = df[df["year"] < vy].dropna(subset=["margin", TARGET])
        val = df[df["year"] == vy].dropna(subset=["margin", TARGET])
        if train.empty or val.empty:
            continue
        fold_true[vy] = val["margin"].values.astype(float)

        # Split train into home-win and away-win
        train_hw = train[train["margin"] > 0].copy()
        train_aw = train[train["margin"] <= 0].copy()
        train_hw["abs_margin"] = train_hw["margin"].abs()
        train_aw["abs_margin"] = train_aw["margin"].abs()

        seed_preds = []
        for seed in SEEDS:
            # Stage 1: classifier for sign
            cp = {**clf_params, "random_seed": seed}
            clf = CatBoostClassifier(**cp)
            clf.fit(Pool(train[features], train[TARGET]),
                    eval_set=Pool(val[features], val[TARGET]))
            clf_probs = clf.predict_proba(val[features])[:, 1]

            # Stage 2a: |margin| regressor for home-wins
            mp = _normalize_params({**margin_params, "random_seed": seed})
            m_hw = CatBoostRegressor(**mp)
            m_hw.fit(Pool(train_hw[features], train_hw["abs_margin"]),
                     eval_set=Pool(val[features], val["margin"].abs()))
            hw_preds = m_hw.predict(val[features])

            # Stage 2b: |margin| regressor for away-wins
            m_aw = CatBoostRegressor(**mp)
            m_aw.fit(Pool(train_aw[features], train_aw["abs_margin"]),
                     eval_set=Pool(val[features], val["margin"].abs()))
            aw_preds = m_aw.predict(val[features])

            # Combine: classifier sign × conditional magnitude
            is_home = clf_probs >= 0.5
            signed = np.where(is_home, np.abs(hw_preds), -np.abs(aw_preds))
            seed_preds.append(signed)

        ensemble_preds[vy] = np.mean(seed_preds, axis=0)

    return _aggregate_direct(ensemble_preds, fold_true, wf_years)


def _aggregate(per_seed_preds, fold_true, wf_years) -> dict:
    """Aggregate seed-averaged preds."""
    ensemble_preds = {}
    for vy in wf_years:
        stacks = [per_seed_preds[s][vy] for s in SEEDS if vy in per_seed_preds[s]]
        if stacks:
            ensemble_preds[vy] = np.mean(stacks, axis=0)
    return _aggregate_direct(ensemble_preds, fold_true, wf_years)


def _aggregate_direct(ensemble_preds, fold_true, wf_years) -> dict:
    all_preds = np.concatenate([ensemble_preds[y] for y in wf_years if y in ensemble_preds])
    all_true = np.concatenate([fold_true[y] for y in wf_years if y in fold_true])
    per_fold = []
    for vy in wf_years:
        if vy not in ensemble_preds:
            continue
        p, t = ensemble_preds[vy], fold_true[vy]
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
    print("TWO-STAGE MARGIN EXPERIMENT")
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
    clf_params = _clf_base_params(opt["best_params"])

    wf_years = list(range(max(2019, min_year + 5), max_year + 1))
    print(f"  {len(features)} features, WF years: {wf_years}")

    configs = {}

    # Baseline: single regressor, MAE
    print("\n==> baseline (MAE)")
    t0 = time.time()
    margin_mae = _margin_base_params()
    configs["baseline_MAE"] = run_baseline(df_complete, features, margin_mae, wf_years)
    print(f"   MAE={configs['baseline_MAE']['mae']:.3f} "
          f"dir={configs['baseline_MAE']['direction_acc']:.4f} ({time.time()-t0:.0f}s)")

    # Baseline: single regressor, Huber:30
    print("\n==> baseline (Huber:30)")
    t0 = time.time()
    margin_huber = _margin_base_params({"loss_function": "Huber:delta=30"})
    configs["baseline_Huber30"] = run_baseline(df_complete, features, margin_huber, wf_years)
    print(f"   MAE={configs['baseline_Huber30']['mae']:.3f} "
          f"dir={configs['baseline_Huber30']['direction_acc']:.4f} ({time.time()-t0:.0f}s)")

    # Two-stage: MAE
    print("\n==> two-stage (MAE)")
    t0 = time.time()
    configs["twostage_MAE"] = run_two_stage(
        df_complete, features, clf_params, margin_mae, wf_years
    )
    print(f"   MAE={configs['twostage_MAE']['mae']:.3f} "
          f"dir={configs['twostage_MAE']['direction_acc']:.4f} ({time.time()-t0:.0f}s)")

    # Two-stage: Huber:30
    print("\n==> two-stage (Huber:30)")
    t0 = time.time()
    configs["twostage_Huber30"] = run_two_stage(
        df_complete, features, clf_params, margin_huber, wf_years
    )
    print(f"   MAE={configs['twostage_Huber30']['mae']:.3f} "
          f"dir={configs['twostage_Huber30']['direction_acc']:.4f} ({time.time()-t0:.0f}s)")

    # Summary
    bl = configs["baseline_MAE"]
    print("\n" + "=" * 80)
    print(f"{'config':<22s} {'MAE':>8s} {'d_MAE':>8s} {'RMSE':>8s} {'dir':>8s} {'d_dir':>8s}")
    print("  " + "-" * 62)
    for name, r in configs.items():
        d_mae = r["mae"] - bl["mae"]
        d_dir = r["direction_acc"] - bl["direction_acc"]
        print(f"  {name:<22s} {r['mae']:>8.3f} {d_mae:>+8.3f} "
              f"{r['rmse']:>8.3f} {r['direction_acc']:>8.4f} {d_dir:>+8.4f}")

    print("\n  Per-fold direction accuracy:")
    print(f"  {'year':>6s}", end="")
    for name in configs:
        print(f" {name:>22s}", end="")
    print()
    for fold_i in range(len(wf_years)):
        yr = wf_years[fold_i]
        print(f"  {yr:>6d}", end="")
        for name in configs:
            folds = configs[name]["per_fold"]
            if fold_i < len(folds):
                print(f" {folds[fold_i]['direction_acc']:>22.3f}", end="")
        print()

    report = {
        "seeds": SEEDS,
        "wf_years": wf_years,
        "n_features": len(features),
        "results": configs,
        "runtime_sec": time.time() - t_start,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {REPORT_PATH}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
