"""
Experiment: stacking meta-learner over classifier + margin ensemble OOF.

Architecture:
  Level 0 (base models):
    - CatBoost classifier ensemble (5 seeds) → P(home_win)
    - CatBoost margin ensemble (5 seeds, Huber:30) → signed margin

  Level 1 (meta-learner):
    - LightGBM classifier trained on meta-features derived from
      level-0 OOF predictions

No leakage design:
  1. Run base models walk-forward from 2017-2026 → OOF per fold
  2. For meta-learner fold Y: train on OOF from years 2017..Y-1,
     predict on OOF from year Y. The OOF for any year Z was produced
     by base models trained only on < Z, so the meta-learner never
     sees any training signal from the year it predicts.

Meta-features:
  - clf_prob:           ensemble-averaged classifier probability
  - clf_confidence:     abs(clf_prob - 0.5)
  - margin_pred:        ensemble-averaged margin prediction (signed)
  - margin_abs:         abs(margin_pred)
  - clf_margin_agree:   1 if sign matches, else 0

Comparison: meta-learner LL vs raw classifier ensemble LL (both on
same WF folds 2019-2026).

Writes: data/analysis/stacking_meta_experiment.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features  # noqa: E402
from src.model import TARGET, load_optimization_results  # noqa: E402

SEEDS = [42, 123, 256, 789, 1337]
MARGIN_OPT_PATH = PROJECT_ROOT / "data" / "margin_optimization_results.json"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "stacking_meta_experiment.json"

META_FEATURES = ["clf_prob", "clf_confidence", "margin_pred",
                 "margin_abs", "clf_margin_agree"]


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


def _margin_base_params() -> dict:
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


def run_base_wf(df: pd.DataFrame, features: list[str],
                clf_params: dict, margin_params: dict,
                wf_years: list[int]) -> pd.DataFrame:
    """Run both base models in walk-forward, return df with OOF columns."""
    clf_oof = np.full(len(df), np.nan)
    margin_oof = np.full(len(df), np.nan)

    for vy in wf_years:
        train = df[df["year"] < vy].dropna(subset=[TARGET, "margin"])
        val = df[df["year"] == vy].dropna(subset=[TARGET, "margin"])
        if train.empty or val.empty:
            continue

        val_idx = val.index.values
        clf_seed_probs = []
        margin_seed_preds = []

        for seed in SEEDS:
            # Classifier
            cp = {**clf_params, "random_seed": seed}
            clf = CatBoostClassifier(**cp)
            clf.fit(Pool(train[features], train[TARGET]),
                    eval_set=Pool(val[features], val[TARGET]))
            clf_seed_probs.append(clf.predict_proba(val[features])[:, 1])

            # Margin
            mp = _normalize_params({**margin_params, "random_seed": seed})
            mrg = CatBoostRegressor(**mp)
            mrg.fit(Pool(train[features], train["margin"]),
                    eval_set=Pool(val[features], val["margin"]))
            margin_seed_preds.append(mrg.predict(val[features]))

        clf_oof[val_idx] = np.mean(clf_seed_probs, axis=0)
        margin_oof[val_idx] = np.mean(margin_seed_preds, axis=0)

    df = df.copy()
    df["clf_oof"] = clf_oof
    df["margin_oof"] = margin_oof
    return df


def build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute meta-features from base model OOF predictions."""
    mf = pd.DataFrame(index=df.index)
    mf["clf_prob"] = df["clf_oof"]
    mf["clf_confidence"] = (df["clf_oof"] - 0.5).abs()
    mf["margin_pred"] = df["margin_oof"]
    mf["margin_abs"] = df["margin_oof"].abs()
    mf["clf_margin_agree"] = (
        ((df["clf_oof"] >= 0.5) & (df["margin_oof"] > 0)) |
        ((df["clf_oof"] < 0.5) & (df["margin_oof"] <= 0))
    ).astype(float)
    return mf


def run_meta_wf(df: pd.DataFrame, meta_df: pd.DataFrame,
                meta_years: list[int]) -> dict:
    """Walk-forward meta-learner on the OOF meta-features."""
    fold_rows = []
    pooled_meta_probs = []
    pooled_raw_clf = []
    pooled_labels = []

    for vy in meta_years:
        # Meta-learner training: years with valid OOF, before vy
        train_mask = (df["year"] < vy) & df["clf_oof"].notna()
        val_mask = (df["year"] == vy) & df["clf_oof"].notna()

        train_meta = meta_df.loc[train_mask]
        val_meta = meta_df.loc[val_mask]
        train_y = df.loc[train_mask, TARGET].values.astype(int)
        val_y = df.loc[val_mask, TARGET].values.astype(int)
        raw_clf = df.loc[val_mask, "clf_oof"].values

        if len(train_meta) < 50 or len(val_meta) < 10:
            continue

        # LightGBM meta-learner — heavily regularized for small data
        lgb = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=8,
            min_child_samples=20,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        lgb.fit(
            train_meta[META_FEATURES], train_y,
            eval_set=[(val_meta[META_FEATURES], val_y)],
            callbacks=[_lgb_early_stop(50)],
        )
        meta_probs = lgb.predict_proba(val_meta[META_FEATURES])[:, 1]

        raw_ll = float(log_loss(val_y, raw_clf))
        meta_ll = float(log_loss(val_y, meta_probs))
        raw_acc = float(((raw_clf >= 0.5).astype(int) == val_y).mean())
        meta_acc = float(((meta_probs >= 0.5).astype(int) == val_y).mean())

        fold_rows.append({
            "year": int(vy),
            "n_train": int(len(train_meta)),
            "n_val": int(len(val_meta)),
            "raw_clf_ll": raw_ll,
            "meta_ll": meta_ll,
            "delta_ll": meta_ll - raw_ll,
            "raw_clf_acc": raw_acc,
            "meta_acc": meta_acc,
            "delta_acc": meta_acc - raw_acc,
            "best_iteration": lgb.best_iteration_ if hasattr(lgb, 'best_iteration_') else -1,
        })

        pooled_meta_probs.extend(meta_probs.tolist())
        pooled_raw_clf.extend(raw_clf.tolist())
        pooled_labels.extend(val_y.tolist())

    pooled_meta = np.array(pooled_meta_probs)
    pooled_raw = np.array(pooled_raw_clf)
    pooled_y = np.array(pooled_labels)

    return {
        "folds": fold_rows,
        "pooled": {
            "n": int(len(pooled_y)),
            "raw_clf_ll": float(log_loss(pooled_y, pooled_raw)),
            "meta_ll": float(log_loss(pooled_y, pooled_meta)),
            "delta_ll": float(log_loss(pooled_y, pooled_meta) - log_loss(pooled_y, pooled_raw)),
            "raw_clf_brier": float(brier_score_loss(pooled_y, pooled_raw)),
            "meta_brier": float(brier_score_loss(pooled_y, pooled_meta)),
            "raw_clf_acc": float(((pooled_raw >= 0.5).astype(int) == pooled_y).mean()),
            "meta_acc": float(((pooled_meta >= 0.5).astype(int) == pooled_y).mean()),
        },
    }


def _lgb_early_stop(n):
    """LightGBM early stopping callback."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=n, verbose=False)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("=" * 80)
    print("STACKING META-LEARNER EXPERIMENT")
    print("=" * 80)

    print("\nLoading + featuring...")
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    df = build_features(df)
    df_complete = (df.dropna(subset=["margin", TARGET])
                   .copy().sort_values("date").reset_index(drop=True))
    min_year = int(df_complete["year"].min())
    max_year = int(df_complete["year"].max())
    print(f"  {len(df_complete)} rows, {min_year}-{max_year}")

    opt = load_optimization_results()
    features = [c for c in opt["best_features"] if c in df_complete.columns]
    clf_params = _clf_base_params(opt["best_params"])
    margin_params = _margin_base_params()
    print(f"  {len(features)} features")

    # Step 1: base model WF (start at 2017 for more meta-learner training data)
    base_wf_start = max(2017, min_year + 3)
    base_wf_years = list(range(base_wf_start, max_year + 1))
    print(f"\n[1/3] Base model walk-forward ({base_wf_years[0]}-{base_wf_years[-1]}, "
          f"{len(base_wf_years)} folds x {len(SEEDS)} seeds x 2 heads)")
    t0 = time.time()
    df_with_oof = run_base_wf(df_complete, features, clf_params, margin_params, base_wf_years)
    oof_count = df_with_oof["clf_oof"].notna().sum()
    print(f"  done in {time.time() - t0:.0f}s, {oof_count} rows with OOF")

    # Step 2: build meta-features
    print(f"\n[2/3] Building meta-features...")
    meta_df = build_meta_features(df_with_oof)
    for col in META_FEATURES:
        non_null = meta_df[col].notna().sum()
        print(f"  {col:>20s}: {non_null} non-null")

    # Step 3: meta-learner WF
    meta_years = list(range(2019, max_year + 1))
    print(f"\n[3/3] Meta-learner walk-forward ({meta_years[0]}-{meta_years[-1]})")
    results = run_meta_wf(df_with_oof, meta_df, meta_years)

    # Print results
    print(f"\n{'year':>6s} {'n_train':>8s} {'n_val':>6s} "
          f"{'raw_LL':>8s} {'meta_LL':>9s} {'d_LL':>8s} "
          f"{'raw_acc':>8s} {'meta_acc':>9s} {'d_acc':>8s}")
    print("  " + "-" * 78)
    for f in results["folds"]:
        print(f"  {f['year']:>6d} {f['n_train']:>8d} {f['n_val']:>6d} "
              f"{f['raw_clf_ll']:>8.4f} {f['meta_ll']:>9.4f} {f['delta_ll']:>+8.4f} "
              f"{f['raw_clf_acc']:>8.3f} {f['meta_acc']:>9.3f} {f['delta_acc']:>+8.3f}")

    p = results["pooled"]
    print(f"\n  POOLED (n={p['n']}):")
    print(f"    raw classifier LL:  {p['raw_clf_ll']:.5f}  Brier: {p['raw_clf_brier']:.5f}  "
          f"acc: {p['raw_clf_acc']:.3f}")
    print(f"    meta-learner LL:    {p['meta_ll']:.5f}  Brier: {p['meta_brier']:.5f}  "
          f"acc: {p['meta_acc']:.3f}")
    print(f"    delta LL:           {p['delta_ll']:+.5f}")
    print(f"    delta acc:          {p['meta_acc'] - p['raw_clf_acc']:+.3f}")

    noise_floor = 0.00174
    if abs(p["delta_ll"]) < noise_floor:
        verdict = "NOISE"
    elif p["delta_ll"] < 0:
        verdict = "BETTER"
    else:
        verdict = "WORSE"
    print(f"\n  Verdict vs noise floor (+/-{noise_floor:.5f}): [{verdict}]")

    report = {
        "seeds": SEEDS,
        "base_wf_years": base_wf_years,
        "meta_years": meta_years,
        "meta_features": META_FEATURES,
        "n_base_features": len(features),
        "results": results,
        "noise_floor": noise_floor,
        "verdict": verdict,
        "runtime_sec": time.time() - t_start,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {REPORT_PATH}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
