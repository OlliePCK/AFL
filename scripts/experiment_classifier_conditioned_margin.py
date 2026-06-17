"""
Experiment: does the classifier's predicted home-win probability, used as
an extra feature, improve the margin model's directional accuracy?

The margin model currently has 0.686 directional accuracy on the holdout
while the classifier has 0.725. That gap suggests the margin model is
under-using the "who wins" signal the classifier captures. This script
tests whether feeding the classifier's OOF probability into the margin
head closes that gap.

Design (no leakage):
  1. Run walk-forward CV for the CLASSIFIER with a single seed, producing
     OOF probs for every year in the WF range. The OOF prob for year Y
     comes from a model that trained only on < Y, so using it as a
     feature on year Y introduces no leakage.
  2. Run walk-forward CV for the MARGIN head TWICE -- same single seed,
     same training rows, same hyperparams -- once baseline (no clf_prob)
     and once with clf_prob appended. Compare MAE, RMSE, direction
     accuracy, per fold.
  3. Report the delta. If direction improves meaningfully (> +0.01) on
     the 2019-2025 WF with no MAE regression, it's worth wiring into
     the production pipeline.

Output:
  data/analysis/classifier_conditioned_margin.json
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
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "classifier_conditioned_margin.json"
MARGIN_OPT_PATH = PROJECT_ROOT / "data" / "margin_optimization_results.json"


def _normalize_margin_params(p: dict) -> dict:
    """Strip the bootstrap param that doesn't match the chosen bootstrap_type."""
    out = dict(p)
    bt = out.get("bootstrap_type", "Bayesian")
    if bt == "Bayesian":
        out.pop("subsample", None)
    else:
        out.pop("bagging_temperature", None)
    return out


def _classifier_base_params(opt_params: dict) -> dict:
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
        "loss_function": "MAE",
        "use_best_model": True,
        "verbose": 0,
        **best,
    }
    return _normalize_margin_params(params)


def run_classifier_wf_ensemble(df: pd.DataFrame, features: list[str], params: dict,
                                wf_years: list[int], seeds: list[int]) -> dict[int, np.ndarray]:
    """Walk-forward classifier CV across multiple seeds, seed-averaged OOF probs."""
    per_seed: dict[int, dict[int, np.ndarray]] = {}
    for seed in seeds:
        per_seed[seed] = {}
        p = {**params, "random_seed": seed}
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
            per_seed[seed][vy] = m.predict_proba(val_df[features])[:, 1]
    oof: dict[int, np.ndarray] = {}
    for vy in wf_years:
        stacks = [per_seed[s][vy] for s in seeds if vy in per_seed[s]]
        if stacks:
            oof[vy] = np.mean(stacks, axis=0)
    return oof


def attach_clf_prob(df: pd.DataFrame, clf_oof: dict[int, np.ndarray],
                    wf_years_clf: list[int]) -> pd.DataFrame:
    """Add 'clf_prob' column using OOF probs for years in the classifier WF range."""
    out = df.copy()
    out["clf_prob"] = np.nan
    for vy in wf_years_clf:
        if vy not in clf_oof:
            continue
        mask = out["year"] == vy
        year_rows = out[mask]
        # Order of OOF probs matches the order of val_df rows produced in run_classifier_wf,
        # which was df[df["year"] == vy].dropna(subset=[TARGET]) — so intersect with non-null target.
        ordered_idx = year_rows.dropna(subset=[TARGET]).index
        assert len(ordered_idx) == len(clf_oof[vy]), \
            f"year {vy}: {len(ordered_idx)} rows vs {len(clf_oof[vy])} probs"
        out.loc[ordered_idx, "clf_prob"] = clf_oof[vy]
    return out


def run_margin_wf_ensemble(df: pd.DataFrame, features: list[str], params: dict,
                            wf_years: list[int], seeds: list[int],
                            require_clf_prob: bool) -> dict:
    """
    Walk-forward margin CV across multiple seeds. Returns seed-averaged preds per fold.
    """
    per_seed_preds: dict[int, dict[int, np.ndarray]] = {s: {} for s in seeds}
    fold_true: dict[int, np.ndarray] = {}

    for vy in wf_years:
        train_df = df[df["year"] < vy].dropna(subset=["margin"])
        val_df = df[df["year"] == vy].dropna(subset=["margin"])
        if require_clf_prob:
            train_df = train_df.dropna(subset=["clf_prob"])
            val_df = val_df.dropna(subset=["clf_prob"])
        if train_df.empty or val_df.empty:
            continue
        fold_true[vy] = val_df["margin"].values.astype(float)
        for seed in seeds:
            p = _normalize_margin_params({**params, "random_seed": seed})
            m = CatBoostRegressor(**p)
            m.fit(
                Pool(train_df[features], train_df["margin"]),
                eval_set=Pool(val_df[features], val_df["margin"]),
            )
            per_seed_preds[seed][vy] = m.predict(val_df[features])

    ensemble_preds: dict[int, np.ndarray] = {}
    for vy in wf_years:
        stacks = [per_seed_preds[s][vy] for s in seeds if vy in per_seed_preds[s]]
        if stacks:
            ensemble_preds[vy] = np.mean(stacks, axis=0)

    return {"preds_by_fold": ensemble_preds, "true_by_fold": fold_true,
            "per_seed_preds_by_fold": per_seed_preds}


def aggregate(result: dict, name: str, wf_years: list[int]) -> dict:
    preds = np.concatenate([result["preds_by_fold"][y] for y in wf_years if y in result["preds_by_fold"]])
    true = np.concatenate([result["true_by_fold"][y] for y in wf_years if y in result["true_by_fold"]])
    per_fold = []
    for y in wf_years:
        if y not in result["preds_by_fold"]:
            continue
        p = result["preds_by_fold"][y]
        t = result["true_by_fold"][y]
        per_fold.append({
            "year": int(y),
            "n": int(len(t)),
            "mae": float(np.mean(np.abs(t - p))),
            "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
            "direction_acc": float(((p > 0) == (t > 0)).mean()),
        })
    return {
        "config": name,
        "n": int(len(true)),
        "mae": float(np.mean(np.abs(true - preds))),
        "rmse": float(np.sqrt(np.mean((true - preds) ** 2))),
        "direction_acc": float(((preds > 0) == (true > 0)).mean()),
        "per_fold": per_fold,
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("=" * 80)
    print("EXPERIMENT -- classifier-conditioned margin")
    print("=" * 80)

    # -------------------- load data --------------------
    print("\n[1/5] Loading master dataset + building features...")
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    df = build_features(df)
    df_complete = df.dropna(subset=["margin", TARGET]).copy().sort_values("date").reset_index(drop=True)
    min_year = int(df_complete["year"].min())
    max_year = int(df_complete["year"].max())
    print(f"  {len(df_complete)} rows, {min_year}-{max_year}")

    opt = load_optimization_results()
    if opt is None:
        print("  FATAL: optimization_results.json (classifier) missing")
        sys.exit(1)
    base_features = [c for c in opt["best_features"] if c in df_complete.columns]
    print(f"  {len(base_features)} base features")

    if not MARGIN_OPT_PATH.exists():
        print(f"  FATAL: {MARGIN_OPT_PATH} missing")
        sys.exit(1)

    clf_params = _classifier_base_params(opt["best_params"])
    margin_params = _margin_base_params()
    print(f"  {len(SEEDS)}-seed ensemble for BOTH classifier and margin (seeds={SEEDS})")

    # -------------------- classifier WF (5-seed ensemble) --------------------
    clf_wf_start = max(2017, min_year + 3)
    clf_wf_years = list(range(clf_wf_start, max_year + 1))
    print(f"\n[2/5] Classifier walk-forward ensemble "
          f"({clf_wf_years[0]}-{clf_wf_years[-1]}, {len(clf_wf_years)} folds x {len(SEEDS)} seeds)")
    t0 = time.time()
    clf_oof = run_classifier_wf_ensemble(df_complete, base_features, clf_params, clf_wf_years, SEEDS)
    print(f"  done in {time.time() - t0:.0f}s")
    total_oof = sum(len(v) for v in clf_oof.values())
    print(f"  OOF probs collected for {total_oof} rows")

    # Attach as a feature to the full df
    df_with_prob = attach_clf_prob(df_complete, clf_oof, clf_wf_years)
    coverage = df_with_prob["clf_prob"].notna().sum()
    print(f"  clf_prob populated on {coverage} rows")

    # -------------------- margin WF: baseline + conditioned (both 5-seed) --------------------
    margin_wf_years = list(range(max(2019, clf_wf_years[0] + 2), max_year + 1))
    print(f"\n[3/5] Margin walk-forward ensemble "
          f"({margin_wf_years[0]}-{margin_wf_years[-1]}, {len(margin_wf_years)} folds x {len(SEEDS)} seeds x 2 conditions)")
    print("  baseline (no clf_prob)...")
    t0 = time.time()
    baseline_result = run_margin_wf_ensemble(
        df_with_prob, base_features, margin_params, margin_wf_years, SEEDS, require_clf_prob=True
    )
    print(f"    {time.time() - t0:.0f}s")

    conditioned_features = base_features + ["clf_prob"]
    print(f"  conditioned (+clf_prob, {len(conditioned_features)} features)...")
    t0 = time.time()
    conditioned_result = run_margin_wf_ensemble(
        df_with_prob, conditioned_features, margin_params, margin_wf_years, SEEDS, require_clf_prob=True
    )
    print(f"    {time.time() - t0:.0f}s")

    # -------------------- aggregate + compare --------------------
    print("\n[4/5] Aggregating...")
    baseline_agg = aggregate(baseline_result, "baseline", margin_wf_years)
    conditioned_agg = aggregate(conditioned_result, "conditioned", margin_wf_years)

    print(f"\n{'metric':<20s} {'baseline':>12s} {'conditioned':>14s} {'delta':>12s}")
    print("  " + "-" * 62)
    for key, fmt in [("mae", "{:.4f}"), ("rmse", "{:.4f}"),
                     ("direction_acc", "{:.4f}"), ("n", "{:d}")]:
        b = baseline_agg[key]
        c = conditioned_agg[key]
        delta = c - b if isinstance(b, (int, float)) else ""
        if key == "n":
            print(f"  {key:<20s} {b:>12d} {c:>14d}")
        else:
            delta_str = f"{delta:+.4f}"
            mark = ""
            if key in ("mae", "rmse") and delta < 0:
                mark = " GOOD"
            elif key == "direction_acc" and delta > 0:
                mark = " GOOD"
            elif key in ("mae", "rmse") and delta > 0:
                mark = " bad"
            elif key == "direction_acc" and delta < 0:
                mark = " bad"
            print(f"  {key:<20s} {fmt.format(b):>12s} {fmt.format(c):>14s} {delta_str:>12s}{mark}")

    print("\n  per fold (year, n, baseline MAE -> conditioned MAE, dir delta):")
    for b_row, c_row in zip(baseline_agg["per_fold"], conditioned_agg["per_fold"]):
        mae_delta = c_row["mae"] - b_row["mae"]
        dir_delta = c_row["direction_acc"] - b_row["direction_acc"]
        print(f"    {b_row['year']}  n={b_row['n']:3d}  "
              f"mae {b_row['mae']:.3f} -> {c_row['mae']:.3f} ({mae_delta:+.3f})  "
              f"dir {b_row['direction_acc']:.3f} -> {c_row['direction_acc']:.3f} "
              f"({dir_delta:+.3f})")

    # -------------------- report --------------------
    print(f"\n[5/5] Writing report -> {REPORT_PATH}")
    report = {
        "seeds": SEEDS,
        "n_base_features": len(base_features),
        "classifier_wf_years": clf_wf_years,
        "margin_wf_years": margin_wf_years,
        "baseline": baseline_agg,
        "conditioned": conditioned_agg,
        "deltas": {
            "mae": conditioned_agg["mae"] - baseline_agg["mae"],
            "rmse": conditioned_agg["rmse"] - baseline_agg["rmse"],
            "direction_acc": conditioned_agg["direction_acc"] - baseline_agg["direction_acc"],
        },
        "training_time_sec": time.time() - t_start,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote {REPORT_PATH}")

    print(f"\nTotal time: {time.time() - t_start:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
