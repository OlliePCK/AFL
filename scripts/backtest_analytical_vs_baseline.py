"""
Sanity check: does the 47-feature analytical model actually beat trivial baselines?

The dashboard runs two models:
  - Betting model (20 features, no odds) -- tuned all day for value detection
  - Analytical model (47 features, WITH odds) -- for tipping accuracy

The analytical model has not been questioned in months. It's worth asking:
  does it beat "always tip the market favorite"?

If it beats market-favorite by ~2pp+ accuracy -> it's earning its keep.
If it ties or loses -> we're running a 47-feature CatBoost ensemble to
replicate "pick the shorter odds." That needs fixing or removing.

Method:
  1. Walk-forward 2019-2025, 5-seed ensemble per fold (same protocol as
     train_analytical_ensemble.py). Train on <vy, predict on vy.
  2. For each bet/match in the pool, compute predictions from 3 strategies:
       A) Analytical model ensemble (mean of 5 seeds) -- predicted_winner
          = home if P(home_win) >= 0.5 else away
       B) Market favorite            = home if home_odds_open < away_odds_open
       C) Tipster consensus          = home if mean_hconfidence > 50
  3. Report pooled accuracy + by-year breakdown.
  4. Same-sample comparison: only rows where all 3 strategies can predict
     (so delta is fair).

Writes: data/analysis/analytical_baseline_report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features  # noqa: E402
from src.model import TARGET, FEATURE_GROUPS, load_optimization_results  # noqa: E402

VAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEEDS = [42, 123, 256, 789, 1337]


def _base_params(opt_params: dict) -> dict:
    params = dict(opt_params)
    params.update({
        "iterations": 1500,
        "early_stopping_rounds": 80,
        "verbose": 0,
        "eval_metric": "Logloss",
        "use_best_model": True,
    })
    if params.get("subsample", 1.0) < 1.0:
        params["bootstrap_type"] = "Bernoulli"
    return params


def run_walk_forward(df: pd.DataFrame, features: list[str],
                     params: dict) -> pd.DataFrame:
    """5-seed ensemble walk-forward. Returns per-match predictions
    augmented with model_prob, and carries odds + tipster columns through
    for baseline comparison."""
    carry_cols = [
        "year", "date", "home_team", "away_team",
        TARGET,
        "home_odds_open", "away_odds_open",
        "mean_hconfidence",
    ]
    rows = []
    available = [c for c in features if c in df.columns]

    for vy in VAL_YEARS:
        train_df = df[df["year"] < vy].dropna(subset=[TARGET])
        val_df = df[df["year"] == vy].dropna(subset=[TARGET])
        if train_df.empty or val_df.empty:
            continue

        seed_probs = []
        for seed in SEEDS:
            p = {**params, "random_seed": seed}
            m = CatBoostClassifier(**p)
            m.fit(
                Pool(train_df[available], train_df[TARGET]),
                eval_set=Pool(val_df[available], val_df[TARGET]),
            )
            probs = m.predict_proba(val_df[available])[:, 1]
            seed_probs.append(probs)
        ensemble = np.mean(seed_probs, axis=0)

        fold = val_df[carry_cols].copy()
        fold["model_prob"] = ensemble
        rows.append(fold)
        print(f"    {vy}  n={len(val_df):3d}  "
              f"ensemble_ll={log_loss(val_df[TARGET], ensemble):.4f}")

    return pd.concat(rows, ignore_index=True)


def strategy_picks(preds: pd.DataFrame) -> pd.DataFrame:
    """Add per-strategy 'home_pick' (1 = pick home, 0 = pick away, NaN = no data)."""
    p = preds.copy()

    # A) Analytical model
    p["pick_model"] = (p["model_prob"] >= 0.5).astype(float)

    # B) Market favorite (lower opening odds -> favorite)
    has_odds = p["home_odds_open"].notna() & p["away_odds_open"].notna()
    p["pick_market"] = np.where(
        has_odds,
        (p["home_odds_open"] < p["away_odds_open"]).astype(float),
        np.nan,
    )

    # C) Tipster consensus (mean_hconfidence on 0-100 scale)
    has_tipster = p["mean_hconfidence"].notna()
    p["pick_tipster"] = np.where(
        has_tipster,
        (p["mean_hconfidence"] > 50).astype(float),
        np.nan,
    )

    # Correct flags
    for strat in ["model", "market", "tipster"]:
        pick = p[f"pick_{strat}"]
        p[f"correct_{strat}"] = np.where(
            pick.isna(), np.nan,
            (pick == p[TARGET].astype(float)).astype(float),
        )
    return p


def acc_summary(correct: pd.Series) -> dict:
    """Summary of a boolean/0-1 correctness column, handling NaNs."""
    valid = correct.dropna()
    if valid.empty:
        return {"n": 0, "accuracy": None, "wins": 0}
    return {
        "n": int(len(valid)),
        "accuracy": float(valid.mean()),
        "wins": int(valid.sum()),
    }


def main() -> None:
    out_path = PROJECT_ROOT / "data" / "analysis" / "analytical_baseline_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 80)
    print("ANALYTICAL MODEL vs TRIVIAL BASELINES  (walk-forward 2019-2025)")
    print("=" * 80)

    print("\n[1/4] Loading + featuring master dataset...")
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv",
        parse_dates=["date"],
    )
    df = build_features(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  featured: {len(df)} completed matches, "
          f"{int(df['year'].min())}-{int(df['year'].max())}")

    # Analytical feature set: Optuna winner + odds group
    opt = load_optimization_results()
    if opt is None:
        print("  FATAL: optimization_results.json not found.")
        sys.exit(1)
    base_features = list(opt["best_features"])
    odds_features = FEATURE_GROUPS["odds"]
    features = base_features + [f for f in odds_features if f not in base_features]
    features = [c for c in features if c in df.columns]
    params = _base_params(opt["best_params"])
    print(f"  analytical features: {len(features)} "
          f"({len(base_features)} base + odds: {odds_features})")

    print(f"\n[2/4] Walk-forward CV -- {len(SEEDS)} seeds x {len(VAL_YEARS)} folds")
    t_wf = time.time()
    preds = run_walk_forward(df, features, params)
    print(f"  [WF total: {time.time() - t_wf:.0f}s, {len(preds)} pooled predictions]")

    print(f"\n[3/4] Computing strategy picks + accuracy...")
    preds = strategy_picks(preds)

    # Full-pool accuracy (each strategy on its own coverage)
    print("\n  Pooled accuracy (each strategy on all rows where it has data):")
    pool_out = {}
    for strat, label in [("model", "Analytical model (47 feat)"),
                         ("market", "Market favorite       "),
                         ("tipster", "Tipster consensus     ")]:
        s = acc_summary(preds[f"correct_{strat}"])
        pool_out[strat] = s
        if s["accuracy"] is not None:
            print(f"    {label}: n={s['n']:4d}  "
                  f"acc={s['accuracy']:.3%}  ({s['wins']}/{s['n']})")
        else:
            print(f"    {label}: no data")

    # Same-sample comparison: only rows where ALL THREE strategies predict
    common = preds.dropna(subset=["correct_model", "correct_market", "correct_tipster"])
    print(f"\n  Same-sample accuracy (n={len(common)} matches where all 3 have data):")
    common_out = {"n": int(len(common))}
    for strat, label in [("model", "Analytical model (47 feat)"),
                         ("market", "Market favorite       "),
                         ("tipster", "Tipster consensus     ")]:
        s = acc_summary(common[f"correct_{strat}"])
        common_out[strat] = s
        print(f"    {label}: acc={s['accuracy']:.3%}  ({s['wins']}/{s['n']})")

    # Pairwise agreement
    print("\n  Pairwise agreement in common sample:")
    pair_out = {}
    for a, b in [("model", "market"), ("model", "tipster"), ("market", "tipster")]:
        same = (common[f"pick_{a}"] == common[f"pick_{b}"]).mean()
        pair_out[f"{a}_vs_{b}"] = float(same)
        print(f"    {a} vs {b}: {same:.3%} of picks agree")

    # By-year breakdown on common sample
    print(f"\n  By-year accuracy (same-sample):")
    print(f"    {'year':>4}  {'n':>4}  {'model':>8}  {'market':>8}  {'tipster':>8}")
    print(f"    {'-'*4}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    year_out = {}
    for yr, g in common.groupby("year"):
        row = {
            "n": int(len(g)),
            "model": float(g["correct_model"].mean()),
            "market": float(g["correct_market"].mean()),
            "tipster": float(g["correct_tipster"].mean()),
        }
        year_out[int(yr)] = row
        print(f"    {int(yr):>4}  {row['n']:>4}  {row['model']:>7.1%}  "
              f"{row['market']:>7.1%}  {row['tipster']:>7.1%}")

    # Signed delta: model vs market (the one that matters most)
    print("\n  Head-to-head: model vs market favorite (same-sample only):")
    both_right = ((common["correct_model"] == 1) & (common["correct_market"] == 1)).sum()
    model_only = ((common["correct_model"] == 1) & (common["correct_market"] == 0)).sum()
    market_only = ((common["correct_model"] == 0) & (common["correct_market"] == 1)).sum()
    both_wrong = ((common["correct_model"] == 0) & (common["correct_market"] == 0)).sum()
    h2h = {
        "both_right": int(both_right),
        "model_only_right": int(model_only),
        "market_only_right": int(market_only),
        "both_wrong": int(both_wrong),
        "model_minus_market_acc": float(
            (common["correct_model"].mean() - common["correct_market"].mean())),
    }
    print(f"    both right:   {both_right:4d}  ({both_right/len(common):.1%})")
    print(f"    model only:   {model_only:4d}  ({model_only/len(common):.1%})")
    print(f"    market only:  {market_only:4d}  ({market_only/len(common):.1%})")
    print(f"    both wrong:   {both_wrong:4d}  ({both_wrong/len(common):.1%})")
    print(f"    model - market accuracy delta: "
          f"{h2h['model_minus_market_acc']:+.4f} "
          f"({h2h['model_minus_market_acc']*100:+.2f}pp)")

    # Verdict
    delta = h2h["model_minus_market_acc"]
    # McNemar-ish heuristic: discordant pairs / sqrt(discordant)
    disc = model_only + market_only
    z = (model_only - market_only) / np.sqrt(disc) if disc > 0 else 0.0
    print(f"\n    McNemar z (approx): {z:+.2f}  "
          f"(|z|>1.96 ~ significant at p=0.05)")
    h2h["mcnemar_z_approx"] = float(z)

    if delta > 0.02:
        verdict = "MODEL WINS -- analytical model beats market by 2pp+, earning its keep."
    elif delta > 0.005:
        verdict = "MODEL MARGINAL -- beats market by <2pp, modest but real."
    elif delta > -0.005:
        verdict = ("MODEL TIES MARKET -- 47 features replicate 'pick shorter odds'. "
                   "Investigate whether odds-only baseline would work.")
    else:
        verdict = "MODEL LOSES -- market-favorite baseline outperforms. Red flag."
    print(f"\n  VERDICT: {verdict}")

    # Write report
    report = {
        "window_start_year": VAL_YEARS[0],
        "window_end_year": VAL_YEARS[-1],
        "seeds": SEEDS,
        "n_features_analytical": len(features),
        "n_total_pooled": int(len(preds)),
        "pooled_own_coverage": pool_out,
        "same_sample_n": int(len(common)),
        "same_sample_accuracy": {k: v for k, v in common_out.items() if k != "n"},
        "pairwise_agreement_common": pair_out,
        "by_year_common_sample": year_out,
        "head_to_head_model_vs_market": h2h,
        "verdict": verdict,
        "runtime_sec": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[4/4] Report: {out_path}")
    print(f"Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
