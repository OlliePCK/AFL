"""
Robustness test for the odds-only analytical model.

Prior finding (backtest_analytical_calibration.py):
  Full 47-feat analytical  LL=0.5853  Brier=0.2011  Acc=68.11%
  Odds-only 3-feat         LL=0.5834  Brier=0.2002  Acc=68.81%
  Raw market implied       LL=0.5968  Brier=0.2055  Acc=68.25%

The odds-only baseline used all CLOSE features (implied_home_close,
overround_close, home_line_close). CLOSE odds aren't live-actionable
until game start — they incorporate late money, team lists, weather.
For a production dashboard we need a feature set that's available
early in the week.

This script tests 4 variants:
  V1  close3  = [implied_home_close, overround_close, home_line_close]
      (prior baseline; best theoretical number; includes leakage if we
      ever snapshot these BEFORE close)
  V2  open2   = [implied_home_open, overround_open]
      (fully live-safe from market open, no spread)
  V3  open+line = [implied_home_open, overround_open, home_line_close]
      (adds the spread, which books post live alongside h2h)
  V4  open+movement = [implied_home_open, overround_open, odds_move,
                        odds_move_magnitude, overround_change]
      (open + open->close movement; carries info about late money
       without exposing close directly, but still requires a close
       snapshot)

Decision question:
  Is V3 (open h2h + close line) close enough to V1 to be worth shipping?
  If V2 alone is good, even simpler.

Writes: data/analysis/odds_only_variants_report.json
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

from src.features import build_features  # noqa: E402
from src.model import TARGET, load_optimization_results  # noqa: E402
from src.value import decimal_odds_to_implied_prob  # noqa: E402

VAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEEDS = [42, 123, 256, 789, 1337]

VARIANTS: dict[str, list[str]] = {
    "V1_close3 (baseline)":
        ["implied_home_close", "overround_close", "home_line_close"],
    "V2_open2":
        ["implied_home_open", "overround_open"],
    "V3_open+line":
        ["implied_home_open", "overround_open", "home_line_close"],
    "V4_open+movement":
        ["implied_home_open", "overround_open", "odds_move",
         "odds_move_magnitude", "overround_change"],
}


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


def wf_ensemble(df: pd.DataFrame, features: list[str], params: dict,
                label: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    available = [c for c in features if c in df.columns]
    if len(available) < len(features):
        missing = [c for c in features if c not in df.columns]
        print(f"    [{label}] WARN missing: {missing}")
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
            seed_probs.append(m.predict_proba(val_df[available])[:, 1])
        out[vy] = np.mean(seed_probs, axis=0)
    return out


def market_implied_open(df_val: pd.DataFrame) -> np.ndarray:
    ho = df_val["home_odds_open"].values
    ao = df_val["away_odds_open"].values
    out = np.full(len(df_val), np.nan)
    for i in range(len(df_val)):
        if np.isnan(ho[i]) or np.isnan(ao[i]) or ho[i] <= 1 or ao[i] <= 1:
            continue
        p_h, _ = decimal_odds_to_implied_prob(float(ho[i]), float(ao[i]))
        out[i] = p_h
    return out


def metrics_on(probs: np.ndarray, labels: np.ndarray) -> dict:
    mask = ~np.isnan(probs) & ~np.isnan(labels)
    if mask.sum() == 0:
        return {"n": 0, "log_loss": None, "brier": None, "accuracy": None}
    p = np.clip(probs[mask], 1e-6, 1 - 1e-6)
    y = labels[mask]
    preds = (p >= 0.5).astype(int)
    return {
        "n": int(mask.sum()),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float((preds == y).mean()),
    }


def main() -> None:
    out_path = PROJECT_ROOT / "data" / "analysis" / "odds_only_variants_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 80)
    print("ODDS-ONLY VARIANTS — robustness / live-safety check")
    print("=" * 80)

    print("\n[1/4] Loading + featuring master dataset...")
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv",
        parse_dates=["date"],
    )
    df = build_features(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  featured: {len(df)} matches")

    opt = load_optimization_results()
    if opt is None:
        print("  FATAL: optimization_results.json not found."); sys.exit(1)
    params = _base_params(opt["best_params"])

    # Train each variant
    print(f"\n[2/4] Walk-forward 2019-2025, 5 seeds, {len(VARIANTS)} variants")
    variant_probs: dict[str, dict[int, np.ndarray]] = {}
    for name, feats in VARIANTS.items():
        t_v = time.time()
        print(f"  Training {name}  features={feats}")
        variant_probs[name] = wf_ensemble(df, feats, params, name)
        print(f"    done ({time.time() - t_v:.0f}s)")

    # Build pooled prediction frame
    print("\n[3/4] Pooling predictions")
    pool = []
    for vy in VAL_YEARS:
        val_df = df[df["year"] == vy].dropna(subset=[TARGET]).reset_index(drop=True)
        sub = val_df[["year", "date", "home_team", "away_team", TARGET,
                      "home_odds_open", "away_odds_open"]].copy()
        for name, by_year in variant_probs.items():
            if vy in by_year:
                sub[f"p_{name}"] = by_year[vy]
            else:
                sub[f"p_{name}"] = np.nan
        sub["p_market_open"] = market_implied_open(val_df)
        pool.append(sub)
    pool = pd.concat(pool, ignore_index=True)
    y = pool[TARGET].values.astype(float)
    print(f"  pooled: {len(pool)} matches")

    # Same-sample filter
    prob_cols = [f"p_{n}" for n in VARIANTS] + ["p_market_open"]
    has_all = pool[prob_cols].notna().all(axis=1)
    common = pool[has_all].copy()
    yc = common[TARGET].values.astype(float)
    print(f"  same-sample: {len(common)} matches have all {len(prob_cols)} probs")

    # Metrics per variant + market
    print("\n[4/4] Metrics on same-sample:")
    print(f"    {'Strategy':>26s}  {'n':>4s}  {'LL':>7s}  {'Brier':>7s}  {'Acc':>7s}")
    print(f"    {'-'*26}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}")
    metrics_out = {}
    for col in prob_cols:
        m = metrics_on(common[col].values, yc)
        metrics_out[col] = m
        label = col.replace("p_", "")
        print(f"    {label:>26s}  {m['n']:>4d}  "
              f"{m['log_loss']:.4f}  {m['brier']:.4f}  {m['accuracy']:.3%}")

    # Deltas vs V1 (prior baseline)
    v1_key = "p_V1_close3 (baseline)"
    print("\n  Variant deltas vs V1_close3:")
    print(f"    {'Variant':>22s}  {'dLL':>8s}  {'dBrier':>8s}  {'dAcc':>8s}")
    deltas = {}
    for col in prob_cols:
        if col == v1_key:
            continue
        d_ll = metrics_out[col]["log_loss"] - metrics_out[v1_key]["log_loss"]
        d_br = metrics_out[col]["brier"] - metrics_out[v1_key]["brier"]
        d_ac = metrics_out[col]["accuracy"] - metrics_out[v1_key]["accuracy"]
        deltas[col] = {"d_ll": d_ll, "d_brier": d_br, "d_acc": d_ac}
        label = col.replace("p_", "")
        print(f"    {label:>22s}  {d_ll:>+7.4f}  {d_br:>+7.4f}  {d_ac:>+7.3%}")

    # Deltas vs V2 (simplest live-safe model)
    v2_key = "p_V2_open2"
    print("\n  Variant deltas vs V2_open2 (simplest live-safe):")
    print(f"    {'Variant':>22s}  {'dLL':>8s}  {'dBrier':>8s}  {'dAcc':>8s}")
    for col in prob_cols:
        if col == v2_key:
            continue
        d_ll = metrics_out[col]["log_loss"] - metrics_out[v2_key]["log_loss"]
        d_br = metrics_out[col]["brier"] - metrics_out[v2_key]["brier"]
        d_ac = metrics_out[col]["accuracy"] - metrics_out[v2_key]["accuracy"]
        label = col.replace("p_", "")
        print(f"    {label:>22s}  {d_ll:>+7.4f}  {d_br:>+7.4f}  {d_ac:>+7.3%}")

    # Verdict for each variant
    noise = 0.00174
    print("\n  Per-variant verdicts (vs V1_close3):")
    verdicts = {}
    for col in prob_cols:
        if col == v1_key:
            continue
        d_ll = deltas[col]["d_ll"]
        label = col.replace("p_", "")
        if d_ll < -noise:
            v = f"BETTER by {-d_ll:.4f} LL"
        elif d_ll < noise:
            v = f"TIE within noise ({d_ll:+.4f} LL)"
        elif d_ll < 3 * noise:
            v = f"SLIGHTLY WORSE by {d_ll:.4f} LL (<3x noise)"
        else:
            v = f"WORSE by {d_ll:.4f} LL"
        verdicts[col] = v
        print(f"    {label:>22s}: {v}")

    report = {
        "window_start_year": VAL_YEARS[0],
        "window_end_year": VAL_YEARS[-1],
        "seeds": SEEDS,
        "variants": VARIANTS,
        "n_common_sample": int(len(common)),
        "metrics": metrics_out,
        "deltas_vs_V1_close3": deltas,
        "verdicts_vs_V1_close3": verdicts,
        "noise_floor_ll": noise,
        "runtime_sec": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {out_path}")
    print(f"Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
