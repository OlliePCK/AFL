"""
Follow-up to backtest_analytical_vs_baseline.py.

Previous finding: analytical 47-feat model ties the market on ACCURACY
(68.13% vs 68.48%). But accuracy only captures the sign of the pick, not
the quality of the probability. Two open questions:

  Q2 (calibration): does the analytical model produce BETTER-CALIBRATED
      probabilities than raw market implied? If yes, the dashboard
      confidence bars earn their keep even when the top pick matches
      market.

  Q3 (feature ablation): if we train a model with ONLY the 3 odds
      features, does it tie the 47-feat full analytical model? If yes,
      the other 44 features are dead weight for tipping. If no, those
      44 features DO add something beyond what odds alone carry -- even
      if it doesn't show up in accuracy.

Method:
  1. Walk-forward 2019-2025, 5-seed ensemble, TWO models:
       a) full analytical (47 features)
       b) odds-only (3 features: implied_home_close, overround_close,
          home_line_close)
  2. For each match collect P(home_win) from:
       - full analytical model
       - odds-only model
       - vig-adjusted market implied prob from opening odds
       - tipster consensus (mean_hconfidence / 100)
  3. Report per-strategy log-loss, Brier, accuracy.
  4. Reliability-bin breakdown -- show whether any probability source is
     better calibrated than the others.

Writes: data/analysis/analytical_calibration_report.json
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
from src.model import TARGET, FEATURE_GROUPS, load_optimization_results  # noqa: E402
from src.value import decimal_odds_to_implied_prob  # noqa: E402

VAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEEDS = [42, 123, 256, 789, 1337]
ODDS_FEATURES = FEATURE_GROUPS["odds"]  # 3 features


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


def wf_ensemble_probs(df: pd.DataFrame, features: list[str], params: dict,
                      name: str) -> dict[int, np.ndarray]:
    """5-seed ensemble walk-forward. Returns probs-by-year dict."""
    out: dict[int, np.ndarray] = {}
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
            seed_probs.append(m.predict_proba(val_df[available])[:, 1])
        out[vy] = np.mean(seed_probs, axis=0)
        ll = log_loss(val_df[TARGET], out[vy])
        print(f"    {name:>14s}  {vy}  n={len(val_df):3d}  LL={ll:.4f}")
    return out


def market_implied(df_val: pd.DataFrame) -> np.ndarray:
    """Vig-adjusted implied prob on home side from opening odds."""
    ho = df_val["home_odds_open"].values
    ao = df_val["away_odds_open"].values
    out = np.full(len(df_val), np.nan)
    for i in range(len(df_val)):
        if np.isnan(ho[i]) or np.isnan(ao[i]) or ho[i] <= 1 or ao[i] <= 1:
            continue
        p_h, _ = decimal_odds_to_implied_prob(float(ho[i]), float(ao[i]))
        out[i] = p_h
    return out


def tipster_prob(df_val: pd.DataFrame) -> np.ndarray:
    """Tipster consensus home probability (mean_hconfidence / 100)."""
    return (df_val["mean_hconfidence"].values / 100.0).astype(float)


def metrics_on(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute LL / Brier / accuracy on non-NaN rows."""
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


def reliability_bins(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int = 10) -> list[dict]:
    """10 equal-width bins over [0,1]. Report mean predicted vs empirical."""
    mask = ~np.isnan(probs) & ~np.isnan(labels)
    p = probs[mask]
    y = labels[mask]
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if sel.sum() == 0:
            out.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": 0,
                        "mean_pred": None, "empirical": None})
            continue
        out.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": int(sel.sum()),
            "mean_pred": float(p[sel].mean()),
            "empirical": float(y[sel].mean()),
        })
    return out


def main() -> None:
    out_path = PROJECT_ROOT / "data" / "analysis" / "analytical_calibration_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 80)
    print("ANALYTICAL CALIBRATION + ODDS-ONLY ABLATION")
    print("=" * 80)

    print("\n[1/5] Loading + featuring master dataset...")
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv",
        parse_dates=["date"],
    )
    df = build_features(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  featured: {len(df)} matches")

    # Feature sets
    opt = load_optimization_results()
    if opt is None:
        print("  FATAL: optimization_results.json not found.")
        sys.exit(1)
    base = list(opt["best_features"])
    full_features = base + [f for f in ODDS_FEATURES if f not in base]
    full_features = [c for c in full_features if c in df.columns]
    odds_only = [c for c in ODDS_FEATURES if c in df.columns]
    params = _base_params(opt["best_params"])

    print(f"  full analytical: {len(full_features)} features")
    print(f"  odds-only:       {len(odds_only)} features {odds_only}")

    # Walk-forward both models
    print(f"\n[2/5] Walk-forward 2019-2025, 5 seeds, 2 models")
    print(f"  Full analytical:")
    t_wf = time.time()
    full_probs = wf_ensemble_probs(df, full_features, params, "FULL")
    print(f"  Odds-only:")
    odds_probs = wf_ensemble_probs(df, odds_only, params, "ODDS_ONLY")
    print(f"  [WF total: {time.time() - t_wf:.0f}s]")

    # Build combined prediction frame
    print("\n[3/5] Pooling + computing market / tipster probabilities")
    pool = []
    for vy in VAL_YEARS:
        if vy not in full_probs:
            continue
        val_df = df[df["year"] == vy].dropna(subset=[TARGET]).reset_index(drop=True)
        sub = val_df[["year", "date", "home_team", "away_team", TARGET,
                      "home_odds_open", "away_odds_open", "mean_hconfidence"]].copy()
        sub["p_full"] = full_probs[vy]
        sub["p_odds_only"] = odds_probs[vy]
        sub["p_market"] = market_implied(val_df)
        sub["p_tipster"] = tipster_prob(val_df)
        pool.append(sub)
    pool = pd.concat(pool, ignore_index=True)
    y = pool[TARGET].values.astype(float)
    print(f"  pooled: {len(pool)} matches")

    # -- Same-sample subset (all 4 probabilities available) --
    has_all = (pool[["p_full", "p_odds_only", "p_market", "p_tipster"]]
               .notna().all(axis=1))
    common = pool[has_all].copy()
    yc = common[TARGET].values.astype(float)
    print(f"  same-sample (all 4 probs): {len(common)} matches")

    # Metrics per strategy on common sample
    print("\n[4/5] Metrics on same-sample pool:")
    print(f"    {'Strategy':>18s}  {'n':>4s}  {'LL':>7s}  {'Brier':>7s}  {'Acc':>6s}")
    print(f"    {'-'*18}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*6}")
    strat_metrics = {}
    for col, label in [
        ("p_full", "Full analytical"),
        ("p_odds_only", "Odds-only model"),
        ("p_market", "Market implied"),
        ("p_tipster", "Tipster prob"),
    ]:
        m = metrics_on(common[col].values, yc)
        strat_metrics[col] = m
        print(f"    {label:>18s}  {m['n']:>4d}  "
              f"{m['log_loss']:.4f}  {m['brier']:.4f}  {m['accuracy']:.3%}")

    # Reliability tables
    print("\n  Reliability (10 equal-width bins, common sample):")
    reliability_out = {}
    for col, label in [("p_full", "Full"),
                       ("p_odds_only", "Odds-only"),
                       ("p_market", "Market"),
                       ("p_tipster", "Tipster")]:
        bins = reliability_bins(common[col].values, yc)
        reliability_out[col] = bins
        print(f"\n    {label}:")
        print(f"      {'bin':>10s}  {'n':>4s}  {'pred':>6s}  {'emp':>6s}  {'diff':>6s}")
        for b in bins:
            if b["n"] == 0:
                continue
            diff = b["empirical"] - b["mean_pred"]
            print(f"      {b['bin']:>10s}  {b['n']:>4d}  "
                  f"{b['mean_pred']:>6.1%}  {b['empirical']:>6.1%}  "
                  f"{diff:>+6.1%}")

    # Head-to-head deltas
    print("\n  Pairwise deltas on same-sample pool:")
    print(f"    {'pair':>30s}  {'dLL':>7s}  {'dBrier':>8s}  {'dAcc':>7s}")
    pair_out = {}
    baselines = ["p_market", "p_tipster"]
    targets = ["p_full", "p_odds_only"]
    for t in targets:
        for b in baselines:
            d_ll = strat_metrics[t]["log_loss"] - strat_metrics[b]["log_loss"]
            d_br = strat_metrics[t]["brier"] - strat_metrics[b]["brier"]
            d_ac = strat_metrics[t]["accuracy"] - strat_metrics[b]["accuracy"]
            pair_out[f"{t}_minus_{b}"] = {
                "d_log_loss": d_ll, "d_brier": d_br, "d_accuracy": d_ac,
            }
            print(f"    {t:>15s} - {b:<12s}  "
                  f"{d_ll:>+6.4f}  {d_br:>+7.4f}  {d_ac:>+7.3%}")
    # Full vs odds-only (THE Q3 answer)
    d_ll = strat_metrics["p_full"]["log_loss"] - strat_metrics["p_odds_only"]["log_loss"]
    d_br = strat_metrics["p_full"]["brier"] - strat_metrics["p_odds_only"]["brier"]
    d_ac = strat_metrics["p_full"]["accuracy"] - strat_metrics["p_odds_only"]["accuracy"]
    pair_out["p_full_minus_p_odds_only"] = {
        "d_log_loss": d_ll, "d_brier": d_br, "d_accuracy": d_ac,
    }
    print(f"    {'p_full':>15s} - {'p_odds_only':<12s}  "
          f"{d_ll:>+6.4f}  {d_br:>+7.4f}  {d_ac:>+7.3%}")

    # Verdicts
    print("\n[5/5] Verdicts:")
    # Q2: is analytical materially better than market?
    q2_ll = strat_metrics["p_full"]["log_loss"] - strat_metrics["p_market"]["log_loss"]
    q2_br = strat_metrics["p_full"]["brier"] - strat_metrics["p_market"]["brier"]
    noise_floor_ll = 0.00174  # 5-seed ensemble std from prior experiments
    if q2_ll < -3 * noise_floor_ll:
        q2 = f"CALIBRATION WIN: full analytical beats market by {-q2_ll:.4f} LL."
    elif q2_ll < -noise_floor_ll:
        q2 = f"CALIBRATION MARGINAL: full analytical beats market by {-q2_ll:.4f} LL (small)."
    elif q2_ll < noise_floor_ll:
        q2 = f"CALIBRATION TIE: full analytical within noise of market ({q2_ll:+.4f} LL)."
    else:
        q2 = f"CALIBRATION LOSS: market beats full analytical by {q2_ll:.4f} LL."
    print(f"  Q2 (analytical vs market): {q2}")

    # Q3: is odds-only as good as full analytical?
    q3_ll = strat_metrics["p_odds_only"]["log_loss"] - strat_metrics["p_full"]["log_loss"]
    if q3_ll < -noise_floor_ll:
        q3 = ("ODDS-ONLY WINS: the 3-feature odds-only model is BETTER than "
              f"the 47-feature analytical by {-q3_ll:.4f} LL. 44 extra features "
              "are net-harmful for tipping.")
    elif q3_ll < noise_floor_ll:
        q3 = ("ODDS-ONLY TIES: the 3-feature odds model matches the 47-feature "
              f"analytical within noise ({q3_ll:+.4f} LL). The 44 extra features "
              "add nothing for tipping.")
    else:
        q3 = ("ODDS-ONLY LOSES: the 47-feature analytical beats odds-only by "
              f"{q3_ll:.4f} LL. The other 44 features DO add signal for tipping.")
    print(f"  Q3 (odds-only vs full):    {q3}")

    report = {
        "window_start_year": VAL_YEARS[0],
        "window_end_year": VAL_YEARS[-1],
        "seeds": SEEDS,
        "n_full_features": len(full_features),
        "n_odds_only_features": len(odds_only),
        "odds_only_features": odds_only,
        "n_common_sample": int(len(common)),
        "metrics_common_sample": strat_metrics,
        "pairwise_deltas_common_sample": pair_out,
        "reliability_bins_common_sample": reliability_out,
        "verdicts": {"q2_calibration": q2, "q3_odds_only_ablation": q3},
        "runtime_sec": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {out_path}")
    print(f"Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
