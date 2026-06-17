"""
Value Bet Filter V2 — QUICKSCAN

Re-scores V1's pooled `all_bets` (edge >= 5%, 2019-2025) under 7 filter
hypotheses to see which deserve a full walk-forward build. Goal: cheap
20-line signal scan before investing ~2 days in a production v2.

The V1 backtest (data/analysis/value_filter_backtest.json) already showed:
  - 969 total bets at edge>=5%, overall ROI -0.73% (market beats model)
  - Current 15-30% band: 317 bets, -4.3% ROI, -$1,231
  - Edge band INVERSION: 10-15% = +4.2% ROI (best), 30-40% = -23.1% (worst)
  - Favorites vs underdogs (15-30%): favs +8.3% ROI / dogs -19.2% ROI
  - Sweet spot: odds 1.5-2.0 + 15-30% band = 73.4% win, +22.7% ROI (n=128)
  - Year variance: only 2022 (+11.7%) and 2024 (+10.0%) profitable

So we already know "favorites + tight odds" signals look promising in the
raw pooled data. This quickscan formalizes 7 hypotheses with proper CIs
and a holdout check to catch overfitting.

Hypotheses tested:
  H1 Favorites-only (odds < 2.0)
  H2 Lower edge threshold (5-15% or 10-20%)
  H3 Moderate favorite + moderate edge (odds 1.5-2.0 + edge 10-20%)
  H4 V3 analytical agrees on direction       *flagged: V3 trained on 2019-25
  H5 Smart-money alignment (odds_move toward our pick)
  H6 Composite: favorites + V3-agree + smart-money
  H7 Kelly sizing variants (25% / 50% / flat $100)

For each hypothesis we report:
  - n_bets, win_rate, ROI, profit
  - Wilson 95% CI on win rate
  - Bootstrap 95% CI on ROI
  - Profitable years (of 7)
  - Holdout ROI on last 30% of bets (chronological)

Pass thresholds:
  ROI >= +5%, n >= 100, profitable years >= 5/7,
  bootstrap CI lower bound > 0%, holdout ROI > 0%

Writes: data/analysis/value_filter_v2_quickscan.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features  # noqa: E402
from src.value import walk_forward_betting  # noqa: E402


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def bootstrap_roi_ci(bets: pd.DataFrame, n_iter: int = 1000,
                     confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI on ROI (% return on wagered)."""
    if bets.empty:
        return (0.0, 0.0)
    n = len(bets)
    rng = np.random.default_rng(seed)
    rois = np.empty(n_iter)
    bet_amts = bets["bet_amount"].values
    profits = bets["profit"].values
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        w = bet_amts[idx].sum()
        p = profits[idx].sum()
        rois[i] = p / w * 100 if w > 0 else 0
    lo = float(np.percentile(rois, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(rois, (1 + confidence) / 2 * 100))
    return (lo, hi)


def summarize_bets(bets: pd.DataFrame) -> dict:
    """All-in-one summary: roi, ci, wilson, yearly breakdown, holdout."""
    if bets.empty:
        return {
            "n_bets": 0, "win_rate": 0.0, "roi": 0.0, "profit": 0.0,
            "avg_edge": 0.0, "avg_odds": 0.0,
            "wilson_low": 0.0, "wilson_high": 0.0,
            "roi_ci_low": 0.0, "roi_ci_high": 0.0,
            "profitable_years": 0, "total_years": 0,
            "holdout_n": 0, "holdout_roi": 0.0,
        }

    wagered = bets["bet_amount"].sum()
    profit = bets["profit"].sum()
    wins = int(bets["won"].sum())
    n = len(bets)
    roi = profit / wagered * 100 if wagered > 0 else 0

    wlo, whi = wilson_ci(wins, n)
    rlo, rhi = bootstrap_roi_ci(bets)

    # Year stability
    yearly = bets.groupby("year").apply(
        lambda g: g["profit"].sum() / g["bet_amount"].sum() * 100
        if g["bet_amount"].sum() > 0 else 0.0
    )
    profitable_years = int((yearly > 0).sum())
    total_years = int(yearly.size)

    # Holdout: last 30% of bets chronologically
    sorted_bets = bets.sort_values("date")
    split_idx = int(len(sorted_bets) * 0.70)
    holdout = sorted_bets.iloc[split_idx:]
    if holdout.empty or holdout["bet_amount"].sum() == 0:
        h_roi = 0.0
    else:
        h_roi = holdout["profit"].sum() / holdout["bet_amount"].sum() * 100

    return {
        "n_bets": int(n),
        "win_rate": float(wins / n),
        "roi": float(roi),
        "profit": float(profit),
        "avg_edge": float(bets["edge"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "wilson_low": float(wlo),
        "wilson_high": float(whi),
        "roi_ci_low": float(rlo),
        "roi_ci_high": float(rhi),
        "profitable_years": profitable_years,
        "total_years": total_years,
        "holdout_n": int(len(holdout)),
        "holdout_roi": float(h_roi),
    }


# ---------------------------------------------------------------------------
# V3 analytical-model prediction (for H4)
# ---------------------------------------------------------------------------
def predict_v3_probs(df: pd.DataFrame) -> np.ndarray:
    """Predict V3 home-win probabilities for every row in df.

    Uses the 5-seed ensemble + isotonic calibrator. Note: V3 was trained on
    2019-2025 so applying it to bets from that window is evaluation-on-train.
    Acceptable for exploration; a clean forward test would need walk-forward
    V3 predictions (train excluding year Y, predict year Y).
    """
    ensemble_dir = PROJECT_ROOT / "data" / "ensemble"
    model_files = sorted(ensemble_dir.glob("analytical_model_*.cbm"))
    if not model_files:
        raise FileNotFoundError("V3 ensemble not found — run train_analytical_odds_only.py")

    schema_path = PROJECT_ROOT / "data" / "analytical_feature_schema.json"
    with open(schema_path) as f:
        features = json.load(f)["features"]

    # Load ensemble
    models = []
    for f in model_files:
        m = CatBoostClassifier()
        m.load_model(str(f))
        models.append(m)

    # Predict
    X = df[features]
    probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

    # Calibrate
    cal_path = PROJECT_ROOT / "data" / "analytical_calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
        probs = calibrator.predict(probs)
        probs = np.clip(probs, 0.02, 0.98)

    return probs


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------
def apply_filter(bets: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    """Apply boolean mask, summarize, and tag with filter name."""
    sub = bets[mask].copy()
    s = summarize_bets(sub)
    s["filter"] = name
    return s


def format_row(s: dict) -> str:
    """One-line pretty print for the console table."""
    flags = []
    if s["n_bets"] >= 100: flags.append("N")
    if s["roi"] >= 5.0: flags.append("R")
    if s["profitable_years"] >= 5: flags.append("Y")
    if s["roi_ci_low"] > 0: flags.append("C")
    if s["holdout_roi"] > 0: flags.append("H")
    flag_str = "".join(flags).ljust(5)
    pass_fail = "PASS" if len(flags) == 5 else f"{len(flags)}/5"

    return (
        f"{s['filter']:<40} {s['n_bets']:>5} "
        f"{s['win_rate']:>6.1%} {s['roi']:>+7.1f}% "
        f"[{s['roi_ci_low']:>+6.1f}, {s['roi_ci_high']:>+6.1f}] "
        f"{s['profitable_years']:>1}/{s['total_years']:<1} "
        f"{s['holdout_roi']:>+7.1f}% {flag_str} {pass_fail}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    out_path = PROJECT_ROOT / "data" / "analysis" / "value_filter_v2_quickscan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 100)
    print("VALUE FILTER V2 -- QUICKSCAN (7 hypotheses)")
    print("=" * 100)

    # -----------------------------------------------------------------------
    # Load + feature master dataset
    # -----------------------------------------------------------------------
    print("\n[1/4] Loading + featuring master dataset...")
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv",
        parse_dates=["date"],
    )
    df = build_features(df)
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    print(f"  {len(df)} completed matches, years {df['year'].min()}-{df['year'].max()}")

    # Build per-match lookup for metadata we need on each bet
    # Key: (date, home_team, away_team) -> row metadata
    df_lookup = df.set_index(
        [df["date"].dt.strftime("%Y-%m-%d"), "home_team", "away_team"]
    )

    # -----------------------------------------------------------------------
    # Walk-forward simulation at edge>=5%
    # -----------------------------------------------------------------------
    print("\n[2/4] Running walk-forward simulation (2019-2025, edge>=5%)...")
    t0 = time.time()
    wf = walk_forward_betting(
        df,
        min_edge=0.05,
        max_edge=1.00,
        kelly_frac=0.25,
        max_bet_pct=0.10,
        odds_source="opening",
        start_year=2019,
        end_year=2025,
        calibrate=True,
        compare_sources=False,
    )
    print(f"  WF done in {time.time() - t0:.0f}s")

    all_bets: pd.DataFrame = wf["all_bets"]
    if all_bets.empty:
        print("No bets generated — aborting.")
        return
    print(f"  Total bets at edge>=5%: {len(all_bets)}")

    # -----------------------------------------------------------------------
    # V3 analytical predictions (for H4)
    # -----------------------------------------------------------------------
    print("\n[3/4] Predicting V3 analytical probabilities...")
    print("  NOTE: V3 trained on 2019-25 so this is evaluation-on-train.")
    print("  Acceptable for exploration; full V2 build may need walk-forward V3.")
    df["v3_home_prob"] = predict_v3_probs(df)

    # Join metadata onto each bet
    print("  Joining odds_move + V3 probs onto bets...")
    all_bets = all_bets.copy()
    all_bets["date_str"] = pd.to_datetime(all_bets["date"]).dt.strftime("%Y-%m-%d")
    all_bets = all_bets.merge(
        df[["date", "home_team", "away_team", "odds_move", "v3_home_prob",
            "home_odds_open", "away_odds_open"]].assign(
            date_str=df["date"].dt.strftime("%Y-%m-%d")
        )[["date_str", "home_team", "away_team", "odds_move",
           "v3_home_prob", "home_odds_open", "away_odds_open"]],
        on=["date_str", "home_team", "away_team"],
        how="left",
    )
    merged_rate = all_bets["odds_move"].notna().mean()
    v3_rate = all_bets["v3_home_prob"].notna().mean()
    print(f"  odds_move merge success: {merged_rate:.0%}  "
          f"V3 prob merge: {v3_rate:.0%}")

    # Tag favorite/underdog on the bet side
    all_bets["role"] = np.where(all_bets["odds"] < 2.0, "favorite", "underdog")

    # Tag smart-money alignment:
    # odds_move = implied_home_close - implied_home_open
    # odds_move > 0 => market moved toward home
    # If we bet home: aligned when odds_move > 0
    # If we bet away: aligned when odds_move < 0
    all_bets["smart_money_aligned"] = np.where(
        all_bets["bet_side"] == "home",
        all_bets["odds_move"] > 0,
        all_bets["odds_move"] < 0,
    )

    # Tag V3 agreement:
    # V3 picks home if v3_home_prob >= 0.5, else away
    # Bet picks home if bet_side == 'home', else away
    v3_picks_home = all_bets["v3_home_prob"] >= 0.5
    bet_picks_home = all_bets["bet_side"] == "home"
    all_bets["v3_agrees"] = (v3_picks_home == bet_picks_home) & all_bets["v3_home_prob"].notna()

    # -----------------------------------------------------------------------
    # Apply 7 hypotheses
    # -----------------------------------------------------------------------
    print("\n[4/4] Applying 7 filter hypotheses...")
    results = []

    # Baseline: current production (15-30% edge, no extra filter)
    mask_current = (all_bets["edge"] >= 0.15) & (all_bets["edge"] < 0.30)
    results.append(apply_filter(all_bets, "BASELINE: current prod (15-30%)", mask_current))

    # Also show the raw edge>=5% pool for reference
    results.append(apply_filter(all_bets, "REF: all edge>=5%", pd.Series(True, index=all_bets.index)))

    # H1: Favorites-only (odds < 2.0), using current 15-30% edge band
    mask_h1a = mask_current & (all_bets["odds"] < 2.0)
    results.append(apply_filter(all_bets, "H1a: favorites only (15-30%)", mask_h1a))

    # H1b: Favorites-only with broader edge band (5-30%)
    mask_h1b = (all_bets["edge"] >= 0.05) & (all_bets["edge"] < 0.30) & (all_bets["odds"] < 2.0)
    results.append(apply_filter(all_bets, "H1b: favorites only (5-30%)", mask_h1b))

    # H2a: Lower edge threshold 5-15%
    mask_h2a = (all_bets["edge"] >= 0.05) & (all_bets["edge"] < 0.15)
    results.append(apply_filter(all_bets, "H2a: edge 5-15% (any role)", mask_h2a))

    # H2b: Edge 10-20%
    mask_h2b = (all_bets["edge"] >= 0.10) & (all_bets["edge"] < 0.20)
    results.append(apply_filter(all_bets, "H2b: edge 10-20% (any role)", mask_h2b))

    # H3: Moderate favorite + moderate edge (odds 1.5-2.0 + edge 10-20%)
    mask_h3 = (
        (all_bets["edge"] >= 0.10) & (all_bets["edge"] < 0.20) &
        (all_bets["odds"] >= 1.5) & (all_bets["odds"] < 2.0)
    )
    results.append(apply_filter(all_bets, "H3: sweet-spot (1.5-2.0 odds, 10-20% edge)", mask_h3))

    # H4a: V3 agrees (any edge band)
    mask_h4a = mask_current & all_bets["v3_agrees"]
    results.append(apply_filter(all_bets, "H4a: V3 agrees (15-30%)", mask_h4a))

    # H4b: V3 agrees + broader edge
    mask_h4b = (all_bets["edge"] >= 0.05) & (all_bets["edge"] < 0.30) & all_bets["v3_agrees"]
    results.append(apply_filter(all_bets, "H4b: V3 agrees (5-30%)", mask_h4b))

    # H5a: Smart-money aligned (current 15-30% band)
    mask_h5a = mask_current & all_bets["smart_money_aligned"]
    results.append(apply_filter(all_bets, "H5a: smart-money aligned (15-30%)", mask_h5a))

    # H5b: Smart-money aligned + broader band
    mask_h5b = (all_bets["edge"] >= 0.05) & (all_bets["edge"] < 0.30) & all_bets["smart_money_aligned"]
    results.append(apply_filter(all_bets, "H5b: smart-money aligned (5-30%)", mask_h5b))

    # H6: Composite — favorites + V3-agree + smart-money
    mask_h6 = (
        (all_bets["edge"] >= 0.05) & (all_bets["edge"] < 0.30) &
        (all_bets["odds"] < 2.0) &
        all_bets["v3_agrees"] &
        all_bets["smart_money_aligned"]
    )
    results.append(apply_filter(all_bets, "H6: composite (fav + V3 + smart$)", mask_h6))

    # H6b: Composite lite — favorites + V3-agree (no smart-money)
    mask_h6b = (
        (all_bets["edge"] >= 0.05) & (all_bets["edge"] < 0.30) &
        (all_bets["odds"] < 2.0) &
        all_bets["v3_agrees"]
    )
    results.append(apply_filter(all_bets, "H6b: fav + V3-agree (no smart$)", mask_h6b))

    # H7: Kelly variants are just bet-sizing — can't truly re-test without
    # re-simulating bankroll. We approximate by computing ROI (profit/wagered)
    # which is scale-invariant, so Kelly fraction doesn't change ROI. Note this.
    # Flat $100 per bet would change the *amount* of profit but not the ROI.
    results.append({
        "filter": "H7: Kelly variants",
        "n_bets": len(all_bets[mask_current]),
        "roi": float("nan"),
        "note": "ROI is scale-invariant to Kelly fraction. Skipping here — "
                 "requires bankroll re-simulation. Will revisit in full build if "
                 "another hypothesis passes.",
    })

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("QUICKSCAN RESULTS")
    print("=" * 100)
    print(f"{'Filter':<40} {'Bets':>5} {'Win%':>6} {'ROI':>8} "
          f"{'ROI 95% CI':>17} {'Yr':>4} {'Hold':>8} {'FLAGS':>5} {'VERDICT':>5}")
    print(f"  Flags: N=n>=100, R=roi>=5%, Y=years>=5, C=CI>0, H=holdout>0")
    print("-" * 100)
    for s in results:
        if "roi" in s and not isinstance(s["roi"], float) or (isinstance(s["roi"], float) and not np.isnan(s.get("roi", 0))):
            print(format_row(s))
        else:
            print(f"{s['filter']:<40}  [skipped: {s.get('note', '')[:60]}]")

    # Pass summary
    passing = [s for s in results if isinstance(s.get("roi"), (int, float))
               and not np.isnan(s["roi"])
               and s["n_bets"] >= 100
               and s["roi"] >= 5.0
               and s["profitable_years"] >= 5
               and s["roi_ci_low"] > 0
               and s["holdout_roi"] > 0]

    print("\n" + "=" * 100)
    if passing:
        print(f"PASSING HYPOTHESES ({len(passing)}):")
        for s in passing:
            print(f"  * {s['filter']}: ROI={s['roi']:+.1f}%, n={s['n_bets']}, "
                  f"years={s['profitable_years']}/{s['total_years']}, "
                  f"holdout={s['holdout_roi']:+.1f}%")
        print("\n  -> Recommend proceeding to full backtest_value_v2.py build "
              "on these filter(s).")
    else:
        # Relaxed: flag those with 4/5 criteria (holdout might fail due to small n)
        near_pass = [s for s in results if isinstance(s.get("roi"), (int, float))
                     and not np.isnan(s["roi"])
                     and sum([
                         s["n_bets"] >= 100,
                         s["roi"] >= 5.0,
                         s["profitable_years"] >= 5,
                         s["roi_ci_low"] > 0,
                         s["holdout_roi"] > 0,
                     ]) >= 4]
        if near_pass:
            print(f"NO STRICT PASS. Near-passes (4/5 criteria):")
            for s in near_pass:
                print(f"  * {s['filter']}: ROI={s['roi']:+.1f}% "
                      f"(CI low {s['roi_ci_low']:+.1f}%), n={s['n_bets']}, "
                      f"years={s['profitable_years']}/{s['total_years']}, "
                      f"holdout={s['holdout_roi']:+.1f}%")
            print("\n  -> Worth one more iteration with refined filters or "
                  "walk-forward V3 before full build.")
        else:
            print("NO HYPOTHESIS PASSES. Consider:")
            print("  1. Abandon value-bet filter refinement as a path.")
            print("  2. Accept the market-beats-model finding as a wall.")
            print("  3. Pivot to D4 (monitoring) or other unsolved problems.")
    print("=" * 100)

    # Write report
    report = {
        "window_start_year": 2019,
        "window_end_year": 2025,
        "odds_source": "opening",
        "kelly_frac": 0.25,
        "total_bets_all_edges": int(len(all_bets)),
        "v3_leakage_caveat": (
            "V3 analytical model was trained on 2019-2025. Applying it to "
            "bets in that window is evaluation-on-training. H4/H6 results "
            "may be optimistic; a full V2 build should use walk-forward V3."
        ),
        "smart_money_merge_success": float(merged_rate),
        "v3_merge_success": float(v3_rate),
        "results": [{k: v for k, v in r.items()} for r in results],
        "passing_hypotheses": [s["filter"] for s in passing],
        "runtime_sec": time.time() - t_start,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {out_path}")
    print(f"Total runtime: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
