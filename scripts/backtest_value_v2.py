"""
Value Bet Filter V2 — FULL WALK-FORWARD BUILD

Supersedes the V1 backtest (scripts/backtest_value_filter.py). The V1 script
showed the current 15-30% edge band was -4.3% ROI — market beat the model.
The V2 quickscan (scripts/backtest_value_v2_quickscan.py) found three filters
that lift ROI by 3-5x over baseline. This script is the clean walk-forward
test of those filters.

V2 vs V1:
  V1                              V2
  --                              --
  Post-hoc mask on pooled bets    Re-simulate bankroll per filter
  V3 applied with leakage         Walk-forward V3 (train <Y, predict Y)
  ROI + year count                + Wilson CI on win rate
                                  + Bootstrap 95% CI on ROI
                                  + 30% holdout (chronological)
                                  + Max drawdown per filter
                                  + Pass/fail against 5 criteria

Filter configurations evaluated (all atop the 15-30% edge band unless noted):
  BASELINE  no filter — current production
  H1        favorites only (bet odds < 2.0)
  H4-WF     V3 analytical agrees on direction (walk-forward V3)
  H5        smart-money aligned (odds_move toward our pick)
  H1+H5     favorites AND smart-money aligned
  H4+H5     V3 agrees AND smart-money aligned
  H1+H4     favorites AND V3 agrees
  H1+H4+H5  triple-stack (quickscan H6 underperformed — confirm)

Also tests broader (5-30%) edge band for the most promising single filter
to see if loosening the threshold adds volume without hurting ROI.

Pass bar (all 5 must hold):
  - ROI >= +5%
  - n >= 100
  - Profitable years >= 5/7
  - Bootstrap 95% CI lower bound > 0%
  - Holdout ROI (last 30%) > 0%

Writes: data/analysis/value_filter_v2_report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features  # noqa: E402
from src.model import BETTING_FEATURE_COLS, load_optimization_results  # noqa: E402
from src.value import (  # noqa: E402
    calculate_edge, decimal_odds_to_implied_prob,
    expected_value, kelly_fraction,
)

START_YEAR = 2019
END_YEAR = 2025
V3_FEATURES = ["implied_home_open", "overround_open", "home_line_close"]
V3_SEEDS = [42, 123, 256, 789, 1337]
INITIAL_BANKROLL = 1000.0


# ===========================================================================
# Statistical helpers
# ===========================================================================
def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def bootstrap_roi_ci(bets: pd.DataFrame, n_iter: int = 2000,
                     confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
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


def max_drawdown(bankroll_series: np.ndarray, initial: float) -> float:
    values = np.concatenate([[initial], bankroll_series])
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return float(drawdown.max())


# ===========================================================================
# Walk-forward V3 (analytical) predictions
# ===========================================================================
def train_v3_for_year(train_df: pd.DataFrame, val_df: pd.DataFrame,
                      params: dict) -> tuple[list[CatBoostClassifier], IsotonicRegression | None]:
    """Train a 5-seed V3 ensemble on train_df with val_df for early stopping."""
    models = []
    val_preds = []
    for seed in V3_SEEDS:
        p = {**params, "random_seed": seed, "iterations": 2000,
             "early_stopping_rounds": 100, "verbose": 0,
             "eval_metric": "Logloss", "use_best_model": True}
        if p.get("subsample", 1.0) < 1.0:
            p["bootstrap_type"] = "Bernoulli"
        m = CatBoostClassifier(**p)
        m.fit(
            Pool(train_df[V3_FEATURES], train_df["home_win"]),
            eval_set=Pool(val_df[V3_FEATURES], val_df["home_win"]),
        )
        models.append(m)
        val_preds.append(m.predict_proba(val_df[V3_FEATURES])[:, 1])

    # Fit isotonic calibrator on val set ensemble mean
    ens_val = np.mean(val_preds, axis=0)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(ens_val, val_df["home_win"].values)
    return models, cal


def predict_v3_ensemble(models: list[CatBoostClassifier],
                        calibrator: IsotonicRegression | None,
                        X: pd.DataFrame) -> np.ndarray:
    """Ensemble mean + isotonic calibrator → clipped probabilities."""
    probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    if calibrator is not None:
        probs = calibrator.predict(probs)
        probs = np.clip(probs, 0.02, 0.98)
    return probs


# ===========================================================================
# Walk-forward betting classifier (reuses src.value pattern)
# ===========================================================================
def train_betting_for_year(train_df: pd.DataFrame, val_df: pd.DataFrame
                            ) -> tuple[CatBoostClassifier, IsotonicRegression | None]:
    """Train betting classifier + isotonic calibrator for one year."""
    available = [c for c in BETTING_FEATURE_COLS if c in train_df.columns]
    target = "home_win"

    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=4,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        early_stopping_rounds=100, eval_metric="Logloss",
        use_best_model=True,
    )
    model.fit(
        Pool(train_df[available], train_df[target]),
        eval_set=Pool(val_df[available], val_df[target]),
    )

    cal = None
    if len(val_df) >= 30:
        val_probs = model.predict_proba(val_df[available])[:, 1]
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(val_probs, val_df[target].values)

    return model, cal


# ===========================================================================
# Cached per-year walk-forward predictions
# ===========================================================================
def run_walk_forward(df: pd.DataFrame) -> list[dict]:
    """For each year in [START_YEAR, END_YEAR], train classifier + V3, predict."""
    print(f"\n[WF] Walk-forward training for {START_YEAR}-{END_YEAR}...")
    available_betting = [c for c in BETTING_FEATURE_COLS if c in df.columns]

    # Load V3 hyperparams (reuse Optuna params — same as production V3 trainer)
    opt = load_optimization_results()
    if opt is None:
        raise RuntimeError("optimization_results.json not found")
    v3_params = dict(opt["best_params"])

    per_year: list[dict] = []
    target = "home_win"

    for year in range(START_YEAR, END_YEAR + 1):
        t0 = time.time()
        train_all = df[df["year"] < year].dropna(subset=[target])
        test_df = df[df["year"] == year].dropna(subset=[target])
        if train_all.empty or test_df.empty:
            print(f"  {year}: skipped (empty split)")
            continue

        val_year = year - 1
        train_inner = train_all[train_all["year"] < val_year]
        val_df = train_all[train_all["year"] == val_year]
        if val_df.empty:
            split_idx = int(len(train_all) * 0.8)
            train_inner = train_all.iloc[:split_idx]
            val_df = train_all.iloc[split_idx:]

        # Leakage check
        assert train_all["year"].max() < year, "LEAKAGE in betting WF"
        assert train_inner["year"].max() < year, "LEAKAGE in V3 WF"

        # 1. Betting classifier
        b_model, b_cal = train_betting_for_year(train_inner, val_df)
        raw_probs = b_model.predict_proba(test_df[available_betting])[:, 1]
        if b_cal is not None:
            probs = np.clip(b_cal.predict(raw_probs), 0.02, 0.98)
        else:
            probs = raw_probs

        # 2. V3 analytical (walk-forward — no leakage)
        v3_models, v3_cal = train_v3_for_year(train_inner, val_df, v3_params)
        v3_probs = predict_v3_ensemble(v3_models, v3_cal, test_df[V3_FEATURES])

        per_year.append({
            "year": year,
            "test_df": test_df.reset_index(drop=True),
            "probs": probs,
            "v3_probs": v3_probs,
            "best_iter_betting": int(b_model.best_iteration_),
        })

        print(f"  {year}: n_test={len(test_df)}, betting_iter={b_model.best_iteration_}, "
              f"v3_mean={v3_probs.mean():.3f} ({time.time() - t0:.0f}s)")

    return per_year


# ===========================================================================
# Filter-aware bet simulation
# ===========================================================================
def simulate_with_filter(per_year: list[dict],
                         filter_fn: Callable[..., bool],
                         min_edge: float = 0.15,
                         max_edge: float = 0.30,
                         kelly_frac: float = 0.25,
                         max_bet_pct: float = 0.10,
                         odds_source: str = "opening") -> pd.DataFrame:
    """Re-simulate betting per year with filter_fn applied post-edge-check.

    Each year starts with fresh $1000 (same convention as V1 backtest).
    The filter_fn receives all available context per candidate bet and
    returns True to accept, False to reject.
    """
    odds_cols = {"closing": ("home_odds_close", "away_odds_close"),
                 "opening": ("home_odds_open", "away_odds_open"),
                 "avg": ("home_odds_avg", "away_odds_avg")}
    ho_col, ao_col = odds_cols[odds_source]

    all_bets = []

    for yr_data in per_year:
        year = yr_data["year"]
        test_df = yr_data["test_df"]
        probs = yr_data["probs"]
        v3_probs = yr_data["v3_probs"]

        bankroll = INITIAL_BANKROLL
        for i, row in test_df.iterrows():
            model_home_prob = float(probs[i])
            model_away_prob = 1.0 - model_home_prob
            v3_prob = float(v3_probs[i])
            odds_move = row.get("odds_move")

            home_odds = row.get(ho_col)
            away_odds = row.get(ao_col)
            if pd.isna(home_odds) or pd.isna(away_odds):
                continue
            home_odds = float(home_odds)
            away_odds = float(away_odds)

            market_home, market_away = decimal_odds_to_implied_prob(home_odds, away_odds)
            home_edge = calculate_edge(model_home_prob, market_home)
            away_edge = calculate_edge(model_away_prob, market_away)

            # Pick the side with the higher edge
            bet_side = None
            edge = 0.0
            bet_odds = 0.0
            bet_prob = 0.0
            if home_edge >= min_edge and home_edge < max_edge and home_edge >= away_edge:
                bet_side, edge, bet_odds, bet_prob = "home", home_edge, home_odds, model_home_prob
            elif away_edge >= min_edge and away_edge < max_edge:
                bet_side, edge, bet_odds, bet_prob = "away", away_edge, away_odds, model_away_prob

            if bet_side is None:
                continue

            # Apply filter
            accept = filter_fn(
                row=row,
                bet_side=bet_side,
                edge=edge,
                bet_odds=bet_odds,
                bet_prob=bet_prob,
                v3_prob=v3_prob,
                odds_move=odds_move,
                min_edge=min_edge,
                max_edge=max_edge,
            )
            if not accept:
                continue

            # Kelly sizing
            bet_fraction = kelly_fraction(bet_prob, bet_odds, kelly_frac)
            bet_fraction = min(bet_fraction, max_bet_pct)
            bet_amount = bankroll * bet_fraction
            if bet_amount < 1.0:
                continue

            # Outcome
            actual_home_win = row["home_win"]
            won = (bet_side == "home" and actual_home_win == 1) or \
                  (bet_side == "away" and actual_home_win == 0)
            profit = bet_amount * (bet_odds - 1) if won else -bet_amount
            bankroll += profit

            all_bets.append({
                "date": row["date"],
                "year": year,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_side": bet_side,
                "bet_team": row["home_team"] if bet_side == "home" else row["away_team"],
                "model_prob": bet_prob,
                "market_prob": market_home if bet_side == "home" else market_away,
                "v3_prob": v3_prob,
                "edge": edge,
                "odds": bet_odds,
                "odds_move": odds_move,
                "ev_per_dollar": expected_value(bet_prob, bet_odds),
                "bet_amount": bet_amount,
                "bet_fraction": bet_fraction,
                "won": won,
                "profit": profit,
                "bankroll": bankroll,
            })

    return pd.DataFrame(all_bets)


# ===========================================================================
# Filter definitions
# ===========================================================================
def in_band(edge: float, min_edge: float, max_edge: float) -> bool:
    return min_edge <= edge < max_edge


def smart_money_aligned(bet_side: str, odds_move) -> bool:
    if pd.isna(odds_move):
        return False
    if bet_side == "home":
        return odds_move > 0
    return odds_move < 0


def v3_agrees(bet_side: str, v3_prob: float) -> bool:
    if pd.isna(v3_prob):
        return False
    v3_picks_home = v3_prob >= 0.5
    bet_picks_home = bet_side == "home"
    return v3_picks_home == bet_picks_home


def filter_baseline(**ctx):
    return True  # Edge band already enforced upstream


def filter_h1(**ctx):
    return ctx["bet_odds"] < 2.0


def filter_h4(**ctx):
    return v3_agrees(ctx["bet_side"], ctx["v3_prob"])


def filter_h5(**ctx):
    return smart_money_aligned(ctx["bet_side"], ctx["odds_move"])


def filter_h1_h5(**ctx):
    return (ctx["bet_odds"] < 2.0 and
            smart_money_aligned(ctx["bet_side"], ctx["odds_move"]))


def filter_h4_h5(**ctx):
    return (v3_agrees(ctx["bet_side"], ctx["v3_prob"]) and
            smart_money_aligned(ctx["bet_side"], ctx["odds_move"]))


def filter_h1_h4(**ctx):
    return (ctx["bet_odds"] < 2.0 and
            v3_agrees(ctx["bet_side"], ctx["v3_prob"]))


def filter_h1_h4_h5(**ctx):
    return (ctx["bet_odds"] < 2.0 and
            v3_agrees(ctx["bet_side"], ctx["v3_prob"]) and
            smart_money_aligned(ctx["bet_side"], ctx["odds_move"]))


FILTERS: dict[str, tuple[str, Callable]] = {
    "baseline (15-30%)": ("0.15 <= edge < 0.30, no filter", filter_baseline),
    "H1: favorites":      ("edge+1530, odds<2.0", filter_h1),
    "H4: V3 agrees (WF)": ("edge+1530, WF V3 direction matches bet", filter_h4),
    "H5: smart-$ aligned":("edge+1530, odds_move toward bet", filter_h5),
    "H1+H5":              ("H1 AND H5", filter_h1_h5),
    "H4+H5":              ("H4 AND H5", filter_h4_h5),
    "H1+H4":              ("H1 AND H4", filter_h1_h4),
    "H1+H4+H5":           ("triple stack", filter_h1_h4_h5),
}


# ===========================================================================
# Metric aggregation & reporting
# ===========================================================================
def summarize(bets: pd.DataFrame, filter_name: str, desc: str) -> dict:
    if bets.empty:
        return {
            "filter": filter_name, "desc": desc,
            "n_bets": 0, "win_rate": 0.0, "roi": 0.0, "profit": 0.0,
            "avg_edge": 0.0, "avg_odds": 0.0,
            "wilson_low": 0.0, "wilson_high": 0.0,
            "roi_ci_low": 0.0, "roi_ci_high": 0.0,
            "profitable_years": 0, "total_years": 0,
            "holdout_n": 0, "holdout_roi": 0.0,
            "max_drawdown": 0.0,
            "yearly": {},
            "passes": False,
        }

    wagered = bets["bet_amount"].sum()
    profit = bets["profit"].sum()
    wins = int(bets["won"].sum())
    n = len(bets)
    roi = profit / wagered * 100 if wagered > 0 else 0

    wlo, whi = wilson_ci(wins, n)
    rlo, rhi = bootstrap_roi_ci(bets)

    # Yearly
    yearly_dict = {}
    profitable = 0
    for yr, g in bets.groupby("year"):
        gw = g["bet_amount"].sum()
        gp = g["profit"].sum()
        gr = gp / gw * 100 if gw > 0 else 0
        yearly_dict[int(yr)] = {
            "n": int(len(g)), "roi": float(gr), "profit": float(gp),
            "win_rate": float(g["won"].mean()),
        }
        if gr > 0:
            profitable += 1

    # Holdout — last 30% chronologically
    sorted_bets = bets.sort_values("date")
    split_idx = int(len(sorted_bets) * 0.70)
    holdout = sorted_bets.iloc[split_idx:]
    if holdout.empty or holdout["bet_amount"].sum() == 0:
        h_roi = 0.0
    else:
        h_roi = holdout["profit"].sum() / holdout["bet_amount"].sum() * 100

    mdd = max_drawdown(bets["bankroll"].values, INITIAL_BANKROLL)

    passes = (n >= 100 and roi >= 5.0 and profitable >= 5
              and rlo > 0 and h_roi > 0)

    return {
        "filter": filter_name, "desc": desc,
        "n_bets": int(n), "win_rate": float(wins / n),
        "roi": float(roi), "profit": float(profit),
        "avg_edge": float(bets["edge"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "wilson_low": float(wlo), "wilson_high": float(whi),
        "roi_ci_low": float(rlo), "roi_ci_high": float(rhi),
        "profitable_years": profitable,
        "total_years": len(yearly_dict),
        "holdout_n": int(len(holdout)),
        "holdout_roi": float(h_roi),
        "max_drawdown": float(mdd),
        "yearly": yearly_dict,
        "passes": bool(passes),
    }


def print_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 120)
    print("V2 WALK-FORWARD BACKTEST — FILTER COMPARISON (15-30% edge band, 25% Kelly)")
    print("=" * 120)
    print(f"{'Filter':<20} {'Bets':>5} {'Win%':>6} {'ROI':>8} "
          f"{'ROI 95% CI':>17} {'Yr':>4} {'Hold':>8} {'MaxDD':>7} {'Verdict':>8}")
    print("-" * 120)
    for s in summaries:
        verdict = "PASS" if s["passes"] else "fail"
        print(
            f"{s['filter']:<20} {s['n_bets']:>5} {s['win_rate']:>6.1%} "
            f"{s['roi']:>+7.1f}% [{s['roi_ci_low']:>+6.1f}, {s['roi_ci_high']:>+6.1f}] "
            f"{s['profitable_years']}/{s['total_years']:<1} "
            f"{s['holdout_roi']:>+7.1f}% {s['max_drawdown']:>6.1%} "
            f"{verdict:>8}"
        )
    print("=" * 120)


def print_yearly_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 120)
    print("YEARLY ROI BY FILTER")
    print("=" * 120)
    years = list(range(START_YEAR, END_YEAR + 1))
    header = f"{'Filter':<20}" + "".join(f"{y:>10}" for y in years) + f"  {'Total':>8}"
    print(header)
    print("-" * 120)
    for s in summaries:
        row = f"{s['filter']:<20}"
        for y in years:
            yr_info = s["yearly"].get(y)
            if yr_info:
                row += f" {yr_info['roi']:>+8.1f}% ({yr_info['n']:>2})".rjust(10)
            else:
                row += f"{'n/a':>10}"
        row += f"  {s['roi']:>+7.1f}%"
        print(row)
    print("=" * 120)


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    out_path = PROJECT_ROOT / "data" / "analysis" / "value_filter_v2_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 100)
    print("VALUE FILTER V2 -- FULL WALK-FORWARD BACKTEST")
    print("=" * 100)

    # Load + feature
    print("\n[1/3] Loading + featuring master dataset...")
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv",
        parse_dates=["date"],
    )
    df = build_features(df)
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    print(f"  {len(df)} completed matches, years {df['year'].min()}-{df['year'].max()}")

    # Walk-forward training (classifier + V3)
    print("\n[2/3] Walk-forward training + caching predictions...")
    t0 = time.time()
    per_year = run_walk_forward(df)
    print(f"  [WF total: {time.time() - t0:.0f}s]")

    # Apply each filter and summarize
    print("\n[3/3] Applying filters + simulating bankrolls...")
    summaries = []
    for name, (desc, fn) in FILTERS.items():
        ts = time.time()
        bets = simulate_with_filter(per_year, fn)
        s = summarize(bets, name, desc)
        summaries.append(s)
        print(f"  {name:<22} n={s['n_bets']:>4} ROI={s['roi']:+.1f}% "
              f"CI=[{s['roi_ci_low']:+.1f},{s['roi_ci_high']:+.1f}] "
              f"years={s['profitable_years']}/{s['total_years']} "
              f"holdout={s['holdout_roi']:+.1f}% ({time.time() - ts:.0f}s)")

    # Reports
    print_table(summaries)
    print_yearly_table(summaries)

    # Summary of passes
    passing = [s for s in summaries if s["passes"]]
    print("\n" + "=" * 100)
    if passing:
        print(f"PASSING FILTERS ({len(passing)}):")
        for s in passing:
            print(f"  * {s['filter']}: ROI={s['roi']:+.1f}% "
                  f"(CI low {s['roi_ci_low']:+.1f}%), n={s['n_bets']}, "
                  f"years={s['profitable_years']}/{s['total_years']}, "
                  f"holdout={s['holdout_roi']:+.1f}%, maxDD={s['max_drawdown']:.0%}")

        best = max(passing, key=lambda s: s["roi"])
        print(f"\n  -> RECOMMEND promoting '{best['filter']}' to production.")
        print(f"     Update src/value.py defaults and scripts/automate.py logic.")
    else:
        near_pass = sorted(
            [s for s in summaries
             if sum([s["n_bets"] >= 100, s["roi"] >= 5.0,
                     s["profitable_years"] >= 5,
                     s["roi_ci_low"] > 0, s["holdout_roi"] > 0]) >= 4],
            key=lambda s: s["roi"], reverse=True,
        )
        if near_pass:
            print(f"NO STRICT PASS. Near-passes (4/5 criteria):")
            for s in near_pass[:3]:
                missing = []
                if s["n_bets"] < 100: missing.append(f"n={s['n_bets']}")
                if s["roi"] < 5.0: missing.append(f"roi={s['roi']:.1f}%")
                if s["profitable_years"] < 5: missing.append(
                    f"years={s['profitable_years']}/{s['total_years']}")
                if s["roi_ci_low"] <= 0: missing.append(f"CI_low={s['roi_ci_low']:.1f}%")
                if s["holdout_roi"] <= 0: missing.append(f"holdout={s['holdout_roi']:.1f}%")
                print(f"  * {s['filter']}: ROI={s['roi']:+.1f}%, miss: {', '.join(missing)}")
            print("\n  -> Consider softer promotion: use best filter as a 'quality gate' "
                  "not a hard constraint, or bet a smaller Kelly fraction on filtered bets.")
        else:
            print("NO FILTER PASSES. Walk-forward V3 likely killed the H4 signal.")
            print("  -> Consider: flat-unit sizing, longer-horizon backtest, "
                  "or accept market-beats-model as the wall.")
    print("=" * 100)

    # Report
    report = {
        "window_start_year": START_YEAR,
        "window_end_year": END_YEAR,
        "edge_band": [0.15, 0.30],
        "kelly_frac": 0.25,
        "max_bet_pct": 0.10,
        "odds_source": "opening",
        "initial_bankroll": INITIAL_BANKROLL,
        "filters_tested": list(FILTERS.keys()),
        "v3_protocol": "walk-forward — train V3 on <Y, predict Y (no leakage)",
        "pass_criteria": {
            "min_bets": 100, "min_roi": 5.0, "min_profitable_years": 5,
            "min_roi_ci_low": 0.0, "min_holdout_roi": 0.0,
        },
        "results": summaries,
        "passing_filters": [s["filter"] for s in passing],
        "runtime_sec": time.time() - t_start,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {out_path}")
    print(f"Total runtime: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
