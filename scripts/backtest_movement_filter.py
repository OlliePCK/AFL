"""
Second-layer filter test: does market odds movement (open->close) add signal
on top of the already-validated fav+10-25% filter?

Hypothesis: if we bet OPENING odds and the market subsequently moves toward
our pick (AGREE), that late money / sharper action confirms our edge. If
the market moves AWAY from our pick (DISAGREE), we're on the wrong side.

Method:
  1. Run walk_forward_betting 2019-2025 with min_edge=0.05 (capture all bets
     we might filter for).
  2. Join each bet back to master dataset to get home/away opening AND
     closing odds.
  3. Compute implied-probability movement toward our bet side:
        move = implied_close_ourside - implied_open_ourside
  4. Classify:
        AGREE    if move >  THRESH
        DISAGREE if move < -THRESH
        NEUTRAL  if |move| <= THRESH
  5. Slice by:
        - current production filter (edge 10-25%, favorites only)
        - full 969-bet pool
        - movement class x edge band
  6. Decide: does layering movement strengthen the filter?

Writes: data/analysis/movement_filter_backtest.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features  # noqa: E402
from src.value import (  # noqa: E402
    walk_forward_betting, decimal_odds_to_implied_prob,
)

MOVE_THRESH = 0.005   # 0.5% implied-prob move -> above noise; matches run_predictions.py


def classify_movement(bet_side: str,
                      home_open: float, away_open: float,
                      home_close: float, away_close: float) -> tuple[str, float]:
    """Classify market movement relative to our bet side.

    Returns (class, signed_move) where signed_move is the implied-probability
    change on OUR side of the bet (positive = market moved TOWARD our pick).
    """
    if any(pd.isna(x) for x in (home_open, away_open, home_close, away_close)):
        return "UNKNOWN", np.nan
    # Vig-adjusted implied probs at open and close
    imp_home_open, imp_away_open = decimal_odds_to_implied_prob(home_open, away_open)
    imp_home_close, imp_away_close = decimal_odds_to_implied_prob(home_close, away_close)
    if bet_side == "home":
        move = imp_home_close - imp_home_open
    else:
        move = imp_away_close - imp_away_open
    if move > MOVE_THRESH:
        cls = "AGREE"
    elif move < -MOVE_THRESH:
        cls = "DISAGREE"
    else:
        cls = "NEUTRAL"
    return cls, float(move)


def summarize(bets: pd.DataFrame) -> dict:
    """Summary stats for a bet DataFrame."""
    if bets.empty:
        return {"n_bets": 0, "roi": 0, "profit": 0, "win_rate": 0, "total_wagered": 0}
    wagered = bets["bet_amount"].sum()
    profit = bets["profit"].sum()
    return {
        "n_bets": int(len(bets)),
        "roi": float(profit / wagered * 100 if wagered > 0 else 0),
        "profit": float(profit),
        "win_rate": float(bets["won"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "total_wagered": float(wagered),
    }


def main() -> None:
    out_path = PROJECT_ROOT / "data" / "analysis" / "movement_filter_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 80)
    print("MOVEMENT FILTER BACKTEST — layering AGREE/DISAGREE on top of fav+10-25%")
    print("=" * 80)

    # --- 1. Featured master dataset (need open+close odds) -----------------
    print("\nLoading + featuring master dataset...")
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv",
        parse_dates=["date"],
    )
    df = build_features(df)
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)

    # --- 2. Walk-forward betting 2019-2025 --------------------------------
    print("\nRunning walk-forward simulation...")
    wf = walk_forward_betting(
        df, min_edge=0.05, max_edge=1.00, kelly_frac=0.25, max_bet_pct=0.10,
        odds_source="opening", start_year=2019, end_year=2025,
        calibrate=True, compare_sources=False,
    )
    bets: pd.DataFrame = wf["all_bets"]
    print(f"  {len(bets)} total bets at edge>=5%")

    # --- 3. Join to master to get open + close odds ------------------------
    key_cols = ["date", "home_team", "away_team"]
    odds_cols = ["home_odds_open", "away_odds_open",
                 "home_odds_close", "away_odds_close"]
    bets = bets.merge(df[key_cols + odds_cols], on=key_cols, how="left")

    # --- 4. Classify movement per bet --------------------------------------
    cls_list, move_list = [], []
    for _, r in bets.iterrows():
        cls, move = classify_movement(
            r["bet_side"],
            r["home_odds_open"], r["away_odds_open"],
            r["home_odds_close"], r["away_odds_close"],
        )
        cls_list.append(cls)
        move_list.append(move)
    bets["move_class"] = cls_list
    bets["move_value"] = move_list
    bets["role"] = np.where(bets["odds"] < 2.0, "favorite", "underdog")

    # Diagnostic: movement class distribution on full 969-bet pool
    print("\nMovement class distribution over all bets:")
    for c, g in bets.groupby("move_class"):
        print(f"  {c:>8s}  n={len(g):>4d}  mean_move={g['move_value'].mean():+.4f}")

    # --- 5. Analyses ------------------------------------------------------
    results = {}

    # (a) Movement class on the CURRENT production filter (fav, 10-25%)
    print("\n" + "=" * 80)
    print("(a) Production filter (favorites, edge 10-25%) split by movement")
    print("=" * 80)
    prod = bets[(bets["role"] == "favorite")
                & (bets["edge"] >= 0.10) & (bets["edge"] < 0.25)]
    print(f"{'Class':>10} {'Bets':>5} {'Win%':>6} {'ROI':>8} {'Profit':>10}")
    print("-" * 45)
    prod_out = {}
    for cls in ["AGREE", "NEUTRAL", "DISAGREE", "UNKNOWN"]:
        sub = prod[prod["move_class"] == cls]
        s = summarize(sub)
        prod_out[cls] = s
        if s["n_bets"] > 0:
            print(f"{cls:>10} {s['n_bets']:>5} {s['win_rate']:>5.1%} "
                  f"{s['roi']:>+7.1f}% ${s['profit']:>9.0f}")
    # Combined views
    prod_out["ALL"] = summarize(prod)
    prod_out["AGREE+NEUTRAL"] = summarize(prod[prod["move_class"].isin(["AGREE", "NEUTRAL"])])
    prod_out["AGREE_ONLY"] = summarize(prod[prod["move_class"] == "AGREE"])
    prod_out["NOT_DISAGREE"] = summarize(prod[prod["move_class"] != "DISAGREE"])
    print()
    for k in ["ALL", "AGREE+NEUTRAL", "AGREE_ONLY", "NOT_DISAGREE"]:
        s = prod_out[k]
        print(f"  {k:>15}  n={s['n_bets']:>4d}  win%={s['win_rate']:>5.1%}  "
              f"ROI={s['roi']:+.1f}%  profit=${s['profit']:.0f}")
    results["production_filter_by_movement"] = prod_out

    # (b) Movement class on the BASE pool (everything, edge>=5%)
    print("\n" + "=" * 80)
    print("(b) FULL bet pool (edge>=5%) split by movement")
    print("=" * 80)
    print(f"{'Class':>10} {'Bets':>5} {'Win%':>6} {'ROI':>8} {'Profit':>10}")
    print("-" * 45)
    full_out = {}
    for cls in ["AGREE", "NEUTRAL", "DISAGREE", "UNKNOWN"]:
        sub = bets[bets["move_class"] == cls]
        s = summarize(sub)
        full_out[cls] = s
        if s["n_bets"] > 0:
            print(f"{cls:>10} {s['n_bets']:>5} {s['win_rate']:>5.1%} "
                  f"{s['roi']:>+7.1f}% ${s['profit']:>9.0f}")
    results["full_pool_by_movement"] = full_out

    # (c) Movement x edge band (does AGREE help at higher edges?)
    print("\n" + "=" * 80)
    print("(c) Edge band x movement class (favorites only)")
    print("=" * 80)
    edge_bands = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
                  (0.25, 0.30)]
    print(f"{'Band':>12} {'Class':>10} {'Bets':>5} {'Win%':>6} {'ROI':>8} {'Profit':>10}")
    print("-" * 60)
    band_out = {}
    for lo, hi in edge_bands:
        key = f"{lo:.2f}-{hi:.2f}"
        band_out[key] = {}
        mask = (bets["role"] == "favorite") & (bets["edge"] >= lo) & (bets["edge"] < hi)
        for cls in ["AGREE", "NEUTRAL", "DISAGREE"]:
            sub = bets[mask & (bets["move_class"] == cls)]
            s = summarize(sub)
            band_out[key][cls] = s
            if s["n_bets"] > 0:
                print(f"{key:>12} {cls:>10} {s['n_bets']:>5} {s['win_rate']:>5.1%} "
                      f"{s['roi']:>+7.1f}% ${s['profit']:>9.0f}")
        print()
    results["edge_band_x_movement_favorites"] = band_out

    # (d) Best combined strategy candidates
    print("\n" + "=" * 80)
    print("(d) Strategy candidates vs CURRENT prod filter baseline")
    print("=" * 80)
    candidates = {
        "CURRENT (fav, 10-25%, all moves)":
            prod,
        "fav, 10-25%, NOT_DISAGREE (AGREE+NEUTRAL+UNKNOWN)":
            prod[prod["move_class"] != "DISAGREE"],
        "fav, 10-25%, AGREE only":
            prod[prod["move_class"] == "AGREE"],
        "fav, 5-25%, NOT_DISAGREE (wider band, no disagree)":
            bets[(bets["role"] == "favorite") & (bets["edge"] >= 0.05) & (bets["edge"] < 0.25)
                 & (bets["move_class"] != "DISAGREE")],
        "fav, any edge>=5%, AGREE only":
            bets[(bets["role"] == "favorite") & (bets["edge"] >= 0.05) & (bets["move_class"] == "AGREE")],
        "all roles, any edge>=5%, AGREE only":
            bets[(bets["edge"] >= 0.05) & (bets["move_class"] == "AGREE")],
    }
    print(f"{'Strategy':>55} {'Bets':>5} {'Win%':>6} {'ROI':>8} {'Profit':>10}")
    print("-" * 95)
    cand_out = {}
    for name, sub in candidates.items():
        s = summarize(sub)
        cand_out[name] = s
        print(f"{name:>55} {s['n_bets']:>5} {s['win_rate']:>5.1%} "
              f"{s['roi']:>+7.1f}% ${s['profit']:>9.0f}")
    results["strategy_candidates"] = cand_out

    # (e) Year stability on the best candidate
    print("\n" + "=" * 80)
    print("(e) Year stability for fav+10-25%+NOT_DISAGREE")
    print("=" * 80)
    best = prod[prod["move_class"] != "DISAGREE"]
    print(f"{'Year':>6} {'Bets':>5} {'Win%':>6} {'ROI':>8} {'Profit':>10}")
    print("-" * 45)
    year_out = {}
    for yr, g in best.groupby("year"):
        s = summarize(g)
        year_out[int(yr)] = s
        print(f"{int(yr):>6} {s['n_bets']:>5} {s['win_rate']:>5.1%} "
              f"{s['roi']:>+7.1f}% ${s['profit']:>9.0f}")
    results["year_stability_best_candidate"] = year_out

    # --- 6. Report --------------------------------------------------------
    report = {
        "move_threshold": MOVE_THRESH,
        "n_bets_total": int(len(bets)),
        "n_bets_with_known_move": int((bets["move_class"] != "UNKNOWN").sum()),
        **results,
        "runtime_sec": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {out_path}")
    print(f"Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
