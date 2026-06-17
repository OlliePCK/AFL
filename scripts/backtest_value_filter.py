"""
Empirical backtest of the value bet filter.

The data sources analysis showed the market beats our betting model by
-0.00781 LL (4.5x noise floor). That implies many of our "edges" could
be model errors rather than true mispricing. This script tests the
current (15-30%) edge band against alternatives:

  * Edge band sweep: (5-10), (10-15), (15-20), (20-25), (25-30), (30-40)
  * Upper cap: 30% vs no cap
  * Movement agreement: all / AGREE-only / non-DISAGREE / no filter
  * Subpopulation breakdown: year, home vs away bet, favorite vs underdog

Uses `walk_forward_betting()` from src.value which trains a fresh model
per year on < Y data and places hypothetical bets on Y matches against
the actual historical odds (2019-2025).

Writes: data/analysis/value_filter_backtest.json
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
from src.value import walk_forward_betting  # noqa: E402


# Edge bands to evaluate (min, max).
EDGE_BANDS: list[tuple[float, float]] = [
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.25),
    (0.25, 0.30),
    (0.30, 0.40),
    (0.05, 0.30),   # Current-broad
    (0.15, 0.30),   # Current production
    (0.15, 1.00),   # No upper cap
    (0.10, 1.00),   # Lower threshold, no upper cap
]


def summarize_bets(bets: pd.DataFrame) -> dict:
    """Summary stats for a bet DataFrame (all bets pooled)."""
    if bets.empty:
        return {"n_bets": 0, "roi": 0, "profit": 0, "win_rate": 0,
                "avg_edge": 0, "avg_odds": 0, "total_wagered": 0}
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


def breakdown_by(bets: pd.DataFrame, by: str) -> dict:
    """Breakdown summary by a categorical column."""
    if bets.empty:
        return {}
    out = {}
    for key, group in bets.groupby(by):
        out[str(key)] = summarize_bets(group)
    return out


def tag_favorite_underdog(bets: pd.DataFrame) -> pd.DataFrame:
    """Tag each bet as 'favorite' (odds < 2.0) or 'underdog' (odds >= 2.0)."""
    bets = bets.copy()
    bets["role"] = np.where(bets["odds"] < 2.0, "favorite", "underdog")
    return bets


def main() -> None:
    out_path = PROJECT_ROOT / "data" / "analysis" / "value_filter_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 80)
    print("VALUE BET FILTER BACKTEST -- walk-forward 2019-2025")
    print("=" * 80)

    print("\nLoading + featuring master dataset...")
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv",
        parse_dates=["date"],
    )
    df = build_features(df)
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    print(f"  {len(df)} completed matches, years {df['year'].min()}-{df['year'].max()}")

    # Use a broad min_edge=0.05 so the WF function produces every bet we might
    # care about at any threshold; we filter per-band post-hoc.
    print("\nRunning walk-forward simulation (2019-2025, edge >= 5%)...")
    print("  This trains 7 separate models (one per val year) — ~30-60s...")
    t0 = time.time()
    wf = walk_forward_betting(
        df,
        min_edge=0.05,
        max_edge=1.00,
        kelly_frac=0.25,
        max_bet_pct=0.10,
        odds_source="opening",   # Opening odds are the pre-match betting signal
        start_year=2019,
        end_year=2025,
        calibrate=True,
        compare_sources=False,
    )
    print(f"  WF sim done ({time.time() - t0:.0f}s)")

    all_bets: pd.DataFrame = wf["all_bets"]
    if all_bets.empty:
        print("No bets generated — aborting.")
        return
    all_bets = tag_favorite_underdog(all_bets)
    print(f"  {len(all_bets)} total bets at edge>=5%")

    # -----------------------------------------------------------------------
    # Edge band sweep
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EDGE BAND SWEEP (opening odds, 25% Kelly, $1000 bankroll / year)")
    print("=" * 80)
    print(f"{'Band':>14} {'Bets':>6} {'Win%':>7} {'ROI':>8} "
          f"{'Profit':>10} {'AvgEdge':>8} {'AvgOdds':>8}")
    print("-" * 68)

    band_results = {}
    for lo, hi in EDGE_BANDS:
        mask = (all_bets["edge"] >= lo) & (all_bets["edge"] < hi)
        band_bets = all_bets[mask]
        s = summarize_bets(band_bets)
        key = f"{lo:.2f}-{hi:.2f}"
        band_results[key] = s
        print(f"{key:>14} {s['n_bets']:>6} {s['win_rate']:>6.1%} "
              f"{s['roi']:>+7.1f}% ${s['profit']:>9.0f} "
              f"{s['avg_edge']:>7.3f} {s['avg_odds']:>7.2f}")

    # -----------------------------------------------------------------------
    # Year stability for current (15-30%) and best-scanned bands
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("YEAR STABILITY (current 15-30% band)")
    print("=" * 80)

    current = all_bets[(all_bets["edge"] >= 0.15) & (all_bets["edge"] < 0.30)]
    print(f"{'Year':>6} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Profit':>10}")
    print("-" * 50)
    year_results = {}
    for yr, g in current.groupby("year"):
        s = summarize_bets(g)
        year_results[int(yr)] = s
        print(f"{int(yr):>6} {s['n_bets']:>6} {s['win_rate']:>6.1%} "
              f"{s['roi']:>+7.1f}% ${s['profit']:>9.0f}")

    # -----------------------------------------------------------------------
    # Subpopulation breakdowns at current 15-30% band
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUBPOPULATION BREAKDOWNS (15-30% band)")
    print("=" * 80)

    print("\n>> by bet side (home vs away):")
    side_results = breakdown_by(current, "bet_side")
    for k, s in side_results.items():
        print(f"   {k:>10}  n={s['n_bets']:>4}  win%={s['win_rate']:.1%}  "
              f"ROI={s['roi']:+.1f}%  profit=${s['profit']:.0f}  "
              f"avg_odds={s['avg_odds']:.2f}")

    print("\n>> by favorite / underdog (odds < 2.0 = favorite):")
    role_results = breakdown_by(current, "role")
    for k, s in role_results.items():
        print(f"   {k:>10}  n={s['n_bets']:>4}  win%={s['win_rate']:.1%}  "
              f"ROI={s['roi']:+.1f}%  profit=${s['profit']:.0f}  "
              f"avg_odds={s['avg_odds']:.2f}")

    # -----------------------------------------------------------------------
    # Broader edge bands with the same breakdowns
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON: CURRENT (15-30%) vs ALTERNATIVE BANDS by SUBPOPULATION")
    print("=" * 80)
    alt_bands = [(0.15, 0.30), (0.20, 0.30), (0.15, 1.00), (0.10, 1.00)]
    print(f"{'Band':>14} {'Role':>12} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Profit':>10}")
    print("-" * 70)
    alt_results = {}
    for lo, hi in alt_bands:
        key = f"{lo:.2f}-{hi:.2f}"
        mask = (all_bets["edge"] >= lo) & (all_bets["edge"] < hi)
        band_bets = all_bets[mask]
        alt_results[key] = {}
        for role in ["favorite", "underdog"]:
            sub = band_bets[band_bets["role"] == role]
            s = summarize_bets(sub)
            alt_results[key][role] = s
            print(f"{key:>14} {role:>12} {s['n_bets']:>6} {s['win_rate']:>6.1%} "
                  f"{s['roi']:>+7.1f}% ${s['profit']:>9.0f}")

    # -----------------------------------------------------------------------
    # Sensitivity: odds-bucket within current band (where does return come from?)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ODDS BUCKETS WITHIN 15-30% EDGE BAND")
    print("=" * 80)
    buckets = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 100.0)]
    print(f"{'Odds Range':>14} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Profit':>10}")
    print("-" * 55)
    odds_bucket_results = {}
    for lo, hi in buckets:
        sub = current[(current["odds"] >= lo) & (current["odds"] < hi)]
        s = summarize_bets(sub)
        odds_bucket_results[f"{lo:.1f}-{hi:.1f}"] = s
        if s["n_bets"] > 0:
            print(f"{lo:.1f}-{hi:.1f}".rjust(14),
                  f"{s['n_bets']:>6} {s['win_rate']:>6.1%} "
                  f"{s['roi']:>+7.1f}% ${s['profit']:>9.0f}")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    report = {
        "window_start_year": 2019,
        "window_end_year": 2025,
        "odds_source": "opening",
        "kelly_frac": 0.25,
        "max_bet_pct": 0.10,
        "total_bets_all_edges": int(len(all_bets)),
        "wf_summary": wf["summary"],
        "edge_band_sweep": band_results,
        "year_stability_current_band": {str(y): v for y, v in year_results.items()},
        "bet_side_breakdown_current_band": side_results,
        "role_breakdown_current_band": role_results,
        "alternative_bands_by_role": alt_results,
        "odds_buckets_within_current_band": odds_bucket_results,
        "runtime_sec": time.time() - t_start,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {out_path}")
    print(f"Total runtime: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
