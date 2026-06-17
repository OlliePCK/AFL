"""
Odds comparison and value detection for AFL betting simulation.

- Converts decimal odds to implied probability (vig-adjusted)
- Compares model probability vs market probability
- Applies fractional Kelly Criterion for bet sizing
- Backtests over historical data
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import PROJECT_ROOT
from src.model import get_feature_cols
from src.utils import setup_logging

log = setup_logging()


def decimal_odds_to_implied_prob(home_odds: float, away_odds: float) -> tuple[float, float]:
    """Convert decimal odds to vig-adjusted (fair) implied probabilities."""
    raw_home = 1.0 / home_odds
    raw_away = 1.0 / away_odds
    overround = raw_home + raw_away
    return raw_home / overround, raw_away / overround


def calculate_edge(model_prob: float, fair_implied_prob: float) -> float:
    """Calculate the edge: model_probability - fair_implied_probability."""
    return model_prob - fair_implied_prob


def kelly_fraction(model_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly Criterion bet sizing.

    Args:
        model_prob: Model's predicted probability of winning
        decimal_odds: Decimal odds offered by bookmaker
        fraction: Kelly fraction (0.25 = quarter Kelly, conservative)

    Returns:
        Fraction of bankroll to bet (0 if negative edge)
    """
    b = decimal_odds - 1.0  # net odds (profit per dollar wagered)
    q = 1.0 - model_prob
    kelly = (model_prob * b - q) / b
    return max(0.0, kelly * fraction)


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """Expected value per dollar wagered."""
    return model_prob * (decimal_odds - 1.0) - (1.0 - model_prob)


def smart_money_aligned(bet_side: str, odds_move) -> bool:
    """Return True iff the market has moved TOWARD our pick since opening.

    Backtest semantics: odds_move = implied_home_close - implied_home_open
        > 0 → home implied prob rose → market shortened home side
        < 0 → home implied prob fell → market shortened away side

    Live semantics: equivalent to implied_move from
    odds_monitor.compute_movement (first-snapshot vs latest-snapshot), or
    (current_implied_home - implied_home_open) where implied_home_open is
    aussportsbetting's opening line.

    V2 backtest (scripts/backtest_value_v2.py, 2019-2025 walk-forward):
      baseline (15-30% edge, no filter)      : 301 bets, +3.3% ROI, 4/7 yrs
      +smart-money aligned (V2 H5)           : 211 bets, +12.8% ROI, 5/7 yrs

    Returns False when odds_move is unavailable (NaN/None) — treated as a
    hard gate: without a movement signal we cannot confirm alignment.
    """
    if odds_move is None:
        return False
    try:
        move = float(odds_move)
    except (TypeError, ValueError):
        return False
    if np.isnan(move):
        return False
    if bet_side == "home":
        return move > 0
    return move < 0


def simulate_betting(df: pd.DataFrame, model: CatBoostClassifier | None = None,
                     initial_bankroll: float = 1000.0,
                     kelly_frac: float = 0.25,
                     min_edge: float = 0.05,
                     max_edge: float = 0.30,
                     max_bet_pct: float = 0.10,
                     odds_source: str = "closing",
                     precomputed_probs: np.ndarray | None = None,
                     apply_smart_money_filter: bool = False) -> pd.DataFrame:
    """Run a betting simulation using model predictions vs bookmaker odds.

    Uses real bookmaker odds when available (from aussportsbetting.com).
    Falls back to Squiggle consensus as a proxy when odds are missing.

    Args:
        df: Featured dataset with odds columns and model predictions
        model: Trained CatBoost model (not needed if precomputed_probs given)
        initial_bankroll: Starting bankroll in dollars
        kelly_frac: Kelly fraction for bet sizing
        min_edge: Minimum edge required to place a bet
        max_edge: Maximum edge — skip extreme disagreements (likely model errors)
        max_bet_pct: Maximum bet as fraction of bankroll
        odds_source: Which odds to use — "closing", "opening", or "avg"
        precomputed_probs: Pre-computed home win probabilities (skips model.predict_proba)
        apply_smart_money_filter: When True, reject bets where odds_move does
            not align with the bet side (V2 H5 filter). Requires the df to
            have an `odds_move` column.

    Returns:
        DataFrame with bet-by-bet simulation results
    """
    if precomputed_probs is not None:
        probs = precomputed_probs
    elif model is not None:
        available_features = [c for c in get_feature_cols() if c in df.columns]
        probs = model.predict_proba(df[available_features])[:, 1]
    else:
        raise ValueError("Either model or precomputed_probs must be provided")

    # Map odds_source to column names
    odds_col_map = {
        "closing": ("home_odds_close", "away_odds_close"),
        "opening": ("home_odds_open", "away_odds_open"),
        "avg": ("home_odds_avg", "away_odds_avg"),
    }
    home_odds_col, away_odds_col = odds_col_map.get(odds_source, odds_col_map["closing"])

    bankroll = initial_bankroll
    bets = []
    odds_used = {"real": 0, "proxy": 0, "skipped": 0}

    for i, (_, row) in enumerate(df.iterrows()):
        model_home_prob = probs[i]
        model_away_prob = 1.0 - model_home_prob

        # Try real bookmaker odds first
        home_odds = row.get(home_odds_col)
        away_odds = row.get(away_odds_col)
        using_real_odds = pd.notna(home_odds) and pd.notna(away_odds)

        if using_real_odds:
            home_odds = float(home_odds)
            away_odds = float(away_odds)
            # Vig-adjusted fair probabilities
            market_home_prob, market_away_prob = decimal_odds_to_implied_prob(home_odds, away_odds)
            odds_used["real"] += 1
        else:
            # Fallback: Squiggle consensus
            market_home_prob = row.get("mean_hconfidence")
            if pd.isna(market_home_prob):
                odds_used["skipped"] += 1
                continue
            market_home_prob = float(market_home_prob) / 100.0
            market_away_prob = 1.0 - market_home_prob
            home_odds = 1.0 / max(market_home_prob, 0.01)
            away_odds = 1.0 / max(market_away_prob, 0.01)
            odds_used["proxy"] += 1

        # Check both sides for value
        home_edge = calculate_edge(model_home_prob, market_home_prob)
        away_edge = calculate_edge(model_away_prob, market_away_prob)

        bet_side = None
        edge = 0
        bet_odds = 0
        bet_prob = 0

        if home_edge >= min_edge and home_edge < max_edge and home_edge >= away_edge:
            bet_side = "home"
            edge = home_edge
            bet_odds = home_odds
            bet_prob = model_home_prob
        elif away_edge >= min_edge and away_edge < max_edge:
            bet_side = "away"
            edge = away_edge
            bet_odds = away_odds
            bet_prob = model_away_prob

        if bet_side is None:
            continue

        # V2 H5 filter: require the market to have moved toward our pick
        # since opening. See smart_money_aligned() for backtest validation.
        if apply_smart_money_filter:
            if not smart_money_aligned(bet_side, row.get("odds_move")):
                continue

        # Kelly bet sizing
        bet_fraction = kelly_fraction(bet_prob, bet_odds, kelly_frac)
        bet_fraction = min(bet_fraction, max_bet_pct)  # Cap max bet
        bet_amount = bankroll * bet_fraction

        if bet_amount < 1.0:  # Minimum bet
            continue

        # Determine outcome
        actual_home_win = row["home_win"]
        won = (bet_side == "home" and actual_home_win == 1) or \
              (bet_side == "away" and actual_home_win == 0)

        profit = bet_amount * (bet_odds - 1) if won else -bet_amount
        bankroll += profit

        bets.append({
            "date": row["date"],
            "year": row["year"],
            "round": row["round"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "bet_side": bet_side,
            "bet_team": row["home_team"] if bet_side == "home" else row["away_team"],
            "model_prob": bet_prob,
            "market_prob": market_home_prob if bet_side == "home" else market_away_prob,
            "edge": edge,
            "odds": bet_odds,
            "ev_per_dollar": expected_value(bet_prob, bet_odds),
            "bet_amount": bet_amount,
            "bet_fraction": bet_fraction,
            "won": won,
            "profit": profit,
            "bankroll": bankroll,
            "real_odds": using_real_odds,
        })

    bets_df = pd.DataFrame(bets)

    if bets_df.empty:
        log.info("No bets placed (no edges found)")
        return bets_df

    # Summary stats
    n_bets = len(bets_df)
    n_wins = bets_df["won"].sum()
    total_wagered = bets_df["bet_amount"].sum()
    total_profit = bets_df["profit"].sum()
    roi = total_profit / total_wagered * 100 if total_wagered > 0 else 0
    final_bankroll = bets_df["bankroll"].iloc[-1]
    real_pct = bets_df["real_odds"].mean() * 100

    log.info(f"=== Betting Simulation Results ===")
    log.info(f"Period: {bets_df['date'].min()} to {bets_df['date'].max()}")
    log.info(f"Odds source: {odds_source} ({real_pct:.0f}% real bookmaker odds)")
    log.info(f"  Real odds: {odds_used['real']}, Proxy: {odds_used['proxy']}, Skipped: {odds_used['skipped']}")
    log.info(f"Bets placed:  {n_bets}")
    log.info(f"Win rate:     {n_wins}/{n_bets} ({n_wins/n_bets:.1%})")
    log.info(f"Total wagered: ${total_wagered:.2f}")
    log.info(f"Total profit:  ${total_profit:.2f}")
    log.info(f"ROI:           {roi:.1f}%")
    log.info(f"Bankroll:      ${initial_bankroll:.2f} -> ${final_bankroll:.2f}")
    log.info(f"Avg edge:      {bets_df['edge'].mean():.3f}")
    log.info(f"Avg bet size:  ${bets_df['bet_amount'].mean():.2f}")

    return bets_df


def plot_bankroll(bets_df: pd.DataFrame, initial_bankroll: float = 1000.0,
                  save_path=None):
    """Plot bankroll over time."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Bankroll curve
    ax = axes[0]
    ax.plot(range(len(bets_df)), bets_df["bankroll"], "b-", linewidth=1.5)
    ax.axhline(y=initial_bankroll, color="gray", linestyle="--", alpha=0.5, label="Starting bankroll")
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Bankroll Over Time")
    ax.legend()

    # Cumulative profit
    ax = axes[1]
    cum_profit = bets_df["profit"].cumsum()
    ax.plot(range(len(bets_df)), cum_profit, "g-" if cum_profit.iloc[-1] > 0 else "r-", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Cumulative Profit ($)")
    ax.set_title("Cumulative Profit/Loss")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info(f"Saved bankroll plot to {save_path}")
    plt.close()


def run_simulation():
    """Run the full simulation pipeline."""
    from src.features import build_features

    # Load data and model
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    model_path = PROJECT_ROOT / "data" / "model.cbm"

    df = pd.read_csv(master_path, parse_dates=["date"])
    df = build_features(df)

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    # Simulate on test period only (2024-2025)
    test_df = df[(df["year"] >= 2024) & (df["year"] <= 2025)].copy()
    test_df = test_df.dropna(subset=["mean_hconfidence"])

    log.info(f"Running simulation on {len(test_df)} matches (2024-2025)")

    bets_df = simulate_betting(test_df, model)

    if not bets_df.empty:
        # Save results
        bets_df.to_csv(PROJECT_ROOT / "data" / "master" / "simulation_results.csv", index=False)

        # Plot
        plots_dir = PROJECT_ROOT / "data" / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_bankroll(bets_df, save_path=plots_dir / "bankroll.png")

    return bets_df


def walk_forward_betting(df: pd.DataFrame,
                         min_edge: float = 0.15,
                         max_edge: float = 0.30,
                         kelly_frac: float = 0.25,
                         max_bet_pct: float = 0.10,
                         odds_source: str = "opening",
                         start_year: int = 2015,
                         end_year: int = 2025,
                         calibrate: bool = True,
                         compare_sources: bool = False) -> dict:
    """Walk-forward betting simulation across multiple years.

    For each year Y from start_year to end_year:
    1. Train a fresh odds-free model on years < Y
    2. Use year Y-1 as CatBoost early-stopping validation
    3. Optionally calibrate on Y-1 out-of-fold predictions
    4. Predict year Y matches
    5. Run simulate_betting() on year Y using actual bookmaker odds

    Returns:
        dict with "yearly", "all_bets", "summary" keys
    """
    from catboost import CatBoostClassifier, Pool
    from sklearn.isotonic import IsotonicRegression
    from src.model import BETTING_FEATURE_COLS

    available = [c for c in BETTING_FEATURE_COLS if c in df.columns]
    target = "home_win"

    yearly_results = []
    all_bets = []
    opening_yearly = []  # For compare_sources

    for year in range(start_year, end_year + 1):
        train_all = df[df["year"] < year].dropna(subset=[target])
        test_df = df[df["year"] == year].dropna(subset=[target])

        if train_all.empty or test_df.empty:
            log.info(f"WF betting {year}: skipped (empty split)")
            continue

        # Split training into train_inner (< Y-1) and val (Y-1)
        val_year = year - 1
        train_inner = train_all[train_all["year"] < val_year]
        val_df = train_all[train_all["year"] == val_year]

        # If val is empty, use last 20% of training data
        if val_df.empty:
            split_idx = int(len(train_all) * 0.8)
            train_inner = train_all.iloc[:split_idx]
            val_df = train_all.iloc[split_idx:]

        # Leakage check
        assert train_all["year"].max() < year, \
            f"Leakage! Train max year {train_all['year'].max()} >= test year {year}"

        # Train fresh model (same hyperparams as production)
        model = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=4,
            l2_leaf_reg=3, random_seed=42, verbose=0,
            early_stopping_rounds=100, eval_metric="Logloss",
            use_best_model=True,
        )
        model.fit(
            Pool(train_inner[available], train_inner[target]),
            eval_set=Pool(val_df[available], val_df[target]),
        )

        # Predict on test year
        raw_probs = model.predict_proba(test_df[available])[:, 1]

        # Optional calibration
        if calibrate and len(val_df) >= 30:
            val_probs = model.predict_proba(val_df[available])[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(val_probs, val_df[target].values)
            probs = np.clip(cal.predict(raw_probs), 0.02, 0.98)
        else:
            probs = raw_probs

        # Check odds coverage
        odds_col = {"closing": "home_odds_close", "opening": "home_odds_open",
                     "avg": "home_odds_avg"}
        odds_coverage = test_df[odds_col.get(odds_source, "home_odds_close")].notna().mean()

        # Simulate betting with pre-computed probs
        bets = simulate_betting(
            test_df, model=None, precomputed_probs=probs,
            kelly_frac=kelly_frac, min_edge=min_edge, max_edge=max_edge,
            max_bet_pct=max_bet_pct, odds_source=odds_source,
        )

        # Per-year metrics
        year_info = {
            "year": year,
            "train_size": len(train_all),
            "test_size": len(test_df),
            "best_iter": model.best_iteration_,
            "odds_coverage": odds_coverage,
        }

        if not bets.empty:
            n_bets = len(bets)
            total_wagered = bets["bet_amount"].sum()
            total_profit = bets["profit"].sum()
            year_info.update({
                "n_bets": n_bets,
                "win_rate": bets["won"].mean(),
                "total_wagered": total_wagered,
                "total_profit": total_profit,
                "roi": total_profit / total_wagered * 100 if total_wagered > 0 else 0,
                "avg_edge": bets["edge"].mean(),
                "max_drawdown": _max_drawdown(bets["bankroll"].values, 1000.0),
            })
            bets["test_year"] = year
            all_bets.append(bets)
        else:
            year_info.update({
                "n_bets": 0, "win_rate": 0, "total_wagered": 0,
                "total_profit": 0, "roi": 0, "avg_edge": 0, "max_drawdown": 0,
            })

        yearly_results.append(year_info)

        log.info(f"WF betting {year}: {year_info['n_bets']} bets, "
                 f"ROI={year_info['roi']:+.1f}%, "
                 f"profit=${year_info['total_profit']:.0f}, "
                 f"odds_cov={odds_coverage:.0%}, "
                 f"iter={model.best_iteration_}")

        # Compare opening vs closing
        if compare_sources and odds_source != "opening":
            open_bets = simulate_betting(
                test_df, model=None, precomputed_probs=probs,
                kelly_frac=kelly_frac, min_edge=min_edge, max_edge=max_edge,
                max_bet_pct=max_bet_pct, odds_source="opening",
            )
            open_info = {"year": year}
            if not open_bets.empty:
                ow = open_bets["bet_amount"].sum()
                op = open_bets["profit"].sum()
                open_info.update({
                    "n_bets": len(open_bets),
                    "roi": op / ow * 100 if ow > 0 else 0,
                    "total_profit": op,
                })
            else:
                open_info.update({"n_bets": 0, "roi": 0, "total_profit": 0})
            opening_yearly.append(open_info)

    # Concatenate all bets
    all_bets_df = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()

    # Build summary
    yearly_df = pd.DataFrame(yearly_results)
    summary = {}
    if not all_bets_df.empty:
        total_w = all_bets_df["bet_amount"].sum()
        total_p = all_bets_df["profit"].sum()
        summary = {
            "total_bets": len(all_bets_df),
            "overall_roi": total_p / total_w * 100 if total_w > 0 else 0,
            "overall_profit": total_p,
            "win_rate": all_bets_df["won"].mean(),
            "avg_edge": all_bets_df["edge"].mean(),
            "profitable_years": int((yearly_df["roi"] > 0).sum()),
            "total_years": len(yearly_df),
            "real_odds_pct": all_bets_df["real_odds"].mean() * 100,
        }

    # Print summary table
    log.info(f"\n{'='*80}")
    log.info(f"WALK-FORWARD BETTING SIMULATION ({start_year}-{end_year})")
    log.info(f"{'='*80}")
    log.info(f"{'Year':>6} {'Bets':>6} {'Win%':>7} {'ROI':>8} "
             f"{'Profit':>10} {'AvgEdge':>8} {'OddsCov':>8}")
    log.info(f"{'-'*80}")
    for r in yearly_results:
        log.info(f"{r['year']:>6} {r['n_bets']:>6} {r['win_rate']:>6.1%} "
                 f"{r['roi']:>+7.1f}% ${r['total_profit']:>9.0f} "
                 f"{r['avg_edge']:>7.3f} {r['odds_coverage']:>7.0%}")
    if summary:
        log.info(f"{'-'*80}")
        log.info(f"{'TOTAL':>6} {summary['total_bets']:>6} {summary['win_rate']:>6.1%} "
                 f"{summary['overall_roi']:>+7.1f}% ${summary['overall_profit']:>9.0f} "
                 f"{summary['avg_edge']:>7.3f}")
        log.info(f"Profitable years: {summary['profitable_years']}/{summary['total_years']}")
    log.info(f"{'='*80}")

    # Opening vs closing comparison
    if compare_sources and opening_yearly:
        log.info(f"\n{'='*80}")
        log.info(f"OPENING vs CLOSING ODDS COMPARISON")
        log.info(f"{'='*80}")
        log.info(f"{'Year':>6} {'Close ROI':>10} {'Open ROI':>10} {'Delta':>8}")
        log.info(f"{'-'*80}")
        for cr, opr in zip(yearly_results, opening_yearly):
            delta = opr["roi"] - cr["roi"]
            log.info(f"{cr['year']:>6} {cr['roi']:>+9.1f}% {opr['roi']:>+9.1f}% "
                     f"{delta:>+7.1f}%")
        log.info(f"{'='*80}")

    result = {
        "yearly": yearly_results,
        "all_bets": all_bets_df,
        "summary": summary,
    }
    if compare_sources:
        result["opening_yearly"] = opening_yearly

    return result


def _max_drawdown(bankroll_series: np.ndarray, initial: float) -> float:
    """Calculate maximum drawdown from peak."""
    values = np.concatenate([[initial], bankroll_series])
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return float(drawdown.max())


def plot_yearly_roi(results: dict, save_path=None):
    """Bar chart per year + cumulative profit line from walk-forward results."""
    import matplotlib.pyplot as plt

    yearly = pd.DataFrame(results["yearly"])
    if yearly.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Bar chart: ROI per year
    colors = ["green" if r > 0 else "red" for r in yearly["roi"]]
    ax1.bar(yearly["year"].astype(str), yearly["roi"], color=colors, alpha=0.7)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylabel("ROI (%)")
    ax1.set_title("Walk-Forward Betting: ROI by Year")
    for i, (yr, roi, nb) in enumerate(zip(yearly["year"], yearly["roi"], yearly["n_bets"])):
        ax1.text(i, roi + (1 if roi >= 0 else -3), f"{nb}b", ha="center", fontsize=8)

    # Cumulative profit line
    all_bets = results.get("all_bets")
    if all_bets is not None and not all_bets.empty:
        cum_profit = all_bets["profit"].cumsum()
        ax2.plot(range(len(cum_profit)), cum_profit, "b-", linewidth=1.5)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Bet number")
        ax2.set_ylabel("Cumulative Profit ($)")
        ax2.set_title("Cumulative Profit Across All Years")

        # Mark year boundaries
        for yr in sorted(all_bets["test_year"].unique()):
            yr_mask = all_bets["test_year"] == yr
            first_pos = yr_mask.values.argmax()
            if first_pos > 0:
                ax2.axvline(x=first_pos, color="gray", linestyle=":", alpha=0.3)
                ax2.text(first_pos, cum_profit.iloc[-1] * 0.9, str(yr),
                         fontsize=7, alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info(f"Saved yearly ROI plot to {save_path}")
    plt.close()
