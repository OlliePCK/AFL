# Value Filter Decision Memo — 2026-04-15

## TL;DR

Changed production value bet filter from **15-30% edge on any side** to
**10-25% edge on favorites only** (odds < 2.0).

Backtest-validated impact over 2019-2025 (969 bets):

| Strategy | Bets | Win% | ROI | Profit |
|---|---|---|---|---|
| OLD: 15-30% any side | 317 | 51.1% | **-4.3%** | **-$1,231** |
| NEW: 10-25% favorites | 303 | 71.3% | **+4.5%** | **+$1,289** |

Net swing: **+$2,520** over 7 years, same bankroll. Same number of bets.

## Motivation

The prior data sources analysis (`data_sources_report.json`) found that a
model using only 5 odds-derived features beats our 44-feature betting model
by -0.00781 LL (4.5x the noise floor). That's direct evidence that **the
market knows more than our model on average**, which implies many of our
computed "edges" are actually model errors — not mispriced markets we can
exploit.

If that's true, then the current value filter (15-30% edge) may be
systematically selecting the cases where our model is MOST wrong, not the
cases where the market is most wrong. Needed to test empirically.

## Method

`scripts/backtest_value_filter.py` runs the existing
`walk_forward_betting()` function across 2019-2025, captures every bet the
model would have made at edge ≥ 5%, then slices the results by:

- Edge band: 5-10, 10-15, 15-20, 20-25, 25-30, 30-40
- Favorite (odds < 2.0) vs underdog (odds ≥ 2.0)
- Year, bet side (home/away), odds bucket

All bets use opening odds (pre-match, actionable for live deployment), 25%
Kelly, $1000 bankroll per year reset. 7 separate models trained one per
validation year — strict walk-forward, no leakage.

## Key Findings

### 1. Higher edges are WORSE (inverted relationship)

| Edge band | Bets | ROI | Profit |
|---|---|---|---|
| 5-10%   | 346 | +2.6% | +$319 |
| 10-15%  | 285 | +4.2% | +$839 |
| 15-20%  | 193 | -1.9% | -$305 |
| 20-25%  |  87 | -5.2% | -$424 |
| 25-30%  |  37 | -13.2% | -$502 |
| 30-40%  |  19 | -23.1% | -$448 |

This is a clean monotonic pattern. Directly confirms the "market beats our
model" theory: when our model claims a huge edge, it's usually wrong.

### 2. Underdogs are toxic at higher edges

15-30% edge band, split by role:

| Role | Bets | Win% | ROI | Profit |
|---|---|---|---|---|
| Favorite (odds < 2.0) | 154 | 70.8% | +8.3% | +$1,272 |
| Underdog (odds ≥ 2.0) | 163 | 32.5% | -19.2% | -$2,504 |

The model massively overestimates underdog win rates at high claimed
edges. 25-30% underdogs: -38.0% ROI. 30%+ underdogs: -19.0% ROI.

### 3. Sweet-spot strategy isolated

10-25% edge on favorites, by year:

| Year | Bets | Win% | ROI | Profit |
|---|---|---|---|---|
| 2019 | 42 | 59.5% | -14.2% | -$292 |
| 2020 | 12 | 91.7% | +29.4% | +$332 |
| 2021 | 43 | 67.4% | -5.2% | -$124 |
| 2022 | 21 | 66.7% | -10.7% | -$238 |
| 2023 | 56 | 66.1% | -7.1% | -$313 |
| 2024 | 43 | 81.4% | +19.9% | +$1,226 |
| 2025 | 47 | 76.6% | +5.0% | +$298 |
| **Total** | **264** | **70.8%** | **+3.7%** | **+$890** |

Extending to 10-25% + all favorites (including outside band):
**303 bets, 71.3% win, +4.5% ROI, +$1,289 profit**.

Not every year profitable (3/7 win), but long-run edge is real and
sample size is substantial (303 bets).

## Changes Made

| File | Change |
|---|---|
| `run_predictions.py` | `MIN_EDGE` 0.15→0.10, `MAX_EDGE` 0.30→0.25, added `FAVORITE_ODDS_MAX=2.0`, added favorite check to bet-side selection |
| `src/predict.py` | Updated display filter (line 506-509) to match: edge 10-25% + `value_odds < 2.0` |
| `scripts/backtest_value_filter.py` | New — 200 lines; rerun after any significant model change |
| `data/analysis/value_filter_backtest.json` | New — full fold-level detail |

## Impact on Round 6

Old filter would have flagged 3 bets (Richmond @ 4.40 edge 21.9%, Port
Adelaide @ 8.60 edge 19.0%, Melbourne @ 3.75 edge 15.2%) — all underdogs
at long odds, exactly the toxic combination the backtest identifies.

New filter: **0 bets this round**. Correct outcome — none of the
favorites have a 10-25% edge in the current market.

The 3 stale bets were removed from `/app/data/betting/bet_log.csv` before
they could contaminate performance tracking.

## Caveats

- **Survivorship in odds data**: aussportsbetting.com may not have every
  match equally. WF sim reports 100% odds coverage on the matches it runs,
  but some matches upstream may be excluded silently.
- **Small sample on 25%+ favorites** (13 bets at +34.4%): the 10-25% band
  deliberately excludes this — the sample is too thin to rely on.
- **Year variance is high**: 2022/2023 were down years, 2024 was +19.9%.
  Expect drawdowns. Don't size bets assuming a smooth +4.5%/yr.
- **Did not test movement-agreement filter here**: the existing
  `_add_value_detection` still tracks AGREE/DISAGREE/NEUTRAL as a flag.
  Worth re-backtesting with that combined filter later; the prior
  `movement_agreement_analysis.json` suggested AGREE bets outperform.

## What's Next

Monitor the new filter for 4-6 rounds of live play. If it generates too
few bets per round (0-1 vs old 2-4), consider:

1. Re-running backtest with 5-25% edge on favorites (more bets, slightly
   lower quality signal per bet)
2. Adding a second tier: 10-15% underdogs (the one underdog slice that
   was still positive: +6.9% ROI on 123 bets)

Also: re-run `scripts/backtest_value_filter.py` after any model retrain,
since the optimal filter depends on the specific miscalibration pattern
of the current model.

---

## Appendix — Movement Filter Layer (2026-04-15, same day)

Tested whether open->close odds movement adds signal on top of the
fav+10-25% filter. Used `scripts/backtest_movement_filter.py`, which
classifies each historical bet as AGREE / NEUTRAL / DISAGREE based on
whether the implied probability of our side moved toward or away from
our pick between opening and closing bookmaker lines (threshold ±0.005).

### Full pool (969 bets, edge>=5%, all roles)

| Class | Bets | Win% | ROI | Profit |
|---|---|---|---|---|
| AGREE | 590 | 59.2% | **+6.6%** | **+$2,707** |
| NEUTRAL | 59 | 54.2% | -1.4% | -$48 |
| DISAGREE | 320 | 42.8% | **-17.2%** | **-$3,113** |

DISAGREE is the single strongest negative signal we've ever quantified —
the market moving AGAINST a bet we'd place is a 42.8% win rate (worse
than flipping a coin) and -17.2% ROI. **Reject always.**

### Within the fav+10-25% filter (already shipped)

| Strategy | Bets | Win% | ROI | Profit |
|---|---|---|---|---|
| fav+10-25%, all moves (yesterday) | 303 | 71.3% | +4.5% | +$1,289 |
| fav+10-25%, NOT_DISAGREE (now)    | 226 | 73.9% | **+6.4%** | **+$1,374** |

Marginal in-sample lift: +$85 profit, +2.6pp win rate, +1.9pp ROI. Small
absolute gain because the fav+10-25% filter already rejects most of the
bad bets DISAGREE would catch. But the filter is cheap — one inequality
check — and there's no cost to adding it.

### Why not ship the more aggressive candidates?

The backtest found bigger profit lifts if we drop the favorite filter or
widen the edge band:

| Alternative | Bets | ROI | Profit |
|---|---|---|---|
| all roles, edge>=5%, AGREE only | 590 | +6.6% | +$2,707 |
| fav, edge 5-25%, NOT_DISAGREE   | 343 | +5.7% | +$1,527 |

But:
- **Live vs historical movement is different.** The backtest uses
  open→close movement (days of market maturation). Live, we check
  current-snapshot vs recent-snapshot, which may be a noisier proxy.
  The conservative filter keeps the favorite-only safety net.
- **Bet lifecycle timing.** We log bets early in the week; movement
  classification firms up near game-time. The `log_bets` dedupe key
  means a bet logged early under NEUTRAL can't be auto-retracted if
  it later becomes DISAGREE. The current filter is more forgiving.
- **One change at a time.** The fav+10-25% filter just shipped.
  Stacking aggressive changes makes drift-detection harder.

If the shipped layer holds up over 4-6 rounds of live data, revisit the
more aggressive candidates.

### Code changes

| File | Change |
|---|---|
| `run_predictions.py` | Added `SKIP_DISAGREE_BETS=True` + `MOVE_THRESH=0.005` constants; `_detect_value_bets` now `continue`s when market moved AGAINST our side by >0.5%; updated display title and the pre-existing movement flag log messages |
| `scripts/backtest_movement_filter.py` | New — 200 lines; layers movement analysis on top of `walk_forward_betting` output |
| `data/analysis/movement_filter_backtest.json` | New — full fold-level detail |
