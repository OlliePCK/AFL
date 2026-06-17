# Analytical Model Decision Memo -- 2026-04-16

## TL;DR

Replaced the 47-feature analytical model with a **3-feature odds-only
model (V3)** using `[implied_home_open, overround_open, home_line_close]`.

Walk-forward validated on 1,430 matches (2019-2025):

| Model | Features | LL | Brier | Accuracy |
|---|---|---|---|---|
| Full analytical (V1) | 47 | 0.5853 | 0.2011 | 68.11% |
| **V3 odds-only** | **3** | **0.5853** | **0.2009** | **68.46%** |
| Raw market implied | 1 | 0.5968 | 0.2055 | 68.25% |

V3 matches V1 on every metric. Both beat raw market by -0.0115 LL
(6.6x the noise floor of +/-0.00174). The other 44 features were dead weight.

## Motivation

The value filter backtests (earlier today) showed the market knows more
than our betting model. That raised the question: does our 47-feature
analytical model know more than the market for *tipping*?

## Experiments

### Exp 1: Accuracy test (`backtest_analytical_vs_baseline.py`)

Three strategies, pooled accuracy on 1,431 matches:

| Strategy | Accuracy |
|---|---|
| Analytical model (47 feat) | 68.13% |
| Market favorite (short odds) | 68.48% |
| Tipster consensus (Squiggle) | 68.34% |

**Model ties market** on accuracy (delta -0.35pp, McNemar z=-0.38).
88.2% of picks agree. But accuracy only checks rank order, not
probability quality.

### Exp 2: Calibration + ablation (`backtest_analytical_calibration.py`)

Tested whether the model produces better *probabilities* than market:

| Source | Log loss | Brier |
|---|---|---|
| Full analytical (47 feat) | 0.5853 | 0.2011 |
| Odds-only (3 feat) | **0.5834** | **0.2002** |
| Raw market implied | 0.5968 | 0.2055 |
| Tipster consensus | 0.5906 | 0.2030 |

- **Q2 (calibration):** Full analytical beats raw market by -0.0115 LL
  (6.6x noise). The model adds real calibration value.
- **Q3 (ablation):** Odds-only 3-feature model TIES or BEATS the full
  47-feature model. The other 44 features contribute nothing.

### Exp 3: Feature robustness (`backtest_odds_only_variants.py`)

Tested which odds features carry the signal:

| Variant | Features | LL | vs V1 delta |
|---|---|---|---|
| V1 (close3 baseline) | implied_close, overround_close, line_close | 0.5834 | -- |
| V2 (opening only) | implied_open, overround_open | 0.5971 | +0.0137 (WORSE) |
| **V3 (open + line)** | implied_open, overround_open, line_close | **0.5853** | +0.0019 (TIE) |
| V4 (open + movement) | implied_open, overround_open, odds_move, magnitude, OR_change | 0.5892 | +0.0058 (WORSE) |
| Raw market | vig-adjusted implied_open | 0.5968 | +0.0134 (WORSE) |

- **`home_line_close` is THE critical feature.** V3 (with line) ties V1.
  V2 (without line) equals raw market.
- **Movement features (V4) hurt** -- noise for tipping (different from
  their role as a *bet filter*, where DISAGREE rejection works).
- Opening h2h alone adds nothing over raw market.

### Live line availability

Probed The Odds API (`probe_odds_api_spreads.py`):
- AFL spreads available from 7/11 Australian bookmakers (TAB, Sportsbet,
  PointsBet, TabTouch, PlayUp, BetR, BetRight)
- 9/9 games had at least one spread price
- Quota cost: 2 credits/call (h2h+spreads) vs 1 for h2h only
- Current usage: 77/500, well within free tier at ~24/month

## Changes Made

| File | Change |
|---|---|
| `src/odds_monitor.py` | Request `markets=h2h,spreads` (was `h2h` only). Parse spreads market, add `home_line`, `away_line`, `home_line_odds`, `away_line_odds`, `n_line_bookmakers` columns. Median across bookmakers. |
| `src/predict.py` | New `_populate_analytical_odds()` -- maps live snapshot to V3 feature names. Moved `fetch_live_odds()` call before analytical model block. `fetch_live_odds()` returns full DataFrame (was stripping to 4 columns). |
| `scripts/train_analytical_odds_only.py` | New -- trains 5-seed ensemble on 3 V3 features. Drop-in replacement for `train_analytical_ensemble.py`. Same artifact paths. |
| `data/analytical_feature_schema.json` | Version 2: 3 features (was 47). |
| `data/ensemble/analytical_model_*.cbm` | Overwritten with V3 models. |
| `data/analytical_calibrator.pkl` | Overwritten with V3 calibrator. |

### Analysis scripts (new, informational only)

| Script | Output |
|---|---|
| `scripts/backtest_analytical_vs_baseline.py` | `data/analysis/analytical_baseline_report.json` |
| `scripts/backtest_analytical_calibration.py` | `data/analysis/analytical_calibration_report.json` |
| `scripts/backtest_odds_only_variants.py` | `data/analysis/odds_only_variants_report.json` |
| `scripts/probe_odds_api_spreads.py` | One-shot probe (no output file) |

## Impact on Round 6

V3 analytical model produces per-match probabilities that diverge from
the betting model where the market disagrees:

- Carlton vs Collingwood: betting model 55% Carlton, V3 46% Carlton
  (market says Collingwood favored by ~10.5 points)
- Melbourne vs Brisbane: betting model 42% Melbourne, V3 15%
  (market says Brisbane heavily favored)

The V3 model now reflects genuine market information instead of running
on NaN odds features (which the OLD 47-feature model was silently doing
for all live predictions).

## Caveats

- **Live line != historical close line.** The model trained on close
  lines; live predictions use the current snapshot. Lines typically
  don't move as much as h2h, so the proxy is good but not exact.
- **Old analytical model was broken for live.** The 47-feature model
  used `implied_home_close`, `overround_close`, `home_line_close` --
  none available for upcoming matches. CatBoost produced a constant
  ~0.58 for every match. V3 fixes this by populating from live odds.
- **V3 produces duplicate probabilities** for matches with similar
  odds+line combinations (only 3 features = limited resolution).
  Acceptable: the calibration is what matters.

## What's Next

1. **Monitor V3 for 2-3 rounds.** Compare its Round 6+ predictions
   against outcomes. If calibration is bad, revert to the old model
   (which is equivalent to "predict 58% for everything").
2. **Retire `train_analytical_ensemble.py`** after V3 is validated.
   Keep as `scripts/_archived/train_analytical_ensemble_v1.py` for
   reference.
3. **Deploy.** Run `bash scripts/deploy.sh` to rebuild Docker image
   with the new odds_monitor (spreads), predict.py (V3 population),
   and model artifacts.
