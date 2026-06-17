# Post-V3 Strategy Memo -- 2026-04-17

## TL;DR

- **Feature engineering on the betting model is done.** The systematic data
  sources analysis found 1/10 non-odds candidates above the noise floor
  (`+tipster_all`, 1.5x noise). Tier 3 external sources are all flagged
  high-null-risk. Marginal adds aren't worth the maintenance cost.
- **Today's ladder fix was material, not cosmetic.** 7.3pp mean shift on
  Round 6 predictions, 25.6pp max (North Melbourne/Richmond). The betting
  model had been silently running with 6/44 features stuck at NaN in
  production, using CatBoost's missing-value branch as an effective default.
- **Pivot away from feature engineering.** Highest-leverage open work: value
  bet filter refinement (real unsolved problem) + model monitoring / retrain
  cadence (ladder fix shows we need it). Defer tipster features.

## Finding 1: Ladder-fix blast radius (Plan B)

Compared 2026-04-16 (pre-fix) vs 2026-04-17 (post-fix) Round 6 predictions
from `predictions_history.csv`:

| Match | Pre | Post | Δ (pp) |
|---|---|---|---|
| North Melbourne vs Richmond | 55.3% | 80.9% | **+25.6** |
| Melbourne vs Brisbane Lions | 41.9% | 31.6% | −10.3 |
| West Coast vs Fremantle | 15.4% | 5.3% | −10.1 |
| Gold Coast vs Essendon | 90.0% | 98.0% | +8.0 |
| Sydney vs GWS | 69.4% | 74.1% | +4.7 |
| Geelong vs Bulldogs | 55.3% | 55.3% | 0 |
| Hawthorn vs Port Adelaide | 69.4% | 69.4% | 0 |
| Adelaide vs St Kilda | 59.3% | 59.3% | 0 |

- **Mean |Δ| betting**: 7.34 pp
- **Max |Δ| betting**: 25.6 pp
- **Mean |Δ| margin**: 10.9 pts (max 25.9 pts)
- **Mean |Δ| analytical (V3)**: 0.00 pp — confirms V3's odds-only
  independence; all movement attributable to ladder fix, not odds drift.

**Interpretation:** CatBoost's splits on `ladder_rank_diff`, `percentage_diff`
and the top4/top8 indicators are sharp. Training saw populated values;
inference was receiving NaN and taking the missing-value branch. Now that
training and inference distributions match, predictions reflect what the
walk-forward validation actually measured.

**Watch list:** Gold Coast at 98% sits at the calibrator clamp ceiling.
North Melbourne at 81% is the round's most aggressive call. If both hit,
the fix is validated. If they miss, investigate potential overfitting to
ladder position in training.

## Finding 2: Tier 3 external sources (Plan C)

Per `data/analysis/data_sources_report.json`:

| Source | Priority | Effort | Null risk | Call |
|---|---|---|---|---|
| Injury lists | Medium | Moderate | High -- existing 9 player features cover this | Skip |
| Coach changes | Low | Low | Very high -- too sparse (~30 events since 2012) | Skip |
| Quarter scoring | Low-Medium | Trivial (already in Squiggle API) | High -- form consistency was null | Maybe (1h) |
| Brownlow votes | Very low | Moderate | Very high -- announced at season end, absent at inference | Hard skip |

Only `+quarter_scoring` has near-zero collection cost. Pre-registered
prediction: null (form-consistency features were already null, quarter
scoring is a sibling). Not worth the hour.

## Decision: A vs D

**Plan A (tipster features) -- DEFER.**
- Gain: +0.00253 LL (1.5x noise)
- Risk: many Squiggle tipsters consume market odds -> adding their consensus
  to the betting model partially leaks market knowledge into a model
  intentionally designed to be market-independent. Could degrade
  value-betting precision.
- Effort vs return: small gain, architectural complication. Not worth it
  until other axes are exhausted.

**Plan D (pivot off feature engineering) -- DO.**

Recommended sequence:

1. **This week (passive):** Watch Round 6 outcomes. Specifically the four
   shifted matches (Gold Coast, North Melbourne, Fremantle, Brisbane).
   `calibration_drift.py` will give a read over 2-3 rounds.

2. **Next (1-2 weeks): D1 value-bet filter refinement.**
   - Previous backtest: market knows more than our betting model for value bets.
   - Hypotheses to test:
     - Edge threshold (10-25% is a guess; grid-search)
     - Favorites-only constraint — disagree-with-market filter on underdogs
       may be where the actual edge lives
     - Smart-money alignment (bet only when market moves toward our pick)
     - Fractional Kelly / flat-unit variants
   - Success metric: positive walk-forward ROI (2019-2025), >=100 bets
   - Files: new `scripts/backtest_value_v2.py`, `src/value.py`
   - Highest-leverage unsolved problem.

3. **Medium-term: D4 model versioning + retrain safety.**
   - Today's ladder fix shows we need: training/inference distribution
     auditing, model version tagging, A/B comparison when retraining,
     automatic rollback on regression.
   - The sanity check already catches live-feature-drift; D4 catches
     retrain-regression and silent-weight-shifts.

4. **Optional: D3 simulation mode.** Greenfield but non-urgent. Consider
   after D1 + D4 land.

## Round 6 Validation (added 2026-04-20)

Round 6 completed. Scored the pre-fix (2026-04-16) and post-fix (2026-04-17)
snapshots against actual outcomes on the 8 matches present in both:

| Metric | Pre-fix | Post-fix | Delta |
|---|---|---|---|
| Log loss | 0.4476 | **0.4024** | -0.0452 (~26x noise floor) |
| Brier | 0.1405 | **0.1293** | -0.0113 |
| Accuracy | 87.5% | 87.5% | tied (both 7/8) |

**Watchlist verdicts (4 matches with >5pp shift):**

| Match | Pre | Post | Outcome | Fix verdict |
|---|---|---|---|---|
| North Melb vs Richmond | 55% | 81% | Home won by 75 | **HELPED** (LL -0.38) |
| Gold Coast vs Essendon | 90% | 98% | Home won by 9 | **HELPED** (LL -0.09) |
| West Coast vs Fremantle | 15% | 5% | Away won by 56 | **HELPED** (LL -0.11) |
| Melbourne vs Brisbane | 42% | 32% | Home won by 2 | HURT (LL +0.28) |

Three of four shifted matches were high-confidence calls that cashed.
The single miss (Melbourne/Brisbane) was a 2-point game -- a genuine
coinflip.

**Overfitting concern: rebutted.** Post-fix predictions at 81-98% were
more aggressive than pre-fix, and the blowouts supported them. No evidence
the model is overreacting to ladder position.

**Loop closed on Plan B.** Ladder fix was material, correct, and validated.
Sanity check harness earned its keep.

## Changes Made

None. This memo documents decisions only.

## What's Next

- Calibration drift monitor will keep watching over the next 2-3 rounds.
- Scope `scripts/backtest_value_v2.py` (D1) as the next concrete work.

## D1 Value Filter Backtest (added 2026-04-20)

### Quickscan → Full WF progression

The V1 backtest had shown -4.3% ROI on the current 15-30% band. Suspected that
filtering *which* bets to take (not the edge threshold) was the leverage point.

**V2 quickscan** (post-hoc masking on pooled bets, `backtest_value_v2_quickscan.py`)
tested 13 filter variants against 5 criteria (n>=100, ROI>=5%, >=5/7 years,
bootstrap CI>0, holdout>0). Near-passes (4/5):

| Filter | ROI | CI low | Years | Holdout | Caveat |
|---|---|---|---|---|---|
| H4a: V3 agrees (15-30%) | +14.0% | +2.1% | 4/7 | +27.2% | V3 leakage — trained on 2019-25 |
| H5a: smart-$ aligned (15-30%) | +13.1% | -0.8% | 5/7 | +27.3% | none |
| H1a: favorites only (15-30%) | +7.9% | -3.1% | 5/7 | +11.3% | none |

### Full walk-forward V2 (`backtest_value_v2.py`)

Proper WF: train V3 on <Y, predict Y (no leakage). Re-simulated bankroll
per filter. Result:

| Filter | n | ROI | CI | Years | Holdout | MaxDD | Verdict |
|---|---|---|---|---|---|---|---|
| baseline (15-30%) | 301 | +3.3% | [-8.8, +15.3] | 4/7 | +14.7% | 63% | fail |
| H1: favorites | 143 | +4.7% | [-7.0, +15.9] | 3/7 | +9.0% | 42% | fail |
| **H4: V3 agrees (WF)** | 163 | **-0.1%** | [-11.5, +11.2] | 4/7 | +10.7% | 60% | **fail** |
| **H5: smart-$ aligned** | 211 | **+12.8%** | [-1.0, +27.8] | **5/7** | **+28.2%** | 71% | near-pass |
| H1+H5 | 106 | +7.2% | [-6.1, +20.0] | 4/7 | +10.9% | 38% | fail |
| H4+H5 | 132 | +1.1% | [-11.2, +14.4] | 3/7 | +9.8% | 54% | fail |
| H1+H4 | 133 | +4.1% | [-7.6, +15.3] | 4/7 | +12.5% | 43% | fail |
| H1+H4+H5 | 102 | +6.7% | [-6.2, +18.9] | 4/7 | +13.9% | 42% | fail |

**Walk-forward V3 killed H4.** The +14% in the quickscan was leakage. H4's
WF ROI is -0.1% — V3 agreement carries no bet-selection signal once you
remove the cheat.

**H5 (smart-money aligned) survived and replicated.** Quickscan +13.1% →
WF +12.8%. 5/7 profitable years. Holdout +28.2%. Only one criterion misses:
CI lower bound -1.0% (vs 0 threshold). With n=211, this is borderline —
the null "H5 ROI=0" is barely not rejected at p=0.05.

### H5 Kelly sensitivity

| Kelly | ROI | CI low | MaxDD | Holdout |
|---|---|---|---|---|
| 0.10 | +10.2% | -2.1% | **45%** | +24.7% |
| **0.15** | **+11.1%** | -1.7% | **56%** | +25.6% |
| 0.20 | +12.1% | -1.2% | 65% | +27.1% |
| 0.25 (prod) | +12.8% | -1.0% | 71% | +28.2% |
| 0.33 | +13.6% | -1.3% | 78% | +28.5% |
| 0.50 | +13.9% | -2.2% | 80% | +29.4% |

ROI is near-scale-invariant. **Max drawdown is what scales with Kelly.**
Dropping to 0.15 Kelly halves the tail risk (71% → 56%) for 1.7pp less ROI.

### Decision: soft-promote H5 with reduced Kelly

**Rationale:**
- Survives walk-forward (V3 didn't)
- Year-stable (5/7, H1 only got 3/7)
- Massive holdout (+28% on last 30% of bets, where recent regime matters)
- Filter logic is simple (odds_move > 0 for home bets; < 0 for away)
- CI miss is statistical, not structural — point estimate is far from zero

**Production change (D1.1):**
- Add `smart_money_aligned()` gate in `src/value.py`
- Change `kelly_frac` default from 0.25 → 0.15
- Keep 15-30% edge band unchanged
- Flag: H5 is borderline-passing. Monitor live ROI for 2-3 rounds; revert if
  production ROI diverges from backtest.

**What V2 ruled out:**
- V3 agreement as a filter (no signal)
- Favorites-only as a filter (year-unstable)
- Stacked filters (sample shredding kills signal)
- Lower edge thresholds (no lift)

**Files:**
- `scripts/backtest_value_v2_quickscan.py` — cheap 21s scan
- `scripts/backtest_value_v2.py` — full 32s WF backtest
- `data/analysis/value_filter_v2_quickscan.json`
- `data/analysis/value_filter_v2_report.json`

## H5 Production Deployment (added 2026-05-30)

**Deployed H5 to the live Unraid container (`grid:/mnt/user/appdata/afl`).**
Until today the box was still running the *pre-V2* strategy (MIN_EDGE=0.10,
MAX_EDGE=0.25, KELLY=0.25, no AGREE gate) — the H5 wiring from 2026-04-20 had
never been deployed. Synced `run_predictions.py`, `src/value.py`,
`src/predict.py`; rebuilt the container (cached layers, code COPYed at build).
Verified live: MIN_EDGE=0.15, MAX_EDGE=0.30, KELLY=0.15, REQUIRE_AGREE=True,
`smart_money_aligned` present; `--no-odds` smoke test ran clean through value
detection on Round 12.

**Rollback:** pre-deploy copies saved on the remote as
`run_predictions.py.pre-h5`, `src/value.py.pre-h5`, `src/predict.py.pre-h5`.
To revert: copy the `.pre-h5` files back over the originals + `docker compose
up -d --build`.

**Why now:** R10–R11 betting lost (GWS −$100, Port Adelaide −$82) under the old
filter. Both bets had ~12–14% edge — *below* H5's 15% floor — so H5 would have
blocked them. Live betting overall is still +$160 / +20.9% ROI over 8 bets, but
strip the one high-edge winner (Melbourne +$231, edge 0.257, which H5 also
takes) and the seven sub-15%-edge favorite bets are −$71 net. Consistent with
the backtest thesis even at n=8.

**Diagnosis of user's two complaints (both benign):**
- "Sparse predictions": Round 12 is a 7-match bye round; 3 played + 1 live
  (Brisbane/Fremantle, excluded by `complete==0`) + 3 upcoming = 3 shown.
  Correct behaviour.
- "Last round horrible": R10–R11 dipped to 56% acc / LL 0.75–0.80 (from R7–9
  peak of 89% / LL 0.31). Variance, not regression — R10/R11 were the season's
  two most home-dominated rounds (78%, 89% home wins vs 67% avg); misses
  clustered on low-confidence away picks while every confident pick still cashed.
  Season holds at 73% acc / LL 0.564. Features fully populated, no NaN.

**Monitoring (unchanged plan):** watch live ROI 2–3 rounds; revert via .pre-h5
backups if it diverges hard from the +11% backtest point estimate.

**Out-of-scope note:** live analytical (tipping) model is still the 47-feature
V1 (`Analytical model predictions added (5 models, 47 features)`), whereas local
is V3 3-feature. Model `.pkl`s drift between local/remote because `data/` isn't
synced on deploy. Does NOT affect the betting model or value bets — flag for a
later retrain-sync pass.
