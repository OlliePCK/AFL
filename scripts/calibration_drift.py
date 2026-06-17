"""
Calibration drift monitoring.

Runs the current production ensemble + calibrator over the completed matches
from the current season and compares predicted probabilities (in 10% bins)
against actual observed hit rates. If any well-populated bin drifts by more
than a threshold, the dashboard metrics flag gets set so we know to retrain.

Writes:
  data/model/calibration_drift.json  -- per-bin drift + overall summary
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CURRENT_YEAR  # noqa: E402
from src.features import build_features  # noqa: E402
from src.model import (  # noqa: E402
    TARGET,
    ensemble_predict_proba,
    get_feature_cols,
    load_betting_ensemble,
    load_calibrator,
)

# Bin edges: 10% bins from 0 to 1, plus a tail extension for extreme probs
BIN_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
BIN_LABELS = [f"{int(BIN_EDGES[i]*100)}-{int(BIN_EDGES[i+1]*100)}%"
              for i in range(len(BIN_EDGES) - 1)]

# A bin needs at least this many samples to be meaningful
MIN_BIN_COUNT = 5

# Drift threshold -- |predicted - observed| above this in a populated bin
# is a *candidate* drift; a bin only counts as truly drifting if the
# predicted_mean falls outside the 95% Wilson confidence interval of the
# observed rate. That keeps early-season thin bins from firing.
DRIFT_THRESHOLD = 0.15


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def main() -> None:
    print("=" * 80)
    print("CALIBRATION DRIFT MONITOR")
    print(f"Year: {CURRENT_YEAR} | Generated: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80)

    # Load ensemble + calibrator
    ens = load_betting_ensemble()
    cal = load_calibrator()
    if not ens or cal is None:
        print("FATAL: production ensemble or calibrator missing")
        sys.exit(1)
    print(f"Loaded {len(ens)} ensemble models + calibrator")

    # Load completed matches from the current season
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    df = build_features(df)
    season = df[(df["year"] == CURRENT_YEAR) & df[TARGET].notna()].copy()
    if season.empty:
        print(f"No completed matches yet for {CURRENT_YEAR}. Skipping drift check.")
        out = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "year": CURRENT_YEAR,
            "n_samples": 0,
            "status": "no_data",
            "bins": [],
            "drift_threshold": DRIFT_THRESHOLD,
            "drifting_bins": [],
            "is_drifting": False,
            "overall": {},
        }
        _write_report(out)
        return

    print(f"Evaluating {len(season)} completed {CURRENT_YEAR} matches")

    feat_cols = [c for c in get_feature_cols() if c in season.columns]
    X = season[feat_cols]
    y = season[TARGET].values.astype(int)

    raw_probs = ensemble_predict_proba(ens, X)
    cal_probs = np.clip(cal.predict(raw_probs), 0.02, 0.98)

    overall = {
        "log_loss": float(log_loss(y, cal_probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y, cal_probs)),
        "accuracy": float(((cal_probs >= 0.5).astype(int) == y).mean()),
    }
    print(f"\nOverall {CURRENT_YEAR} metrics:")
    print(f"  log_loss  = {overall['log_loss']:.4f}")
    print(f"  brier     = {overall['brier']:.4f}")
    print(f"  accuracy  = {overall['accuracy']:.3f}")

    # Bin by predicted probability and compute observed hit rate per bin
    bin_idx = np.clip(np.digitize(cal_probs, BIN_EDGES, right=False) - 1,
                      0, len(BIN_LABELS) - 1)
    bins_out = []
    drifting_bins = []
    for i, label in enumerate(BIN_LABELS):
        mask = bin_idx == i
        n = int(mask.sum())
        if n == 0:
            bins_out.append({
                "bin": label,
                "n": 0,
                "predicted_mean": None,
                "observed_rate": None,
                "drift": None,
                "sufficient": False,
            })
            continue
        predicted_mean = float(cal_probs[mask].mean())
        observed = float(y[mask].mean())
        drift = observed - predicted_mean
        sufficient = n >= MIN_BIN_COUNT
        successes = int(y[mask].sum())
        ci_low, ci_high = wilson_ci(successes, n)
        # Significant if the Wilson CI does not cover the predicted mean
        # AND the absolute drift exceeds the threshold AND we have enough samples.
        outside_ci = bool((predicted_mean < ci_low) or (predicted_mean > ci_high))
        is_drifting_bin = bool(sufficient and abs(drift) > DRIFT_THRESHOLD and outside_ci)
        bins_out.append({
            "bin": label,
            "n": n,
            "predicted_mean": round(predicted_mean, 4),
            "observed_rate": round(observed, 4),
            "drift": round(drift, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "sufficient": sufficient,
            "drift_flag": is_drifting_bin,
        })
        if is_drifting_bin:
            drifting_bins.append({
                "bin": label,
                "n": n,
                "predicted_mean": predicted_mean,
                "observed_rate": observed,
                "drift": drift,
                "ci": [ci_low, ci_high],
            })

    print(f"\n{'bin':>10s} {'n':>5s} {'pred':>8s} {'obs':>8s} "
          f"{'drift':>8s} {'CI':>18s}  status")
    print("  " + "-" * 72)
    for row in bins_out:
        if row["n"] == 0:
            print(f"  {row['bin']:>10s} {'0':>5s} {'--':>8s} {'--':>8s} "
                  f"{'--':>8s} {'--':>18s}  empty")
            continue
        if row["drift_flag"]:
            flag = "DRIFT"
        elif not row["sufficient"]:
            flag = "thin"
        elif abs(row["drift"]) > DRIFT_THRESHOLD:
            flag = "noisy"
        else:
            flag = "ok"
        ci_str = f"[{row['ci_low']:.2f},{row['ci_high']:.2f}]"
        print(f"  {row['bin']:>10s} {row['n']:>5d} "
              f"{row['predicted_mean']:>8.3f} {row['observed_rate']:>8.3f} "
              f"{row['drift']:>+8.3f} {ci_str:>18s}  {flag}")

    is_drifting = len(drifting_bins) > 0
    print(f"\n{'DRIFT DETECTED' if is_drifting else 'Calibration healthy'}: "
          f"{len(drifting_bins)} bin(s) beyond +/- {DRIFT_THRESHOLD}")

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "year": CURRENT_YEAR,
        "n_samples": int(len(season)),
        "status": "ok",
        "bins": bins_out,
        "drift_threshold": DRIFT_THRESHOLD,
        "drifting_bins": drifting_bins,
        "is_drifting": is_drifting,
        "overall": overall,
    }
    _write_report(out)


def _write_report(out: dict) -> None:
    out_path = PROJECT_ROOT / "data" / "model" / "calibration_drift.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
