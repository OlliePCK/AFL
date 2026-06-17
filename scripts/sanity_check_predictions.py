"""
Prediction sanity check harness.

Runs after predictions are generated to catch silent model failures.

Motivated by the April 2026 incident where the 47-feature analytical model
was silently producing a constant ~0.58 for every match because its 'close'
odds features (implied_home_close, overround_close, home_line_close) were
NaN for all upcoming matches. CatBoost fell back to a global prior with no
observable error. We now V3-populate those features from the live snapshot,
but we also want a guardrail that screams if the same class of bug recurs.

Checks (evaluated over the CURRENT upcoming round only):

  1. Near-constant probability output (stddev < 0.02 across the round)
     -> likely NaN-feature bug for either model.
  2. Feature schema columns NaN on >50% of upcoming matches
     -> upstream version of check 1; lists the offending features.
  3. Calibrated probs pinned at the [0.02, 0.98] clamps on >30% of matches
     -> calibrator saturating; suggests retrain needed.
  4. home_line populated on fewer than 80% of upcoming matches
     -> V3 degradation alert; analytical model would fall back to odds-only.
  5. Betting vs analytical home_prob disagreeing by >30pp on any match
     -> informational; surfaces individual red-flag matches.

Exit codes:
  0 - all checks pass (or only info-level observations)
  1 - one or more WARN/FAIL conditions triggered

Writes:
  data/model/prediction_sanity.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (  # noqa: E402
    get_analytical_feature_cols,
    get_feature_cols,
)

# ---- thresholds --------------------------------------------------------------

CONSTANT_PROB_STDDEV = 0.02          # check 1
FEATURE_NAN_MAX_RATE = 0.50          # check 2
CLAMP_LO, CLAMP_HI = 0.02, 0.98      # check 3 (matches production clamp)
CLAMP_MAX_RATE = 0.30                # check 3
LINE_MIN_COVERAGE = 0.80             # check 4
MODEL_DISAGREE_THRESHOLD = 0.30      # check 5

UPCOMING_PATH = PROJECT_ROOT / "data" / "master" / "upcoming_predictions.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "master" / "upcoming_features.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "model" / "prediction_sanity.json"


# ---- terminal colours --------------------------------------------------------

class C:
    # ANSI colours; fall back to empty strings if not a TTY.
    _tty = sys.stdout.isatty()
    RESET = "\033[0m" if _tty else ""
    BOLD = "\033[1m" if _tty else ""
    GREEN = "\033[32m" if _tty else ""
    YELLOW = "\033[33m" if _tty else ""
    RED = "\033[31m" if _tty else ""
    CYAN = "\033[36m" if _tty else ""
    DIM = "\033[2m" if _tty else ""


def status_tag(level: str) -> str:
    if level == "pass":
        return f"{C.GREEN}[PASS]{C.RESET}"
    if level == "warn":
        return f"{C.YELLOW}[WARN]{C.RESET}"
    if level == "fail":
        return f"{C.RED}[FAIL]{C.RESET}"
    return f"{C.DIM}[INFO]{C.RESET}"


# ---- helpers -----------------------------------------------------------------

def _load_upcoming() -> pd.DataFrame:
    if not UPCOMING_PATH.exists():
        print(f"{status_tag('fail')} upcoming_predictions.csv not found at {UPCOMING_PATH}")
        sys.exit(1)
    df = pd.read_csv(UPCOMING_PATH)
    if df.empty:
        print(f"{status_tag('warn')} upcoming_predictions.csv is empty — nothing to check")
        sys.exit(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _pick_current_round(df: pd.DataFrame) -> pd.DataFrame:
    """The current round is whichever upcoming round starts earliest.

    Restricting checks to one round avoids false positives from future-round
    predictions that intentionally lack odds (analytical model falls back to a
    global prior for those, which would trip check 1 across the whole file).
    """
    round_order = (
        df.dropna(subset=["roundname", "date"])
        .groupby("roundname")["date"].min()
        .sort_values()
    )
    if round_order.empty:
        return df
    current = round_order.index[0]
    return df[df["roundname"] == current].reset_index(drop=True)


# ---- individual checks -------------------------------------------------------

def check_constant_output(current: pd.DataFrame) -> dict:
    """Check 1: flag any model that produces near-constant probabilities."""
    result = {"name": "constant_output", "level": "pass", "details": {}}
    for col, label in [("home_win_prob", "betting"),
                       ("analytical_home_prob", "analytical")]:
        if col not in current.columns:
            result["details"][label] = {"status": "missing", "stddev": None}
            continue
        probs = current[col].dropna()
        if len(probs) < 2:
            result["details"][label] = {"status": "too_few", "stddev": None,
                                         "n": int(len(probs))}
            continue
        stddev = float(probs.std(ddof=0))
        status = "ok"
        if stddev < CONSTANT_PROB_STDDEV:
            status = "constant"
            result["level"] = "fail"
        result["details"][label] = {
            "status": status,
            "stddev": round(stddev, 5),
            "min": round(float(probs.min()), 4),
            "max": round(float(probs.max()), 4),
            "n": int(len(probs)),
        }
    return result


def check_feature_nan_coverage(current: pd.DataFrame) -> dict:
    """Check 2: audit feature NaN rates on upcoming matches using the
    upcoming_features.csv snapshot written by run_predictions.py."""
    result = {"name": "feature_nan_coverage", "level": "pass", "offenders": {}}

    if not FEATURES_PATH.exists():
        result["level"] = "warn"
        result["skipped"] = (f"{FEATURES_PATH.name} not found — run predictions "
                              f"on this code version to populate it")
        return result

    features_df = pd.read_csv(FEATURES_PATH)
    if "game_id" not in features_df.columns or "game_id" not in current.columns:
        result["level"] = "warn"
        result["skipped"] = "game_id missing from features or predictions file"
        return result

    ids = set(current["game_id"].astype(str).tolist())
    sub = features_df[features_df["game_id"].astype(str).isin(ids)]
    if sub.empty:
        result["level"] = "warn"
        result["skipped"] = "no upcoming rows matched in features snapshot"
        return result

    n = len(sub)
    schemas = {
        "betting": get_feature_cols(),
        "analytical": get_analytical_feature_cols() or [],
    }

    for label, features in schemas.items():
        offenders: list[dict] = []
        for feat in features:
            if feat not in sub.columns:
                offenders.append({"feature": feat, "nan_rate": 1.0,
                                   "reason": "column_missing"})
                continue
            nan_rate = float(sub[feat].isna().mean())
            if nan_rate > FEATURE_NAN_MAX_RATE:
                offenders.append({"feature": feat,
                                   "nan_rate": round(nan_rate, 3),
                                   "reason": "nan"})
        if offenders:
            result["level"] = "warn"
        result["offenders"][label] = {"n_upcoming": n, "offenders": offenders,
                                        "total_features": len(features)}
    return result


def check_clamp_saturation(current: pd.DataFrame) -> dict:
    """Check 3: flag if too many probs are pinned at the calibrator clamps."""
    result = {"name": "clamp_saturation", "level": "pass", "details": {}}
    for col, label in [("home_win_prob", "betting"),
                       ("analytical_home_prob", "analytical")]:
        if col not in current.columns:
            result["details"][label] = {"status": "missing", "rate": None}
            continue
        probs = current[col].dropna()
        if len(probs) == 0:
            result["details"][label] = {"status": "empty", "rate": None}
            continue
        tol = 1e-6
        n_low = int((probs <= CLAMP_LO + tol).sum())
        n_high = int((probs >= CLAMP_HI - tol).sum())
        rate = (n_low + n_high) / len(probs)
        status = "ok"
        if rate > CLAMP_MAX_RATE:
            status = "saturated"
            result["level"] = "warn"
        result["details"][label] = {
            "status": status,
            "rate": round(rate, 3),
            "n_low": n_low,
            "n_high": n_high,
            "n": int(len(probs)),
        }
    return result


def check_line_coverage(current: pd.DataFrame) -> dict:
    """Check 4: flag if home_line is missing on too many upcoming matches."""
    result = {"name": "line_coverage", "level": "pass"}
    if "home_line" not in current.columns:
        result["level"] = "warn"
        result["reason"] = "home_line column absent"
        return result
    n = len(current)
    populated = int(current["home_line"].notna().sum())
    rate = populated / n if n else 0.0
    result["n_upcoming"] = n
    result["n_with_line"] = populated
    result["coverage"] = round(rate, 3)
    if rate < LINE_MIN_COVERAGE:
        result["level"] = "warn"
    return result


def check_model_disagreement(current: pd.DataFrame) -> dict:
    """Check 5: surface any match where the two models disagree by >30pp."""
    result = {"name": "model_disagreement", "level": "pass", "flagged": []}
    needed = {"home_win_prob", "analytical_home_prob", "home_team", "away_team"}
    if not needed.issubset(current.columns):
        result["level"] = "info"
        result["skipped"] = f"missing columns: {sorted(needed - set(current.columns))}"
        return result
    for _, row in current.iterrows():
        b, a = row.get("home_win_prob"), row.get("analytical_home_prob")
        if pd.isna(b) or pd.isna(a):
            continue
        diff = abs(float(b) - float(a))
        if diff >= MODEL_DISAGREE_THRESHOLD:
            result["flagged"].append({
                "match": f"{row['home_team']} vs {row['away_team']}",
                "betting": round(float(b), 3),
                "analytical": round(float(a), 3),
                "delta_pp": round(diff * 100, 1),
            })
    # Disagreement is informational — never fails, but we note WARN if >=2
    # matches disagree strongly (suggests systematic divergence, not a one-off).
    if len(result["flagged"]) >= 2:
        result["level"] = "warn"
    elif result["flagged"]:
        result["level"] = "info"
    return result


# ---- report rendering --------------------------------------------------------

def _print_header(round_name: str, n: int):
    print("=" * 78)
    print(f"{C.BOLD}PREDICTION SANITY CHECK{C.RESET}  ({round_name}, {n} matches)")
    print(f"{C.DIM}generated: {datetime.now().isoformat(timespec='seconds')}{C.RESET}")
    print("=" * 78)


def _print_check_1(r: dict):
    print(f"\n{status_tag(r['level'])} {C.BOLD}1. Constant-probability guardrail{C.RESET}")
    print(f"    (fail if stddev < {CONSTANT_PROB_STDDEV})")
    for label, d in r["details"].items():
        if d["stddev"] is None:
            print(f"      {label:>11s}: {C.DIM}{d['status']}{C.RESET}")
            continue
        tag = C.RED if d["status"] == "constant" else C.GREEN
        print(f"      {label:>11s}: stddev={tag}{d['stddev']:.4f}{C.RESET}  "
              f"range=[{d['min']:.3f}, {d['max']:.3f}]  n={d['n']}")


def _print_check_2(r: dict):
    print(f"\n{status_tag(r['level'])} {C.BOLD}2. Feature NaN coverage{C.RESET}")
    print(f"    (warn if any schema feature NaN on >{int(FEATURE_NAN_MAX_RATE*100)}% of upcoming matches)")
    if "skipped" in r:
        print(f"      {C.DIM}skipped: {r['skipped']}{C.RESET}")
        return
    for label, d in r["offenders"].items():
        n_off = len(d["offenders"])
        total = d["total_features"]
        n = d["n_upcoming"]
        if n_off == 0:
            print(f"      {label:>11s}: {C.GREEN}clean{C.RESET}  "
                  f"({total} features, {n} upcoming)")
            continue
        print(f"      {label:>11s}: {C.YELLOW}{n_off}/{total} features flagged{C.RESET}  ({n} upcoming)")
        for off in d["offenders"][:10]:
            reason = off.get("reason", "nan")
            rate = off.get("nan_rate", 1.0)
            print(f"          - {off['feature']:<32s} {rate*100:>5.1f}% NaN  ({reason})")
        if len(d["offenders"]) > 10:
            print(f"          {C.DIM}... +{len(d['offenders']) - 10} more{C.RESET}")


def _print_check_3(r: dict):
    print(f"\n{status_tag(r['level'])} {C.BOLD}3. Calibrator clamp saturation{C.RESET}")
    print(f"    (warn if >{int(CLAMP_MAX_RATE*100)}% of probs pinned at [{CLAMP_LO}, {CLAMP_HI}])")
    for label, d in r["details"].items():
        if d["rate"] is None:
            print(f"      {label:>11s}: {C.DIM}{d['status']}{C.RESET}")
            continue
        tag = C.YELLOW if d["status"] == "saturated" else C.GREEN
        print(f"      {label:>11s}: {tag}{d['rate']*100:>5.1f}%{C.RESET}  "
              f"(low={d['n_low']}, high={d['n_high']}, n={d['n']})")


def _print_check_4(r: dict):
    print(f"\n{status_tag(r['level'])} {C.BOLD}4. home_line coverage (V3 input){C.RESET}")
    print(f"    (warn if <{int(LINE_MIN_COVERAGE*100)}% of upcoming matches have a line)")
    if "reason" in r:
        print(f"      {C.YELLOW}{r['reason']}{C.RESET}")
        return
    tag = C.GREEN if r["coverage"] >= LINE_MIN_COVERAGE else C.YELLOW
    print(f"      {tag}{r['coverage']*100:>5.1f}%{C.RESET}  "
          f"({r['n_with_line']}/{r['n_upcoming']} matches have home_line)")


def _print_check_5(r: dict):
    print(f"\n{status_tag(r['level'])} {C.BOLD}5. Model disagreement (>{int(MODEL_DISAGREE_THRESHOLD*100)}pp){C.RESET}")
    if "skipped" in r:
        print(f"      {C.DIM}skipped: {r['skipped']}{C.RESET}")
        return
    if not r["flagged"]:
        print(f"      {C.GREEN}no matches with >{int(MODEL_DISAGREE_THRESHOLD*100)}pp disagreement{C.RESET}")
        return
    for f in r["flagged"]:
        print(f"      - {f['match']:<46s} "
              f"betting={f['betting']:.2f}  analytical={f['analytical']:.2f}  "
              f"Δ={f['delta_pp']:+.1f}pp")


def _print_footer(level: str, checks: list[dict]):
    print("\n" + "=" * 78)
    levels = [c["level"] for c in checks]
    n_fail = sum(1 for lv in levels if lv == "fail")
    n_warn = sum(1 for lv in levels if lv == "warn")
    if level == "fail":
        print(f"{C.RED}{C.BOLD}RESULT: FAIL{C.RESET}  "
              f"({n_fail} fail, {n_warn} warn) — investigate before shipping predictions")
    elif level == "warn":
        print(f"{C.YELLOW}{C.BOLD}RESULT: WARN{C.RESET}  "
              f"({n_warn} warn) — review flagged items")
    else:
        print(f"{C.GREEN}{C.BOLD}RESULT: PASS{C.RESET}  all guardrails green")
    print("=" * 78)


def _overall_level(checks: list[dict]) -> str:
    levels = {c["level"] for c in checks}
    if "fail" in levels:
        return "fail"
    if "warn" in levels:
        return "warn"
    return "pass"


def _write_report(checks: list[dict], level: str, round_name: str, n: int):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "round": round_name,
        "n_upcoming": n,
        "overall_level": level,
        "checks": checks,
        "thresholds": {
            "constant_prob_stddev": CONSTANT_PROB_STDDEV,
            "feature_nan_max_rate": FEATURE_NAN_MAX_RATE,
            "clamp_lo": CLAMP_LO,
            "clamp_hi": CLAMP_HI,
            "clamp_max_rate": CLAMP_MAX_RATE,
            "line_min_coverage": LINE_MIN_COVERAGE,
            "model_disagree_threshold": MODEL_DISAGREE_THRESHOLD,
        },
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")


# ---- main --------------------------------------------------------------------

def main() -> int:
    upcoming = _load_upcoming()
    current = _pick_current_round(upcoming)
    round_name = str(current["roundname"].iloc[0]) if "roundname" in current.columns else "unknown"
    n = len(current)

    _print_header(round_name, n)

    checks = [
        check_constant_output(current),
        check_feature_nan_coverage(current),
        check_clamp_saturation(current),
        check_line_coverage(current),
        check_model_disagreement(current),
    ]

    _print_check_1(checks[0])
    _print_check_2(checks[1])
    _print_check_3(checks[2])
    _print_check_4(checks[3])
    _print_check_5(checks[4])

    level = _overall_level(checks)
    _print_footer(level, checks)
    _write_report(checks, level, round_name, n)

    return 0 if level == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
