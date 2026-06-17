"""
CatBoost model for AFL match prediction.

- Binary classifier: home team wins (1) or loses (0)
- Chronological train/validate/test split (no random splits)
- Evaluation: accuracy, log loss, AUC-ROC, calibration
"""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import PROJECT_ROOT
from src.utils import setup_logging

log = setup_logging()

SCHEMA_PATH = PROJECT_ROOT / "data" / "feature_schema.json"

# Production feature set — Elo + venue + rolling form + market odds + player availability.
# 22 features: validated on 2024-2025 test set (0.692 acc, 0.556 log loss, 0.773 AUC).
FEATURE_COLS = [
    "elo_diff", "elo_expected",
    "home_interstate", "away_interstate",
    "home_at_home_ground", "away_at_home_ground",
    "venue_win_rate_diff",
    # Rolling form (5-game window)
    "win_rate_diff_5", "avg_margin_diff_5",
    "score_for_diff_5", "score_against_diff_5",
    # Market odds (NaN-safe — CatBoost handles missing values)
    "implied_home_close", "overround_close",
    # Player availability — lineup disruption (Stage 1)
    "lineup_changes_diff", "lineup_continuity_diff",
    "home_ruck_missing", "away_ruck_missing",
    # Player availability — quality-weighted absences (Stage 2)
    "missing_rating_diff", "net_quality_change_diff",
    "missing_mid_rating_diff", "missing_fwd_rating_diff",
    "missing_def_rating_diff",
]

# Betting model features — excludes market odds (implied_home_close, overround_close)
# so the model produces independent probability estimates to compare against market prices.
BETTING_FEATURE_COLS = [f for f in FEATURE_COLS
                        if f not in ("implied_home_close", "overround_close")]

# Analytical features — includes odds movement for post-hoc analysis.
# NOT for betting (odds_move requires closing odds, unavailable at bet time).
ANALYTICAL_FEATURE_COLS = FEATURE_COLS + [
    "odds_move", "odds_move_magnitude", "overround_change",
]

# Extended features — for experimentation
EXTENDED_FEATURE_COLS = FEATURE_COLS + [
    "avg_margin_diff_10", "win_rate_diff_10",
    "h2h_home_win_rate",
    "rest_diff", "is_final", "season_progress",
    "avg_I50_diff_5", "avg_CL_diff_5", "avg_D_diff_5",
    "avg_T_diff_5", "avg_M_diff_5", "avg_HO_diff_5",
]

# Feature groups for optimization — each group is toggled on/off by Optuna.
# Base = BETTING_FEATURE_COLS (20 features). Groups add extras already in the dataset.
FEATURE_GROUPS = {
    "form_10g": [
        "win_rate_diff_10", "avg_margin_diff_10",
        "score_for_diff_10", "score_against_diff_10",
    ],
    "rest": ["rest_diff", "home_had_bye", "away_had_bye"],
    "ladder": [
        "ladder_rank_diff", "percentage_diff",
        "home_top4", "away_top4", "home_top8", "away_top8",
    ],
    "h2h": ["h2h_home_win_rate", "h2h_meetings"],
    "detailed_stats": [
        "avg_D_diff_5", "avg_I50_diff_5", "avg_CL_diff_5", "avg_T_diff_5",
        "avg_HO_diff_5", "avg_CG_diff_5", "avg_R50_diff_5", "avg_M_diff_5",
        "avg_FF_diff_5", "avg_FA_diff_5",
    ],
    "season_context": ["season_progress"],
    # Weather — collected via Visual Crossing, NaN-safe (CatBoost handles).
    # rain_mm / wind_kmh have real AFL impact on low-scoring wet games.
    "weather": [
        "rain_mm", "wind_kmh", "wind_gust_kmh", "humidity", "temp_c",
        "is_wet", "wind_strong", "is_roofed",
    ],
    # Travel distance — continuous km from team home city to venue.
    # Upgrades the binary home_interstate / away_interstate flags.
    "travel": [
        "home_travel_km", "away_travel_km", "travel_distance_diff",
    ],
    # Elo velocity — Elo momentum over last 5 games.
    "elo_velocity": ["elo_velocity_diff"],
    # Scoring efficiency — I50 conversion and kicking accuracy.
    "efficiency": ["i50_conversion_diff_5", "kick_accuracy_diff_5"],
    # Form consistency — margin volatility and win/loss streak.
    "consistency": ["margin_volatility_diff_5", "form_streak_diff"],
    # Odds — market-implied features (NOT for betting model, analytical only).
    # home_line_close is the single most predictive feature tested (-0.00683 LL).
    "odds": ["implied_home_close", "overround_close", "home_line_close"],
}

TARGET = "home_win"


def get_feature_cols() -> list[str]:
    """Load active feature list from saved schema, falling back to FEATURE_COLS."""
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH) as f:
            return json.load(f)["features"]
    return FEATURE_COLS


def save_feature_schema(features: list[str]):
    """Persist the active feature list alongside the trained model."""
    schema = {"version": 1, "features": features, "target": TARGET}
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)
    log.info(f"Saved feature schema ({len(features)} features) to {SCHEMA_PATH}")


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/validate/test sets.

    Train: 2012-2022
    Validate: 2023
    Test: 2024-2025
    """
    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET]).copy()

    # Use only columns that exist
    feature_cols = get_feature_cols()
    available_features = [c for c in feature_cols if c in df.columns]
    cols = available_features + [TARGET, "margin", "year", "date", "home_team", "away_team",
                                  "home_score", "away_score", "game_id"]
    df = df[[c for c in cols if c in df.columns]]

    train = df[df["year"] <= 2022]
    val = df[df["year"] == 2023]
    test = df[(df["year"] >= 2024) & (df["year"] <= 2025)]

    log.info(f"Train: {len(train)} ({train['year'].min()}-{train['year'].max()})")
    log.info(f"Validate: {len(val)} ({val['year'].min()}-{val['year'].max()})")
    log.info(f"Test: {len(test)} ({test['year'].min()}-{test['year'].max()})")

    return train, val, test


def train_model(train: pd.DataFrame, val: pd.DataFrame,
                feature_cols: list[str] | None = None,
                params: dict | None = None) -> CatBoostClassifier:
    """Train a CatBoost binary classifier."""
    feature_cols = feature_cols or get_feature_cols()
    available_features = [c for c in feature_cols if c in train.columns]

    X_train = train[available_features]
    y_train = train[TARGET]
    X_val = val[available_features]
    y_val = val[TARGET]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    default_params = {
        "iterations": 2000, "learning_rate": 0.05, "depth": 4,
        "l2_leaf_reg": 3, "random_seed": 42, "verbose": 100,
        "early_stopping_rounds": 100, "eval_metric": "Logloss",
        "use_best_model": True,
    }
    if params:
        default_params.update(params)

    model = CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=val_pool)

    # Persist the feature schema used for this model
    save_feature_schema(available_features)

    log.info(f"Best iteration: {model.best_iteration_}")
    return model


def evaluate_model(model: CatBoostClassifier, df: pd.DataFrame, label: str,
                   feature_cols: list[str] | None = None) -> dict:
    """Evaluate model on a dataset. Returns metrics dict."""
    feature_cols = feature_cols or get_feature_cols()
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features]
    y = df[TARGET]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs)
    auc = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs)

    # Baseline: always predict home team wins
    baseline_acc = y.mean() if y.mean() > 0.5 else 1 - y.mean()

    metrics = {
        "accuracy": acc,
        "log_loss": ll,
        "auc_roc": auc,
        "brier_score": brier,
        "baseline_accuracy": baseline_acc,
        "n_matches": len(df),
    }

    log.info(f"=== {label} Results ===")
    log.info(f"Accuracy:  {acc:.3f} (baseline: {baseline_acc:.3f}, lift: +{acc - baseline_acc:.3f})")
    log.info(f"Log Loss:  {ll:.4f}")
    log.info(f"AUC-ROC:   {auc:.3f}")
    log.info(f"Brier:     {brier:.4f}")

    return metrics


def plot_feature_importance(model: CatBoostClassifier, save_path: Path | None = None):
    """Plot top feature importances."""
    available_features = [c for c in get_feature_cols() if c in model.feature_names_]
    importances = model.feature_importances_
    names = model.feature_names_

    sorted_idx = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx][::-1])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([names[i] for i in sorted_idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info(f"Saved feature importance plot to {save_path}")
    plt.close()


def calibrate_model(oof_probs: list[float], oof_labels: list[float]) -> object:
    """Fit isotonic regression calibrator on out-of-fold predictions.

    Returns the fitted calibrator and saves it to disk.
    """
    import pickle
    from sklearn.isotonic import IsotonicRegression

    probs = np.array(oof_probs)
    labels = np.array(oof_labels)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs, labels)

    # Measure improvement
    raw_brier = brier_score_loss(labels, probs)
    cal_probs = calibrator.predict(probs)
    cal_brier = brier_score_loss(labels, cal_probs)

    log.info(f"Calibration: Brier {raw_brier:.4f} -> {cal_brier:.4f} "
             f"(on {len(labels)} OOF samples)")

    # Save
    cal_path = PROJECT_ROOT / "data" / "calibrator.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    log.info(f"Calibrator saved to {cal_path}")

    return calibrator


def load_calibrator():
    """Load saved calibrator from disk. Returns None if not found."""
    import pickle
    cal_path = PROJECT_ROOT / "data" / "calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            return pickle.load(f)
    return None


def plot_calibration(model: CatBoostClassifier, df: pd.DataFrame,
                     label: str, save_path: Path | None = None,
                     calibrator=None):
    """Plot calibration curve: predicted probability vs actual win rate.

    If a calibrator is provided, shows both raw and calibrated curves.
    """
    available_features = [c for c in get_feature_cols() if c in df.columns]
    X = df[available_features]
    y = df[TARGET]
    probs = model.predict_proba(X)[:, 1]

    prob_true, prob_pred = calibration_curve(y, probs, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(prob_pred, prob_true, "s-", label="Raw model")

    if calibrator is not None:
        cal_probs = calibrator.predict(probs)
        cal_true, cal_pred = calibration_curve(y, cal_probs, n_bins=10, strategy="uniform")
        ax.plot(cal_pred, cal_true, "o-", label="Calibrated")
        raw_brier = brier_score_loss(y, probs)
        cal_brier = brier_score_loss(y, cal_probs)
        ax.set_title(f"Calibration ({label}) — Brier: {raw_brier:.4f} raw, {cal_brier:.4f} cal")
    else:
        ax.set_title(f"Calibration Curve ({label})")

    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual win rate")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info(f"Saved calibration plot to {save_path}")
    plt.close()


def walk_forward_cv(df: pd.DataFrame,
                    val_years: list[int] = [2019, 2020, 2021, 2022, 2023],
                    feature_cols: list[str] | None = None,
                    params: dict | None = None,
                    ) -> dict:
    """Walk-forward cross-validation: train on all years before val_year,
    validate on val_year. Collects per-fold and pooled metrics, plus
    out-of-fold predictions for calibration.

    Years 2024-2025 are never touched (held out for final test).
    """
    feature_cols = feature_cols or get_feature_cols()
    available = [c for c in feature_cols if c in df.columns]

    default_params = {
        "iterations": 2000, "learning_rate": 0.05, "depth": 4,
        "l2_leaf_reg": 3, "random_seed": 42, "verbose": 0,
        "early_stopping_rounds": 100, "eval_metric": "Logloss",
        "use_best_model": True,
    }
    if params:
        default_params.update(params)

    fold_results = []
    all_probs, all_labels = [], []

    for val_year in val_years:
        train_df = df[df["year"] < val_year].dropna(subset=[TARGET])
        val_df = df[df["year"] == val_year].dropna(subset=[TARGET])

        if train_df.empty or val_df.empty:
            log.info(f"WF fold {val_year}: skipped (empty split)")
            continue

        m = CatBoostClassifier(**default_params)
        m.fit(Pool(train_df[available], train_df[TARGET]),
              eval_set=Pool(val_df[available], val_df[TARGET]))

        probs = m.predict_proba(val_df[available])[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(val_df[TARGET], preds)
        ll = log_loss(val_df[TARGET], probs)
        auc = roc_auc_score(val_df[TARGET], probs)

        fold_results.append({
            "val_year": val_year,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "accuracy": acc,
            "log_loss": ll,
            "auc_roc": auc,
            "best_iter": m.best_iteration_,
        })
        all_probs.extend(probs.tolist())
        all_labels.extend(val_df[TARGET].tolist())

        log.info(f"WF {val_year}: Acc={acc:.3f}  LL={ll:.4f}  AUC={auc:.3f}  "
                 f"(train={len(train_df)}, val={len(val_df)}, iter={m.best_iteration_})")

    # Pooled metrics
    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)
    pooled = {
        "accuracy": accuracy_score(all_labels_arr, (all_probs_arr >= 0.5).astype(int)),
        "log_loss": log_loss(all_labels_arr, all_probs_arr),
        "auc_roc": roc_auc_score(all_labels_arr, all_probs_arr),
    }

    mean_acc = np.mean([f["accuracy"] for f in fold_results])
    mean_ll = np.mean([f["log_loss"] for f in fold_results])

    log.info(f"=== Walk-Forward CV Summary ({len(fold_results)} folds) ===")
    log.info(f"Mean Accuracy: {mean_acc:.3f}  Mean LL: {mean_ll:.4f}")
    log.info(f"Pooled Accuracy: {pooled['accuracy']:.3f}  "
             f"Pooled LL: {pooled['log_loss']:.4f}  "
             f"Pooled AUC: {pooled['auc_roc']:.3f}")

    return {
        "folds": fold_results,
        "pooled": pooled,
        "oof_probs": all_probs,
        "oof_labels": all_labels,
    }


def run_full_pipeline():
    """End-to-end: load data, build features, walk-forward CV, train, evaluate."""
    from src.features import build_features

    # Load master dataset
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    log.info(f"Loaded {len(df)} matches")

    # Build features
    df = build_features(df)

    # Walk-forward cross-validation (does not touch 2024-2025)
    cv_results = walk_forward_cv(df)

    # Prepare splits for final production model
    train, val, test = prepare_data(df)

    # Train
    model = train_model(train, val)

    # Evaluate on all splits
    train_metrics = evaluate_model(model, train, "Train")
    val_metrics = evaluate_model(model, val, "Validation")
    test_metrics = evaluate_model(model, test, "Test")

    # Post-hoc calibration using walk-forward out-of-fold predictions
    calibrator = calibrate_model(cv_results["oof_probs"], cv_results["oof_labels"])

    # Plots
    plots_dir = PROJECT_ROOT / "data" / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_feature_importance(model, plots_dir / "feature_importance.png")
    plot_calibration(model, test, "Test", plots_dir / "calibration_test.png",
                     calibrator=calibrator)

    # Save model
    model_path = PROJECT_ROOT / "data" / "model.cbm"
    model.save_model(str(model_path))
    log.info(f"Model saved to {model_path}")

    # Save featured dataset
    df.to_csv(PROJECT_ROOT / "data" / "master" / "afl_featured_dataset.csv", index=False)

    # Train margin regression model
    margin_model = train_margin_model(train, val)
    margin_model_path = PROJECT_ROOT / "data" / "margin_model.cbm"
    margin_model.save_model(str(margin_model_path))
    evaluate_margin_model(margin_model, test, "Test")

    return model, df, {"train": train_metrics, "val": val_metrics, "test": test_metrics}


# ──────────────────────────────────────────────────────────────────────
# HYPERPARAMETER + FEATURE OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────

def load_optimization_results() -> dict | None:
    """Load saved optimization results from disk. Returns None if not found."""
    opt_path = PROJECT_ROOT / "data" / "optimization_results.json"
    if opt_path.exists():
        with open(opt_path) as f:
            return json.load(f)
    return None


def optimize_model(df: pd.DataFrame, n_trials: int = 80,
                   val_years: list[int] | None = None) -> dict:
    """Run Optuna hyperparameter + feature group optimization.

    Uses walk-forward CV with pooled log loss as the objective.
    Searches CatBoost hyperparameters and feature group toggles jointly.

    Args:
        df: Featured DataFrame with all columns (output of build_features()).
        n_trials: Number of Optuna trials (default 80, ~45 min).
        val_years: Walk-forward folds (default [2019..2025]).

    Returns:
        dict with best_params, best_features, best_feature_groups, best_log_loss.
    """
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    if val_years is None:
        val_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    df_complete = df.dropna(subset=[TARGET]).copy()

    # Run baseline with current defaults for comparison
    log.info("Running baseline walk-forward CV (current defaults)...")
    baseline = walk_forward_cv(df_complete, val_years=val_years,
                               feature_cols=BETTING_FEATURE_COLS)
    baseline_ll = baseline["pooled"]["log_loss"]
    baseline_acc = baseline["pooled"]["accuracy"]
    log.info(f"Baseline: log_loss={baseline_ll:.4f}, accuracy={baseline_acc:.3f}")

    def _objective(trial):
        # Hyperparameters
        trial_params = {
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "iterations": 1500,
            "early_stopping_rounds": 80,
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "Logloss",
            "use_best_model": True,
        }
        # CatBoost requires bootstrap_type=Bernoulli when subsample < 1
        if trial_params["subsample"] < 1.0:
            trial_params["bootstrap_type"] = "Bernoulli"

        # Feature group toggles
        feature_cols = list(BETTING_FEATURE_COLS)
        for group_name, group_cols in FEATURE_GROUPS.items():
            if trial.suggest_categorical(f"use_{group_name}", [True, False]):
                feature_cols.extend(group_cols)

        available = [c for c in feature_cols if c in df_complete.columns]

        # Walk-forward CV with pruning
        all_probs, all_labels = [], []
        for i, val_year in enumerate(val_years):
            train_df = df_complete[df_complete["year"] < val_year]
            val_df = df_complete[df_complete["year"] == val_year]
            if train_df.empty or val_df.empty:
                continue

            m = CatBoostClassifier(**trial_params)
            m.fit(Pool(train_df[available], train_df[TARGET]),
                  eval_set=Pool(val_df[available], val_df[TARGET]))

            probs = m.predict_proba(val_df[available])[:, 1]
            all_probs.extend(probs.tolist())
            all_labels.extend(val_df[TARGET].tolist())

            # Report intermediate result for pruning
            running_ll = log_loss(all_labels, all_probs)
            trial.report(running_ll, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return log_loss(all_labels, all_probs)

    # Create and run study
    log.info(f"\n{'=' * 80}")
    log.info(f"OPTUNA OPTIMIZATION — {n_trials} trials, {len(val_years)} WF folds")
    log.info(f"{'=' * 80}")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    # Extract results
    best = study.best_trial
    best_params = {
        k: best.params[k] for k in
        ["depth", "learning_rate", "l2_leaf_reg", "min_data_in_leaf",
         "subsample", "colsample_bylevel"]
    }

    best_feature_groups = [
        g for g in FEATURE_GROUPS if best.params.get(f"use_{g}", False)
    ]
    best_features = list(BETTING_FEATURE_COLS)
    for g in best_feature_groups:
        best_features.extend(FEATURE_GROUPS[g])

    best_ll = best.value

    # Summary
    log.info(f"\n{'=' * 80}")
    log.info("OPTIMIZATION COMPLETE")
    log.info(f"{'=' * 80}")
    log.info(f"Best log_loss: {best_ll:.4f} (baseline: {baseline_ll:.4f}, "
             f"improvement: {baseline_ll - best_ll:+.4f})")
    log.info(f"Best accuracy estimate: ~{accuracy_score(baseline['oof_labels'], (np.array(baseline['oof_probs']) >= 0.5).astype(int)):.3f} -> check retrain")
    log.info(f"Feature groups selected: {best_feature_groups or 'none (base only)'}")
    log.info(f"Total features: {len(best_features)} (was {len(BETTING_FEATURE_COLS)})")
    log.info(f"Hyperparameters: {best_params}")
    log.info(f"Completed trials: {len(study.trials)}, pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    # Save results
    results = {
        "best_params": best_params,
        "best_features": best_features,
        "best_feature_groups": best_feature_groups,
        "best_log_loss": best_ll,
        "baseline_log_loss": baseline_ll,
        "n_trials": n_trials,
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }
    opt_path = PROJECT_ROOT / "data" / "optimization_results.json"
    with open(opt_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved optimization results to {opt_path}")

    return results


# ──────────────────────────────────────────────────────────────────────
# RETRAIN ON LATEST DATA
# ──────────────────────────────────────────────────────────────────────
def retrain_production_model():
    """Retrain the production model on ALL completed match data.

    Uses an expanding window: trains on all completed matches except the most
    recent season (used as validation for early stopping). As the current
    season progresses, completed matches are added to training.
    """
    from src.features import build_features

    # Load and build features
    master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
    df = pd.read_csv(master_path, parse_dates=["date"])
    log.info(f"Loaded {len(df)} matches")
    df = build_features(df)

    # Only completed matches (have a result)
    df_complete = df.dropna(subset=[TARGET]).copy()
    max_year = int(df_complete["year"].max())
    min_year = int(df_complete["year"].min())

    # Use optimized params/features if available
    opt_results = load_optimization_results()
    if opt_results:
        feature_cols = opt_results["best_features"]
        catboost_params = opt_results["best_params"]
        log.info(f"Using optimized params ({len(feature_cols)} features, "
                 f"CV log loss {opt_results['best_log_loss']:.4f})")
    else:
        feature_cols = BETTING_FEATURE_COLS
        catboost_params = {}
        log.info("No optimization results -- using default BETTING_FEATURE_COLS")

    available = [c for c in feature_cols if c in df_complete.columns]

    # Dynamic train/val split: use all completed data.
    # Val = most recent full season (for early stopping only).
    # Train = everything before that.
    # The production model trains on train+val early-stopping boundary,
    # so all completed data contributes to the final model's trees.
    df_complete = df_complete.sort_values("date").reset_index(drop=True)
    val_size = 207  # ~1 season of matches for early stopping
    train = df_complete.iloc[:-val_size]
    val = df_complete.iloc[-val_size:]
    train_years = f"{min_year}-{int(train['year'].max())}"
    val_years_range = f"{int(val['year'].min())}-{int(val['year'].max())}"
    train_label = train_years
    val_label = val_years_range

    log.info("=" * 80)
    log.info(f"RETRAINING PRODUCTION MODEL (train {train_label}, val {val_label})")
    log.info("=" * 80)
    log.info(f"Train: {len(train)} matches ({train_label}), Val: {len(val)} matches ({val_label})")
    log.info(f"Total completed matches: {len(df_complete)} (through {max_year})")

    if val.empty:
        log.error("No validation data available. Run --update first.")
        return None

    # Baseline: evaluate OLD model on val data
    old_metrics = None
    old_model_path = PROJECT_ROOT / "data" / "model.cbm"
    if old_model_path.exists():
        old_model = CatBoostClassifier()
        old_model.load_model(str(old_model_path))
        log.info(f"\n--- Old Model on {val_label} data ---")
        old_metrics = evaluate_model(old_model, val, f"Old Model -> {val_label}")

    # Walk-forward CV for calibrator (use all available years)
    wf_start = max(2019, min_year + 5)
    wf_years = list(range(wf_start, max_year + 1))
    log.info(f"\n--- Walk-Forward CV ({len(wf_years)} folds: {wf_years[0]}-{wf_years[-1]}) ---")
    cv_results = walk_forward_cv(df_complete, val_years=wf_years,
                                 feature_cols=feature_cols, params=catboost_params)

    # Save feature schema
    save_feature_schema(feature_cols)

    # Train production model on ALL completed data (train + val)
    # Val is used for early stopping only — the final model sees everything
    log.info("\n--- Training Production Model (all completed data) ---")
    all_completed = df_complete  # everything with a result
    new_model = train_model(train, val, feature_cols=feature_cols, params=catboost_params)

    # Evaluate on val set
    log.info(f"\n--- New Model on {val_label} data ---")
    new_metrics = evaluate_model(new_model, val, f"New Model -> {val_label}",
                                 feature_cols=feature_cols)

    # Before/after comparison
    if old_metrics:
        log.info("\n" + "=" * 60)
        log.info(f"BEFORE vs AFTER COMPARISON (on {val_label} data)")
        log.info("=" * 60)
        log.info(f"{'Metric':<15} {'Old':>10} {'New':>10} {'Delta':>10}")
        log.info("-" * 45)
        for key in ["accuracy", "log_loss", "auc_roc", "brier_score"]:
            old_v = old_metrics[key]
            new_v = new_metrics[key]
            delta = new_v - old_v
            better = "+" if (delta > 0 and key in ("accuracy", "auc_roc")) or \
                           (delta < 0 and key in ("log_loss", "brier_score")) else ""
            log.info(f"{key:<15} {old_v:>10.4f} {new_v:>10.4f} {delta:>+10.4f} {better}")

    # Calibrate using walk-forward OOF predictions
    log.info("\n--- Calibration ---")
    calibrator = calibrate_model(cv_results["oof_probs"], cv_results["oof_labels"])

    # Save model
    model_path = PROJECT_ROOT / "data" / "model.cbm"
    new_model.save_model(str(model_path))
    log.info(f"New model saved to {model_path}")

    # Plots
    plots_dir = PROJECT_ROOT / "data" / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_feature_importance(new_model, plots_dir / "feature_importance.png")
    plot_calibration(new_model, val, f"Validation {val_label}",
                     plots_dir / "calibration_val.png", calibrator=calibrator)

    # Train margin model with the same splits
    log.info("\n--- Margin Model ---")
    margin_model = train_margin_model(train, val, feature_cols=feature_cols,
                                      params=catboost_params)
    margin_model_path = PROJECT_ROOT / "data" / "margin_model.cbm"
    margin_model.save_model(str(margin_model_path))
    evaluate_margin_model(margin_model, val, f"Validation {val_label}",
                          feature_cols=feature_cols)

    # Save featured dataset
    df.to_csv(PROJECT_ROOT / "data" / "master" / "afl_featured_dataset.csv", index=False)

    # Save model metrics JSON for dashboard
    metrics_data = {
        "new_model": {k: round(v, 4) for k, v in new_metrics.items()
                      if k in ("accuracy", "log_loss", "auc_roc", "brier_score")},
        "old_model": {k: round(v, 4) for k, v in old_metrics.items()
                      if k in ("accuracy", "log_loss", "auc_roc", "brier_score")} if old_metrics else {},
        "train_period": train_label,
        "val_period": val_label,
    }
    if opt_results:
        metrics_data["optimization"] = {
            "n_trials": opt_results.get("n_trials"),
            "best_log_loss_cv": opt_results.get("best_log_loss"),
            "feature_groups": opt_results.get("best_feature_groups", []),
            "n_features": len(opt_results.get("best_features", [])),
        }
    metrics_path = PROJECT_ROOT / "data" / "model" / "model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    log.info(f"Saved model metrics to {metrics_path}")

    log.info("\n" + "=" * 80)
    log.info("RETRAIN COMPLETE -- new model.cbm, calibrator.pkl, margin_model.cbm saved")
    log.info("=" * 80)

    return new_model


# ──────────────────────────────────────────────────────────────────────
# MARGIN REGRESSION MODEL
# ──────────────────────────────────────────────────────────────────────
def train_margin_model(train: pd.DataFrame, val: pd.DataFrame,
                       feature_cols: list[str] | None = None,
                       params: dict | None = None):
    """Train a CatBoost regressor to predict margin (home_score - away_score)."""
    from catboost import CatBoostRegressor

    feature_cols = feature_cols or BETTING_FEATURE_COLS
    available_features = [c for c in feature_cols if c in train.columns]
    X_train = train[available_features]
    y_train = train["margin"]
    X_val = val[available_features]
    y_val = val["margin"]

    default_params = {
        "iterations": 2000, "learning_rate": 0.05, "depth": 4,
        "l2_leaf_reg": 3, "random_seed": 42, "verbose": 100,
        "early_stopping_rounds": 100, "eval_metric": "MAE",
        "use_best_model": True,
    }
    if params:
        default_params.update(params)
    # Margin model always uses MAE
    default_params["eval_metric"] = "MAE"

    model = CatBoostRegressor(**default_params)
    model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val))
    log.info(f"Margin model best iteration: {model.best_iteration_}")
    return model


def evaluate_margin_model(model, df: pd.DataFrame, label: str,
                          feature_cols: list[str] | None = None):
    """Evaluate the margin regression model."""
    feature_cols = feature_cols or BETTING_FEATURE_COLS
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features]
    y_true = df["margin"]
    y_pred = model.predict(X)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # Directional accuracy: does predicted margin sign match actual?
    direction_acc = np.mean((y_pred > 0) == (y_true > 0))

    log.info(f"=== Margin Model {label} ===")
    log.info(f"MAE:       {mae:.1f} points")
    log.info(f"RMSE:      {rmse:.1f} points")
    log.info(f"Direction: {direction_acc:.3f} (predicting winner from margin)")


# ──────────────────────────────────────────────────────────────────────
# BETTING ENSEMBLE MODEL
# ──────────────────────────────────────────────────────────────────────

def train_betting_ensemble(train: pd.DataFrame, val: pd.DataFrame,
                           n_models: int = 5,
                           seeds: list[int] | None = None,
                           feature_cols: list[str] | None = None,
                           params: dict | None = None) -> list[CatBoostClassifier]:
    """Train an ensemble of CatBoost models.

    Each model uses a different random seed. At prediction time,
    average the probabilities across all models for more stable estimates.

    Returns list of fitted models.
    """
    feature_cols = feature_cols or BETTING_FEATURE_COLS
    available = [c for c in feature_cols if c in train.columns]
    X_train, y_train = train[available], train[TARGET]
    X_val, y_val = val[available], val[TARGET]

    if seeds is None:
        seeds = [42, 123, 256, 789, 1337][:n_models]

    default_params = {
        "iterations": 2000, "learning_rate": 0.05, "depth": 4,
        "l2_leaf_reg": 3, "verbose": 0,
        "early_stopping_rounds": 100, "eval_metric": "Logloss",
        "use_best_model": True,
    }
    if params:
        default_params.update(params)

    models = []
    for i, seed in enumerate(seeds):
        default_params["random_seed"] = seed
        m = CatBoostClassifier(**default_params)
        m.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val))
        models.append(m)
        log.info(f"Ensemble model {i+1}/{n_models}: seed={seed}, iter={m.best_iteration_}")

    # Evaluate ensemble on validation
    val_probs = np.mean([m.predict_proba(X_val)[:, 1] for m in models], axis=0)
    acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
    ll = log_loss(y_val, val_probs)
    auc = roc_auc_score(y_val, val_probs)
    log.info(f"Ensemble val: Acc={acc:.3f}, LL={ll:.4f}, AUC={auc:.3f}")

    return models


def save_betting_ensemble(models: list[CatBoostClassifier],
                          feature_cols: list[str] | None = None):
    """Save ensemble models to disk."""
    ensemble_dir = PROJECT_ROOT / "data" / "ensemble"
    ensemble_dir.mkdir(exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(str(ensemble_dir / f"betting_model_{i}.cbm"))
    # Save feature list
    feature_cols = feature_cols or BETTING_FEATURE_COLS
    available = [c for c in feature_cols if c in models[0].feature_names_]
    save_feature_schema(available)
    log.info(f"Saved {len(models)} ensemble models to {ensemble_dir}")


def load_betting_ensemble() -> list[CatBoostClassifier] | None:
    """Load ensemble models from disk. Returns None if not found."""
    ensemble_dir = PROJECT_ROOT / "data" / "ensemble"
    if not ensemble_dir.exists():
        return None
    model_files = sorted(ensemble_dir.glob("betting_model_*.cbm"))
    if not model_files:
        return None
    models = []
    for f in model_files:
        m = CatBoostClassifier()
        m.load_model(str(f))
        models.append(m)
    log.info(f"Loaded {len(models)} ensemble models")
    return models


def ensemble_predict_proba(models: list[CatBoostClassifier],
                           X: pd.DataFrame) -> np.ndarray:
    """Average predicted probabilities across ensemble models."""
    probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return probs


def load_margin_ensemble():
    """Load margin-regression ensemble from disk. Returns None if not found."""
    from catboost import CatBoostRegressor
    ensemble_dir = PROJECT_ROOT / "data" / "ensemble"
    if not ensemble_dir.exists():
        return None
    model_files = sorted(ensemble_dir.glob("margin_model_*.cbm"))
    if not model_files:
        return None
    models = []
    for f in model_files:
        m = CatBoostRegressor()
        m.load_model(str(f))
        models.append(m)
    log.info(f"Loaded {len(models)} margin ensemble models")
    return models


def ensemble_predict_margin(models, X: pd.DataFrame) -> np.ndarray:
    """Average predicted margins across regressor ensemble models."""
    return np.mean([m.predict(X) for m in models], axis=0)


def load_analytical_ensemble() -> list[CatBoostClassifier] | None:
    """Load analytical ensemble (includes odds features) from disk."""
    ensemble_dir = PROJECT_ROOT / "data" / "ensemble"
    if not ensemble_dir.exists():
        return None
    model_files = sorted(ensemble_dir.glob("analytical_model_*.cbm"))
    if not model_files:
        return None
    models = []
    for f in model_files:
        m = CatBoostClassifier()
        m.load_model(str(f))
        models.append(m)
    log.info(f"Loaded {len(models)} analytical ensemble models")
    return models


def load_analytical_calibrator():
    """Load the analytical model's calibrator from disk."""
    import pickle
    cal_path = PROJECT_ROOT / "data" / "analytical_calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            return pickle.load(f)
    return None


ANALYTICAL_SCHEMA_PATH = PROJECT_ROOT / "data" / "analytical_feature_schema.json"


def get_analytical_feature_cols() -> list[str] | None:
    """Load analytical model feature list. Returns None if not trained."""
    if ANALYTICAL_SCHEMA_PATH.exists():
        with open(ANALYTICAL_SCHEMA_PATH) as f:
            return json.load(f)["features"]
    return None


def retrain_production_ensembles() -> dict | None:
    """Retrain both the classifier and margin ensembles end-to-end.

    This is the production retrain path -- called by the weekly cron.
    Delegates training to the scripts/train_ensemble.py and
    scripts/train_margin_ensemble.py entry points (they are the single
    source of truth for how an ensemble is built), then rolls up the
    resulting ensemble reports into a model_metrics.json payload the
    dashboard can consume.

    Returns the metrics dict on success, None on any training failure.
    Training failures are logged -- the old ensemble on disk is not
    touched, so the production pipeline keeps running on stale models.
    """
    import subprocess
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

    log.info("=" * 80)
    log.info("RETRAINING PRODUCTION ENSEMBLES (classifier + margin)")
    log.info("=" * 80)

    scripts_dir = PROJECT_ROOT / "scripts"
    classifier_script = scripts_dir / "train_ensemble.py"
    margin_script = scripts_dir / "train_margin_ensemble.py"

    if not classifier_script.exists() or not margin_script.exists():
        log.error(f"Ensemble training scripts missing from {scripts_dir}")
        return None

    # Snapshot the existing metrics so we can expose them as "old_model"
    # in the dashboard's before/after comparison.
    metrics_path = PROJECT_ROOT / "data" / "model" / "model_metrics.json"
    old_new_model: dict | None = None
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                prev = json.load(f)
            old_new_model = prev.get("new_model")
        except Exception as e:
            log.warning(f"Could not read previous metrics: {e}")

    def _run(script: Path) -> bool:
        import sys as _sys
        log.info(f"--- running {script.name} ---")
        result = subprocess.run(
            [_sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
        )
        if result.returncode != 0:
            log.error(f"{script.name} exited {result.returncode}")
            return False
        return True

    if not _run(classifier_script):
        log.error("Classifier ensemble retrain FAILED -- aborting retrain")
        return None
    if not _run(margin_script):
        log.error("Margin ensemble retrain FAILED -- classifier ensemble was updated")
        return None

    # Read the two reports back out
    clf_report_path = PROJECT_ROOT / "data" / "analysis" / "ensemble_report.json"
    margin_report_path = PROJECT_ROOT / "data" / "analysis" / "margin_ensemble_report.json"
    if not clf_report_path.exists():
        log.error(f"Classifier report missing at {clf_report_path}")
        return None
    with open(clf_report_path) as f:
        clf_report = json.load(f)

    margin_report = None
    if margin_report_path.exists():
        with open(margin_report_path) as f:
            margin_report = json.load(f)

    # Classifier holdout metrics -- compute AUC separately since the
    # ensemble report doesn't currently store it.
    val_period = clf_report["final_holdout"]["val_period"]
    calibrated_ll = clf_report["final_holdout"]["ensemble_log_loss_calibrated"]
    calibrated_brier = clf_report["final_holdout"]["ensemble_brier_calibrated"]
    ens_accuracy = clf_report["final_holdout"]["ensemble_accuracy"]

    # Compute AUC by re-running the freshly-trained ensemble on the same
    # val split (last 207 rows). This is cheap and gives us a matching
    # AUC number for the dashboard.
    try:
        from src.features import build_features
        master_path = PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv"
        df = pd.read_csv(master_path, parse_dates=["date"])
        df = build_features(df)
        complete = df.dropna(subset=[TARGET]).sort_values("date").reset_index(drop=True)
        val = complete.iloc[-207:]
        schema_features = get_feature_cols()
        feat_cols = [c for c in schema_features if c in val.columns]
        X_val = val[feat_cols]
        y_val = val[TARGET].values

        ens = load_betting_ensemble()
        cal = load_calibrator()
        if ens and cal is not None:
            raw = ensemble_predict_proba(ens, X_val)
            calibrated = np.clip(cal.predict(raw), 0.02, 0.98)
            auc = float(roc_auc_score(y_val, calibrated))
        else:
            auc = float("nan")

        # Write the featured dataset for downstream analyses (the old
        # retrain function also did this).
        df.to_csv(PROJECT_ROOT / "data" / "master" / "afl_featured_dataset.csv",
                  index=False)
    except Exception as e:
        log.warning(f"Could not compute holdout AUC or write featured dataset: {e}")
        auc = float("nan")

    new_model_metrics = {
        "accuracy": round(float(ens_accuracy), 4),
        "log_loss": round(float(calibrated_ll), 4),
        "auc_roc": round(float(auc), 4) if not np.isnan(auc) else 0.0,
        "brier_score": round(float(calibrated_brier), 4),
    }

    # Derive a sensible train period string from the WF years in the
    # report: train = earliest year through (first val year - 1).
    wf_years = clf_report.get("wf_years", [])
    if wf_years:
        train_end = int(val_period.split("-")[0]) - 1
        train_start = int(wf_years[0]) - 7  # WF starts 7y after min_year
        train_period = f"{max(2012, train_start)}-{train_end}"
    else:
        train_period = val_period  # fallback

    metrics_data = {
        "new_model": new_model_metrics,
        "old_model": old_new_model if old_new_model else new_model_metrics,
        "train_period": train_period,
        "val_period": val_period,
        "ensemble": {
            "n_models": len(clf_report.get("seeds", [])),
            "wf_pooled_log_loss": clf_report["wf_cv"]["ensemble"]["log_loss"],
            "wf_seed_mean_log_loss": clf_report["wf_cv"]["seed_mean_log_loss"],
            "wf_seed_std_log_loss": clf_report["wf_cv"]["seed_std_log_loss"],
            "wf_ensemble_delta_vs_seed_mean": clf_report["wf_cv"]["ensemble_delta_vs_seed_mean"],
        },
    }
    if margin_report:
        metrics_data["margin_ensemble"] = {
            "n_models": len(margin_report.get("seeds", [])),
            "wf_pooled_mae": margin_report["wf_cv"]["ensemble"]["mae"],
            "holdout_mae": margin_report["final_holdout"]["ensemble_mae"],
            "holdout_direction_acc": margin_report["final_holdout"]["ensemble_direction_acc"],
        }

    # Preserve optimization block from previous metrics.json if present
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                prev = json.load(f)
            if "optimization" in prev:
                metrics_data["optimization"] = prev["optimization"]
            if "train_period" in prev and prev["train_period"]:
                metrics_data["train_period"] = prev["train_period"]
        except Exception:
            pass

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    log.info(f"Saved model metrics to {metrics_path}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("RETRAIN COMPLETE")
    log.info("=" * 60)
    log.info(f"Classifier:  {new_model_metrics}")
    if margin_report:
        log.info(f"Margin:      pooled MAE {margin_report['wf_cv']['ensemble']['mae']:.3f}, "
                 f"holdout MAE {margin_report['final_holdout']['ensemble_mae']:.3f}")
    log.info("=" * 60)

    return metrics_data
