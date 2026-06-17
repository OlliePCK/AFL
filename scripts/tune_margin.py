"""
Item B -- Optuna hyperparameter tuning for the margin regression model.

Baseline (current hardcoded params, Optuna feature set):
  MAE        26.36 points  (2025-2026 holdout, n=207)
  RMSE       33.31 points
  Direction  0.696

We're fixing the feature set to the Optuna-winning classifier config
(form_10g, rest, ladder, detailed_stats, season_context — 44 features).
The classifier Item A analysis just confirmed those are the real signal
so there's no reason to re-tune feature groups for the margin head too.
Instead, run ~60 Optuna trials over CatBoostRegressor hyperparameters
with MAE as the objective and walk-forward CV identical to the
classifier's tuning.

Writes:
  data/margin_optimization_results.json  -- best params + per-trial log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features  # noqa: E402
from src.model import load_optimization_results, TARGET  # noqa: E402

VAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEED = 42
N_TRIALS = 60
TIMEOUT_SEC = 900  # 15 min safety cap


def _normalize_params(p: dict) -> dict:
    """Ensure bootstrap_type and its dependent params are mutually consistent.

    CatBoost rules:
      - bootstrap_type='Bayesian'  -> uses bagging_temperature, NOT subsample
      - bootstrap_type='Bernoulli' -> uses subsample, NOT bagging_temperature
      - bootstrap_type='MVS'       -> uses subsample, NOT bagging_temperature
    """
    out = dict(p)
    bt = out.get("bootstrap_type", "Bayesian")
    if bt == "Bayesian":
        out.pop("subsample", None)
    else:
        out.pop("bagging_temperature", None)
    return out


def walk_forward_mae(params: dict, df: pd.DataFrame,
                     features: list[str]) -> dict:
    """Walk-forward CV for margin regression. Returns pooled MAE + folds."""
    fold_rows = []
    pooled_preds, pooled_true = [], []

    for vy in VAL_YEARS:
        train_df = df[df["year"] < vy].dropna(subset=["margin"])
        val_df = df[df["year"] == vy].dropna(subset=["margin"])
        if train_df.empty or val_df.empty:
            continue

        X_train = train_df[features]
        y_train = train_df["margin"]
        X_val = val_df[features]
        y_val = val_df["margin"].values

        m = CatBoostRegressor(**_normalize_params(params))
        m.fit(
            Pool(X_train, y_train),
            eval_set=Pool(X_val, val_df["margin"]),
        )
        y_pred = m.predict(X_val)
        mae = float(np.mean(np.abs(y_val - y_pred)))
        fold_rows.append({"year": int(vy), "n": len(val_df), "mae": mae})
        pooled_preds.extend(y_pred.tolist())
        pooled_true.extend(y_val.tolist())

    pooled_preds = np.asarray(pooled_preds)
    pooled_true = np.asarray(pooled_true)
    return {
        "pooled_mae": float(np.mean(np.abs(pooled_true - pooled_preds))),
        "pooled_rmse": float(np.sqrt(np.mean((pooled_true - pooled_preds) ** 2))),
        "pooled_direction_acc": float(((pooled_preds > 0) == (pooled_true > 0)).mean()),
        "folds": fold_rows,
    }


def objective(trial: optuna.Trial, df: pd.DataFrame,
              features: list[str]) -> float:
    params = {
        "iterations": 2000,
        "early_stopping_rounds": 100,
        "eval_metric": "MAE",
        "loss_function": "MAE",
        "use_best_model": True,
        "verbose": 0,
        "random_seed": SEED,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
    }
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 1.0
        )
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

    # Do walk-forward and report per-fold MAE so pruning works
    maes = []
    for i, vy in enumerate(VAL_YEARS):
        train_df = df[df["year"] < vy].dropna(subset=["margin"])
        val_df = df[df["year"] == vy].dropna(subset=["margin"])
        if train_df.empty or val_df.empty:
            continue

        m = CatBoostRegressor(**_normalize_params(params))
        m.fit(
            Pool(train_df[features], train_df["margin"]),
            eval_set=Pool(val_df[features], val_df["margin"]),
        )
        y_pred = m.predict(val_df[features])
        mae = float(np.mean(np.abs(val_df["margin"].values - y_pred)))
        maes.append(mae)

        trial.report(float(np.mean(maes)), i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(maes))


def main() -> None:
    t_start = time.time()
    print("=" * 80)
    print("ITEM B -- Optuna tuning for CatBoost margin regressor")
    print("=" * 80)

    # Load features (same classifier Optuna-winning set)
    print("\n[1/3] Loading data + features...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "master" / "afl_master_dataset.csv",
                     parse_dates=["date"])
    df = build_features(df)
    complete = df.dropna(subset=["margin"]).sort_values("date").reset_index(drop=True)
    print(f"  {len(complete)} completed matches, "
          f"{int(complete['year'].min())}-{int(complete['year'].max())}")

    opt = load_optimization_results()
    if opt is None:
        print("FATAL: data/optimization_results.json missing")
        sys.exit(1)
    features = [c for c in opt["best_features"] if c in complete.columns]
    print(f"  Using {len(features)} classifier-winning features")

    # Baseline with current hardcoded params
    print(f"\n[2/3] Measuring current-params baseline (walk-forward CV)...")
    baseline_params = {
        "iterations": 2000, "learning_rate": 0.05, "depth": 4,
        "l2_leaf_reg": 3, "random_seed": SEED, "verbose": 0,
        "early_stopping_rounds": 100, "eval_metric": "MAE",
        "loss_function": "MAE", "use_best_model": True,
    }
    baseline = walk_forward_mae(baseline_params, complete, features)
    print(f"  pooled MAE:       {baseline['pooled_mae']:.3f}")
    print(f"  pooled RMSE:      {baseline['pooled_rmse']:.3f}")
    print(f"  pooled direction: {baseline['pooled_direction_acc']:.3f}")

    # Optuna
    print(f"\n[3/3] Running {N_TRIALS} Optuna trials "
          f"(timeout {TIMEOUT_SEC}s)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        best = study.best_value if study.best_value is not None else float("inf")
        elapsed = time.time() - t_start
        val_str = f"{trial.value:.3f}" if trial.value is not None else "pruned"
        print(f"  trial {trial.number:3d}  mae={val_str}  best={best:.3f}  [{elapsed:.0f}s]")

    study.optimize(
        lambda t: objective(t, complete, features),
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SEC,
        callbacks=[_cb],
        gc_after_trial=True,
    )

    best_params = dict(study.best_params)
    best_mae = float(study.best_value)
    improvement = baseline["pooled_mae"] - best_mae
    print("\n" + "=" * 80)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"  baseline MAE: {baseline['pooled_mae']:.3f}")
    print(f"  tuned MAE:    {best_mae:.3f}")
    print(f"  delta:        {-improvement:+.3f}  ({improvement:+.3f} points better)")
    print(f"  best params:  {json.dumps(best_params, indent=2)}")

    # Full WF eval with best params to capture pooled metrics
    full_best_params = _normalize_params({
        "iterations": 2000,
        "early_stopping_rounds": 100,
        "eval_metric": "MAE",
        "loss_function": "MAE",
        "use_best_model": True,
        "verbose": 0,
        "random_seed": SEED,
        **best_params,
    })
    tuned = walk_forward_mae(full_best_params, complete, features)
    print(f"\nTuned walk-forward pooled:")
    print(f"  MAE:       {tuned['pooled_mae']:.3f}")
    print(f"  RMSE:      {tuned['pooled_rmse']:.3f}")
    print(f"  Direction: {tuned['pooled_direction_acc']:.3f}")

    out = {
        "baseline": {"params": baseline_params, "walk_forward": baseline},
        "tuned": {
            "params": full_best_params,
            "best_params": best_params,
            "best_trial_number": int(study.best_trial.number),
            "n_trials_completed": len([t for t in study.trials if t.state.is_finished()]),
            "walk_forward": tuned,
        },
        "features": features,
        "val_years": VAL_YEARS,
        "seed": SEED,
        "runtime_sec": time.time() - t_start,
        "trial_log": [
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    out_path = PROJECT_ROOT / "data" / "margin_optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
