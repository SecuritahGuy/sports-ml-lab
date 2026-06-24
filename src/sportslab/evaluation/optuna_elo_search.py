"""Optuna joint hyperparameter search for all Elo parameters.

Searches K, HFA, reg, decay, qb_bonus, k_off, k_def, MOV type/scale/cap
simultaneously using rolling-origin 3-fold validation as objective.
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, NEUTRAL_COLUMN, TARGET_COLUMN
from sportslab.features.ratings import compute_od_elo_features

MOV_TYPES = ["none", "capped_linear", "log", "sqrt", "capped_log"]
N_TRIALS = 200


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _fit_platt(train_prob: np.ndarray, train_y: np.ndarray) -> Pipeline:
    platt = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    platt.fit(train_prob.reshape(-1, 1), train_y)
    return platt


def _objective(trial, df_raw, team_overrides) -> float:
    """Compute average validation log loss for a parameter set."""
    k = trial.suggest_int("k", 20, 60)
    hfa = trial.suggest_int("hfa", 10, 50)
    reg = trial.suggest_float("reg", 0.0, 0.5)
    decay = trial.suggest_int("decay", 16, 64)
    qb_bonus = trial.suggest_float("qb_bonus", 0.0, 0.5)
    k_off = trial.suggest_int("k_off", 20, 80)
    k_def = trial.suggest_int("k_def", 10, 60)
    mov_type = trial.suggest_categorical("mov_type", MOV_TYPES)

    if mov_type == "none":
        mov_scale = 0.0
        mov_cap = None
    else:
        mov_scale = trial.suggest_float("mov_scale", 0.01, 0.15)
        if mov_type in ("capped_linear", "capped_log"):
            mov_cap = trial.suggest_float("mov_cap", 1.5, 5.0)
        else:
            mov_cap = None

    # Combine base reg with qb bonus for full override dict
    combined_overrides = team_overrides.copy() if team_overrides else {}
    if qb_bonus > 0 and team_overrides:
        for team, base_reg in team_overrides.items():
            combined_overrides[team] = min(reg + qb_bonus, 1.0)

    df_elo = compute_od_elo_features(
        df_raw,
        k_factor=k,
        home_advantage=hfa,
        preseason_regression=reg,
        mov_type=mov_type,
        mov_scale=mov_scale,
        mov_cap=mov_cap,
        decay_half_life=decay,
        k_off=k_off,
        k_def=k_def,
        team_regression_overrides=combined_overrides if combined_overrides else None,
    )

    df_all = _filter_df(df_elo)
    y = df_all[TARGET_COLUMN].astype(float).values

    fold_lls = []
    for train_seasons, val_season in ROLLING_FOLDS:
        train_mask = df_all["season"].isin(train_seasons).values
        val_mask = (df_all["season"] == val_season).values

        if train_mask.sum() < 10 or val_mask.sum() < 10:
            return float("inf")

        train_elo = df_all["elo_prob"].values[train_mask]
        train_y = y[train_mask].astype(int)
        val_elo = df_all["elo_prob"].values[val_mask]
        val_y = y[val_mask]

        platt = _fit_platt(train_elo, train_y)
        val_proba = platt.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        val_ll = compute_classification_metrics(val_y, val_proba)["log_loss"]
        fold_lls.append(val_ll)

    return float(np.mean(fold_lls))


def run_optuna_search(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/optuna_elo_search.md",
    n_trials: int = N_TRIALS,
    study_name: str = "elo_joint_search",
    storage: str | None = None,
) -> str:
    """Run Optuna joint Elo parameter search.

    Parallel-safe: uses optuna storage if provided, otherwise in-memory.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)
    print(f"Loaded {len(df_raw)} rows from {fp}")

    # Compute base team regression overrides (before qb_bonus is applied)
    base_reg = 0.1
    team_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=base_reg,
        qb_change_bonus=0.0,
    )

    print(f"\n=== Optuna Joint Elo Search ({n_trials} trials) ===")
    print("Search space: K, HFA, reg, decay, qb_bonus, k_off, k_def, MOV type/scale/cap")
    print("Objective: avg rolling-origin 3-fold validation log loss")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
    )

    study.optimize(
        lambda trial: _objective(trial, df_raw, team_overrides),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_params = best.params

    print(f"\n{'=' * 60}")
    print(f"Best trial: {best.number}")
    print(f"Best avg val LL: {best.value:.4f}")
    print("Best params:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")
    print(f"{'=' * 60}")

    # === One-shot 2025 holdout ===
    print("\n=== 2025 Holdout Evaluation ===")
    k = best_params["k"]
    hfa = best_params["hfa"]
    reg = best_params["reg"]
    decay = best_params["decay"]
    qb_bonus = best_params["qb_bonus"]
    k_off = best_params["k_off"]
    k_def = best_params["k_def"]
    mov_type = best_params["mov_type"]
    mov_scale = best_params.get("mov_scale", 0.0)
    mov_cap = best_params.get("mov_cap", None)

    combined_overrides = team_overrides.copy() if team_overrides else {}
    if qb_bonus > 0 and team_overrides:
        for team, base_reg_val in team_overrides.items():
            combined_overrides[team] = min(reg + qb_bonus, 1.0)

    df_elo = compute_od_elo_features(
        df_raw,
        k_factor=k,
        home_advantage=hfa,
        preseason_regression=reg,
        mov_type=mov_type,
        mov_scale=mov_scale,
        mov_cap=mov_cap,
        decay_half_life=decay,
        k_off=k_off,
        k_def=k_def,
        team_regression_overrides=combined_overrides if combined_overrides else None,
    )

    df_all = _filter_df(df_elo)
    y = df_all[TARGET_COLUMN].astype(float).values

    # Incumbent: recompute with fixed params
    inc_reg = 0.1
    inc_team_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=inc_reg,
        qb_change_bonus=0.0,
    )
    inc_combined = inc_team_overrides.copy() if inc_team_overrides else {}
    qb_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=inc_reg,
        qb_change_bonus=0.2,
    )
    if qb_overrides:
        inc_combined.update(qb_overrides)

    df_inc = compute_od_elo_features(
        df_raw,
        k_factor=36,
        home_advantage=40,
        preseason_regression=inc_reg,
        mov_type="capped_linear",
        mov_scale=0.05,
        mov_cap=2.0,
        decay_half_life=32,
        k_off=52,
        k_def=20,
        team_regression_overrides=inc_combined if inc_combined else None,
    )
    df_inc_f = _filter_df(df_inc)
    inc_y = df_inc_f[TARGET_COLUMN].astype(float).values

    hold_mask = (df_all["season"] == HOLDOUT_SEASON).values
    train_mask = df_all["season"].isin([2021, 2022, 2023, 2024]).values

    hold_y = y[hold_mask]
    train_elo = df_all["elo_prob"].values[train_mask]
    train_y = y[train_mask].astype(int)
    hold_elo = df_all["elo_prob"].values[hold_mask]

    platt = _fit_platt(train_elo, train_y)
    hold_proba = platt.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_m = compute_classification_metrics(hold_y, hold_proba)

    # Incumbent holdout
    inc_hold_y = inc_y[hold_mask]
    inc_train_elo = df_inc_f["elo_prob"].values[train_mask]
    inc_train_y = inc_y[train_mask].astype(int)
    inc_hold_elo = df_inc_f["elo_prob"].values[hold_mask]
    inc_platt = _fit_platt(inc_train_elo, inc_train_y)
    inc_hold_proba = inc_platt.predict_proba(inc_hold_elo.reshape(-1, 1))[:, 1]
    inc_hold_m = compute_classification_metrics(inc_hold_y, inc_hold_proba)

    print(f"  Optuna best + Platt: holdout LL = {hold_m['log_loss']:.4f}")
    print(f"  Incumbent + Platt:   holdout LL = {inc_hold_m['log_loss']:.4f}")

    baselines = {}
    random_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y.mean()
    prior_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))
    baselines["random"] = {"log_loss": random_ll}
    baselines["home_prior"] = {"log_loss": prior_ll, "rate": prior_rate}

    # === Write report ===
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Optuna Joint Elo Parameter Search\n\n")
        f.write("*Jointly optimizing all Elo parameters simultaneously")
        f.write(" with rolling-origin 3-fold validation.*\n\n")

        f.write("## Method\n\n")
        f.write(f"- **Optimizer**: Optuna TPESampler, {n_trials} trials\n")
        f.write("- **Search space** (10 parameters):\n")
        f.write("  - K: 20–60\n")
        f.write("  - HFA: 10–50\n")
        f.write("  - reg (base): 0.0–0.5\n")
        f.write("  - decay half-life: 16–64 games\n")
        f.write("  - qb_bonus: 0.0–0.5\n")
        f.write("  - k_off: 20–80\n")
        f.write("  - k_def: 10–60\n")
        f.write(f"  - MOV type: {MOV_TYPES}\n")
        f.write("  - MOV scale: 0.01–0.15 (log, only if MOV != none)\n")
        f.write("  - MOV cap: 1.5–5.0 (only for capped types)\n")
        f.write("- **Objective**: Average validation log loss across 3 rolling-origin folds\n")
        f.write("- **Calibration**: Platt scaling on each fold\n\n")

        f.write("## Best Params\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for k, v in sorted(best_params.items()):
            f.write(f"| {k} | {v} |\n")
        f.write(f"| best_val_ll | {best.value:.4f} |\n")
        f.write("\n")

        f.write("## Holdout Results\n\n")
        f.write("| Model | Hold LL | Brier | AUC | Acc |\n")
        f.write("|-------|---------|-------|-----|-----|\n")
        f.write(f"| Random | {random_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_ll:.4f} | — | 0.5000 | — |\n")
        f.write(
            f"| Incumbent (O/D Elo + Platt) | {inc_hold_m['log_loss']:.4f}"
            f" | {inc_hold_m['brier_score']:.4f}"
            f" | {inc_hold_m['roc_auc']:.4f}"
            f" | {inc_hold_m['accuracy']:.4f} |\n"
        )
        f.write(
            f"| **Optuna best + Platt** | {hold_m['log_loss']:.4f}"
            f" | {hold_m['brier_score']:.4f}"
            f" | {hold_m['roc_auc']:.4f}"
            f" | {hold_m['accuracy']:.4f} |\n\n"
        )

        improvement = inc_hold_m["log_loss"] - hold_m["log_loss"]
        if improvement > 0:
            f.write(f"**Optuna best beats incumbent by {improvement:.4f} holdout log loss!** ")
            if improvement > 0.005:
                f.write("**New research champion.**\n")
            else:
                f.write("Marginal improvement — user discretion on promotion.\n")
        else:
            f.write(
                f"**Incumbent retains champion.** "
                f"Optuna search could not beat incumbent "
                f"({-improvement:.4f} worse on holdout).\n"
            )

        # Top 10 trials
        f.write("\n## Top 10 Trials\n\n")
        f.write("| Trial | Avg Val LL | K | HFA | Reg | Decay | QB Bonus | k_off | k_def | MOV |\n")
        f.write("|-------|-----------|----|-----|------|-------|---------|-------|-------|-----|\n")
        best_trials = study.best_trials[:10]
        for t in best_trials:
            p = t.params
            f.write(
                f"| {t.number} | {t.value:.4f}"
                f" | {p.get('k', '?')} | {p.get('hfa', '?')}"
                f" | {p.get('reg', '?'):.2f}"
                f" | {p.get('decay', '?')}"
                f" | {p.get('qb_bonus', 0):.2f}"
                f" | {p.get('k_off', '?')} | {p.get('k_def', '?')}"
                f" | {p.get('mov_type', '?')} |\n"
            )

    print(f"\nReport written to: {rp}")
    return str(rp)
