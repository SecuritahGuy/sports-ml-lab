"""Optuna combinatorial feature selection experiment.

Searches over all available feature groups (16 groups: 10 situational,
4 QB, 2 coach) using boolean inclusion masks, with rolling-origin
3-fold validation as the objective. Confirms or improves upon the
greedy forward-selection champion (qb_changed + rolling_mov_3).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.coach import compute_coach_features
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025

ROLLING_FOLDS: List[Tuple[List[int], int]] = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2

N_TRIALS = 500

FEATURE_GROUPS: Dict[str, List[str]] = {
    "rolling_mov_3": ["home_rolling_mov_3", "away_rolling_mov_3"],
    "rolling_mov_5": ["home_rolling_mov_5", "away_rolling_mov_5"],
    "rolling_pts_for": ["home_rolling_pts_for", "away_rolling_pts_for"],
    "rolling_pts_against": ["home_rolling_pts_against", "away_rolling_pts_against"],
    "win_streak": ["home_win_streak", "away_win_streak"],
    "ytd_win_pct": ["home_ytd_win_pct", "away_ytd_win_pct"],
    "turf_flag": ["turf_flag"],
    "high_altitude": ["high_altitude_flag"],
    "prime_time": ["prime_time_flag"],
    "rest_diff_squared": ["rest_diff_squared"],
    "qb_changed": ["home_qb_changed", "away_qb_changed"],
    "qb_win_pct": ["home_qb_win_pct_pre", "away_qb_win_pct_pre"],
    "games_since_change": ["home_games_since_qb_change", "away_games_since_qb_change"],
    "new_qb": ["home_new_qb_flag", "away_new_qb_flag"],
    "coach_tenure": ["home_coach_tenure", "away_coach_tenure"],
    "coach_win_pct": ["home_coach_win_pct", "away_coach_win_pct"],
}


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def _build_feature_table() -> pd.DataFrame:
    fp = Path("data/features/nfl/feature_table.parquet")
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")
    df_raw = pd.read_parquet(fp)

    print("Computing Elo features...")
    team_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=BEST_REG,
        qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=team_overrides,
        decay_half_life=BEST_DECAY,
    )
    print("Computing QB features...")
    df = compute_qb_features(df)
    print("Computing situational features...")
    df = compute_situational_features(df)
    print("Computing coach features...")
    df = compute_coach_features(df)

    df = _filter_df(df)
    print(f"  After filter: {len(df)} rows")
    return df


def _compute_rolling_val_ll(
    df: pd.DataFrame,
    elo_prob: np.ndarray,
    y: np.ndarray,
    feat_columns: List[str],
) -> float:
    fold_lls = []
    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df["season"].isin(train_seasons).values
        is_val = (df["season"] == val_season).values
        train_elo = elo_prob[is_train]
        val_elo = elo_prob[is_val]
        train_y_ = y[is_train].astype(int)
        val_y_ = y[is_val]
        if feat_columns:
            train_feat = df.loc[is_train, feat_columns].values
            val_feat = df.loc[is_val, feat_columns].values
            x_train = np.column_stack([train_elo, train_feat])
            x_val = np.column_stack([val_elo, val_feat])
        else:
            x_train = train_elo.reshape(-1, 1)
            x_val = val_elo.reshape(-1, 1)
        pipe = _logistic_model()
        pipe.fit(x_train, train_y_)
        proba = pipe.predict_proba(x_val)[:, 1]
        fold_lls.append(compute_classification_metrics(val_y_, proba)["log_loss"])
    return float(np.mean(fold_lls))


def _eval_holdout_ll(
    df: pd.DataFrame,
    elo_prob: np.ndarray,
    y: np.ndarray,
    feat_cols: List[str],
) -> float:
    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = (df["season"] == HOLDOUT_SEASON).values
    train_elo = elo_prob[is_train]
    hold_elo = elo_prob[is_hold]
    train_y_ = y[is_train].astype(int)
    hold_y_ = y[is_hold]
    if feat_cols:
        train_feat = df.loc[is_train, feat_cols].values
        hold_feat = df.loc[is_hold, feat_cols].values
        x_train = np.column_stack([train_elo, train_feat])
        x_hold = np.column_stack([hold_elo, hold_feat])
    else:
        x_train = train_elo.reshape(-1, 1)
        x_hold = hold_elo.reshape(-1, 1)
    pipe = _logistic_model()
    pipe.fit(x_train, train_y_)
    proba = pipe.predict_proba(x_hold)[:, 1]
    return compute_classification_metrics(hold_y_, proba)["log_loss"]


def run_optuna_feature_selection(
    n_trials: int = N_TRIALS,
    report_path: str = "reports/experiments/optuna_feature_selection.md",
) -> str:
    print("\n=== Building feature table ===")
    df = _build_feature_table()

    elo_prob = df["elo_prob"].values
    y = df[TARGET_COLUMN].astype(float).values

    available_groups: Dict[str, List[str]] = {}
    for name, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if present:
            available_groups[name] = present

    group_names = sorted(available_groups.keys())
    print(f"  Active groups: {group_names}")

    # Baseline: Platt only (no feature groups)
    platt_ll = _compute_rolling_val_ll(df, elo_prob, y, [])
    print(f"\n  Platt baseline avg val LL: {platt_ll:.4f}")

    # Incumbent: qb_changed + rolling_mov_3
    inc_cols = available_groups.get("qb_changed", []) + available_groups.get("rolling_mov_3", [])
    incumbent_ll = _compute_rolling_val_ll(df, elo_prob, y, inc_cols)
    print(f"  Incumbent (qb_changed + mov3) avg val LL: {incumbent_ll:.4f}")

    def objective(trial: optuna.Trial) -> float:
        selected_cols: List[str] = []
        for name in group_names:
            include = trial.suggest_categorical(name, [0, 1])
            if include:
                selected_cols.extend(available_groups[name])
        val_ll = _compute_rolling_val_ll(df, elo_prob, y, selected_cols)
        if val_ll > platt_ll + 0.02:
            raise optuna.TrialPruned()
        return val_ll

    print(f"\n=== Running {n_trials} Optuna trials ===")
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Best config
    best = study.best_trial
    best_mask = {name: int(best.params.get(name, 0)) for name in group_names}
    best_cols: List[str] = []
    best_active: List[str] = []
    for name in group_names:
        if best_mask.get(name, 0):
            best_cols.extend(available_groups[name])
            best_active.append(name)

    best_val_ll = _compute_rolling_val_ll(df, elo_prob, y, best_cols)

    print(f"\n  Best trial: val LL = {best_val_ll:.4f}")
    print(f"  Active groups: {best_active}")
    print(f"  Total params: {len(best_cols)}")

    # Feature importance
    importance = optuna.importance.get_param_importances(study)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Holdout
    print("\n=== 2025 Holdout ===")
    platt_hold_ll = _eval_holdout_ll(df, elo_prob, y, [])
    print(f"  Platt incumbent: {platt_hold_ll:.4f}")

    inc_hold_ll = _eval_holdout_ll(df, elo_prob, y, inc_cols)
    print(f"  Incumbent (qb_changed + mov3): {inc_hold_ll:.4f}")

    opt_hold_ll = _eval_holdout_ll(df, elo_prob, y, best_cols)
    print(f"  Optuna best: {opt_hold_ll:.4f}")

    # Report
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Optuna Feature Selection\n\n")
        f.write("*Combinatorial search over 16 feature groups using Optuna TPE.*\n\n")

        f.write("## Method\n\n")
        f.write(
            f"Rolling-origin 3-fold validation. Each feature group is a"
            f" boolean inclusion variable (0/1). {n_trials} trials with TPE"
            f" sampler, MedianPruner. Objective: avg val log loss.\n\n"
        )
        f.write(f"Incumbent: qb_changed + rolling_mov_3 (val LL = {incumbent_ll:.4f}).\n\n")

        f.write("### Feature Groups\n\n")
        f.write("| Group | Columns |\n")
        f.write("|-------|---------|\n")
        for name in group_names:
            cols = available_groups[name]
            f.write(f"| `{name}` | {', '.join(cols)} |\n")

        f.write("\n### Elo Params\n\n")
        f.write(
            f"K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG},"
            f" decay={BEST_DECAY}, qb_bonus={BEST_QB_BONUS}\n\n"
        )

        f.write("## Results\n\n")
        f.write(f"**Trials completed:** {len(study.trials)}\n")
        pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        f.write(f"**Pruned trials:** {pruned}\n\n")

        f.write("### Best Configuration\n\n")
        f.write("| Group | Included |\n")
        f.write("|-------|----------|\n")
        for name in group_names:
            val = "✓" if best_mask.get(name, 0) else "✗"
            f.write(f"| `{name}` | {val} |\n")
        active_display = ", ".join(best_active) if best_active else "(none)"
        f.write(f"\n**Active groups:** {active_display}\n")
        f.write(f"\n**Total feature columns:** {len(best_cols)}\n")
        f.write(f"\n**Validation LL:** {best_val_ll:.4f}\n")

        f.write("\n### Validation Comparison\n\n")
        f.write("| Model | Avg Val LL |\n")
        f.write("|-------|-----------|\n")
        f.write(f"| Platt baseline | {platt_ll:.4f} |\n")
        f.write(f"| Incumbent | {incumbent_ll:.4f} |\n")
        f.write(f"| Optuna best | {best_val_ll:.4f} |\n")

        f.write("\n### Holdout Comparison\n\n")
        f.write("| Model | Holdout LL |\n")
        f.write("|-------|-----------|\n")
        f.write(f"| Platt baseline | {platt_hold_ll:.4f} |\n")
        f.write(f"| Incumbent | {inc_hold_ll:.4f} |\n")
        f.write(f"| Optuna best | {opt_hold_ll:.4f} |\n")

        f.write("\n### Feature Importance (TPE param importance)\n\n")
        f.write("| Group | Importance |\n")
        f.write("|-------|-----------|\n")
        for name, imp in sorted_importance:
            f.write(f"| `{name}` | {imp:.4f} |\n")

        # Decision
        beats_val = best_val_ll < incumbent_ll
        beats_hold = opt_hold_ll < inc_hold_ll

        f.write("\n## Decision\n\n")
        if beats_val and beats_hold:
            f.write(
                f"**PROMOTED:** Optuna best beats the incumbent on both"
                f" validation ({best_val_ll:.4f} vs {incumbent_ll:.4f})"
                f" and holdout ({opt_hold_ll:.4f} vs {inc_hold_ll:.4f}).\n"
            )
        elif beats_val:
            f.write(
                f"**Diagnostic:** Optuna best beats incumbent on validation"
                f" ({best_val_ll:.4f} vs {incumbent_ll:.4f})"
                f" but not holdout ({opt_hold_ll:.4f} vs {inc_hold_ll:.4f}).\n"
            )
        else:
            f.write(
                f"**No improvement.** Optuna best ({best_val_ll:.4f})"
                f" does not beat incumbent ({incumbent_ll:.4f})"
                f" on validation.\n"
            )
            f.write("Greedy forward selection result is validated.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
