"""Dynamic Bayesian Elo experiment: MLE-estimated state-space team strength.

Tests whether a correctly-formulated state-space model (margin observations,
standard Kalman filter, MLE-estimated variance parameters) beats the
v3.0.0 Frozen QB Overlay incumbent.
"""

from pathlib import Path

import numpy as np
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
from sportslab.features.dynamic_elo import DynamicBayesianElo
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025
FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]

INC_K, INC_HFA, INC_REG, INC_DECAY, INC_QB_BONUS = 36, 40, 0.1, 32, 0.2
INCUMBENT_FEATURES = [
    "home_qb_changed",
    "away_qb_changed",
    "home_rolling_mov_3",
    "away_rolling_mov_3",
]

OVERLAY_GAMMA = 1.0
OVERLAY_CAP = 40
ELO_TO_LOGIT = np.log(10) / 400.0


def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def _build_gate_mask(df):
    h_c = df.get("home_qb_changed", pd.Series(0)).fillna(0).values.astype(float)
    a_c = df.get("away_qb_changed", pd.Series(0)).fillna(0).values.astype(float)
    h_s = df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    a_s = df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    return (h_c == 1) | (a_c == 1) | (h_s < 17) | (a_s < 17)


def _fit_margins_and_predict(df, train_mask, val_mask, initial_sigma=20):
    """Fit DynamicBayesianElo on train_mask games, predict margins on train+val."""
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    model = DynamicBayesianElo(initial_sigma=initial_sigma)
    model.fit(train_df)

    train_margins = model.training_margins()
    val_margins = model.predict(val_df)

    return train_margins, val_margins, model


def run_dynamic_elo_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/dynamic_elo.md",
) -> str:
    rp = Path(report_path)

    print("=== Loading + building feature pipeline ===")
    df_raw = pd.read_parquet(ft_path)
    df = compute_qb_features(df_raw)
    df = compute_qb_adjustments(df)
    df = compute_situational_features(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy().reset_index(drop=True)
    df = df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    y = df[TARGET_COLUMN].values
    gate_mask = _build_gate_mask(df)
    home_qb_adj = df.get("home_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    away_qb_adj = df.get("away_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    print(f"  {len(df)} eligible games, {gate_mask.sum()} with active overlay")

    # --- Incumbent ---
    print("\n=== Incumbent (standard Elo) ===")
    inc_overrides = build_team_regression_overrides(
        df, preseason_regression=INC_REG, qb_change_bonus=INC_QB_BONUS,
    )
    df_inc = compute_elo_features(
        df, k_factor=INC_K, home_advantage=INC_HFA,
        preseason_regression=INC_REG,
        team_regression_overrides=inc_overrides,
        decay_half_life=INC_DECAY,
    )
    inc_prob = df_inc["elo_prob"].values

    inc_fll = []
    hold = df["season"] == HOLDOUT_SEASON
    for train_s, val_s in FOLDS:
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values
        x_train = np.column_stack([inc_prob[tr]] + [
            df.loc[tr, c].values for c in INCUMBENT_FEATURES
        ])
        x_val = np.column_stack([inc_prob[va]] + [
            df.loc[va, c].values for c in INCUMBENT_FEATURES
        ])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(x_train, y[tr].astype(int))
        bp = pipe.predict_proba(x_val)[:, 1]
        fp = _sigmoid(_logit(bp) + OVERLAY_GAMMA * (
            np.clip(home_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
            - np.clip(away_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
        ) * ELO_TO_LOGIT * gate_mask[va].astype(float))
        inc_fll.append(compute_classification_metrics(y[va], fp)["log_loss"])

    x_train = np.column_stack([inc_prob[~hold]] + [
        df.loc[~hold, c].values for c in INCUMBENT_FEATURES
    ])
    x_hold = np.column_stack([inc_prob[hold]] + [
        df.loc[hold, c].values for c in INCUMBENT_FEATURES
    ])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_train, y[~hold].astype(int))
    bp = pipe.predict_proba(x_hold)[:, 1]
    fp_h = _sigmoid(_logit(bp) + OVERLAY_GAMMA * (
        np.clip(home_qb_adj[hold], -OVERLAY_CAP, OVERLAY_CAP)
        - np.clip(away_qb_adj[hold], -OVERLAY_CAP, OVERLAY_CAP)
    ) * ELO_TO_LOGIT * gate_mask[hold].astype(float))
    inc_hm = compute_classification_metrics(y[hold], fp_h)
    inc_val = float(np.mean(inc_fll))
    print(f"  Platt: val={inc_val:.4f} hold={inc_hm['log_loss']:.4f}")

    # --- Dynamic Bayesian Elo (per-fold Platt) ---
    print("\n=== Dynamic Bayesian Elo ===")
    de_fll = []
    de_fold_params = []

    for train_s, val_s in FOLDS:
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values

        train_margins, val_margins, model = _fit_margins_and_predict(df, tr, va)
        m = model
        de_fold_params.append((val_s, m.sigma_evolution, m.sigma_observation, m.hfa))
        print(f"  Fold {val_s}: sigma_evo={m.sigma_evolution:.3f} "
              f"sigma_obs={m.sigma_observation:.3f} HFA={m.hfa:.3f}")

        # Per-fold Platt on margin predictions
        x_train = np.column_stack([train_margins] + [
            df.loc[tr, c].values for c in INCUMBENT_FEATURES
        ])
        x_val = np.column_stack([val_margins] + [
            df.loc[va, c].values for c in INCUMBENT_FEATURES
        ])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(x_train, y[tr].astype(int))
        bp = pipe.predict_proba(x_val)[:, 1]
        fp = _sigmoid(_logit(bp) + OVERLAY_GAMMA * (
            np.clip(home_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
            - np.clip(away_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
        ) * ELO_TO_LOGIT * gate_mask[va].astype(float))
        de_fll.append(compute_classification_metrics(y[va], fp)["log_loss"])

    # Holdout
    train_margins, val_margins, hold_model = _fit_margins_and_predict(
        df, ~hold, hold
    )
    de_fold_params.append((HOLDOUT_SEASON, hold_model.sigma_evolution,
                           hold_model.sigma_observation, hold_model.hfa))
    print(f"  Holdout: sigma_evo={hold_model.sigma_evolution:.3f} "
          f"sigma_obs={hold_model.sigma_observation:.3f} HFA={hold_model.hfa:.3f}")

    x_train = np.column_stack([train_margins] + [
        df.loc[~hold, c].values for c in INCUMBENT_FEATURES
    ])
    x_hold = np.column_stack([val_margins] + [
        df.loc[hold, c].values for c in INCUMBENT_FEATURES
    ])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_train, y[~hold].astype(int))
    bp = pipe.predict_proba(x_hold)[:, 1]
    fp_h = _sigmoid(_logit(bp) + OVERLAY_GAMMA * (
        np.clip(home_qb_adj[hold], -OVERLAY_CAP, OVERLAY_CAP)
        - np.clip(away_qb_adj[hold], -OVERLAY_CAP, OVERLAY_CAP)
    ) * ELO_TO_LOGIT * gate_mask[hold].astype(float))
    de_hm = compute_classification_metrics(y[hold], fp_h)
    de_val = float(np.mean(de_fll))
    print(f"  Platt: val={de_val:.4f} hold={de_hm['log_loss']:.4f}")

    # --- Report ---
    print(f"\n=== Report: {rp} ===")
    with open(rp, "w") as f:
        f.write("# Dynamic Bayesian Elo\n\n")
        f.write("State-space model with MLE-estimated variance parameters.\n\n")
        f.write("**Model:**\n")
        f.write("- State: latent team strength theta (Nx1) on margin-of-victory scale\n")
        f.write("- Transition: theta_k = theta_{k-1} + epsilon\n")
        f.write("- Observation: y_k = theta_k[home] - theta_k[away] + HFA + v_k\n")
        f.write("- Estimation: Maximum likelihood via prediction-error decomposition\n")
        f.write("- Pre-game margin predicted from filtered state -> Platt-calibrated\n\n")

        d_v = de_val - inc_val
        d_h = de_hm['log_loss'] - inc_hm['log_loss']
        f.write("| Variant | Val LL | Hold LL |\n")
        f.write("|---------|--------|--------|\n")
        f.write(f"| Incumbent | {inc_val:.4f} | {inc_hm['log_loss']:.4f} |\n")
        f.write(f"| Dynamic Elo | {de_val:.4f} (Delta={d_v:+.4f}) | "
                f"{de_hm['log_loss']:.4f} (Delta={d_h:+.4f}) |\n")

        f.write("\n## Per-Fold Parameters\n\n")
        f.write("| Fold | sigma_evolution | sigma_observation | HFA |\n")
        f.write("|------|----------------|-------------------|-----|\n")
        for v, fp, op, hp in de_fold_params:
            label = f"Holdout ({HOLDOUT_SEASON})" if v == HOLDOUT_SEASON else f"Fold {v}"
            f.write(f"| {label} | {fp:.3f} | {op:.3f} | {hp:.3f} |\n")

        f.write("\n## Platt Metrics\n\n")
        f.write("| Model | Val LL | Fold1 | Fold2 | Fold3 | Hold LL | Brier | AUC | Acc |\n")
        f.write("|-------|--------|-------|-------|-------|---------|-------|-----|-----|\n")
        f.write(f"| Incumbent | {inc_val:.4f} | {inc_fll[0]:.4f} | "
                f"{inc_fll[1]:.4f} | {inc_fll[2]:.4f} | "
                f"{inc_hm['log_loss']:.4f} | {inc_hm['brier_score']:.4f} | "
                f"{inc_hm['roc_auc']:.3f} | {inc_hm['accuracy']:.3f} |\n")
        f.write(f"| Dynamic Elo | {de_val:.4f} | {de_fll[0]:.4f} | "
                f"{de_fll[1]:.4f} | {de_fll[2]:.4f} | "
                f"{de_hm['log_loss']:.4f} | {de_hm['brier_score']:.4f} | "
                f"{de_hm['roc_auc']:.3f} | {de_hm['accuracy']:.3f} |\n")

        if d_v <= -0.001 and d_h <= -0.001:
            f.write("\n## ✅ PROMOTED\n")
        else:
            f.write(f"\n## ❌ NOT PROMOTED — Dval={d_v:+.4f}, Dhold={d_h:+.4f}\n")

        f.write("\n---\nReport: dynamic_elo_experiment.py\n")

    print(f"  Report written to {rp}")
    return str(rp)
