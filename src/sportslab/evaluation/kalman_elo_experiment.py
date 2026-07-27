"""Kalman-filter Elo experiment vs standard Elo incumbent.

Tests whether uncertainty-aware ratings (Kalman gain proportional to
sigma²) improve on the fixed-K Elo spine, using the same feature pipeline
(qb_changed + rolling_mov_3 + Platt + QB overlay).
"""

from itertools import product
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
from sportslab.features.kalman_elo import compute_kalman_elo_features
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025
FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]

# Incumbent champion config
INC_K, INC_HFA, INC_REG, INC_DECAY, INC_QB_BONUS = 36, 40, 0.1, 32, 0.2
INCUMBENT_FEATURES = ["home_qb_changed", "away_qb_changed",
                      "home_rolling_mov_3", "away_rolling_mov_3"]

# Kalman grid
GRID_K = [32, 36, 40]
GRID_HFA = [35, 40, 45]
GRID_INIT_SIGMA = [200, 400]
GRID_OBS_NOISE = [50, 100, 200]
GRID_REG = [0.05, 0.1, 0.2]

# QB overlay
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
    h_changed = df.get("home_qb_changed", pd.Series(0)).fillna(0).values.astype(float)
    a_changed = df.get("away_qb_changed", pd.Series(0)).fillna(0).values.astype(float)
    h_starts = df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    a_starts = df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    return (h_changed == 1) | (a_changed == 1) | (h_starts < 17) | (a_starts < 17)


def _raw_elo_with_overlay(df, prob_col, gate_mask, y):
    """Compute raw log loss with QB overlay (no Platt, no features)."""
    home_qb_adj = df.get("home_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    away_qb_adj = df.get("away_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    prob = df[prob_col].values

    hold = df["season"] == HOLDOUT_SEASON
    fls = []
    for train_s, val_s in FOLDS:
        va = (df["season"] == val_s).values
        lt = _logit(prob[va])
        o = OVERLAY_GAMMA * (
            np.clip(home_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
            - np.clip(away_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
        ) * ELO_TO_LOGIT
        fp = _sigmoid(lt + o * gate_mask[va].astype(float))
        fls.append(compute_classification_metrics(y[va], fp)["log_loss"])

    lt_h = _logit(prob[hold])
    o_h = OVERLAY_GAMMA * (
        np.clip(home_qb_adj[hold], -OVERLAY_CAP, OVERLAY_CAP)
        - np.clip(away_qb_adj[hold], -OVERLAY_CAP, OVERLAY_CAP)
    ) * ELO_TO_LOGIT
    fp_h = _sigmoid(lt_h + o_h * gate_mask[hold].astype(float))
    hm = compute_classification_metrics(y[hold], fp_h)
    return fls, hm


def _platt_pipeline(df, prob_col, gate_mask, y):
    """Fit Platt(prob + features) + QB overlay."""
    fcols = [prob_col] + INCUMBENT_FEATURES
    home_qb_adj = df.get("home_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    away_qb_adj = df.get("away_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)

    fls = []
    for train_s, val_s in FOLDS:
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values
        xt = np.column_stack([df.loc[tr, c].values for c in fcols])
        xv = np.column_stack([df.loc[va, c].values for c in fcols])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(xt, y[tr].astype(int))
        bp = pipe.predict_proba(xv)[:, 1]
        fp = _sigmoid(_logit(bp) + OVERLAY_GAMMA * (
            np.clip(home_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
            - np.clip(away_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
        ) * ELO_TO_LOGIT * gate_mask[va].astype(float))
        fls.append(compute_classification_metrics(y[va], fp)["log_loss"])

    is_train = df["season"] != HOLDOUT_SEASON
    is_hold = df["season"] == HOLDOUT_SEASON
    xt = np.column_stack([df.loc[is_train, c].values for c in fcols])
    xh = np.column_stack([df.loc[is_hold, c].values for c in fcols])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(xt, y[is_train].astype(int))
    bp = pipe.predict_proba(xh)[:, 1]
    fp = _sigmoid(_logit(bp) + OVERLAY_GAMMA * (
        np.clip(home_qb_adj[is_hold], -OVERLAY_CAP, OVERLAY_CAP)
        - np.clip(away_qb_adj[is_hold], -OVERLAY_CAP, OVERLAY_CAP)
    ) * ELO_TO_LOGIT * gate_mask[is_hold].astype(float))
    hm = compute_classification_metrics(y[is_hold], fp)
    return fls, hm


def run_kalman_elo_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/kalman_elo.md",
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
    print(f"  {len(df)} eligible games, {gate_mask.sum()} with active overlay")

    # --- Incumbent (standard Elo) ---
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
    df["elo_prob"] = df_inc["elo_prob"].values
    inc_raw_fll, inc_raw_hm = _raw_elo_with_overlay(df_inc, "elo_prob", gate_mask, y)
    inc_raw_val = float(np.mean(inc_raw_fll))
    inc_platt_fll, inc_platt_hm = _platt_pipeline(df, "elo_prob", gate_mask, y)
    inc_platt_val = float(np.mean(inc_platt_fll))
    print(f"  Raw: val={inc_raw_val:.4f} hold={inc_raw_hm['log_loss']:.4f}")
    print(f"  Platt: val={inc_platt_val:.4f} hold={inc_platt_hm['log_loss']:.4f}")

    # --- Kalman Elo grid ---
    n_combos = len(list(product(GRID_K, GRID_HFA, GRID_INIT_SIGMA, GRID_OBS_NOISE, GRID_REG)))
    print(f"\n=== Kalman Elo grid ({n_combos} combos) ===")
    kalman_results = []
    for i, (k, hfa, init_s, obs_n, reg) in enumerate(product(
        GRID_K, GRID_HFA, GRID_INIT_SIGMA, GRID_OBS_NOISE, GRID_REG,
    )):
        df_k = compute_kalman_elo_features(
            df, k_factor=k, home_advantage=hfa,
            initial_sigma=init_s, obs_noise=obs_n,
            preseason_regression=reg,
        )
        raw_fll, raw_hm = _raw_elo_with_overlay(df_k, "kalman_prob", gate_mask, y)
        raw_val = float(np.mean(raw_fll))
        kalman_results.append({
            "label": f"K={k}_HFA={hfa}_is={init_s}_on={obs_n}_reg={reg}",
            "k": k, "hfa": hfa, "init_sigma": init_s, "obs_noise": obs_n, "reg": reg,
            "raw_val": raw_val, "raw_hold": raw_hm["log_loss"],
            "raw_fll": raw_fll,
        })
        if (i + 1) % 40 == 0:
            print(f"  Progress: {i+1}")

    kr = pd.DataFrame(kalman_results)
    best_kalman = kr.loc[kr["raw_val"].idxmin()]
    best_val = best_kalman['raw_val']
    best_hold = best_kalman['raw_hold']
    print(f"\n  Best raw val: {best_kalman['label']} (val={best_val:.4f} hold={best_hold:.4f})")

    # --- Test best Kalman with Platt ---
    best_cfg = best_kalman
    df_best = compute_kalman_elo_features(
        df, k_factor=best_cfg["k"], home_advantage=best_cfg["hfa"],
        initial_sigma=best_cfg["init_sigma"], obs_noise=best_cfg["obs_noise"],
        preseason_regression=best_cfg["reg"],
    )
    df["kalman_prob"] = df_best["kalman_prob"].values
    kal_platt_fll, kal_platt_hm = _platt_pipeline(df, "kalman_prob", gate_mask, y)
    kal_platt_val = float(np.mean(kal_platt_fll))

    print(f"  Platt: val={kal_platt_val:.4f} hold={kal_platt_hm['log_loss']:.4f}")

    # --- Report ---
    print(f"\n=== Report: {rp} ===")
    with open(rp, "w") as f:
        f.write("# Kalman-Filter Elo\n\n")
        f.write("Uncertainty-aware Elo ratings using Kalman gain (sigma² / total_var) ")
        f.write("instead of fixed K-factor update proportions.\n\n")
        f.write(f"**Grid:** {len(kalman_results)} combos (K∈{GRID_K} × HFA∈{GRID_HFA} × ")
        f.write(f"init_sigma∈{GRID_INIT_SIGMA} × obs_noise∈{GRID_OBS_NOISE} × reg∈{GRID_REG})\n\n")

        f.write("| Config | Raw Val LL | Fold1 | Fold2 | Fold3 | Raw Hold LL |\n")
        f.write("|--------|-----------|-------|-------|-------|-------------|\n")
        f.write(f"| Incumbent (Std Elo) | {inc_raw_val:.4f} | {inc_raw_fll[0]:.4f} | "
                f"{inc_raw_fll[1]:.4f} | {inc_raw_fll[2]:.4f} | {inc_raw_hm['log_loss']:.4f} |\n")
        # Show top 10 Kalman
        for _, r in kr.nsmallest(10, "raw_val").iterrows():
            f.write(f"| {r['label']} | {r['raw_val']:.4f} "
                    f"| {r['raw_fll'][0]:.4f} | {r['raw_fll'][1]:.4f} | {r['raw_fll'][2]:.4f} "
                    f"| {r['raw_hold']:.4f} |\n")

        f.write("\n## Platt + Features + QB Overlay\n\n")
        f.write("| Model | Val LL | Fold1 | Fold2 | Fold3 | Hold LL | Brier | AUC | Acc |\n")
        f.write("|-------|--------|-------|-------|-------|---------|-------|-----|-----|\n")
        d_v = kal_platt_val - inc_platt_val
        d_h = kal_platt_hm['log_loss'] - inc_platt_hm['log_loss']
        f.write(f"| Incumbent | {inc_platt_val:.4f} | {inc_platt_fll[0]:.4f} | "
                f"{inc_platt_fll[1]:.4f} | {inc_platt_fll[2]:.4f} | "
                f"{inc_platt_hm['log_loss']:.4f} | {inc_platt_hm['brier_score']:.4f} | "
                f"{inc_platt_hm['roc_auc']:.3f} | {inc_platt_hm['accuracy']:.3f} |\n")
        f.write(f"| Kalman Elo (best) | {kal_platt_val:.4f} (Δ={d_v:+.4f}) | "
                f"{kal_platt_fll[0]:.4f} | {kal_platt_fll[1]:.4f} | "
                f"{kal_platt_fll[2]:.4f} | "
                f"{kal_platt_hm['log_loss']:.4f} (Δ={d_h:+.4f}) | "
                f"{kal_platt_hm['brier_score']:.4f} | {kal_platt_hm['roc_auc']:.3f} | "
                f"{kal_platt_hm['accuracy']:.3f} |\n")

        f.write(f"\n**Best Kalman config:** {best_cfg['label']}\n")
        if d_v <= -0.001 and d_h <= -0.001:
            f.write("\n## ✅ PROMOTED — beats incumbent on both val and holdout\n")
        else:
            f.write(f"\n## ❌ NOT PROMOTED — Δval={d_v:+.4f}, Δhold={d_h:+.4f}\n")

        f.write("\n---\nReport: kalman_elo_experiment.py\n")

    print(f"  Report written to {rp}")
    return str(rp)
