"""Neural network with expanded feature set experiment.

The MLP calibrator already beat the logistic incumbent using only the 5
incumbent features. This experiment tests whether the MORE FLEXIBLE calibrator
can also exploit additional pregame-safe features that were individually
REJECTED for the logistic model (collinear/noisy at logistic scale but
potentially useful to a nonlinear model):

  - rest_diff (home - away rest days)
  - div_game (divisional matchup)
  - is_dome (dome/retractable stadium)
  - week (calendar week, early-season signal)
  - roof_enc / surface_enc (stadium type)
  - home/away_rolling_mov_5 (longer form window)

Follows the canonical fold-safe rolling-origin pattern with the frozen QB
overlay applied on top. Selection by avg rolling-origin validation LL; final
2025 holdout scored once.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.neural_network_experiment import (
    BEST_DECAY,
    BEST_HFA,
    BEST_K,
    BEST_QB_BONUS,
    BEST_REG,
    INCUMBENT_FEATS,
    MIN_PROMOTION_DELTA,
    SEED,
    _apply_overlay,
    _build_gate,
    _mlp_proba,
    _train_mlp,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, TARGET_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

# Additional pregame-safe candidates (all available before kickoff).
EXTRA_CANDIDATES = [
    "rest_diff",
    "div_game",
    "is_dome",
    "week",
    "roof_enc",
    "surface_enc",
    "home_rolling_mov_5",
    "away_rolling_mov_5",
]

# Feature-set variants to test on the MLP calibrator.
FEATURE_VARIANTS = [
    ("incumbent_only", INCUMBENT_FEATS),
    ("+rest_div_dome", INCUMBENT_FEATS + ["rest_diff", "div_game", "is_dome"]),
    ("+week_roof_surface", INCUMBENT_FEATS + ["week", "roof_enc", "surface_enc"]),
    ("+mov5", INCUMBENT_FEATS + ["home_rolling_mov_5", "away_rolling_mov_5"]),
    ("+all_extra", INCUMBENT_FEATS + EXTRA_CANDIDATES),
]

MLP_HIDDEN = (16, 16, 16)
MLP_DROPOUT = 0.1
MLP_WD = 1e-4


def run_neural_network_features_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/neural_network_features.md",
) -> str:
    print("=== Neural Network + Expanded Features ===")
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS
    )
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_qb_adjustments(df)
    df = compute_situational_features(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df["is_neutral"].fillna(False).values
    df = df[mask].copy().reset_index(drop=True)
    y = df[TARGET_COLUMN].astype(float).values
    print(f"  Eligible games: {len(df)}")

    gate = _build_gate(df)
    ha = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
    aa = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
    elo_prob = df["elo_prob"].values.astype(float)

    # Reference: logistic incumbent (matches NN_VARIANTS incumbent)
    from sklearn.linear_model import LogisticRegression

    def _feature_matrix(feat_cols, idx):
        avail = [c for c in feat_cols if c in df.columns]
        return np.column_stack([elo_prob[idx]] + [df[c].values[idx] for c in avail])

    def _fit_predict(feat_cols, train_idx, all_idx):
        xtr = _feature_matrix(feat_cols, train_idx)
        xall = _feature_matrix(feat_cols, all_idx)
        model, scaler = _train_mlp(xtr, y[train_idx], MLP_HIDDEN, MLP_DROPOUT, MLP_WD)
        raw = _mlp_proba(model, scaler, xall)
        return _apply_overlay(raw, gate, ha, aa)

    # Logistic baseline (incumbent features) for reference
    def _logistic_incumbent():
        xtr = _feature_matrix(INCUMBENT_FEATS, slice(None))
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(
            StandardScaler().fit_transform(xtr),
            y.astype(int),
        )
        return lr

    val_results = {}
    hold_results = {}

    for name, feat_cols in FEATURE_VARIANTS:
        fold_lls = []
        for train_s, val_s in ROLLING_FOLDS:
            tr = df["season"].isin(train_s).values
            va = (df["season"] == val_s).values
            if tr.sum() == 0 or va.sum() == 0:
                fold_lls.append(1.0)
                continue
            pp = _fit_predict(feat_cols, tr, slice(None))
            vy = y[va]
            valid = ~np.isnan(vy)
            fold_lls.append(float(sk_log_loss(vy[valid].astype(int), pp[va][valid])))
        val_results[name] = round(float(np.mean(fold_lls)), 4)

        tr = (df["season"] < HOLDOUT_SEASON).values
        va = (df["season"] == HOLDOUT_SEASON).values
        pp = _fit_predict(feat_cols, tr, slice(None))
        vy = y[va]
        valid = ~np.isnan(vy)
        hold_results[name] = float(sk_log_loss(vy[valid].astype(int), pp[va][valid]))

    iv = val_results["incumbent_only"]
    ih = hold_results["incumbent_only"]
    print("\n--- Validation (avg rolling-origin) ---")
    for name, _ in FEATURE_VARIANTS:
        print(f"  {name:20s}  val={val_results[name]:.4f}  Δ={val_results[name]-iv:+.4f}")
    print("\n--- 2025 Holdout ---")
    for name, _ in FEATURE_VARIANTS:
        print(f"  {name:20s}  hold={hold_results[name]:.4f}  Δ={hold_results[name]-ih:+.4f}")

    best_v = min(v for k, v in val_results.items() if k != "incumbent_only")
    best_v_name = min(
        (k for k in val_results if k != "incumbent_only"), key=lambda k: val_results[k]
    )
    best_h = min(v for k, v in hold_results.items() if k != "incumbent_only")
    best_h_name = min(
        (k for k in hold_results if k != "incumbent_only"), key=lambda k: hold_results[k]
    )
    improved = (best_v < iv - MIN_PROMOTION_DELTA) and (best_h < ih - MIN_PROMOTION_DELTA)

    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        w = f.write
        w("# Neural Network + Expanded Features\n\n")
        w("Tests whether the MLP calibrator can exploit pregame-safe features "
          "that were rejected for the logistic model.\n\n")
        w("| Feature set | Val LL | Δ Val | Holdout LL | Δ Holdout |\n")
        w("|-------------|--------|-------|-----------|-----------|\n")
        for name, _ in FEATURE_VARIANTS:
            w(f"| {name} | {val_results[name]:.4f} | {val_results[name]-iv:+.4f} | "
              f"{hold_results[name]:.4f} | {hold_results[name]-ih:+.4f} |\n")
        w("\n## Decision\n\n")
        if improved:
            w(f"**✅ PROMOTED** — {best_v_name} / {best_h_name} beat the MLP "
              "incumbent on both validation and holdout.\n\n")
        else:
            w("**❌ REJECTED** — no expanded feature set beat the MLP incumbent "
              "on both validation and holdout by >= 0.001.\n\n")
            w(f"Best validation: {best_v_name} ({best_v:.4f}, Δ={best_v-iv:+.4f})\n")
            w(f"Best holdout: {best_h_name} ({best_h:.4f}, Δ={best_h-ih:+.4f})\n")
    print(f"  Report: {rp}")
    return str(report_path)


if __name__ == "__main__":
    run_neural_network_features_experiment()
