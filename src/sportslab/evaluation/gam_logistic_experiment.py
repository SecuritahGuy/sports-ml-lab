"""GAM/spline-based logistic experiment.

Research question:
    Does a nonlinear (spline) transformation of elo_prob improve the v3.0.0
    incumbent when combined with the frozen QB overlay?

Architecture (fold-safe):
    For each rolling-origin fold:
        1. Compute Elo features chronologically (no future leakage)
        2. Build feature matrix with spline-transformed elo_prob + raw features
        3. Fit logistic regression (GAM-style) on train seasons only
        4. Apply frozen QB overlay on top of GAM base probability
        5. Score on validation season

    The GAM replaces the incumbent's Platt stage. The overlay is applied
    identically to the v3.0.0 champion.

    Selection: average validation log loss across 3 folds.
    2025 holdout: one-shot evaluation after selection.

Comparison baseline: v3.0.0 champion (val LL 0.6305, holdout LL 0.6200).
Promotion requires Δ >= 0.001 on BOTH val and holdout.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import compute_metrics
from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

MIN_PROMOTION_DELTA = 0.001
SEED = 42

# v3.0.0 champion reference
V3_VAL_LL = 0.6305
V3_HOLDOUT_LL = 0.6200

# Incumbent Elo spine params
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2

# Base features (used directly, not spline-transformed)
LINEAR_FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
    "rest_diff", "is_dome", "div_game",
]

# QB overlay params (v3.0.0 champion)
QB_GATE_GAMMA = 1.0
QB_GATE_CAP = 40
ELO_TO_LOGIT = np.log(10.0) / 400.0

# Hyperparameter grid
N_KNOTS_VALUES = [3, 4, 5]
DEGREE_VALUES = [2, 3]
C_VALUES = [0.1, 1.0, 10.0]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def _build_gate_mask(df: pd.DataFrame) -> np.ndarray:
    h_changed = df.get("home_qb_changed", pd.Series(0)).values
    a_changed = df.get("away_qb_changed", pd.Series(0)).values
    h_starts = df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values
    a_starts = df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values
    return (h_changed == 1) | (a_changed == 1) | (h_starts < 17) | (a_starts < 17)


def _get_features(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    if avail:
        return df[avail].values
    return np.empty((len(df), 0))


def run_gam_logistic_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/gam_logistic.md",
    output_csv: Optional[str] = None,
) -> str:
    print("=== GAM/Spline Logistic Experiment ===")

    # ── 1. Load data and build features ──
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw, k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG, team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_qb_adjustments(df)
    df = compute_situational_features(df)

    # Filter eligible
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    # Pre-compute arrays
    elo_prob = df["elo_prob"].values.astype(float).reshape(-1, 1)
    y = df[TARGET_COLUMN].astype(float).values
    linear_feat = _get_features(df, LINEAR_FEATURE_COLS)
    home_qb_adj = df["home_qb_adj"].values.astype(float)
    away_qb_adj = df["away_qb_adj"].values.astype(float)
    gate_mask = _build_gate_mask(df)

    print(f"  Linear features: {LINEAR_FEATURE_COLS}")
    print(f"  Spline feature: elo_prob")

    # ── 2. Rolling-origin validation ──
    print("\n=== Rolling-Origin Validation ===")
    print(f"  Folds: {ROLLING_FOLDS}")

    fold_data: Dict[int, Dict] = {}

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        print(f"\n  Fold {fold_idx + 1}: train {train_seasons} → val {val_season}")

        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values

        fold_data[fold_idx] = {
            "elo_prob": elo_prob,
            "linear_feat": linear_feat,
            "y": y,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "home_qb_adj": home_qb_adj,
            "away_qb_adj": away_qb_adj,
            "gate_mask": gate_mask,
        }

    # ── 3. Baseline: incumbent (fold-safe) ──
    print("\n=== Incumbent Baseline ===")
    inc_fold_lls = []
    for fold_idx in range(len(ROLLING_FOLDS)):
        fd = fold_data[fold_idx]
        train_mask = fd["train_mask"]
        val_mask = fd["val_mask"]
        train_y_int = fd["y"][train_mask].astype(int)
        val_y = fd["y"][val_mask]

        train_elo = fd["elo_prob"][train_mask]
        train_lin = fd["linear_feat"][train_mask]
        x_train = np.column_stack([train_elo, train_lin])

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_train, train_y_int)

        x_all = np.column_stack([fd["elo_prob"], fd["linear_feat"]])
        base_prob = pipe.predict_proba(x_all)[:, 1]
        base_logit = _logit(base_prob)

        capped_h = np.clip(fd["home_qb_adj"], -QB_GATE_CAP, QB_GATE_CAP)
        capped_a = np.clip(fd["away_qb_adj"], -QB_GATE_CAP, QB_GATE_CAP)
        overlay = QB_GATE_GAMMA * (capped_h - capped_a) * ELO_TO_LOGIT
        final_logit = base_logit + overlay * fd["gate_mask"].astype(float)
        final_prob = _sigmoid(final_logit)

        val_prob = final_prob[val_mask]
        valid = ~np.isnan(val_y)
        m = compute_metrics(val_y[valid], val_prob[valid])
        inc_fold_lls.append(m.get("log_loss", 1.0))

    inc_avg_val_ll = float(np.mean(inc_fold_lls))
    print(f"  Incumbent: avg val LL = {inc_avg_val_ll:.4f}")

    # ── 4. GAM grid search ──
    print("\n=== GAM Grid Search ===")
    results: List[Dict] = []

    for n_knots in N_KNOTS_VALUES:
        for degree in DEGREE_VALUES:
            for C in C_VALUES:
                fold_lls = []

                for fold_idx in range(len(ROLLING_FOLDS)):
                    fd = fold_data[fold_idx]
                    train_mask = fd["train_mask"]
                    val_mask = fd["val_mask"]
                    train_y_int = fd["y"][train_mask].astype(int)
                    val_y = fd["y"][val_mask]

                    # Fit SplineTransformer on training elo_prob only
                    elo_train = fd["elo_prob"][train_mask]
                    spline = SplineTransformer(
                        n_knots=n_knots, degree=degree, knots="uniform",
                    )
                    spline.fit(elo_train)

                    # Transform elo_prob for all data
                    elo_spline_all = spline.transform(fd["elo_prob"])

                    # Build full feature matrix
                    x_all_gam = np.column_stack([elo_spline_all, fd["linear_feat"]])
                    x_train_gam = x_all_gam[train_mask]

                    # Fit logistic
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("lr", LogisticRegression(C=C, max_iter=2000, random_state=SEED)),
                    ])
                    pipe.fit(x_train_gam, train_y_int)

                    # Get base probability
                    gam_prob = pipe.predict_proba(x_all_gam)[:, 1]
                    gam_logit = _logit(gam_prob)

                    # Apply frozen QB overlay
                    capped_h = np.clip(fd["home_qb_adj"], -QB_GATE_CAP, QB_GATE_CAP)
                    capped_a = np.clip(fd["away_qb_adj"], -QB_GATE_CAP, QB_GATE_CAP)
                    overlay = QB_GATE_GAMMA * (capped_h - capped_a) * ELO_TO_LOGIT
                    final_logit = gam_logit + overlay * fd["gate_mask"].astype(float)
                    final_prob = _sigmoid(final_logit)

                    val_prob = final_prob[val_mask]
                    valid = ~np.isnan(val_y)
                    m = compute_metrics(val_y[valid], val_prob[valid])
                    fold_lls.append(m.get("log_loss", 1.0))

                avg_ll = float(np.mean(fold_lls))
                name = f"Spline k={n_knots} d={degree} C={C}"
                results.append({
                    "name": name,
                    "n_knots": n_knots,
                    "degree": degree,
                    "C": C,
                    "avg_val_ll": avg_ll,
                    "fold_lls": fold_lls,
                })
                fll_str = f"{fold_lls[0]:.4f}, {fold_lls[1]:.4f}, {fold_lls[2]:.4f}"
                print(f"  {name}: avg val LL = {avg_ll:.4f} ({fll_str})")

    # Sort by val LL
    results.sort(key=lambda r: r["avg_val_ll"])
    best = results[0]

    beats_val = best["avg_val_ll"] < inc_avg_val_ll - MIN_PROMOTION_DELTA
    print(f"\n  Best GAM: {best['name']} (val LL {best['avg_val_ll']:.4f})")
    print(f"  Incumbent val LL: {inc_avg_val_ll:.4f}")
    print(f"  Beats incumbent: {beats_val}")

    # ── 5. 2025 Holdout ──
    print("\n=== 2025 Holdout ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[hold_mask]
    valid_hold = ~np.isnan(hold_y)

    # Fit incumbent on all 2021-2024
    train_mask_hold = df["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_h = elo_prob[train_mask_hold]
    train_lin_h = linear_feat[train_mask_hold]
    train_y_h = y[train_mask_hold].astype(int)

    inc_pipe_h = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    inc_pipe_h.fit(np.column_stack([train_elo_h, train_lin_h]), train_y_h)
    inc_base_h = inc_pipe_h.predict_proba(np.column_stack([elo_prob, linear_feat]))[:, 1]
    inc_logit_h = _logit(inc_base_h)
    capped_hh = np.clip(home_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    capped_aa = np.clip(away_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    ov_h = QB_GATE_GAMMA * (capped_hh - capped_aa) * ELO_TO_LOGIT
    hold_inc_prob = _sigmoid(inc_logit_h + ov_h * gate_mask.astype(float))

    hold_inc_p = hold_inc_prob[hold_mask][valid_hold]
    hold_y_c = hold_y[valid_hold]
    inc_hold_m = compute_classification_metrics(hold_y_c, hold_inc_p)
    inc_hold_ll = inc_hold_m["log_loss"]
    print(f"  Incumbent: holdout LL = {inc_hold_ll:.4f}")

    # Fit best GAM on all 2021-2024
    best_spline = SplineTransformer(
        n_knots=best["n_knots"], degree=best["degree"], knots="uniform",
    )
    best_spline.fit(train_elo_h)
    elo_spline_all = best_spline.transform(elo_prob)

    x_tr_gam = np.column_stack([elo_spline_all[train_mask_hold], train_lin_h])
    x_ho_gam = np.column_stack([elo_spline_all, linear_feat])

    gam_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=best["C"], max_iter=2000, random_state=SEED)),
    ])
    gam_pipe.fit(x_tr_gam, train_y_h)
    gam_base_h = gam_pipe.predict_proba(x_ho_gam)[:, 1]
    gam_logit_h = _logit(gam_base_h)
    hold_gam_prob = _sigmoid(gam_logit_h + ov_h * gate_mask.astype(float))

    sel_hold_p = hold_gam_prob[hold_mask][valid_hold]
    sel_hold_m = compute_classification_metrics(hold_y_c, sel_hold_p)
    sel_hold_ll = sel_hold_m["log_loss"]
    print(f"  Best GAM ({best['name']}): holdout LL = {sel_hold_ll:.4f}")

    beats_hold = sel_hold_ll < inc_hold_ll - MIN_PROMOTION_DELTA

    # Evaluate ALL variants on holdout (diagnostic)
    hold_results: Dict[str, Dict] = {}
    for r in results:
        spline = SplineTransformer(
            n_knots=r["n_knots"], degree=r["degree"], knots="uniform",
        )
        spline.fit(train_elo_h)
        es_all = spline.transform(elo_prob)
        x_tr = np.column_stack([es_all[train_mask_hold], train_lin_h])
        x_all = np.column_stack([es_all, linear_feat])

        p = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=r["C"], max_iter=2000, random_state=SEED)),
        ])
        p.fit(x_tr, train_y_h)
        base_h = p.predict_proba(x_all)[:, 1]
        logit_h = _logit(base_h)
        prob = _sigmoid(logit_h + ov_h * gate_mask.astype(float))

        prob_h = prob[hold_mask][valid_hold]
        m = compute_classification_metrics(hold_y_c, prob_h)
        hold_results[r["name"]] = m

    sorted_hold = sorted(hold_results.items(), key=lambda x: x[1]["log_loss"])
    best_hold_name = sorted_hold[0][0]
    best_hold_ll = hold_results[best_hold_name]["log_loss"]

    # ── 6. Promotion check ──
    promotes = beats_val and beats_hold

    # ── 7. Slices ──
    qb_h = df.loc[hold_mask, "home_qb_changed"].fillna(0).values.astype(bool)
    qb_a = df.loc[hold_mask, "away_qb_changed"].fillna(0).values.astype(bool)
    qc_mask = (qb_h | qb_a)[valid_hold]
    no_qc_mask = ~qc_mask

    def _slice_ll(y_s, p_s):
        if len(y_s) < 2:
            return None
        return compute_metrics(y_s, p_s).get("log_loss")

    inc_qc = _slice_ll(hold_y_c[qc_mask], hold_inc_p[qc_mask])
    inc_nqc = _slice_ll(hold_y_c[no_qc_mask], hold_inc_p[no_qc_mask])
    sel_qc = _slice_ll(hold_y_c[qc_mask], sel_hold_p[qc_mask])
    sel_nqc = _slice_ll(hold_y_c[no_qc_mask], sel_hold_p[no_qc_mask])

    # ── 8. Write report ──
    print(f"\n=== Writing report → {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# GAM/Spline Logistic Experiment\n\n")
        _w("## Research Question\n\n")
        _w("Does a nonlinear (spline) transformation of `elo_prob` improve ")
        _w("the v3.0.0 Frozen QB Overlay champion?\n\n")

        _w("## Architecture\n\n")
        _w("```\n")
        _w("For each rolling-origin fold:\n")
        _w("  1. Compute Elo features chronologically\n")
        _w("  2. Build X = [SplineTransformer(elo_prob), qb_changed, mov_3,\n")
        _w("                rest_diff, is_dome, div_game]\n")
        _w("  3. Fit LogisticRegression on train seasons only\n")
        _w("  4. Apply frozen QB overlay (v3.0.0 params) on top\n")
        _w("  5. Score on validation season\n")
        _w("```\n\n")

        _w("## Features\n\n")
        _w("| Feature | Transform |\n")
        _w("|---------|----------|\n")
        _w("| elo_prob | SplineTransformer (knots=3-5, degree=2-3, uniform) |\n")
        for col in LINEAR_FEATURE_COLS:
            _w(f"| {col} | Raw (linear) |\n")

        _w("\n## Hyperparameter Grid\n\n")
        _w(f"Knots: {N_KNOTS_VALUES}\n")
        _w(f"Degree: {DEGREE_VALUES}\n")
        _w(f"C: {C_VALUES}\n")
        _w(f"Total variants: {len(N_KNOTS_VALUES) * len(DEGREE_VALUES) * len(C_VALUES)}\n\n")

        _w("## Validation Results\n\n")
        _w("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        _w("|-------|-----------|-------|-------|-------|\n")
        _w(f"| v3.0.0 Incumbent | {inc_avg_val_ll:.4f}")
        _w(f" | {inc_fold_lls[0]:.4f} | {inc_fold_lls[1]:.4f} | {inc_fold_lls[2]:.4f} |\n")
        for r in results:
            _w(f"| {r['name']} | {r['avg_val_ll']:.4f}")
            _w(f" | {r['fold_lls'][0]:.4f} | {r['fold_lls'][1]:.4f} | {r['fold_lls'][2]:.4f} |\n")

        _w("\n## 2025 Holdout Results\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy | Selection |\n")
        _w("|-------|----------|-------|-----|----------|-----------|\n")
        _w(f"| v3.0.0 Incumbent | {inc_hold_ll:.4f}")
        _w(f" | {inc_hold_m['brier_score']:.4f}")
        _w(f" | {inc_hold_m['roc_auc']:.4f}")
        _w(f" | {inc_hold_m['accuracy']:.4f} | baseline |\n")
        _w(f"| Best GAM ({best['name']}) | {sel_hold_ll:.4f}")
        _w(f" | {sel_hold_m['brier_score']:.4f}")
        _w(f" | {sel_hold_m['roc_auc']:.4f}")
        _w(f" | {sel_hold_m['accuracy']:.4f} | validation |\n")

        if best_hold_name != best["name"]:
            bh_m = hold_results[best_hold_name]
            _w(f"| {best_hold_name} | {bh_m['log_loss']:.4f}")
            _w(f" | {bh_m['brier_score']:.4f}")
            _w(f" | {bh_m['roc_auc']:.4f}")
            _w(f" | {bh_m['accuracy']:.4f} | diagnostic |\n")

        _w("\n## QB-Change Slices\n\n")
        _w("| Variant | QB-Change LL | No-QB-Change LL | QC Δ | NoQC Δ |\n")
        _w("|--------|-------------|-----------------|------|--------|\n")
        qc_d = (sel_qc - inc_qc) if (sel_qc is not None and inc_qc is not None) else None
        nqc_d = (sel_nqc - inc_nqc) if (sel_nqc is not None and inc_nqc is not None) else None
        _w(f"| v3.0.0 | {inc_qc or 0:.4f} | {inc_nqc or 0:.4f} | — | — |\n")
        qc_s = f"{qc_d:+.4f}" if qc_d is not None else "N/A"
        nqc_s = f"{nqc_d:+.4f}" if nqc_d is not None else "N/A"
        _w(f"| Best GAM | {sel_qc or 0:.4f} | {sel_nqc or 0:.4f} | {qc_s} | {nqc_s} |\n")

        _w("\n## Decision\n\n")
        if promotes:
            _w(f"**✅ PROMOTED: {best['name']}**\n\n")
        else:
            _w("**❌ REJECTED**\n\n")
            v_delta = inc_avg_val_ll - best["avg_val_ll"]
            h_delta = inc_hold_ll - sel_hold_ll
            _w(f"| Criterion | Met? | Details |\n")
            _w(f"|-----------|------|--------|\n")
            _w(f"| Beats val by >= {MIN_PROMOTION_DELTA} | ")
            _w(f"{'✅' if beats_val else '❌'} | Δ = {v_delta:.4f} |\n")
            _w(f"| Beats holdout by >= {MIN_PROMOTION_DELTA} | ")
            _w(f"{'✅' if beats_hold else '❌'} | Δ = {h_delta:.4f} |\n")

        _w("\n---\n")
        _w("*Report generated by `sportslab gam-logistic`. ")
        _w(f"Seasons: 2021–{HOLDOUT_SEASON}, ")
        _w(f"Folds: {len(ROLLING_FOLDS)}.*\n")

    # ── 9. Export CSV if requested ──
    if output_csv is not None:
        out_df = df[hold_mask].copy()
        out_df["incumbent_home_win_prob"] = hold_inc_p
        out_df["gam_model_prob"] = sel_hold_p
        out_df["best_variant"] = best["name"]
        out_df.to_csv(output_csv, index=False)
        print(f"  CSV: {output_csv}")

    print(f"\nReport: {rp}")
    return str(report_path)
