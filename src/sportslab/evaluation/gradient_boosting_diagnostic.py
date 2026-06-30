"""Gradient boosting diagnostic experiment.

Research question:
    Can a strictly regularized HistGradientBoostingClassifier, using only
    pregame-safe features, match or exceed the v3.0.0 Frozen QB Overlay?

Architecture (fold-safe):
    For each rolling-origin fold:
        1. Compute Elo features chronologically (no future leakage)
        2. Build feature matrix from pregame-safe features
        3. Fit HistGradientBoostingClassifier on train seasons only
        4. Score on validation season

    Unlike the incumbent, the gradient boosting model does NOT use the
    frozen QB overlay. It learns all relationships from features directly.

    Selection: average validation log loss across 3 folds.
    2025 holdout: one-shot evaluation after selection.

    Diagnosis: slices, calibration, worst predictions.

    This experiment is DIAGNOSTIC by default -- tree models have consistently
    overfit this dataset (previous experiments: expressive_models rejected,
    AutoGluon rejected).
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import (
    calibration_buckets,
    compute_metrics,
    confidence_buckets,
    worst_predictions,
)
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
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

SEED = 42

# v3.0.0 champion reference
V3_VAL_LL = 0.6305
V3_HOLDOUT_LL = 0.6200

# Elo params (for feature building only)
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2

# Feature columns (all pregame-safe, low cardinality)
FEATURE_COLS = [
    "elo_prob",
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
    "rest_diff", "is_dome", "div_game",
]

# HGB hyperparameter grid (conservative)
LEARNING_RATE_VALUES = [0.01, 0.05, 0.1]
MAX_DEPTH_VALUES = [2, 3]
MIN_SAMPLES_LEAF_VALUES = [20, 50, 100]
MAX_BINS_VALUES = [64, 128]


def _get_features(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    if avail:
        return df[avail].values
    return np.empty((len(df), 0))


def run_gradient_boosting_diagnostic(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/gradient_boosting_diagnostic.md",
    output_csv: Optional[str] = None,
) -> str:
    print("=== Gradient Boosting Diagnostic ===")

    # ── 1. Load and build features ──
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
    df = compute_situational_features(df)

    # Filter eligible
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    # Build feature matrix
    x_mat = _get_features(df, FEATURE_COLS)
    y = df[TARGET_COLUMN].astype(float).values

    print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"  Feature matrix shape: {x_mat.shape}")

    # ── 2. Baseline: fold-safe incumbent ──
    print("\n=== Incumbent Baseline ===")
    inc_fold_lls = []
    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values

        train_elo = df["elo_prob"].values[train_mask].reshape(-1, 1)
        train_lin = _get_features(df[train_mask], FEATURE_COLS[1:])
        train_y_int = y[train_mask].astype(int)
        x_tr = np.column_stack([train_elo, train_lin])

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_tr, train_y_int)

        val_elo = df["elo_prob"].values[val_mask].reshape(-1, 1)
        val_lin = _get_features(df[val_mask], FEATURE_COLS[1:])
        x_val = np.column_stack([val_elo, val_lin])
        val_prob = pipe.predict_proba(x_val)[:, 1]
        val_y = y[val_mask]

        valid = ~np.isnan(val_y)
        m = compute_metrics(val_y[valid], val_prob[valid])
        inc_fold_lls.append(m.get("log_loss", 1.0))

    inc_avg_val_ll = float(np.mean(inc_fold_lls))
    print(f"  Incumbent (logistic on same features): avg val LL = {inc_avg_val_ll:.4f}")

    # ── 3. HGB grid search ──
    print("\n=== HGB Grid Search ===")
    results: List[Dict] = []
    total = (
        len(LEARNING_RATE_VALUES)
        * len(MAX_DEPTH_VALUES)
        * len(MIN_SAMPLES_LEAF_VALUES)
        * len(MAX_BINS_VALUES)
    )
    done = 0

    for lr in LEARNING_RATE_VALUES:
        for depth in MAX_DEPTH_VALUES:
            for min_leaf in MIN_SAMPLES_LEAF_VALUES:
                for max_bins in MAX_BINS_VALUES:
                    fold_lls = []

                    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
                        train_mask = df["season"].isin(train_seasons).values
                        val_mask = (df["season"] == val_season).values

                        x_train = x_mat[train_mask]
                        y_train = y[train_mask].astype(int)
                        x_val = x_mat[val_mask]
                        y_val = y[val_mask]

                        hgb = HistGradientBoostingClassifier(
                            learning_rate=lr,
                            max_depth=depth,
                            min_samples_leaf=min_leaf,
                            max_bins=max_bins,
                            max_iter=500,
                            early_stopping=True,
                            validation_fraction=0.2,
                            n_iter_no_change=15,
                            tol=1e-4,
                            random_state=SEED,
                        )
                        hgb.fit(x_train, y_train)
                        val_prob = hgb.predict_proba(x_val)[:, 1]

                        valid = ~np.isnan(y_val)
                        m = compute_metrics(y_val[valid], val_prob[valid])
                        fold_lls.append(m.get("log_loss", 1.0))

                    avg_ll = float(np.mean(fold_lls))
                    name = f"HGB lr={lr} d={depth} ml={min_leaf} b={max_bins}"
                    results.append({
                        "name": name,
                        "lr": lr,
                        "depth": depth,
                        "min_leaf": min_leaf,
                        "max_bins": max_bins,
                        "avg_val_ll": avg_ll,
                        "fold_lls": fold_lls,
                    })
                    done += 1
                    if done <= 5 or done == total or done % 12 == 0:
                        print(f"  [{done}/{total}] {name}: avg val LL = {avg_ll:.4f}")

    results.sort(key=lambda r: r["avg_val_ll"])
    best = results[0]

    print(f"\n  Best HGB: {best['name']} (val LL {best['avg_val_ll']:.4f})")
    print(f"  Incumbent val LL: {inc_avg_val_ll:.4f}")

    # ── 4. 2025 Holdout ──
    print("\n=== 2025 Holdout ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[hold_mask]
    valid_hold = ~np.isnan(hold_y)

    # Fit incumbent on all 2021-2024
    train_mask_h = df["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_h = df["elo_prob"].values[train_mask_h].reshape(-1, 1)
    train_lin_h = _get_features(df[train_mask_h], FEATURE_COLS[1:])
    train_y_h = y[train_mask_h].astype(int)

    inc_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    inc_pipe.fit(np.column_stack([train_elo_h, train_lin_h]), train_y_h)

    hold_elo = df["elo_prob"].values[hold_mask].reshape(-1, 1)
    hold_lin = _get_features(df[hold_mask], FEATURE_COLS[1:])
    hold_inc_prob = inc_pipe.predict_proba(np.column_stack([hold_elo, hold_lin]))[:, 1]

    hold_y_c = hold_y[valid_hold]
    hold_inc_p = hold_inc_prob[valid_hold]
    inc_hold_m = compute_classification_metrics(hold_y_c, hold_inc_p)
    inc_hold_ll = inc_hold_m["log_loss"]
    print(f"  Incumbent: holdout LL = {inc_hold_ll:.4f}")

    # Fit best HGB on all 2021-2024
    x_train_h = x_mat[train_mask_h]
    y_train_h = y[train_mask_h].astype(int)
    x_hold = x_mat[hold_mask]

    hgb_best = HistGradientBoostingClassifier(
        learning_rate=best["lr"],
        max_depth=best["depth"],
        min_samples_leaf=best["min_leaf"],
        max_bins=best["max_bins"],
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        tol=1e-4,
        random_state=SEED,
    )
    hgb_best.fit(x_train_h, y_train_h)
    hold_hgb_prob = hgb_best.predict_proba(x_hold)[:, 1][valid_hold]

    sel_hold_m = compute_classification_metrics(hold_y_c, hold_hgb_prob)
    sel_hold_ll = sel_hold_m["log_loss"]
    print(f"  Best HGB ({best['name']}): holdout LL = {sel_hold_ll:.4f}")

    # Evaluate ALL variants
    hold_results: Dict[str, Dict] = {}
    for r in results:
        clf = HistGradientBoostingClassifier(
            learning_rate=r["lr"], max_depth=r["depth"],
            min_samples_leaf=r["min_leaf"], max_bins=r["max_bins"],
            max_iter=500, early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=15, tol=1e-4, random_state=SEED,
        )
        clf.fit(x_train_h, y_train_h)
        prob = clf.predict_proba(x_hold)[:, 1][valid_hold]
        hold_results[r["name"]] = compute_classification_metrics(hold_y_c, prob)

    sorted_hold = sorted(hold_results.items(), key=lambda x: x[1]["log_loss"])
    best_hold_name = sorted_hold[0][0]

    # Num iterations used
    n_iter = hgb_best.n_iter_

    # ── 5. Calibration / diagnostics ──
    cal_buckets = calibration_buckets(hold_y_c, hold_hgb_prob)
    conf_buckets = confidence_buckets(hold_y_c, hold_hgb_prob)
    worst = worst_predictions(hold_y_c, hold_hgb_prob,
                              df.loc[hold_mask, "game_id"].values[valid_hold],
                              df.loc[hold_mask, "home_team"].values[valid_hold])

    # ── 6. Write report ──
    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# Gradient Boosting Diagnostic\n\n")
        _w("## Research Question\n\n")
        _w("Can a strictly regularized HistGradientBoostingClassifier match ")
        _w("the v3.0.0 Frozen QB Overlay on pregame-safe features?\n\n")
        _w("**Status: DIAGNOSTIC.** Previous tree experiments (expressive_models, ")
        _w("AutoGluon) consistently overfit this dataset. Strong regularization ")
        _w("is applied but promotion is not expected.\n\n")

        _w("## Architecture\n\n")
        _w("```\n")
        _w("For each rolling-origin fold:\n")
        _w("  1. Compute Elo features chronologically\n")
        _w("  2. Build X = [elo_prob, qb_changed, mov_3, rest_diff, is_dome, div_game]\n")
        _w("  3. Fit HistGradientBoosting with early stopping on train only\n")
        _w("  4. Score on validation season\n")
        _w("```\n\n")

        _w("## Features\n\n")
        for col in FEATURE_COLS:
            _w(f"- {col}\n")

        _w("\n## Hyperparameter Grid\n\n")
        _w(f"learning_rate: {LEARNING_RATE_VALUES}\n")
        _w(f"max_depth: {MAX_DEPTH_VALUES}\n")
        _w(f"min_samples_leaf: {MIN_SAMPLES_LEAF_VALUES}\n")
        _w(f"max_bins: {MAX_BINS_VALUES}\n")
        _w(f"Total: {total} variants\n\n")

        _w("## Validation Results (Top 10)\n\n")
        _w("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        _w("|-------|-----------|-------|-------|-------|\n")
        _w(f"| Logistic (incumbent) | {inc_avg_val_ll:.4f}")
        _w(f" | {inc_fold_lls[0]:.4f} | {inc_fold_lls[1]:.4f} | ")
        _w(f"{inc_fold_lls[2]:.4f} |\n")
        for r in results[:10]:
            _w(f"| {r['name']} | {r['avg_val_ll']:.4f}")
            _w(f" | {r['fold_lls'][0]:.4f} | {r['fold_lls'][1]:.4f} | ")
            _w(f"{r['fold_lls'][2]:.4f} |\n")
        if len(results) > 10:
            _w(f"\n... ({len(results) - 10} more variants)\n\n")

        _w("\n## 2025 Holdout Results\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy | Selection |\n")
        _w("|-------|----------|-------|-----|----------|-----------|\n")
        _w(f"| Logistic (incumbent) | {inc_hold_ll:.4f} | {inc_hold_m['brier_score']:.4f}")
        _w(f" | {inc_hold_m['roc_auc']:.4f} | {inc_hold_m['accuracy']:.4f} | baseline |\n")
        _w(f"| Best HGB (val-sel) | {sel_hold_ll:.4f} | {sel_hold_m['brier_score']:.4f}")
        _w(f" | {sel_hold_m['roc_auc']:.4f} | {sel_hold_m['accuracy']:.4f} | validation |\n")

        if best_hold_name != best["name"]:
            bh = hold_results[best_hold_name]
            _w(f"| {best_hold_name} | {bh['log_loss']:.4f} | {bh['brier_score']:.4f}")
            _w(f" | {bh['roc_auc']:.4f} | {bh['accuracy']:.4f} | diagnostic |\n")

        _w(f"\nBest HGB iterations: {n_iter}\n")

        _w("\n## Calibration (Holdout, Best HGB)\n\n")
        _w("| Bucket | N | Mean Pred | Mean Actual | Cal Error |\n")
        _w("|--------|---|-----------|-------------|-----------|\n")
        for b in cal_buckets:
            _w(f"| {b['bucket']} | {b['n']} | {b['mean_pred']:.4f}")
            _w(f" | {b['mean_actual']:.4f} | {b['cal_error']:.4f} |\n")

        _w("\n## Confidence (Holdout, Best HGB)\n\n")
        _w("| Bucket | N | Log Loss |\n")
        _w("|--------|---|----------|\n")
        for b in conf_buckets:
            _w(f"| {b['bucket']} | {b['n']} | {b['log_loss']:.4f} |\n")

        _w("\n## Worst Predictions\n\n")
        _w("| Game | Actual | Pred | LL Contrib |\n")
        _w("|------|--------|------|------------|\n")
        for w in worst:
            _w(f"| {w['game_id']} | {w['actual']} | {w['pred']:.4f} | ")
            _w(f"{w['log_loss_contrib']:.4f} |\n")

        _w("\n## Decision\n\n")
        beats_val = best["avg_val_ll"] < inc_avg_val_ll - 0.001
        beats_hold = sel_hold_ll < inc_hold_ll - 0.001
        promotes = beats_val and beats_hold

        if promotes:
            _w(f"**PROMOTED: {best['name']}**\n\n")
        else:
            _w("**REJECTED -- DIAGNOSTIC ONLY**\n\n")
            vd = inc_avg_val_ll - best["avg_val_ll"]
            hd = inc_hold_ll - sel_hold_ll
            _w(f"Val Delta: {vd:+.4f} (need >= 0.001)\n")
            _w(f"Holdout Delta: {hd:+.4f} (need >= 0.001)\n")

        _w("\n---\n")
        _w("*Report generated by `sportslab gradient-boosting`. ")
        _w(f"Seasons: 2021--{HOLDOUT_SEASON}, ")
        _w(f"Folds: {len(ROLLING_FOLDS)}, ")
        _w(f"Variants: {total}.*\n")

    # ── 7. Export CSV if requested ──
    if output_csv is not None:
        out_df = df[hold_mask].copy()
        out_df["incumbent_home_win_prob"] = hold_inc_p
        out_df["hgb_prob"] = hold_hgb_prob
        out_df.to_csv(output_csv, index=False)
        print(f"  CSV: {output_csv}")

    print(f"\nReport: {rp}")
    return str(report_path)
