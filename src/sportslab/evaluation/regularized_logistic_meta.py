"""Regularized logistic meta-model experiment — promotion-eligible.

Research question:
    Can a more regularized logistic layer improve on v3.0.0 Frozen QB Overlay by
    tuning C/penalty and adding a tiny set of low-risk pregame features?

Architecture (fold-safe):
    For each rolling-origin fold:
        1. Fit incumbent Platt + QB overlay on train seasons only
        2. Compute meta-input = logit(v3.0.0 incumbent prob)
        3. Build meta-features = [meta-input] + optional small feature group
        4. Fit regularized logistic meta-model on train seasons only
        5. Score meta-model on validation season

    Selection: average validation log loss across 3 folds.
    2025 holdout: one-shot evaluation after selection.

    Feature groups are tested separately (not combined) to isolate each signal.
    No market/score/result/target columns used as features.

Comparison baseline: v3.0.0 champion (val LL 0.6305, holdout LL 0.6200).
Promotion requires delta >= 0.001 on BOTH val and holdout.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import compute_metrics
from sportslab.evaluation.fold_safe import (
    check_promotion,
    load_feature_table,
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

# Incumbent base features (Platt stage inputs)
INCUMBENT_FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]

# QB overlay params (v3.0.0)
QB_GATE_GAMMA = 1.0
QB_GATE_CAP = 40
ELO_TO_LOGIT = np.log(10.0) / 400.0

# Hyperparameter grids
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
L1_RATIO_VALUES = [0.25, 0.5, 0.75]

# Feature sets — each tested separately against v3 baseline
FEATURE_SETS: Dict[str, List[str]] = {
    "v3_logit_only": [],
    "+ week": ["week_sin", "week_cos"],
    "+ context": ["div_game", "is_dome", "rest_diff"],
    "+ early_season": ["early_season"],
    "+ qb_starts": ["home_qb_team_starts_pre", "away_qb_team_starts_pre"],
}

ALL_FEATURE_COLS = [
    c for cols in FEATURE_SETS.values() for c in cols
]

META_FEATURE_FRIENDLY: Dict[str, str] = {
    "v3_logit_only": "Only v3 logit (no extra features)",
    "+ week": "v3 logit + week_sin, week_cos",
    "+ context": "v3 logit + div_game, is_dome, rest_diff",
    "+ early_season": "v3 logit + early_season flag",
    "+ qb_starts": "v3 logit + home/away QB team starts pre",
}

PROHIBITED_COLS = {"home_score", "away_score", "home_win", "result", "is_tie",
                    "market_home_prob_novig", "spread_line", "home_moneyline",
                    "away_moneyline", "over_odds", "under_odds", "total_line"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def _build_gate_mask(df: pd.DataFrame) -> np.ndarray:
    h_changed = df.get("home_qb_changed", pd.Series(0)).values.astype(float)
    a_changed = df.get("away_qb_changed", pd.Series(0)).values.astype(float)
    h_starts = df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    a_starts = df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    return (h_changed == 1) | (a_changed == 1) | (h_starts < 17) | (a_starts < 17)


def _incumbent_prob_all(
    df: pd.DataFrame,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Fit fold-safe Platt + QB overlay on train_mask, predict for all rows."""
    elo = df["elo_prob"].values.astype(float)
    feats = df[INCUMBENT_FEATURE_COLS].values
    y = df[TARGET_COLUMN].values.astype(float)

    x_tr = (
        np.column_stack([elo[train_mask], feats[train_mask]])
        if len(INCUMBENT_FEATURE_COLS) > 0
        else elo[train_mask].reshape(-1, 1)
    )
    y_tr = y[train_mask].astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(x_tr, y_tr)

    x_all = (
        np.column_stack([elo, feats])
        if len(INCUMBENT_FEATURE_COLS) > 0
        else elo.reshape(-1, 1)
    )
    base_prob = pipe.predict_proba(x_all)[:, 1]
    base_logit = _logit(base_prob)

    home_qb_adj = df["home_qb_adj"].values.astype(float)
    away_qb_adj = df["away_qb_adj"].values.astype(float)
    gate_mask = _build_gate_mask(df)

    capped_h = np.clip(home_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    capped_a = np.clip(away_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    net_adj = capped_h - capped_a
    overlay = QB_GATE_GAMMA * net_adj * ELO_TO_LOGIT
    final_logit = base_logit + overlay * gate_mask.astype(float)
    return _sigmoid(final_logit)


def _week_sin_cos(week: np.ndarray, max_week: int = 18) -> np.ndarray:
    theta = 2.0 * np.pi * week / max_week
    return np.column_stack([np.sin(theta), np.cos(theta)])


def _build_meta_features(
    df: pd.DataFrame,
    v3_prob: np.ndarray,
    extra_cols: List[str],
) -> np.ndarray:
    """Stack v3 logit with extra feature columns."""
    v3_logit = _logit(v3_prob).reshape(-1, 1)
    if not extra_cols:
        return v3_logit
    avail = [c for c in extra_cols if c in df.columns]
    if not avail:
        return v3_logit
    extra = df[avail].values.astype(float)
    return np.column_stack([v3_logit, extra])


def _early_season_flag(df: pd.DataFrame) -> np.ndarray:
    weeks = df["week"].values.astype(float)
    return (weeks <= 4).astype(float)


def run_regularized_logistic_meta(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/regularized_logistic_meta.md",
    output_csv: Optional[str] = None,
) -> str:
    """Run the regularized logistic meta-model experiment."""
    print("=== Regularized Logistic Meta-Model Experiment ===")

    # ── 1. Load data and build features ──
    df_raw = load_feature_table(ft_path)

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

    # Add week sin/cos and early_season (used as meta-features)
    w_sin, w_cos = _week_sin_cos(df["week"].values).T
    df["week_sin"] = w_sin
    df["week_cos"] = w_cos
    df["early_season"] = _early_season_flag(df)

    # Prohibited columns check
    for col in PROHIBITED_COLS:
        assert col not in ALL_FEATURE_COLS, (
            f"Prohibited column {col} should not be in feature sets"
        )

    # Filter eligible (non-tie, non-neutral)
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    y = df[TARGET_COLUMN].values.astype(float)

    # ── 2. Rolling-origin validation ──
    print("\n=== Rolling-Origin Validation ===")
    print(f"  Folds: {ROLLING_FOLDS}")

    # Per-fold precompute: incumbent prob for all data
    fold_v3_prob: Dict[int, np.ndarray] = {}
    fold_inc_fold_lls: Dict[int, List[float]] = {}

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values

        v3_prob = _incumbent_prob_all(df, train_mask)
        fold_v3_prob[fold_idx] = v3_prob

        # Record incumbent val performance for this fold
        val_y = y[val_mask]
        inc_val_prob = v3_prob[val_mask]
        valid = ~np.isnan(val_y)
        m = compute_metrics(val_y[valid], inc_val_prob[valid])
        fold_inc_fold_lls[fold_idx] = [m.get("log_loss", 1.0)]

        inc_val_ll = m.get("log_loss", 1.0)
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        print(f"  Fold {fold_idx + 1}: train {train_seasons} → val {val_season}"
              f" ({n_train} train, {n_val} val)  incumbent val LL: {inc_val_ll:.4f}")

    # ── 3. Grid search over feature sets and hyperparams ──
    print("\n=== Grid Search ===")
    all_results: List[Dict] = []
    model_registry: Dict[str, Pipeline] = {}

    for feat_name, extra_cols in FEATURE_SETS.items():
        print(f"\n  --- Feature set: {feat_name} ---")

        for c_val in C_VALUES:
            # L2
            for penalty in ["l2"]:
                name = f"L2 C={c_val} {feat_name}"
                solver = "lbfgs"
                fold_lls = []
                for fold_idx in range(len(ROLLING_FOLDS)):
                    train_seasons, val_season = ROLLING_FOLDS[fold_idx]
                    train_mask = df["season"].isin(train_seasons).values
                    val_mask = (df["season"] == val_season).values

                    v3_prob = fold_v3_prob[fold_idx]
                    meta_x = _build_meta_features(df, v3_prob, extra_cols)
                    x_tr = meta_x[train_mask]
                    x_val = meta_x[val_mask]
                    y_tr = y[train_mask].astype(int)
                    y_val = y[val_mask]

                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("lr", LogisticRegression(
                            C=c_val, penalty=penalty, solver=solver,
                            max_iter=2000, random_state=SEED,
                        )),
                    ])
                    pipe.fit(x_tr, y_tr)
                    val_prob = pipe.predict_proba(x_val)[:, 1]
                    valid = ~np.isnan(y_val)
                    m = compute_metrics(y_val[valid], val_prob[valid])
                    fold_lls.append(m.get("log_loss", 1.0))
                    model_registry[f"{name}_fold{fold_idx}"] = pipe

                avg_ll = float(np.mean(fold_lls))
                all_results.append({
                    "name": name,
                    "C": c_val,
                    "penalty": penalty,
                    "l1_ratio": None,
                    "feat_set": feat_name,
                    "avg_val_ll": round(avg_ll, 4),
                    "fold_lls": [round(v, 4) for v in fold_lls],
                })
                fll = ", ".join(f"{v:.4f}" for v in fold_lls)
                print(f"    {name}: {avg_ll:.4f} ({fll})")

            # L1 (saga)
            name_l1 = f"L1 C={c_val} {feat_name}"
            fold_lls_l1 = []
            for fold_idx in range(len(ROLLING_FOLDS)):
                train_seasons, val_season = ROLLING_FOLDS[fold_idx]
                train_mask = df["season"].isin(train_seasons).values
                val_mask = (df["season"] == val_season).values

                v3_prob = fold_v3_prob[fold_idx]
                meta_x = _build_meta_features(df, v3_prob, extra_cols)
                x_tr = meta_x[train_mask]
                x_val = meta_x[val_mask]
                y_tr = y[train_mask].astype(int)
                y_val = y[val_mask]

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(
                        C=c_val, penalty="l1", solver="saga",
                        max_iter=2000, random_state=SEED,
                    )),
                ])
                pipe.fit(x_tr, y_tr)
                val_prob = pipe.predict_proba(x_val)[:, 1]
                valid = ~np.isnan(y_val)
                m = compute_metrics(y_val[valid], val_prob[valid])
                fold_lls_l1.append(m.get("log_loss", 1.0))
                model_registry[f"{name_l1}_fold{fold_idx}"] = pipe

            avg_ll_l1 = float(np.mean(fold_lls_l1))
            all_results.append({
                "name": name_l1,
                "C": c_val,
                "penalty": "l1",
                "l1_ratio": None,
                "feat_set": feat_name,
                "avg_val_ll": round(avg_ll_l1, 4),
                "fold_lls": [round(v, 4) for v in fold_lls_l1],
            })
            fll = ", ".join(f"{v:.4f}" for v in fold_lls_l1)
            print(f"    {name_l1}: {avg_ll_l1:.4f} ({fll})")

            # ElasticNet
            for l1r in L1_RATIO_VALUES:
                name_en = f"EN l1={l1r} C={c_val} {feat_name}"
                fold_lls_en = []
                for fold_idx in range(len(ROLLING_FOLDS)):
                    train_seasons, val_season = ROLLING_FOLDS[fold_idx]
                    train_mask = df["season"].isin(train_seasons).values
                    val_mask = (df["season"] == val_season).values

                    v3_prob = fold_v3_prob[fold_idx]
                    meta_x = _build_meta_features(df, v3_prob, extra_cols)
                    x_tr = meta_x[train_mask]
                    x_val = meta_x[val_mask]
                    y_tr = y[train_mask].astype(int)
                    y_val = y[val_mask]

                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("lr", LogisticRegression(
                            C=c_val, penalty="elasticnet", solver="saga",
                            l1_ratio=l1r, max_iter=2000, random_state=SEED,
                        )),
                    ])
                    pipe.fit(x_tr, y_tr)
                    val_prob = pipe.predict_proba(x_val)[:, 1]
                    valid = ~np.isnan(y_val)
                    m = compute_metrics(y_val[valid], val_prob[valid])
                    fold_lls_en.append(m.get("log_loss", 1.0))
                    model_registry[f"{name_en}_fold{fold_idx}"] = pipe

                avg_ll_en = float(np.mean(fold_lls_en))
                all_results.append({
                    "name": name_en,
                    "C": c_val,
                    "penalty": "elasticnet",
                    "l1_ratio": l1r,
                    "feat_set": feat_name,
                    "avg_val_ll": round(avg_ll_en, 4),
                    "fold_lls": [round(v, 4) for v in fold_lls_en],
                })
                fll = ", ".join(f"{v:.4f}" for v in fold_lls_en)
                print(f"    {name_en}: {avg_ll_en:.4f} ({fll})")

    # Sort by val LL
    all_results.sort(key=lambda r: r["avg_val_ll"])

    # ── 4. Incumbent baseline val LL ──
    inc_val_ll = float(np.mean([
        fold_inc_fold_lls[fi][0] for fi in range(len(ROLLING_FOLDS))
    ]))
    inc_fold_lls = [fold_inc_fold_lls[fi][0] for fi in range(len(ROLLING_FOLDS))]
    print(f"\n  Incumbent baseline: avg val LL = {inc_val_ll:.4f}")

    best = all_results[0]
    beats_val = best["avg_val_ll"] < inc_val_ll - MIN_PROMOTION_DELTA
    print(f"  Best meta-model: {best['name']} (val LL {best['avg_val_ll']:.4f})")
    print(f"  Beats incumbent on val: {beats_val}")

    # ── 5. 2025 holdout ──
    print("\n=== 2025 Holdout ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[hold_mask]
    valid_hold = ~np.isnan(hold_y)

    # Fit incumbent on all 2021-2024
    all_train_mask = df["season"].isin([2021, 2022, 2023, 2024]).values
    hold_v3_prob = _incumbent_prob_all(df, all_train_mask)
    hold_inc_prob = hold_v3_prob[hold_mask][valid_hold]
    hold_y_clean = hold_y[valid_hold]
    inc_hold_m = compute_classification_metrics(hold_y_clean, hold_inc_prob)
    inc_hold_ll = inc_hold_m["log_loss"]
    print(f"  Incumbent: holdout LL = {inc_hold_ll:.4f}")

    # Fit best meta-model on all 2021-2024
    best_extra_cols = FEATURE_SETS[best["feat_set"]]
    hold_meta_x = _build_meta_features(df, hold_v3_prob, best_extra_cols)
    x_tr_hold = hold_meta_x[all_train_mask]
    y_tr_hold = y[all_train_mask].astype(int)

    best_l1r = best["l1_ratio"] if best["penalty"] == "elasticnet" else None
    hold_solver = "saga" if best["penalty"] in ("l1", "elasticnet") else "lbfgs"

    meta_pipe_hold = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=best["C"], penalty=best["penalty"],
            solver=hold_solver, l1_ratio=best_l1r,
            max_iter=2000, random_state=SEED,
        )),
    ])
    meta_pipe_hold.fit(x_tr_hold, y_tr_hold)
    hold_meta_prob = meta_pipe_hold.predict_proba(hold_meta_x)[:, 1]
    sel_hold_prob = hold_meta_prob[hold_mask][valid_hold]
    sel_hold_m = compute_classification_metrics(hold_y_clean, sel_hold_prob)
    sel_hold_ll = sel_hold_m["log_loss"]
    print(f"  Best meta-model ({best['name']}): holdout LL = {sel_hold_ll:.4f}")

    # Evaluate ALL variants on holdout
    hold_results: Dict[str, Dict] = {}
    for r in all_results:
        extra_cols = FEATURE_SETS[r["feat_set"]]
        meta_x_v = _build_meta_features(df, hold_v3_prob, extra_cols)
        x_tr_v = meta_x_v[all_train_mask]
        pen = r["penalty"]
        l1r = r["l1_ratio"] if pen == "elasticnet" else None
        solv = "saga" if pen in ("l1", "elasticnet") else "lbfgs"
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=r["C"], penalty=pen, solver=solv, l1_ratio=l1r,
                max_iter=2000, random_state=SEED,
            )),
        ])
        pipe.fit(x_tr_v, y_tr_hold)
        prob = pipe.predict_proba(meta_x_v)[:, 1][hold_mask][valid_hold]
        m = compute_classification_metrics(hold_y_clean, prob)
        hold_results[r["name"]] = m

    sorted_hold = sorted(hold_results.items(), key=lambda x: x[1]["log_loss"])
    best_hold_name = sorted_hold[0][0]
    best_hold_ll = sorted_hold[0][1]["log_loss"]
    print(f"  Best on holdout (diagnostic): {best_hold_name} ({best_hold_ll:.4f})")

    # ── 6. Promotion check ──
    verdict = check_promotion(
        val_ll=best["avg_val_ll"],
        holdout_ll=sel_hold_ll,
    )

    # ── 7. QB-change slices ──
    qb_h = df.loc[hold_mask, "home_qb_changed"].fillna(0).values.astype(bool)
    qb_a = df.loc[hold_mask, "away_qb_changed"].fillna(0).values.astype(bool)
    qb_change_mask = (qb_h | qb_a)[valid_hold]
    no_qb_change_mask = ~qb_change_mask

    def _slice_ll(y_slice, prob_slice):
        if len(y_slice) < 2:
            return None
        return compute_metrics(y_slice, prob_slice).get("log_loss")

    inc_qc_ll = _slice_ll(hold_y_clean[qb_change_mask], hold_inc_prob[qb_change_mask])
    inc_nqc_ll = _slice_ll(hold_y_clean[no_qb_change_mask], hold_inc_prob[no_qb_change_mask])
    sel_qc_ll = _slice_ll(hold_y_clean[qb_change_mask], sel_hold_prob[qb_change_mask])
    sel_nqc_ll = _slice_ll(hold_y_clean[no_qb_change_mask], sel_hold_prob[no_qb_change_mask])

    # ── 8. Coefficient analysis ──
    meta_feature_names = ["v3_logit"] + META_FEATURE_FRIENDLY[best["feat_set"]].replace(
        "v3 logit + ", "").replace("v3 logit + ", "").split(", ")
    if meta_feature_names == ["v3_logit"]:
        pass  # already correct
    coef = meta_pipe_hold.named_steps["lr"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": meta_feature_names[:len(coef)],
        "coefficient": coef.round(4),
    })

    # ── 9. Write report ──
    print(f"\n=== Writing report → {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# Regularized Logistic Meta-Model Experiment\n\n")
        _w("## Research Question\n\n")
        _w("Can a more regularized logistic layer improve on v3.0.0 Frozen QB Overlay ")
        _w("by tuning C/penalty and adding a tiny set of low-risk pregame features?\n\n")

        _w("## Architecture\n\n")
        _w("```\n")
        _w("For each rolling-origin fold:\n")
        _w("  1. Fit fold-safe incumbent (Platt + QB overlay) on train seasons\n")
        _w("  2. Compute meta-input = logit(v3.0.0 incumbent prob)\n")
        _w("  3. Build meta-features = [meta-input] + optional small feature group\n")
        _w("  4. Fit regularized logistic meta-model on train seasons only\n")
        _w("  5. Score meta-model on validation season\n")
        _w("```\n\n")
        _w("Feature groups are tested separately to isolate each signal.\n\n")

        _w("## Fold Structure\n\n")
        _w("| Fold | Training Seasons | Validation Season |\n")
        _w("|------|-----------------|-------------------|\n")
        for fi, (ts, vs) in enumerate(ROLLING_FOLDS):
            _w(f"| {fi + 1} | {', '.join(str(s) for s in ts)} | {vs} |\n")

        _w("\n## Holdout\n\n")
        _w(f"2025 season ({HOLDOUT_SEASON}) held out entirely.\n\n")

        _w("## Feature Sets\n\n")
        _w("Each feature set is tested separately. No combination across groups.\n\n")
        _w("| Group | Features | Rationale |\n")
        _w("|-------|----------|----------|\n")
        for fname, desc in META_FEATURE_FRIENDLY.items():
            _w(f"| {fname} | {desc} | -\n")

        _w("\n## Hyperparameter Grid\n\n")
        _w(f"- L2 (lbfgs): C in {C_VALUES}\n")
        _w(f"- L1 (saga): C in {C_VALUES}\n")
        _w(f"- ElasticNet (saga): C in {C_VALUES} × l1_ratio in {L1_RATIO_VALUES}\n")
        n_l2 = len(C_VALUES) * 1
        n_l1 = len(C_VALUES) * 1
        n_en = len(C_VALUES) * len(L1_RATIO_VALUES)
        total_per_feat = n_l2 + n_l1 + n_en
        total_all = total_per_feat * len(FEATURE_SETS)
        _w(f"Per feature set: {total_per_feat} variants ({n_l2} L2 + {n_l1} L1 + {n_en} EN)\n")
        _w(f"Total: {total_all}\n\n")

        _w("## No-Go Columns\n\n")
        _w("The following column types are explicitly excluded from all feature sets:\n")
        _w("- Market: moneyline, spread, odds, no-vig probability\n")
        _w("- Score/result: home_score, away_score, result, home_win\n")
        _w("- Target: is_tie\n\n")

        _w("## Validation Results\n\n")
        _w("Sorted by average validation log loss (lower is better). "
           "Top 10 shown.\n\n")
        _w("| Rank | Model | Feat Set | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        _w("|------|-------|----------|-----------|-------|-------|-------|\n")
        _w(f"| baseline | v3.0.0 Incumbent | — | {inc_val_ll:.4f} "
           f"| {inc_fold_lls[0]:.4f} | {inc_fold_lls[1]:.4f} | {inc_fold_lls[2]:.4f} |\n")
        for rank, r in enumerate(all_results[:10], 1):
            _w(f"| {rank} | {r['name']} | {r['feat_set']} | {r['avg_val_ll']:.4f} "
               f"| {r['fold_lls'][0]:.4f} | {r['fold_lls'][1]:.4f} "
               f"| {r['fold_lls'][2]:.4f} |\n")
        if len(all_results) > 10:
            _w(f"\n... and {len(all_results) - 10} more variants (full list available in logs).\n")

        _w("\n## 2025 Holdout Results\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy | Selection |\n")
        _w("|-------|----------|-------|-----|----------|-----------|\n")
        _w(f"| v3.0.0 Incumbent | {inc_hold_ll:.4f} | {inc_hold_m['brier_score']:.4f} "
           f"| {inc_hold_m['roc_auc']:.4f} | {inc_hold_m['accuracy']:.4f} | baseline |\n")
        _w(f"| {best['name']} | {sel_hold_ll:.4f} "
           f"| {sel_hold_m['brier_score']:.4f} | {sel_hold_m['roc_auc']:.4f} "
           f"| {sel_hold_m['accuracy']:.4f} | validation-selected |\n")
        if best_hold_name != best["name"]:
            bh_m = hold_results[best_hold_name]
            _w(f"| {best_hold_name} | {bh_m['log_loss']:.4f} "
               f"| {bh_m['brier_score']:.4f} | {bh_m['roc_auc']:.4f} "
               f"| {bh_m['accuracy']:.4f} | diagnostic (best holdout) |\n")

        _w("\n## QB-Change Slices (2025 Holdout)\n\n")
        _w("| Variant | QB-Change LL | No-QB-Change LL | QC Δ | NoQC Δ |\n")
        _w("|--------|-------------|-----------------|------|--------|\n")
        qc_ok = sel_qc_ll is not None and inc_qc_ll is not None
        nqc_ok = sel_nqc_ll is not None and inc_nqc_ll is not None
        qc_d = f"{sel_qc_ll - inc_qc_ll:+.4f}" if qc_ok else "N/A"
        nqc_d = f"{sel_nqc_ll - inc_nqc_ll:+.4f}" if nqc_ok else "N/A"
        iqc = str(inc_qc_ll) if inc_qc_ll is not None else "N/A"
        inqc = str(inc_nqc_ll) if inc_nqc_ll is not None else "N/A"
        sqc = str(sel_qc_ll) if sel_qc_ll is not None else "N/A"
        snqc = str(sel_nqc_ll) if sel_nqc_ll is not None else "N/A"
        _w(f"| v3.0.0 Incumbent | {iqc} | {inqc} | — | — |\n")
        _w(f"| {best['name']} | {sqc} | {snqc} | {qc_d} | {nqc_d} |\n")

        _w("\n## Coefficients (Holdout Fit)\n\n")
        _w("| Feature | Coefficient |\n")
        _w("|---------|------------|\n")
        for _, row in coef_df.iterrows():
            _w(f"| {row['feature']} | {row['coefficient']} |\n")

        _w("\n## Decision\n\n")
        if verdict["promoted"]:
            _w(f"**✅ PROMOTED: {best['name']}**\n\n")
            _w("| Criterion | Met? |\n")
            _w("|-----------|------|\n")
            _w(f"| Beats incumbent on val LL by >= {MIN_PROMOTION_DELTA} | ✅ |\n")
            _w(f"| Beats incumbent on holdout LL by >= {MIN_PROMOTION_DELTA} | ✅ |\n")
        else:
            _w("**❌ REJECTED**\n\n")
            _w("| Criterion | Met? | Details |\n")
            _w("|-----------|------|--------|\n")
            v_check = "✅" if verdict["beats_val"] else "❌"
            h_check = "✅" if verdict["beats_holdout"] else "❌"
            _w(f"| Beats incumbent on val LL by >= {MIN_PROMOTION_DELTA} "
               f"| {v_check} | Δ = {verdict['val_delta']:.4f} |\n")
            _w(f"| Beats incumbent on holdout LL by >= {MIN_PROMOTION_DELTA} "
               f"| {h_check} | Δ = {verdict['holdout_delta']:.4f} |\n")

        _w("\n### Validation Delta\n\n")
        _w(f"Best meta-model val LL: {best['avg_val_ll']:.4f}\n")
        _w(f"Incumbent val LL: {inc_val_ll:.4f}\n")
        _w(f"Improvement: {verdict['val_delta']:+.4f}\n\n")

        _w("### Holdout Delta\n\n")
        _w(f"Best meta-model holdout LL: {sel_hold_ll:.4f}\n")
        _w(f"Incumbent holdout LL: {inc_hold_ll:.4f}\n")
        _w(f"Improvement: {verdict['holdout_delta']:+.4f}\n\n")

        _w("### Leakage Risk\n\n")
        _w("- **None.** All feature columns are pregame-safe (no score/result/market).\n")
        _w("- Holdout (2025) is never accessed during validation selection.\n")
        _w("- Platt + overlay fitted per fold on train seasons only.\n")
        _w("- Meta-model fitted per fold on train seasons only.\n")
        _w("- Rolling-origin validation ensures chronological ordering.\n\n")

        _w("### Calibration Notes\n\n")
        inc_brier = inc_hold_m["brier_score"]
        sel_brier = sel_hold_m["brier_score"]
        _w(f"- Incumbent holdout Brier: {inc_brier:.4f}\n")
        _w(f"- Meta-model holdout Brier: {sel_brier:.4f}\n")
        _w("- Calibration should be verified with reliability diagrams.\n\n")

        _w("---\n")
        _w("*Report generated by `sportslab regularized-logistic-meta`. ")
        _w(f"Seasons: 2021–{HOLDOUT_SEASON}, Folds: {len(ROLLING_FOLDS)}, "
           f"Variants: {total_all}.*\n")

    # ── 10. Export CSV if requested ──
    if output_csv is not None:
        out_df = df[hold_mask].copy()
        out_df["incumbent_home_win_prob"] = hold_inc_prob
        out_df["meta_model_prob"] = sel_hold_prob
        out_df["best_variant"] = best["name"]
        out_df.to_csv(output_csv, index=False)
        print(f"  CSV: {output_csv}")

    print(f"\nReport: {rp}")
    return str(report_path)
