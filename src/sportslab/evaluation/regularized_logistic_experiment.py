"""Regularized logistic meta-model experiment.

Research question:
    Can a regularized logistic regression on the v3.0.0 incumbent logit plus
    additional pregame features (rest_diff, is_dome, div_game, week, game_type)
    improve on the v3.0.0 Frozen QB Overlay champion?

Architecture (fold-safe):
    For each rolling-origin fold:
        1. Fit fold-safe incumbent (Platt + QB overlay) using only train seasons
        2. Compute incumbent_logit from the fold-safe incumbent
        3. Build meta-features = [incumbent_logit] + pregame features
        4. Fit logistic regression meta-model (tuning C + penalty) on train only
        5. Score meta-model on validation season

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
from sklearn.preprocessing import StandardScaler

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

# Incumbent base features
INCUMBENT_FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]

# QB overlay params
QB_GATE_GAMMA = 1.0
QB_GATE_CAP = 40

ELO_TO_LOGIT = np.log(10.0) / 400.0

# Meta-model feature columns
META_FEATURE_COLS = [
    "rest_diff",
    "is_dome",
    "div_game",
]

# Hyperparameter grid
C_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
PENALTY_VALUES = ["l2", "l1"]

N_WORST = 20


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def _encode_week_sin_cos(week: np.ndarray, max_week: int = 18) -> np.ndarray:
    theta = 2.0 * np.pi * week / max_week
    return np.column_stack([np.sin(theta), np.cos(theta)])


def _build_gate_mask(df: pd.DataFrame) -> np.ndarray:
    h_changed = df.get("home_qb_changed", pd.Series(0)).values.astype(float)
    a_changed = df.get("away_qb_changed", pd.Series(0)).values.astype(float)
    h_starts = df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    a_starts = df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    return (h_changed == 1) | (a_changed == 1) | (h_starts < 17) | (a_starts < 17)


def _get_features(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    if avail:
        return df[avail].values
    return np.empty((len(df), 0))


def run_regularized_logistic_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/regularized_logistic.md",
    output_csv: Optional[str] = None,
) -> str:
    print("=== Regularized Logistic Meta-Model Experiment ===")

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

    # Pre-compute arrays (used across all folds)
    elo_prob = df["elo_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values
    inc_feat = _get_features(df, INCUMBENT_FEATURE_COLS)
    home_qb_adj = df["home_qb_adj"].values.astype(float)
    away_qb_adj = df["away_qb_adj"].values.astype(float)
    gate_mask = _build_gate_mask(df)

    # Meta-features (pregame-safe, low cardinality)
    meta_vals = _get_features(df, META_FEATURE_COLS)
    week_sincos = _encode_week_sin_cos(df["week"].values)
    playoff_flag = (df.get("game_type_enc", pd.Series(0)).values > 0).astype(float)
    meta_extra = np.column_stack([meta_vals, week_sincos, playoff_flag])

    meta_names = META_FEATURE_COLS + ["week_sin", "week_cos", "game_type_is_playoff"]
    print(f"  Meta-features: {meta_names}")

    # ── 2. Rolling-origin validation (fold-safe) ──
    print("\n=== Rolling-Origin Validation ===")
    print(f"  Folds: {ROLLING_FOLDS}")

    fold_data: Dict[int, Dict] = {}

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        print(f"\n  Fold {fold_idx + 1}: train {train_seasons} → val {val_season}")

        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())

        # ── 2a. Fit fold-safe incumbent Platt ──
        train_elo = elo_prob[train_mask]
        train_y_int = y[train_mask].astype(int)
        train_feat = inc_feat[train_mask]

        x_train = (
            np.column_stack([train_elo, train_feat])
            if train_feat.size else train_elo.reshape(-1, 1)
        )

        inc_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        inc_pipe.fit(x_train, train_y_int)

        # Generate incumbent probs for ALL data
        x_all = (
            np.column_stack([elo_prob, inc_feat])
            if inc_feat.size else elo_prob.reshape(-1, 1)
        )
        base_prob = inc_pipe.predict_proba(x_all)[:, 1]
        base_logit = _logit(base_prob)

        # ── 2b. Apply frozen QB overlay ──
        capped_h = np.clip(home_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
        capped_a = np.clip(away_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
        cur_net_adj = capped_h - capped_a
        overlay = QB_GATE_GAMMA * cur_net_adj * ELO_TO_LOGIT
        final_logit = base_logit + overlay * gate_mask.astype(float)
        incumbent_prob = _sigmoid(final_logit)

        # ── 2c. Build meta-training data ──
        inc_logit = _logit(incumbent_prob).reshape(-1, 1)
        meta_features = np.column_stack([inc_logit, meta_extra])

        fold_data[fold_idx] = {
            "meta_X": meta_features,
            "y": y,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "incumbent_prob": incumbent_prob,
            "n_train": n_train,
            "n_val": n_val,
        }

    # ── 3. Run all hyperparameter combinations ──
    print("\n=== Grid Search ===")
    results: List[Dict] = []

    for c_val in C_VALUES:
        for penalty in PENALTY_VALUES:
            solver = "liblinear" if penalty == "l1" else "lbfgs"
            fold_lls = []
            fold_models: List[Pipeline] = []

            for fold_idx in range(len(ROLLING_FOLDS)):
                fd = fold_data[fold_idx]
                train_y_int = fd["y"][fd["train_mask"]].astype(int)
                val_y = fd["y"][fd["val_mask"]]

                # Fit meta-model on training seasons only
                meta_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(
                        C=c_val, penalty=penalty, solver=solver,
                        max_iter=2000, random_state=SEED,
                    )),
                ])

                x_tr = fd["meta_X"][fd["train_mask"]]
                x_val = fd["meta_X"][fd["val_mask"]]
                meta_pipe.fit(x_tr, train_y_int)
                val_prob = meta_pipe.predict_proba(x_val)[:, 1]

                valid = ~np.isnan(val_y)
                m = compute_metrics(val_y[valid], val_prob[valid])
                fold_lls.append(m.get("log_loss", 1.0))
                fold_models.append(meta_pipe)

            avg_ll = float(np.mean(fold_lls))
            name = f"Logistic C={c_val} {penalty}"
            results.append({
                "name": name,
                "C": c_val,
                "penalty": penalty,
                "avg_val_ll": avg_ll,
                "fold_lls": fold_lls,
                "fold_models": fold_models,
            })

            fll_str = f"{fold_lls[0]:.4f}, {fold_lls[1]:.4f}, {fold_lls[2]:.4f}"
            print(f"  {name}: avg val LL = {avg_ll:.4f} ({fll_str})")

    # Sort by val LL
    results.sort(key=lambda r: r["avg_val_ll"])

    # ── 4. Baseline: run incumbent through same folds ──
    print("\n=== Incumbent Baseline (Fold-Safe) ===")
    inc_fold_lls = []
    for fold_idx in range(len(ROLLING_FOLDS)):
        fd = fold_data[fold_idx]
        val_y = fd["y"][fd["val_mask"]]
        inc_prob = fd["incumbent_prob"][fd["val_mask"]]
        valid = ~np.isnan(val_y)
        m = compute_metrics(val_y[valid], inc_prob[valid])
        inc_fold_lls.append(m.get("log_loss", 1.0))

    inc_avg_val_ll = float(np.mean(inc_fold_lls))
    inc_lls = f"{inc_fold_lls[0]:.4f}, {inc_fold_lls[1]:.4f}, {inc_fold_lls[2]:.4f}"
    print(f"  Incumbent baseline: avg val LL = {inc_avg_val_ll:.4f} ({inc_lls})")

    # ── 5. Select best variant ──
    best = results[0]
    beats_val = best["avg_val_ll"] < inc_avg_val_ll - MIN_PROMOTION_DELTA
    print(f"\n  Best meta-model: {best['name']} (val LL {best['avg_val_ll']:.4f})")
    print(f"  Incumbent val LL: {inc_avg_val_ll:.4f}")
    print(f"  Beats incumbent on val: {beats_val} (Δ = {inc_avg_val_ll - best['avg_val_ll']:.4f})")

    # ── 6. 2025 holdout ──
    print("\n=== 2025 Holdout ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[hold_mask]
    valid_hold = ~np.isnan(hold_y)

    # Fit incumbent on all 2021-2024
    train_mask_hold = df["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_hold = elo_prob[train_mask_hold]
    train_y_hold = y[train_mask_hold].astype(int)
    train_feat_hold = inc_feat[train_mask_hold]
    x_train_hold = (
        np.column_stack([train_elo_hold, train_feat_hold])
        if train_feat_hold.size else train_elo_hold.reshape(-1, 1)
    )
    inc_pipe_hold = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    inc_pipe_hold.fit(x_train_hold, train_y_hold)
    x_all_hold = (
        np.column_stack([elo_prob, inc_feat])
        if inc_feat.size else elo_prob.reshape(-1, 1)
    )
    hold_base_prob = inc_pipe_hold.predict_proba(x_all_hold)[:, 1]
    hold_base_logit = _logit(hold_base_prob)
    hold_capped_h = np.clip(home_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    hold_capped_a = np.clip(away_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    hold_overlay = QB_GATE_GAMMA * (hold_capped_h - hold_capped_a) * ELO_TO_LOGIT
    hold_incumbent_prob = _sigmoid(hold_base_logit + hold_overlay * gate_mask.astype(float))

    hold_inc_prob = hold_incumbent_prob[hold_mask][valid_hold]
    hold_y_clean = hold_y[valid_hold]
    inc_hold_m = compute_classification_metrics(hold_y_clean, hold_inc_prob)
    inc_hold_ll = inc_hold_m["log_loss"]
    print(f"  Incumbent (reproduced): holdout LL = {inc_hold_ll:.4f}")

    # Fit best meta-model on all 2021-2024
    inc_logit_hold = _logit(hold_incumbent_prob).reshape(-1, 1)
    meta_hold = np.column_stack([inc_logit_hold, meta_extra])

    hold_train_mask = train_mask_hold
    x_tr_hold = meta_hold[hold_train_mask]
    y_tr_hold = y[hold_train_mask].astype(int)

    meta_pipe_hold = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=best["C"], penalty=best["penalty"],
            solver="liblinear" if best["penalty"] == "l1" else "lbfgs",
            max_iter=2000, random_state=SEED,
        )),
    ])
    meta_pipe_hold.fit(x_tr_hold, y_tr_hold)
    hold_meta_prob = meta_pipe_hold.predict_proba(meta_hold)[:, 1]
    sel_hold_prob = hold_meta_prob[hold_mask][valid_hold]
    sel_hold_m = compute_classification_metrics(hold_y_clean, sel_hold_prob)
    sel_hold_ll = sel_hold_m["log_loss"]
    print(f"  Best meta-model ({best['name']}): holdout LL = {sel_hold_ll:.4f}")

    beats_hold = sel_hold_ll < inc_hold_ll - MIN_PROMOTION_DELTA

    # Evaluate ALL variants on holdout (diagnostic)
    hold_results: Dict[str, Dict] = {}
    for r in results:
        c_val = r["C"]
        penalty = r["penalty"]
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        meta_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=c_val, penalty=penalty, solver=solver,
                max_iter=2000, random_state=SEED,
            )),
        ])
        meta_pipe.fit(x_tr_hold, y_tr_hold)
        prob = meta_pipe.predict_proba(meta_hold)[:, 1][hold_mask][valid_hold]
        m = compute_classification_metrics(hold_y_clean, prob)
        hold_results[r["name"]] = m

    sorted_hold = sorted(hold_results.items(), key=lambda x: x[1]["log_loss"])
    best_hold_name = sorted_hold[0][0]
    best_hold_ll = sorted_hold[0][1]["log_loss"]
    print(f"  Best on holdout (diagnostic): {best_hold_name} ({best_hold_ll:.4f})")

    # ── 7. Promotion check ──
    promotes = beats_val and beats_hold

    # ── 8. Slices ──
    qb_h = df.loc[hold_mask, "home_qb_changed"].fillna(0).values.astype(bool)
    qb_a = df.loc[hold_mask, "away_qb_changed"].fillna(0).values.astype(bool)
    qb_change_mask_hold = (qb_h | qb_a)[valid_hold]
    no_qb_change_mask_hold = ~qb_change_mask_hold

    def _slice_ll(y_slice, prob_slice):
        if len(y_slice) < 2:
            return None
        return compute_metrics(y_slice, prob_slice).get("log_loss")

    # Incumbent slices
    inc_qc_ll = _slice_ll(
        hold_y_clean[qb_change_mask_hold], hold_inc_prob[qb_change_mask_hold])
    inc_nqc_ll = _slice_ll(
        hold_y_clean[no_qb_change_mask_hold], hold_inc_prob[no_qb_change_mask_hold])

    # Meta-model slices
    sel_qc_ll = _slice_ll(
        hold_y_clean[qb_change_mask_hold], sel_hold_prob[qb_change_mask_hold])
    sel_nqc_ll = _slice_ll(
        hold_y_clean[no_qb_change_mask_hold], sel_hold_prob[no_qb_change_mask_hold])

    # ── 9. Coefficient analysis ──
    coef = meta_pipe_hold.named_steps["lr"].coef_[0]
    extra_coefs = ["week_sin", "week_cos", "game_type_is_playoff"]
    coef_names = ["incumbent_logit"] + META_FEATURE_COLS + extra_coefs
    coef_df = pd.DataFrame({"feature": coef_names[:len(coef)], "coefficient": coef.round(4)})

    # ── 10. Write report ──
    print(f"\n=== Writing report → {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# Regularized Logistic Meta-Model Experiment\n\n")
        _w("## Research Question\n\n")
        _w("Can a regularized logistic regression on the v3.0.0 incumbent logit ")
        _w("plus additional pregame features improve on the v3.0.0 Frozen QB Overlay champion?\n\n")

        _w("## Architecture\n\n")
        _w("```\n")
        _w("For each rolling-origin fold:\n")
        _w("  1. Fit fold-safe incumbent (Platt + QB overlay) on train seasons\n")
        _w("  2. Compute incumbent_logit = logit(incumbent_prob)\n")
        _w("  3. Build meta-features = [incumbent_logit] + rest_diff + is_dome + ")
        _w("div_game + week_sin + week_cos + game_type_is_playoff\n")
        _w("  4. Fit logistic regression meta-model (tuning C + penalty) on train\n")
        _w("  5. Score on validation season\n")
        _w("```\n\n")

        _w("## Fold Structure\n\n")
        _w("| Fold | Training Seasons | Validation Season |\n")
        _w("|------|-----------------|-------------------|\n")
        for fi, (ts, vs) in enumerate(ROLLING_FOLDS):
            _w(f"| {fi + 1} | {', '.join(str(s) for s in ts)} | {vs} |\n")

        _w("\n## Holdout\n\n")
        _w(f"2025 season (year {HOLDOUT_SEASON}) is held out entirely.\n\n")

        _w("## Meta-Features\n\n")
        _w("| Feature | Description |\n")
        _w("|---------|-------------|\n")
        _w("| incumbent_logit | logit(v3.0.0 probability) — the incumbent signal |\n")
        _w("| rest_diff | home_rest - away_rest (rest advantage) |\n")
        _w("| is_dome | 1 if game is in domed stadium |\n")
        _w("| div_game | 1 if division matchup |\n")
        _w("| week_sin | sin(2π * week / 18) — cyclic week encoding |\n")
        _w("| week_cos | cos(2π * week / 18) — cyclic week encoding |\n")
        _w("| game_type_is_playoff | 1 if playoff game |\n\n")

        _w("## Hyperparameter Grid\n\n")
        _w(f"C values: {C_VALUES}\n")
        _w(f"Penalties: {PENALTY_VALUES}\n")
        _w(f"Total variants: {len(C_VALUES) * len(PENALTY_VALUES)}\n\n")

        _w("## Validation Results\n\n")
        _w("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        _w("|-------|-----------|-------|-------|-------|\n")
        _w(f"| v3.0.0 Incumbent | {inc_avg_val_ll:.4f} | {inc_fold_lls[0]:.4f} | {inc_fold_lls[1]:.4f} | {inc_fold_lls[2]:.4f} |\n")
        for r in results:
            _w(f"| {r['name']} | {r['avg_val_ll']:.4f} | {r['fold_lls'][0]:.4f} | {r['fold_lls'][1]:.4f} | {r['fold_lls'][2]:.4f} |\n")

        _w("\n## 2025 Holdout Results\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy | Selection |\n")
        _w("|-------|----------|-------|-----|----------|-----------|\n")
        _w(f"| v3.0.0 Incumbent | {inc_hold_ll:.4f} | {inc_hold_m['brier_score']:.4f} | {inc_hold_m['roc_auc']:.4f} | {inc_hold_m['accuracy']:.4f} | baseline |\n")
        _w(f"| Best meta-model ({best['name']}) | {sel_hold_ll:.4f} | {sel_hold_m['brier_score']:.4f} | {sel_hold_m['roc_auc']:.4f} | {sel_hold_m['accuracy']:.4f} | validation-selected |\n")

        if best_hold_name != best["name"]:
            bh_m = hold_results[best_hold_name]
            _w(f"| {best_hold_name} | {bh_m['log_loss']:.4f} | {bh_m['brier_score']:.4f} | {bh_m['roc_auc']:.4f} | {bh_m['accuracy']:.4f} | diagnostic (best holdout) |\n")

        _w("\n## QB-Change Slices\n\n")
        _w("| Variant | QB-Change LL | No-QB-Change LL | QC Δ | NoQC Δ |\n")
        _w("|--------|-------------|-----------------|------|--------|\n")
        qc_delta = (sel_qc_ll - inc_qc_ll) if (sel_qc_ll is not None and inc_qc_ll is not None) else None
        nqc_delta = (sel_nqc_ll - inc_nqc_ll) if (sel_nqc_ll is not None and inc_nqc_ll is not None) else None
        qc_str = f"{qc_delta:+.4f}" if qc_delta is not None else "N/A"
        nqc_str = f"{nqc_delta:+.4f}" if nqc_delta is not None else "N/A"
        _w(f"| v3.0.0 Incumbent | {inc_qc_ll or 'N/A':.4f} | {inc_nqc_ll or 'N/A':.4f} | — | — |\n")
        _w(f"| Best meta-model | {sel_qc_ll or 'N/A':.4f} | {sel_nqc_ll or 'N/A':.4f} | {qc_str} | {nqc_str} |\n")

        _w("\n## Coefficients (Holdout Fit)\n\n")
        _w("| Feature | Coefficient |\n")
        _w("|---------|------------|\n")
        for _, row in coef_df.iterrows():
            _w(f"| {row['feature']} | {row['coefficient']} |\n")

        _w("\n## Decision\n\n")
        if promotes:
            _w(f"**✅ PROMOTED: {best['name']}**\n\n")
            _w("| Criterion | Met? |\n")
            _w("|-----------|------|\n")
            val_line = (f"| Beats incumbent on val LL ({best['avg_val_ll']:.4f} < "
                        f"{inc_avg_val_ll:.4f}) by >= {MIN_PROMOTION_DELTA} | ✅ |\n")
            _w(val_line)
            hold_line = (f"| Beats incumbent on holdout LL ({sel_hold_ll:.4f} < "
                         f"{inc_hold_ll:.4f}) by >= {MIN_PROMOTION_DELTA} | ✅ |\n")
            _w(hold_line)
        else:
            _w("**❌ REJECTED**\n\n")
            _w("| Criterion | Met? | Details |\n")
            _w("|-----------|------|--------|\n")
            val_delta = inc_avg_val_ll - best['avg_val_ll']
            hold_delta = inc_hold_ll - sel_hold_ll
            v_check = "✅" if beats_val else "❌"
            h_check = "✅" if beats_hold else "❌"
            _w(f"| Beats incumbent on val LL by >= {MIN_PROMOTION_DELTA} | {v_check} | Δ = {val_delta:.4f} |\n")
            _w(f"| Beats incumbent on holdout LL by >= {MIN_PROMOTION_DELTA} | {h_check} | Δ = {hold_delta:.4f} |\n")

        _w("\n### Validation Delta\n\n")
        _w(f"Best meta-model val LL: {best['avg_val_ll']:.4f}\n")
        _w(f"Incumbent val LL: {inc_avg_val_ll:.4f}\n")
        _w(f"Improvement: {val_delta:+.4f}\n\n")

        _w("### Holdout Delta\n\n")
        _w(f"Best meta-model holdout LL: {sel_hold_ll:.4f}\n")
        _w(f"Incumbent holdout LL: {inc_hold_ll:.4f}\n")
        _w(f"Improvement: {hold_delta:+.4f}\n\n")

        if not promotes:
            _w("### Recommended Next Steps\n\n")
            _w("1. Try GAM/spline-based logistic with nonlinear elo_prob transformations\n")
            _w("2. Test gradient boosting diagnostic with strong regularization\n")
            _w("3. Investigate whether incumbent_logit dominates features (check coefficients)\n")
            _w("4. Consider dynamic calibration (season-aware Platt, temperature scaling)\n\n")

        _w("---\n")
        _w("*Report generated by `sportslab regularized-logistic`. ")
        _w(f"Seasons: 2021–{HOLDOUT_SEASON}, ")
        _w(f"Folds: {len(ROLLING_FOLDS)}, ")
        _w(f"Variants: {len(C_VALUES) * len(PENALTY_VALUES)}.*\n")

    # ── 11. Export CSV if requested ──
    if output_csv is not None:
        out_df = df[hold_mask].copy()
        out_df["incumbent_home_win_prob"] = hold_inc_prob
        out_df["meta_model_prob"] = hold_meta_prob[hold_mask]
        out_df["best_variant"] = best["name"]
        out_df.to_csv(output_csv, index=False)
        print(f"  CSV: {output_csv}")

    print(f"\nReport: {rp}")
    return str(report_path)
