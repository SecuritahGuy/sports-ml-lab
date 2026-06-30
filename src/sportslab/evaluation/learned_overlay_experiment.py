"""Learned overlay experiment — regularized logistic vs hand-tuned QB overlay.

Research question:
    Can a single regularized logistic regression (L1, L2, ElasticNet) with
    base features + QB adjustment signals learn a better combination of
    signals than the hand-tuned two-layer v3.0.0 (Platt + QB overlay)?

Architecture (fold-safe):
    For each rolling-origin fold:
        1. Compute all features chronologically (no future leakage)
        2. Fit v3.0.0 incumbent pipeline (Platt + frozen QB overlay)
        3. For each regularized logistic variant:
           a. Build feature matrix X from candidate feature set
           b. Fit LogisticRegression(C, penalty) on train only (fold-safe)
           c. Score on validation season
        4. Compare regularized logistic vs v3.0.0 incumbent

    Selection: average validation log loss across 3 folds.
    2025 holdout: one-shot evaluation after selection.

    Baseline: v3.0.0 Frozen QB Overlay (val LL 0.6305, holdout LL 0.6200).
    Promotion requires Delta >= 0.001 on BOTH val and holdout.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Elo spine params (v3.0.0)
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2

# Overlay params (v3.0.0)
OVERLAY_GAMMA = 1.0
OVERLAY_CAP = 40
ELO_TO_LOGIT = np.log(10) / 400.0

# Feature sets to test
FEATURE_SETS: Dict[str, List[str]] = {
    "base": [
        "elo_prob",
        "home_qb_changed", "away_qb_changed",
        "home_rolling_mov_3", "away_rolling_mov_3",
    ],
    "base+adj": [
        "elo_prob",
        "home_qb_changed", "away_qb_changed",
        "home_rolling_mov_3", "away_rolling_mov_3",
        "home_qb_adj", "away_qb_adj",
    ],
    "base+depth": [
        "elo_prob",
        "home_qb_changed", "away_qb_changed",
        "home_rolling_mov_3", "away_rolling_mov_3",
        "home_qb_team_starts_pre", "away_qb_team_starts_pre",
        "home_games_since_qb_change", "away_games_since_qb_change",
    ],
    "all": [
        "elo_prob",
        "home_qb_changed", "away_qb_changed",
        "home_rolling_mov_3", "away_rolling_mov_3",
        "home_qb_adj", "away_qb_adj",
        "home_qb_team_starts_pre", "away_qb_team_starts_pre",
        "home_games_since_qb_change", "away_games_since_qb_change",
    ],
    "adj_only": [
        "elo_prob",
        "home_qb_adj", "away_qb_adj",
    ],
}

# Regularization grid
C_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0]
PENALTIES = ["l2", "l1"]


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-15, 1 - 1e-15) / np.clip(1 - p, 1e-15, 1 - 1e-15))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _get_features(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns: {missing}")
    if avail:
        return df[avail].values
    return np.empty((len(df), 0))


def _build_v3_incumbent(
    df_train: pd.DataFrame, df_val: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    """Build v3.0.0 two-layer incumbent (Platt + QB overlay) and return
    validation probabilities + log loss."""
    # Layer 1: Platt on base features
    base_cols = FEATURE_SETS["base"]
    x_tr = _get_features(df_train, base_cols)
    y_tr = df_train[TARGET_COLUMN].astype(float).values.astype(int)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(x_tr, y_tr)

    x_val = _get_features(df_val, base_cols)
    base_prob = pipe.predict_proba(x_val)[:, 1]

    # Layer 2: Frozen QB overlay
    h_changed = df_val.get("home_qb_changed", pd.Series(0)).values.astype(float)
    a_changed = df_val.get("away_qb_changed", pd.Series(0)).values.astype(float)
    h_starts = df_val.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    a_starts = df_val.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    gate_mask = (h_changed == 1) | (a_changed == 1) | (h_starts < 17) | (a_starts < 17)

    h_adj = df_val.get("home_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    a_adj = df_val.get("away_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    capped_h = np.clip(h_adj, -OVERLAY_CAP, OVERLAY_CAP)
    capped_a = np.clip(a_adj, -OVERLAY_CAP, OVERLAY_CAP)
    net_adj = capped_h - capped_a
    overlay = OVERLAY_GAMMA * net_adj * ELO_TO_LOGIT
    final_logit = _logit(base_prob) + overlay * gate_mask.astype(float)
    final_prob = _sigmoid(final_logit)

    val_y = df_val[TARGET_COLUMN].astype(float).values
    valid = ~np.isnan(val_y)
    m = compute_metrics(val_y[valid], final_prob[valid])
    return final_prob, float(m.get("log_loss", 1.0))


def run_learned_overlay_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/learned_overlay.md",
    output_csv: Optional[str] = None,
) -> str:
    print("=== Learned Overlay Experiment ===")

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
    df = compute_qb_adjustments(df)  # needed for overlay + adj features
    df = compute_qb_features(df)
    df = compute_situational_features(df)

    # Filter eligible
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    # Verify feature columns exist
    all_req = set()
    for cols in FEATURE_SETS.values():
        all_req.update(cols)
    missing = [c for c in all_req if c not in df.columns]
    if missing:
        print(f"  ERROR: missing feature columns: {missing}")
        return ""
    print(f"  All feature columns present ({len(all_req)} unique)")

    # ── 2. Fold-safe v3.0.0 baseline ──
    print("\n=== v3.0.0 Incumbent (fold-safe) ===")
    v3_fold_lls = []
    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values
        _, fold_ll = _build_v3_incumbent(
            df[train_mask], df[val_mask],
        )
        v3_fold_lls.append(fold_ll)
    v3_avg_val_ll = float(np.mean(v3_fold_lls))
    print(f"  v3.0.0 inc: avg val LL = {v3_avg_val_ll:.4f}")
    print(f"    per fold: {[f'{ll:.4f}' for ll in v3_fold_lls]}")

    # ── 3. Regularized logistic grid ──
    print("\n=== Regularized Logistic Grid ===")
    results: List[Dict] = []
    total = len(FEATURE_SETS) * len(C_VALUES) * len(PENALTIES)
    done = 0

    for set_name, feat_cols in FEATURE_SETS.items():
        for c_val in C_VALUES:
            for penalty in PENALTIES:
                fold_lls = []

                for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
                    train_mask = df["season"].isin(train_seasons).values
                    val_mask = (df["season"] == val_season).values

                    df_train = df[train_mask]
                    df_val = df[val_mask]

                    x_tr = _get_features(df_train, feat_cols)
                    y_tr = df_train[TARGET_COLUMN].astype(float).values.astype(int)
                    x_val = _get_features(df_val, feat_cols)
                    y_val = df_val[TARGET_COLUMN].astype(float).values

                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("lr", LogisticRegression(
                            C=c_val, penalty=penalty, solver="saga",
                            max_iter=5000, random_state=SEED,
                        )),
                    ])
                    pipe.fit(x_tr, y_tr)
                    prob = pipe.predict_proba(x_val)[:, 1]

                    valid = ~np.isnan(y_val)
                    m = compute_metrics(y_val[valid], prob[valid])
                    fold_lls.append(m.get("log_loss", 1.0))

                avg_ll = float(np.mean(fold_lls))
                name = f"{set_name} C={c_val} {penalty}"
                results.append({
                    "name": name,
                    "set_name": set_name,
                    "C": c_val,
                    "penalty": penalty,
                    "avg_val_ll": avg_ll,
                    "fold_lls": fold_lls,
                })
                done += 1
                if done <= 5 or done == total or done % 25 == 0:
                    delta = avg_ll - v3_avg_val_ll
                    best_delta = results[0]["avg_val_ll"] - v3_avg_val_ll
                    marker = "*BEST*" if delta < 0 and delta < best_delta else ""
                    print(f"  [{done}/{total}] {name}: {avg_ll:.4f} (Delta {delta:+.4f}) {marker}")

    results.sort(key=lambda r: r["avg_val_ll"])

    # Best per feature set
    best_per_set: Dict[str, Dict] = {}
    for r in results:
        sn = r["set_name"]
        if sn not in best_per_set or r["avg_val_ll"] < best_per_set[sn]["avg_val_ll"]:
            best_per_set[sn] = r

    best = results[0]
    print(f"\n  Overall best: {best['name']} (val LL {best['avg_val_ll']:.4f})")
    print(f"  v3.0.0 incumbent: {v3_avg_val_ll:.4f}")
    print(f"  Delta: {best['avg_val_ll'] - v3_avg_val_ll:+.4f}")
    print("\n  Best per feature set:")
    for sn, r in sorted(best_per_set.items()):
        d = r["avg_val_ll"] - v3_avg_val_ll
        print(f"    {sn}: {r['avg_val_ll']:.4f} (Delta {d:+.4f})")

    # ── 4. 2025 Holdout ──
    print("\n=== 2025 Holdout ===")
    train_mask_h = df["season"].isin([2021, 2022, 2023, 2024]).values
    hold_mask = (df["season"] == HOLDOUT_SEASON).values

    # v3.0.0 holdout
    _, inc_hold_ll = _build_v3_incumbent(df[train_mask_h], df[hold_mask])
    print(f"  v3.0.0 inc: holdout LL = {inc_hold_ll:.4f}")

    # Regularized logistic holdout for ALL variants
    hold_results: Dict[str, Dict] = {}
    for r in results:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=r["C"], penalty=r["penalty"], solver="saga",
                max_iter=5000, random_state=SEED,
            )),
        ])
        x_tr = _get_features(df[train_mask_h], FEATURE_SETS[r["set_name"]])
        y_tr = df[train_mask_h][TARGET_COLUMN].astype(float).values.astype(int)
        pipe.fit(x_tr, y_tr)
        x_hold = _get_features(df[hold_mask], FEATURE_SETS[r["set_name"]])
        prob = pipe.predict_proba(x_hold)[:, 1]

        hold_y = df[hold_mask][TARGET_COLUMN].astype(float).values
        valid = ~np.isnan(hold_y)
        hold_results[r["name"]] = compute_classification_metrics(hold_y[valid], prob[valid])

    sorted_hold = sorted(hold_results.items(), key=lambda x: x[1]["log_loss"])
    best_hold_name = sorted_hold[0][0]
    best_hold_ll = hold_results[best_hold_name]["log_loss"]

    # Best per-set holdout
    hold_best_per_set: Dict[str, Tuple[str, float]] = {}
    for r in results:
        sn = r["set_name"]
        hl = hold_results[r["name"]]["log_loss"]
        if sn not in hold_best_per_set or hl < hold_best_per_set[sn][1]:
            hold_best_per_set[sn] = (r["name"], hl)

    # Val-selected holdout
    val_sel_prob = None
    pipe_vs = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=best["C"], penalty=best["penalty"], solver="saga",
            max_iter=5000, random_state=SEED,
        )),
    ])
    x_tr_vs = _get_features(df[train_mask_h], FEATURE_SETS[best["set_name"]])
    y_tr_vs = df[train_mask_h][TARGET_COLUMN].astype(float).values.astype(int)
    pipe_vs.fit(x_tr_vs, y_tr_vs)
    x_hold_vs = _get_features(df[hold_mask], FEATURE_SETS[best["set_name"]])
    val_sel_prob = pipe_vs.predict_proba(x_hold_vs)[:, 1]

    hold_y = df[hold_mask][TARGET_COLUMN].astype(float).values
    valid = ~np.isnan(hold_y)
    val_sel_metrics = compute_classification_metrics(hold_y[valid], val_sel_prob[valid])
    val_sel_ll = val_sel_metrics["log_loss"]

    print(f"  Best (val-sel): {best['name']}: holdout LL = {val_sel_ll:.4f}")
    print(f"  Best (holdout): {best_hold_name}: holdout LL = {best_hold_ll:.4f}")

    # ── 5. Write report ──
    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# Learned Overlay Experiment\n\n")
        _w("## Research Question\n\n")
        _w("Can a single regularized logistic model (L1/L2) learn a better ")
        _w("combination of base features + QB adjustment signals than the ")
        _w("hand-tuned two-layer v3.0.0 (Platt + frozen QB overlay)?\n\n")

        _w("## Architecture\n\n")
        _w("```\n")
        _w("v3.0.0 (two-layer):\n")
        _w("  Layer 1: Platt logistic on [elo_prob, qb_changed, mov_3]\n")
        _w("  Layer 2: Hand-tuned QB overlay (gamma=1, cap=40, gate=starts<17 OR changed)\n")
        _w("  Final: Layer 1 -> Layer 2 (logit-space additive)\n\n")
        _w("Challenger (single layer):\n")
        _w("  Regularized logistic on [base + QB adj + depth features]\n")
        _w("  All signals learned jointly with L1/L2 shrinkage\n")
        _w("```\n\n")

        _w("## Feature Sets\n\n")
        for sn, cols in FEATURE_SETS.items():
            _w(f"- **{sn}**: {', '.join(cols)}\n")

        _w("\n## Hyperparameter Grid\n\n")
        _w(f"C values: {C_VALUES}\n")
        _w(f"Penalties: {PENALTIES}\n")
        _w(f"Feature sets: {len(FEATURE_SETS)}\n")
        _w(f"Total variants: {total}\n\n")

        _w("## Validation Results\n\n")
        _w("| Feature Set | Best Config | Avg Val LL | Delta vs v3.0.0 |\n")
        _w("|------------|-------------|-----------|------------------|\n")
        for sn, r in sorted(best_per_set.items()):
            d = r["avg_val_ll"] - v3_avg_val_ll
            _w(f"| {sn} | C={r['C']} {r['penalty']} | {r['avg_val_ll']:.4f} | {d:+.4f} |\n")
        _w(f"| v3.0.0 inc | (hand-tuned overlay) | {v3_avg_val_ll:.4f} | -- |\n")

        _w("\n### Top 10 Overall\n\n")
        _w("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        _w("|-------|-----------|-------|-------|-------|\n")
        for r in results[:10]:
            _w(f"| {r['name']} | {r['avg_val_ll']:.4f}")
            _w(f" | {r['fold_lls'][0]:.4f} | {r['fold_lls'][1]:.4f} | ")
            _w(f"{r['fold_lls'][2]:.4f} |\n")
        if len(results) > 10:
            _w(f"\n... ({len(results) - 10} more variants)\n\n")

        _w("## 2025 Holdout Results\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy | Selection |\n")
        _w("|-------|----------|-------|-----|----------|-----------|\n")
        _w(f"| v3.0.0 inc | {inc_hold_ll:.4f} | -- | -- | -- | baseline |\n")
        _w(f"| {best['name']} | {val_sel_ll:.4f}")
        _w(f" | {val_sel_metrics['brier_score']:.4f}")
        _w(f" | {val_sel_metrics['roc_auc']:.4f}")
        _w(f" | {val_sel_metrics['accuracy']:.4f} | validation |\n")

        _w("\n### Best per Feature Set (Holdout)\n\n")
        _w("| Feature Set | Best Config | Holdout LL |\n")
        _w("|------------|-------------|------------|\n")
        for sn in sorted(hold_best_per_set.keys()):
            name, hl = hold_best_per_set[sn]
            _w(f"| {sn} | {name} | {hl:.4f} |\n")
        _w(f"| v3.0.0 inc | (hand-tuned overlay) | {inc_hold_ll:.4f} |\n")

        _w("\n### All Variants (Holdout)\n\n")
        _w("| Model | Log Loss | Brier | AUC |\n")
        _w("|-------|----------|-------|-----|\n")
        for name, m in sorted_hold:
            _w(f"| {name} | {m['log_loss']:.4f} | {m['brier_score']:.4f} | {m['roc_auc']:.4f} |\n")

        _w("\n## Decision\n\n")
        beats_val = best["avg_val_ll"] < v3_avg_val_ll - MIN_PROMOTION_DELTA
        beats_hold = val_sel_ll < inc_hold_ll - MIN_PROMOTION_DELTA
        promotes = beats_val and beats_hold

        vd = v3_avg_val_ll - best["avg_val_ll"]
        hd = inc_hold_ll - val_sel_ll

        if promotes:
            _w(f"**PROMOTED: {best['name']}**\n\n")
            _w(f"Val Delta: {vd:+.4f} (incumbent was better)\n")
            _w(f"Holdout Delta: {hd:+.4f}\n")
        else:
            _w("**REJECTED**\n\n")
            _w(f"Val Delta: {vd:+.4f} (need >= {MIN_PROMOTION_DELTA})\n")
            _w(f"Holdout Delta: {hd:+.4f} (need >= {MIN_PROMOTION_DELTA})\n")
            if not beats_val:
                _w("Regularized logistic does NOT beat v3.0.0 on validation.\n")
            if not beats_hold:
                _w("Val-selected variant does NOT beat v3.0.0 on holdout.\n")

        _w("\n### Post-Hoc (Diagnostic Only)\n\n")
        if best_hold_name != best["name"]:
            _w(f"The best holdout variant ({best_hold_name}) differs from the ")
            _w("validation-selected variant. This is a diagnostic finding only.")
        else:
            _w("The best validation variant is also best on holdout.")

        _w("\n\n---\n")
        _w("*Report generated by `sportslab learned-overlay`. ")
        _w(f"Seasons: 2021--{HOLDOUT_SEASON}, ")
        _w(f"Folds: {len(ROLLING_FOLDS)}, ")
        _w(f"Variants: {total}.*\n")

    # ── 6. Export CSV if requested ──
    if output_csv is not None:
        out_df = df[hold_mask].copy()
        out_df["v3_incumbent_prob"] = _sigmoid(_logit(
            _build_v3_incumbent(df[train_mask_h], df[hold_mask])[0]
        ))
        out_df["learned_overlay_prob"] = val_sel_prob
        out_df.to_csv(output_csv, index=False)
        print(f"  CSV: {output_csv}")

    print(f"\nReport: {rp}")
    return str(report_path)
