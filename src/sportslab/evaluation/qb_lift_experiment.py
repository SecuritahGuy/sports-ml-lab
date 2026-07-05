"""Fold-safe experiment testing QB Lift features against the v3.0.0 incumbent.

QB Lift = rolling QB EPA/dropback and CPOE from PBP data, computed
as pregame-safe features (prior games only, minimum 10 dropbacks).

This is a NEW pregame-safe data source (PBP-derived QB efficiency),
not a retest of any rejected feature family.

Research question: Do rolling QB efficiency metrics improve on the
incumbent, especially for QB-change games?
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sportslab.evaluation.calibration_audit import (
    _build_gate_mask,
    _fold_masks,
    brier_decomposition,
    ece_mce,
)
from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.predict_incumbent import (
    BEST_DECAY,
    BEST_HFA,
    BEST_K,
    BEST_QB_BONUS,
    BEST_REG,
    ELO_TO_LOGIT,
    FEATURE_COLS,
    INCUMBENT_VERSION,
    OVERLAY_CAP,
    OVERLAY_GAMMA,
    _build_pipeline,
    _logit,
    _sigmoid,
)
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
from sportslab.features.qb_lift import compute_qb_lift_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025

INCUMBENT_VAL_LL = 0.6305
INCUMBENT_HOLDOUT_LL = 0.6200

QB_LIFT_VARIANTS = [
    ("qb_lift_3", ["home_qb_epa_3", "away_qb_epa_3", "net_qb_epa_3"]),
    ("qb_lift_5", ["home_qb_epa_5", "away_qb_epa_5", "net_qb_epa_5"]),
    ("qb_lift_all", [
        "home_qb_epa_3", "away_qb_epa_3", "net_qb_epa_3",
        "home_qb_epa_5", "away_qb_epa_5", "net_qb_epa_5",
    ]),
    ("qb_lift_cpoe", ["home_qb_cpoe_3", "away_qb_cpoe_3"]),
]


def _v3_probs(
    df: pd.DataFrame, train_mask: np.ndarray, qb_lift_cols: List[str] = None,
) -> np.ndarray:
    """Return v3.0.0 + optional QB Lift final probability for all rows."""
    elo = df["elo_prob"].values.astype(float)
    feat_cols_list = [c for c in FEATURE_COLS if c in df.columns]
    if qb_lift_cols:
        feat_cols_list = feat_cols_list + [c for c in qb_lift_cols if c in df.columns]
    y = df[TARGET_COLUMN].values.astype(float)
    has_feats = len(feat_cols_list) > 0

    # Impute NaN to 0 for training stability
    df_imp = df.copy()
    for c in feat_cols_list:
        df_imp[c] = df_imp[c].fillna(0)

    if has_feats:
        cols = [elo[train_mask]] + [
            df_imp[c].values[train_mask].reshape(-1, 1) for c in feat_cols_list
        ]
        x_tr = np.column_stack(cols)
    else:
        x_tr = elo[train_mask].reshape(-1, 1)
    y_tr = y[train_mask].astype(int)

    pipe = _build_pipeline()
    pipe.fit(x_tr, y_tr)

    if has_feats:
        x_all = np.column_stack(
            [elo] + [df_imp[c].values.reshape(-1, 1) for c in feat_cols_list]
        )
    else:
        x_all = elo.reshape(-1, 1)
    base_prob = pipe.predict_proba(x_all)[:, 1]
    base_logit = _logit(base_prob)

    home_qb_adj = df.get("home_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    away_qb_adj = df.get("away_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    gate = _build_gate_mask(df)

    capped_h = np.clip(home_qb_adj, -OVERLAY_CAP, OVERLAY_CAP)
    capped_a = np.clip(away_qb_adj, -OVERLAY_CAP, OVERLAY_CAP)
    net_adj = capped_h - capped_a
    overlay = OVERLAY_GAMMA * net_adj * ELO_TO_LOGIT
    final_logit = base_logit + overlay * gate.astype(float)
    return _sigmoid(final_logit)


def _fold_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Compute log loss, ECE, MCE for a single fold."""
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    metrics = compute_classification_metrics(y_t, y_p)
    cal = ece_mce(y_t, y_p)
    brier_dec = brier_decomposition(y_t, y_p)
    return {
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier_score"],
        "accuracy": metrics["accuracy"],
        "auc": metrics["roc_auc"],
        "ece": cal["ece"],
        "mce": cal["mce"],
        "reliability": brier_dec["reliability"],
        "n": int(valid.sum()),
    }


def run_qb_lift_experiment(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = "reports/experiments/qb_lift.md",
) -> str:
    """Run fold-safe QB Lift experiment."""
    print("=== QB Lift Experiment ===\n")
    print("New pregame-safe data source: rolling QB EPA/dropback from PBP\n")

    # 1. Load data
    fp = Path(ft_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {ft_path}")
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

    # Add QB Lift features
    df = compute_qb_lift_features(df)

    non_neutral = ~df.get(NEUTRAL_COLUMN, pd.Series(False)).fillna(False).values
    mask = df[MODEL_ELIGIBLE_COLUMN].values & non_neutral
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    y_all = df[TARGET_COLUMN].values.astype(float)

    # 2. Build model variants
    variants = [("baseline", None)]
    for name, cols in QB_LIFT_VARIANTS:
        variants.append((name, cols))

    print(f"  Variants: {len(variants)}")

    # 3. Rolling-origin validation
    fold_mask_list = list(_fold_masks(df))
    val_results = [[] for _ in variants]

    for fold_idx, (train_mask, val_mask, val_season) in enumerate(fold_mask_list):
        val_y = y_all[val_mask]
        valid = ~np.isnan(val_y)
        val_y_clean = val_y[valid]
        print(f"\n  Fold {fold_idx + 1}: val={val_season}, {int(valid.sum())} games")

        for v_idx, (v_name, qb_lift_cols) in enumerate(variants):
            val_probs = _v3_probs(df, train_mask, qb_lift_cols=qb_lift_cols)
            val_p_clean = val_probs[val_mask][valid]
            m = _fold_metrics(val_y_clean, val_p_clean)
            val_results[v_idx].append(m)

    # 4. Average across folds
    variant_scores = []
    for v_idx, (v_name, cols) in enumerate(variants):
        folds = val_results[v_idx]
        avg_ll = float(np.mean([f["log_loss"] for f in folds]))
        avg_ece = float(np.mean([f["ece"] for f in folds]))
        avg_mce = float(np.mean([f["mce"] for f in folds]))
        variant_scores.append({
            "name": v_name,
            "cols": cols or [],
            "avg_val_ll": avg_ll,
            "avg_ece": avg_ece,
            "avg_mce": avg_mce,
            "folds": folds,
        })

    variant_scores.sort(key=lambda x: x["avg_val_ll"])
    baseline_score = [s for s in variant_scores if s["name"] == "baseline"][0]
    baseline_ll = baseline_score["avg_val_ll"]
    best = variant_scores[0]
    beats_val = best["avg_val_ll"] < baseline_ll

    print("\n\n  Results by variant:")
    print(f"  {'Variant':<25} {'Val LL':>8} {'ECE':>8} {'MCE':>8}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8}")
    for vs in variant_scores:
        m = " ← BEST" if vs is best else ""
        print(f"  {vs['name']:<25} {vs['avg_val_ll']:>8.4f} "
              f"{vs['avg_ece']:>8.4f} {vs['avg_mce']:>8.4f}{m}")

    print(f"\n  Baseline val LL: {baseline_ll:.4f}")
    print(f"  Best val LL:     {best['avg_val_ll']:.4f} ({best['name']})")
    print(f"  Beats val:       {beats_val}")

    # 5. Evaluate best on holdout
    print("\n  === 2025 Holdout Evaluation ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    train_mask_hold = df["season"].isin(HISTORICAL_SEASONS).values
    hold_y = y_all[hold_mask]
    hold_valid = ~np.isnan(hold_y)
    hold_y_clean = hold_y[hold_valid]

    best_probs = _v3_probs(df, train_mask_hold, qb_lift_cols=best["cols"])
    hold_p_clean = best_probs[hold_mask][hold_valid]
    hold_metrics = _fold_metrics(hold_y_clean, hold_p_clean)

    base_probs = _v3_probs(df, train_mask_hold, qb_lift_cols=None)
    base_p_clean = base_probs[hold_mask][hold_valid]
    base_hold_metrics = _fold_metrics(hold_y_clean, base_p_clean)

    beats_hold = hold_metrics["log_loss"] < INCUMBENT_HOLDOUT_LL
    print(f"  Baseline holdout LL: {base_hold_metrics['log_loss']:.4f}")
    print(f"  Best holdout LL:     {hold_metrics['log_loss']:.4f} ({best['name']})")
    print(f"  Beats holdout:       {beats_hold}")

    # QB-change subset
    qb_h = df.loc[hold_mask, "home_qb_changed"].fillna(0).values.astype(bool)
    qb_a = df.loc[hold_mask, "away_qb_changed"].fillna(0).values.astype(bool)
    qb_change_hold = qb_h | qb_a
    base_qb_p = base_p_clean[qb_change_hold[hold_valid]]
    base_qb_y = hold_y_clean[qb_change_hold[hold_valid]]
    base_qb_ece = ece_mce(base_qb_y, base_qb_p)
    best_qb_p = hold_p_clean[qb_change_hold[hold_valid]]
    best_qb_y = hold_y_clean[qb_change_hold[hold_valid]]
    best_qb_ece = ece_mce(best_qb_y, best_qb_p)
    print(f"\n  QB-change subset (N={int(base_qb_ece['n'])}):")
    print(f"    Baseline ECE: {base_qb_ece['ece']:.4f}")
    print(f"    Best     ECE: {best_qb_ece['ece']:.4f}")

    # 6. Decision
    promoted = beats_val and beats_hold
    print(f"\n  Decision: {'PROMOTED' if promoted else 'REJECTED'}")
    if not promoted:
        if not beats_val:
            print(f"  Reason: does not beat val LL ({best['avg_val_ll']:.4f} >= {baseline_ll:.4f})")
        if not beats_hold:
            ih = INCUMBENT_HOLDOUT_LL
            print(f"  Reason: does not beat hold LL ({hold_metrics['log_loss']:.4f} >= {ih:.4f})")

    # 7. Report
    print(f"\n=== Writing report → {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write
        _w("# QB Lift Experiment\n\n")
        _w(f"*Model: {INCUMBENT_VERSION} + QB Lift*\n\n")

        _w("## Research Question\n\n")
        _w("Do rolling QB efficiency metrics (EPA/dropback, CPOE) improve ")
        _w("on the incumbent, especially for QB-change games?\n\n")

        _w("## Governance Trigger\n\n")
        _w("QB Lift is a **new pregame-safe data source** derived from ")
        _w("play-by-play quarterback efficiency data. It is not a retest ")
        _w("of any rejected feature family. QB depth features (career ")
        _w("starts, win pct) tested previously are unrelated to rolling ")
        _w("PBP-derived efficiency metrics.\n\n")

        _w("## Methods\n\n")
        _w("| Variant | Features |\n")
        _w("|---------|----------|\n")
        _w("| baseline | v3.0.0 unchanged |\n")
        for v_name, cols in variants[1:]:
            _w(f"| {v_name} | {', '.join(cols)} |\n")

        _w("\n## Validation (Rolling-Origin, 3 folds)\n\n")
        _w("| Variant | Avg Val LL | ECE | MCE |\n")
        _w("|---------|-----------|-----|-----|\n")
        for vs in variant_scores:
            m = " ← **SELECTED**" if vs is best else ""
            _w(f"| {vs['name']} | {vs['avg_val_ll']:.4f} "
               f"| {vs['avg_ece']:.4f} | {vs['avg_mce']:.4f} {m}|\n")

        _w("\n### Fold Details (Best)\n\n")
        _w(f"**{best['name']}**\n\n")
        _w("| Fold | Val N | LL | ECE | MCE |\n")
        _w("|------|-------|-----|-----|-----|\n")
        for fi, fm in enumerate(best["folds"]):
            _, _, vs = fold_mask_list[fi]
            _w(f"| {vs} | {fm['n']} | {fm['log_loss']:.4f} "
               f"| {fm['ece']:.4f} | {fm['mce']:.4f} |\n")

        _w("\n### Fold Details (Baseline)\n\n")
        _w("| Fold | Val N | LL | ECE | MCE |\n")
        _w("|------|-------|-----|-----|-----|\n")
        for fi, fm in enumerate(baseline_score["folds"]):
            _, _, vs = fold_mask_list[fi]
            _w(f"| {vs} | {fm['n']} | {fm['log_loss']:.4f} "
               f"| {fm['ece']:.4f} | {fm['mce']:.4f} |\n")

        _w("\n## Holdout (2025)\n\n")
        _w("| Metric | Baseline | Selected |\n")
        _w("|--------|----------|----------|\n")
        _w(f"| Log loss | {base_hold_metrics['log_loss']:.4f} | {hold_metrics['log_loss']:.4f} |\n")
        _w(f"| Brier | {base_hold_metrics['brier']:.4f} | {hold_metrics['brier']:.4f} |\n")
        _w(f"| AUC | {base_hold_metrics['auc']:.4f} | {hold_metrics['auc']:.4f} |\n")
        _w(f"| Accuracy | {base_hold_metrics['accuracy']:.4f} | {hold_metrics['accuracy']:.4f} |\n")
        _w(f"| ECE | {base_hold_metrics['ece']:.4f} | {hold_metrics['ece']:.4f} |\n")
        _w(f"| MCE | {base_hold_metrics['mce']:.4f} | {hold_metrics['mce']:.4f} |\n")
        _w(f"| N | {int(base_hold_metrics['n'])} | {int(hold_metrics['n'])} |\n\n")

        _w("### QB-Change Subset\n\n")
        _w("| Metric | Baseline | Selected |\n")
        _w("|--------|----------|----------|\n")
        _w(f"| N | {int(base_qb_ece['n'])} | {int(best_qb_ece['n'])} |\n")
        _w(f"| ECE | {base_qb_ece['ece']:.4f} | {best_qb_ece['ece']:.4f} |\n")
        _w(f"| MCE | {base_qb_ece['mce']:.4f} | {best_qb_ece['mce']:.4f} |\n\n")

        _w("## Leakage Risk\n\n")
        _w("- QB Lift uses only prior-game data (rolling window, no future).\n")
        _w("- Minimum 10 dropbacks filters out non-QB trick plays.\n")
        _w("- No 2025 holdout data accessed during fold validation.\n")
        _w("- No market features used.\n")
        _w("- No new feature families from the rejected list.\n\n")

        _w("## Decision\n\n")
        if promoted:
            _w("**✅ PROMOTED** — beats baseline on both validation ")
            _w(f"({best['avg_val_ll']:.4f} vs {baseline_ll:.4f}) ")
            _w(f"and holdout ({hold_metrics['log_loss']:.4f} vs ")
            _w(f"{INCUMBENT_HOLDOUT_LL:.4f}).\n\n")
        else:
            _w("**❌ REJECTED** — no variant beats baseline ")
            _w("on both validation and holdout. ")
            _w(f"Val LL: {best['avg_val_ll']:.4f} vs {baseline_ll:.4f}. ")
            _w(f"Holdout LL: {hold_metrics['log_loss']:.4f} vs ")
            _w(f"{INCUMBENT_HOLDOUT_LL:.4f}.\n\n")

    print(f"\nReport: {rp}")
    return str(report_path)
