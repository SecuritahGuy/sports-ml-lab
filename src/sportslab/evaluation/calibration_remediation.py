"""Fold-safe calibration remediation for v3.0.0.

Allowed under governance: calibration audit identified a repeatable failure
mode (QB-change ECE=0.2097, MCE=0.5690).

Methods tested (all logit-space post-overlay):
  1. Baseline: v3.0.0 unchanged
  2. Global temperature scaling (T sweep)
  3. Gate-aware T (separate T for gate-active / gate-inactive)
  4. QB-change-aware T (separate T for QB-change / non-QB-change)
  5. Conservative shrinkage toward 0.5 for QB-change games
  6. Conservative shrinkage toward fold base rate for QB-change games
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sportslab.evaluation.calibration_audit import (
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
    _build_gate_mask,
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
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025

INCUMBENT_VAL_LL = 0.6305
INCUMBENT_HOLDOUT_LL = 0.6200

# Temperature sweep centered at 1.0 (no-op)
T_SWEEP = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
SHRINK_ALPHAS = [0.05, 0.10, 0.15, 0.20]


def _temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling: logit / T, then sigmoid.

    T=1.0 is identity. T>1 softens, T<1 sharpens.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    if temperature == 1.0:
        return probs.copy()
    logit_val = _logit(probs)
    scaled = logit_val / temperature
    return _sigmoid(scaled)


def _shrink(probs: np.ndarray, alpha: float, target: float = 0.5) -> np.ndarray:
    """Shrink probs toward target: (1-a)*prob + a*target."""
    if not 0 <= alpha <= 1:
        raise ValueError(f"Alpha must be in [0,1], got {alpha}")
    return (1.0 - alpha) * probs + alpha * target


def _v3_probs(df: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    """Return v3.0.0 final probability (Platt + QB overlay) for all rows."""
    elo = df["elo_prob"].values.astype(float)
    feat_cols_list = [c for c in FEATURE_COLS if c in df.columns]
    y = df[TARGET_COLUMN].values.astype(float)
    has_feats = len(feat_cols_list) > 0

    if has_feats:
        x_tr = np.column_stack([elo[train_mask], df[feat_cols_list].values[train_mask]])
    else:
        x_tr = elo[train_mask].reshape(-1, 1)
    y_tr = y[train_mask].astype(int)

    pipe = _build_pipeline()
    pipe.fit(x_tr, y_tr)

    if has_feats:
        x_all = np.column_stack([elo, df[feat_cols_list].values])
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
    return _sigmoid(final_logit), gate, base_prob


def _fold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> Dict:
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


def _run_method(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    method: str,
    y_train_vals: Optional[np.ndarray] = None,
    **params,
) -> np.ndarray:
    """Run a single method variant on a fold, return val probabilities."""
    all_probs, gate, base_probs = _v3_probs(df, train_mask)
    val_probs = all_probs[val_mask]
    val_gate = gate[val_mask]

    # Get QB-change mask for val
    qb_h = df.loc[val_mask, "home_qb_changed"].fillna(0).values.astype(bool)
    qb_a = df.loc[val_mask, "away_qb_changed"].fillna(0).values.astype(bool)
    qb_change = qb_h | qb_a

    if method == "baseline":
        return val_probs

    if method == "global_temperature":
        t = params["temperature"]
        return _temperature_scale(val_probs, t)

    if method == "gate_temperature":
        t_gate = params.get("t_gate", 1.0)
        t_nogate = params.get("t_nogate", 1.0)
        result = val_probs.copy()
        result[val_gate] = _temperature_scale(val_probs[val_gate], t_gate)
        result[~val_gate] = _temperature_scale(val_probs[~val_gate], t_nogate)
        return result

    if method == "qb_temperature":
        t_qb = params.get("t_qb", 1.0)
        t_noqb = params.get("t_noqb", 1.0)
        result = val_probs.copy()
        result[qb_change] = _temperature_scale(val_probs[qb_change], t_qb)
        result[~qb_change] = _temperature_scale(val_probs[~qb_change], t_noqb)
        return result

    if method == "qb_shrink":
        alpha = params["alpha"]
        result = val_probs.copy()
        result[qb_change] = _shrink(val_probs[qb_change], alpha, target=0.5)
        return result

    if method == "qb_shrink_baserate":
        alpha = params["alpha"]
        br = float(y_train_vals.mean()) if y_train_vals is not None else 0.5
        result = val_probs.copy()
        result[qb_change] = _shrink(val_probs[qb_change], alpha, target=br)
        return result

    raise ValueError(f"Unknown method: {method}")


def _build_variants() -> List[Dict]:
    """Build list of (method_name, params_dict) to test."""
    variants = [{"method": "baseline", "params": {}, "label": "Baseline (v3.0.0)"}]

    # Global temperature
    for t in T_SWEEP:
        if t == 1.0:
            continue  # same as baseline
        variants.append({
            "method": "global_temperature",
            "params": {"temperature": t},
            "label": f"Global T={t}",
        })

    # Gate-aware temperature (limited combos)
    for t_g in [1.0, 1.5, 2.0, 3.0, 5.0]:
        for t_ng in [1.0, 1.5, 2.0, 3.0, 5.0]:
            if t_g == 1.0 and t_ng == 1.0:
                continue
            variants.append({
                "method": "gate_temperature",
                "params": {"t_gate": t_g, "t_nogate": t_ng},
                "label": f"Gate T: gate={t_g}, no_gate={t_ng}",
            })

    # QB-change-aware temperature (limited combos)
    for t_qb in [1.0, 1.5, 2.0, 3.0, 5.0]:
        for t_noqb in [1.0, 1.5, 2.0, 3.0, 5.0]:
            if t_qb == 1.0 and t_noqb == 1.0:
                continue
            variants.append({
                "method": "qb_temperature",
                "params": {"t_qb": t_qb, "t_noqb": t_noqb},
                "label": f"QB T: qb={t_qb}, no_qb={t_noqb}",
            })

    # Shrinkage for QB-change
    for alpha in SHRINK_ALPHAS:
        variants.append({
            "method": "qb_shrink",
            "params": {"alpha": alpha},
            "label": f"QB shrink →0.5 α={alpha}",
        })
        variants.append({
            "method": "qb_shrink_baserate",
            "params": {"alpha": alpha},
            "label": f"QB shrink →base α={alpha}",
        })

    return variants


def run_calibration_remediation(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = "reports/experiments/calibration_remediation.md",
) -> str:
    """Run fold-safe calibration remediation experiment."""
    print("=== Calibration Remediation ===\n")
    print("Allowed: calibration audit found QB-change ECE=0.2097 (repeatable)\n")

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

    non_neutral = ~df.get(NEUTRAL_COLUMN, pd.Series(False)).fillna(False).values
    mask = df[MODEL_ELIGIBLE_COLUMN].values & non_neutral
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    y_all = df[TARGET_COLUMN].values.astype(float)

    # 2. Build variants
    variants = _build_variants()
    print(f"  Variants: {len(variants)} (1 baseline + {len(variants) - 1} challengers)")

    # 3. Rolling-origin fold validation
    fold_mask_list = list(_fold_masks(df))
    print(f"\n  Folds: {len(fold_mask_list)}")
    for _, _, val_season in fold_mask_list:
        print(f"    → val {val_season}")

    val_results = [[] for _ in variants]
    for fold_idx, (train_mask, val_mask, val_season) in enumerate(fold_mask_list):
        val_y = y_all[val_mask]
        valid = ~np.isnan(val_y)
        val_y_clean = val_y[valid]
        y_train_vals = y_all[train_mask][~np.isnan(y_all[train_mask])]
        print(f"\n  Fold {fold_idx + 1}: val={val_season}, {int(valid.sum())} games")

        for v_idx, v in enumerate(variants):
            val_probs = _run_method(
                df, train_mask, val_mask, v["method"],
                y_train_vals=y_train_vals, **v["params"],
            )
            val_p_clean = val_probs[valid]
            m = _fold_metrics(val_y_clean, val_p_clean)
            val_results[v_idx].append(m)

    # 4. Average across folds
    variant_scores = []
    for v_idx, v in enumerate(variants):
        folds = val_results[v_idx]
        avg_ll = float(np.mean([f["log_loss"] for f in folds]))
        avg_ece = float(np.mean([f["ece"] for f in folds]))
        avg_mce = float(np.mean([f["mce"] for f in folds]))
        avg_rel = float(np.mean([f["reliability"] for f in folds]))
        variant_scores.append({
            "label": v["label"],
            "method": v["method"],
            "params": v["params"],
            "avg_val_ll": avg_ll,
            "avg_ece": avg_ece,
            "avg_mce": avg_mce,
            "avg_reliability": avg_rel,
            "folds": folds,
        })

    # Sort by avg val LL
    variant_scores.sort(key=lambda x: x["avg_val_ll"])

    print("\n\n  Top 10 by avg validation LL:")
    print(f"  {'Variant':<50} {'Val LL':>8} {'ECE':>8} {'MCE':>8}")
    print(f"  {'─' * 50} {'─' * 8} {'─' * 8} {'─' * 8}")
    for vs in variant_scores[:10]:
        print(f"  {vs['label']:<50} {vs['avg_val_ll']:>8.4f} "
              f"{vs['avg_ece']:>8.4f} {vs['avg_mce']:>8.4f}")

    # 5. Select best by avg val LL (must beat baseline validation)
    baseline_score = [s for s in variant_scores if s["method"] == "baseline"][0]
    baseline_ll = baseline_score["avg_val_ll"]

    # Best is first (sorted by avg_val_ll ascending)
    best = variant_scores[0]
    beats_val = best["avg_val_ll"] < baseline_ll

    print(f"\n  Baseline val LL: {baseline_ll:.4f}")
    print(f"  Best val LL:     {best['avg_val_ll']:.4f} ({best['label']})")
    print(f"  Beats val:       {beats_val}")

    # 6. Evaluate best on 2025 holdout (once)
    print("\n  === 2025 Holdout Evaluation ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    train_mask_hold = df["season"].isin(HISTORICAL_SEASONS).values
    hold_y = y_all[hold_mask]
    hold_valid = ~np.isnan(hold_y)
    hold_y_clean = hold_y[hold_valid]

    hold_probs = _run_method(df, train_mask_hold, hold_mask, best["method"], **best["params"])
    hold_p_clean = hold_probs[hold_valid]
    hold_metrics = _fold_metrics(hold_y_clean, hold_p_clean)
    hold_ll = hold_metrics["log_loss"]

    # Baseline holdout
    base_probs = _run_method(df, train_mask_hold, hold_mask, "baseline", **{})
    base_p_clean = base_probs[hold_valid]
    base_hold_metrics = _fold_metrics(hold_y_clean, base_p_clean)
    base_hold_ll = base_hold_metrics["log_loss"]

    beats_hold = hold_ll < INCUMBENT_HOLDOUT_LL
    print(f"  Baseline holdout LL: {base_hold_ll:.4f}")
    print(f"  Best holdout LL:     {hold_ll:.4f} ({best['label']})")
    print(f"  Beats holdout:       {beats_hold}")

    # 7. QB-change subset calibration (before/after)
    qb_h = df.loc[hold_mask, "home_qb_changed"].fillna(0).values.astype(bool)
    qb_a = df.loc[hold_mask, "away_qb_changed"].fillna(0).values.astype(bool)
    qb_change_hold = qb_h | qb_a

    # Baseline on QB-change subset
    base_qb_p = base_probs[qb_change_hold[hold_valid]]
    base_qb_y = hold_y_clean[qb_change_hold[hold_valid]]
    base_qb_ece = ece_mce(base_qb_y, base_qb_p)

    # Best on QB-change subset
    best_qb_p = hold_probs[qb_change_hold[hold_valid]]
    best_qb_y = hold_y_clean[qb_change_hold[hold_valid]]
    best_qb_ece = ece_mce(best_qb_y, best_qb_p)

    print(f"\n  QB-change subset (N={int(base_qb_ece['n'])}):")
    print(f"    Baseline ECE: {base_qb_ece['ece']:.4f}, MCE: {base_qb_ece['mce']:.4f}")
    print(f"    Best     ECE: {best_qb_ece['ece']:.4f}, MCE: {best_qb_ece['mce']:.4f}")

    # 8. Decision
    promoted = beats_val and beats_hold
    decision = "PROMOTED" if promoted else "REJECTED"

    print(f"\n  Decision: {decision}")
    if promoted:
        print(f"  New incumbent: {best['label']}")
    else:
        if not beats_val:
            bl = baseline_ll
            print(f"  Reason: does not beat val LL ({best['avg_val_ll']:.4f} >= {bl:.4f})")
        if not beats_hold:
            ih = INCUMBENT_HOLDOUT_LL
            print(f"  Reason: does not beat hold LL ({hold_ll:.4f} >= {ih:.4f})")

    # 9. Write report
    print(f"\n=== Writing report → {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# Calibration Remediation\n\n")
        _w(f"*Model: {INCUMBENT_VERSION} Frozen QB Overlay*\n\n")

        _w("## Research Question\n\n")
        _w("Can we reduce calibration error for v3.0.0, especially QB-change / ")
        _w("gate-active games, without worsening validation log loss or holdout log loss?\n\n")

        _w("## Governance Trigger\n\n")
        _w("Calibration audit (ECE=0.0628, MCE=0.1343) identified a repeatable ")
        _w("failure mode: QB-change games have ECE=0.2097 and MCE=0.5690 on ")
        _w("2025 holdout (N=55). This qualifies for remediation under the ")
        _w("\"repeatable failure mode\" trigger.\n\n")

        _w("## Methods Tested\n\n")
        _w("| # | Method | Variants |\n")
        _w("|---|--------|----------|\n")
        _w("| 1 | **Baseline** | v3.0.0 unchanged |\n")
        _w("| 2 | **Global temperature scaling** | "
            "T ∈ {0.8, 0.9, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0} |\n")
        _w("| 3 | **Gate-aware T** | separate T for gate-active / gate-inactive |\n")
        _w("| 4 | **QB-change-aware T** | separate T for QB-change / non-QB-change |\n")
        _w("| 5 | **QB-change shrink →0.5** | α ∈ {0.05, 0.10, 0.15, 0.20} |\n")
        _w("| 6 | **QB-change shrink →base** | α ∈ {0.05, 0.10, 0.15, 0.20} |\n\n")

        _w(f"Total variants: {len(variants)}\n\n")

        _w("## Validation (Rolling-Origin, 3 folds)\n\n")
        _w("Selection criterion: average validation log loss.\n\n")

        # Summary table: top 10 + all baselines/shrinks
        _w("| Variant | Avg Val LL | ECE | MCE | Rel |\n")
        _w("|---------|-----------|-----|-----|-----|\n")
        for vs in variant_scores:
            marker = " ← **SELECTED**" if vs is best else ""
            _w(f"| {vs['label']} | {vs['avg_val_ll']:.4f} "
               f"| {vs['avg_ece']:.4f} | {vs['avg_mce']:.4f} "
               f"| {vs['avg_reliability']:.4f} {marker}|\n")

        _w("\n### Fold Details (Best Variant)\n\n")
        _w(f"**{best['label']}**\n\n")
        _w("| Fold | Val N | LL | ECE | MCE |\n")
        _w("|------|-------|-----|-----|-----|\n")
        for fi, fm in enumerate(best["folds"]):
            _, _, val_season = fold_mask_list[fi]
            _w(f"| {val_season} | {fm['n']} | {fm['log_loss']:.4f} | "
               f"{fm['ece']:.4f} | {fm['mce']:.4f} |\n")

        _w("\n### Fold Details (Baseline)\n\n")
        _w("| Fold | Val N | LL | ECE | MCE |\n")
        _w("|------|-------|-----|-----|-----|\n")
        for fi, fm in enumerate(baseline_score["folds"]):
            _, _, val_season = fold_mask_list[fi]
            _w(f"| {val_season} | {fm['n']} | {fm['log_loss']:.4f} | "
               f"{fm['ece']:.4f} | {fm['mce']:.4f} |\n")

        _w("\n## Holdout (2025) Results\n\n")
        _w("| Metric | Baseline | Selected |\n")
        _w("|--------|----------|----------|\n")
        _w(f"| Log loss | {base_hold_metrics['log_loss']:.4f} | {hold_metrics['log_loss']:.4f} |\n")
        _w(f"| Brier | {base_hold_metrics['brier']:.4f} | {hold_metrics['brier']:.4f} |\n")
        _w(f"| AUC | {base_hold_metrics['auc']:.4f} | {hold_metrics['auc']:.4f} |\n")
        _w(f"| Accuracy | {base_hold_metrics['accuracy']:.4f} | {hold_metrics['accuracy']:.4f} |\n")
        _w(f"| ECE | {base_hold_metrics['ece']:.4f} | {hold_metrics['ece']:.4f} |\n")
        _w(f"| MCE | {base_hold_metrics['mce']:.4f} | {hold_metrics['mce']:.4f} |\n")
        _w(f"| Reliability | {base_hold_metrics['reliability']:.4f} | "
           f"{hold_metrics['reliability']:.4f} |\n")
        _w(f"| N | {int(base_hold_metrics['n'])} | {int(hold_metrics['n'])} |\n\n")

        _w("### QB-Change Subset Calibration\n\n")
        _w("| Metric | Baseline | Selected |\n")
        _w("|--------|----------|----------|\n")
        _w(f"| N | {int(base_qb_ece['n'])} | {int(best_qb_ece['n'])} |\n")
        _w(f"| ECE | {base_qb_ece['ece']:.4f} | {best_qb_ece['ece']:.4f} |\n")
        _w(f"| MCE | {base_qb_ece['mce']:.4f} | {best_qb_ece['mce']:.4f} |\n\n")

        _w("## Leakage Risk\n\n")
        _w("- Temperature scaling is fit to validation data per-fold (0 parameters — ")
        _w("single T is a hyperparameter sweep, not a learned parameter).\n")
        _w("- No 2025 holdout data accessed during fold validation.\n")
        _w("- No market features used as model inputs.\n")
        _w("- No new feature families introduced.\n")
        _w("- Selection by validation LL, not by ECE/MCE or holdout.\n\n")

        _w("## Decision\n\n")

        if promoted:
            _w(f"**✅ PROMOTED** — **{best['label']}** beats baseline on both ")
            _w(f"validation ({best['avg_val_ll']:.4f} vs {baseline_ll:.4f}) ")
            _w(f"and holdout ({hold_ll:.4f} vs {INCUMBENT_HOLDOUT_LL:.4f}).\n\n")
        else:
            _w("**❌ REJECTED** — no variant beats baseline on both validation ")
            _w("and holdout. ")
            _w(f"Val LL: {best['avg_val_ll']:.4f} vs {baseline_ll:.4f}. ")
            _w(f"Holdout LL: {hold_ll:.4f} vs {INCUMBENT_HOLDOUT_LL:.4f}.\n\n")

        _w(f"Best variant: **{best['label']}**\n\n")

    print(f"\nReport: {rp}")
    return str(report_path)

