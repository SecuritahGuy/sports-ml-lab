"""Uncertainty/calibration audit for the v3.0.0 Frozen QB Overlay champion.

Computes:
  - ECE (Expected Calibration Error) and MCE (Maximum Calibration Error)
  - Brier score decomposition (uncertainty, resolution, reliability)
  - Reliability diagram (text-based ASCII)
  - Confidence distribution / sharpness histogram
  - Over/underconfidence rates
  - Subset-specific calibration (QB-change, early season, extreme probs)
  - Calibration stability across validation folds

No network access required. Works with existing feature table.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

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
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
N_BINS = 10
REPORT_WIDTH = 72


# ── Core calibration metrics ──


def ece_mce(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS,
) -> Dict:
    """Expected Calibration Error and Maximum Calibration Error.

    Equal-width bins over [0, 1). Returns per-bucket details plus
    weighted-average ECE and max-over-buckets MCE.
    """
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    n = len(y_t)
    eps = 1e-12
    y_p = np.clip(y_p, eps, 1.0 - eps)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.floor(y_p * n_bins).astype(int), 0, n_bins - 1)
    buckets = []
    total_cal_error = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        k = int(mask.sum())
        if k == 0:
            continue
        pred = float(y_p[mask].mean())
        actual = float(y_t[mask].mean())
        cal_err = abs(pred - actual)
        total_cal_error += cal_err * k
        buckets.append({
            "bin": f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})",
            "n": k,
            "mean_pred": round(pred, 4),
            "mean_actual": round(actual, 4),
            "cal_error": round(cal_err, 4),
        })
    ece_val = total_cal_error / n if n > 0 else float("nan")
    mce_val = max(b["cal_error"] for b in buckets) if buckets else float("nan")
    return {
        "ece": round(ece_val, 4),
        "mce": round(mce_val, 4),
        "n_bins": n_bins,
        "buckets": buckets,
        "n": n,
    }


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Decompose Brier score into uncertainty, resolution, reliability.

    Brier = Uncertainty - Resolution + Reliability

    Reference: Murphy (1973), "A New Vector Partition of the Probability Score"
    """
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    n = len(y_t)

    base_rate = float(y_t.mean())
    brier_raw = float(np.mean((y_t - y_p) ** 2))

    # Uncertainty: base_rate * (1 - base_rate)
    uncertainty = base_rate * (1.0 - base_rate)

    # Murphy decomposition via equal-width bins
    n_bins = N_BINS
    indices = np.clip(np.floor(y_p * n_bins).astype(int), 0, n_bins - 1)
    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        mask = indices == i
        k = int(mask.sum())
        if k == 0:
            continue
        pred = y_p[mask].mean()
        actual = y_t[mask].mean()
        reliability += ((pred - actual) ** 2) * k
        resolution += ((actual - base_rate) ** 2) * k
    reliability /= n
    resolution /= n

    # Brier = Uncertainty - Resolution + Reliability
    brier_decomp = uncertainty - resolution + reliability

    return {
        "brier_score": round(brier_raw, 4),
        "uncertainty": round(uncertainty, 4),
        "resolution": round(resolution, 4),
        "reliability": round(reliability, 4),
        "brier_decomposed": round(brier_decomp, 4),
    }


def sharpness_buckets(y_prob: np.ndarray) -> Dict:
    """Sharpness: distribution of predicted probabilities across 10 bins."""
    y_p = y_prob[~np.isnan(y_prob)]
    n = len(y_p)
    bin_edges = np.linspace(0.0, 1.0, 11)
    counts = []
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        k = int(((y_p >= lo) & (y_p < hi)).sum())
        pct = round(k / n * 100, 1) if n > 0 else 0.0
        counts.append({
            "bin": f"[{lo:.1f}, {hi:.1f})",
            "n": k,
            "pct": pct,
        })
    # Edge: predictions exactly 1.0
    k_one = int((y_p == 1.0).sum())
    if k_one > 0:
        counts[-1]["n"] += k_one
        counts[-1]["pct"] = round((counts[-1]["n"]) / n * 100, 1)
    return {"bins": counts, "n": n}


def home_favorite_directional_error(y_true, y_prob):
    """Home-favorite directional error (predictions > 0.5 only).

    Overconfident: pred > actual (for home-favored predictions > 0.5)
    Underconfident: pred < actual (for home-favored predictions > 0.5)
    This is NOT a general calibration over/underconfidence metric.
    It only counts predictions where the model favors the home team
    (prob > 0.5), and within those, whether the model was too
    confident (overconfident) or too timid (underconfident).
    """
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    n = len(y_t)
    over = int(((y_p > y_t) & (y_p > 0.5)).sum())
    under = int(((y_p < y_t) & (y_p > 0.5)).sum())
    correct = int(((y_p >= 0.5) == y_t).sum())
    return {
        "n": n,
        "overconfident": over,
        "overconfidence_rate": round(over / n, 4) if n > 0 else 0.0,
        "underconfident": under,
        "underconfidence_rate": round(under / n, 4) if n > 0 else 0.0,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n > 0 else 0.0,
    }


def reliability_diagram_text(buckets: List[Dict], width: int = 50) -> str:
    """Build an ASCII reliability diagram."""
    lines = []
    lines.append("Reliability Diagram")
    lines.append("  (bars show fraction of positives per bucket; ideal = diagonal)")
    lines.append("")
    lines.append(f"{'Bucket':<16} {'N':>5} {'Pred':>6} {'Actual':>7} {'Err':>5}  Chart")
    lines.append("-" * (16 + 5 + 6 + 7 + 5 + 3 + width))
    for b in buckets:
        pred = b["mean_pred"]
        actual = b["mean_actual"]
        err = b["cal_error"]
        n = b["n"]
        bar_len = max(1, int(actual * width))
        bar = "█" * bar_len + "░" * (width - bar_len)
        lines.append(
            f"{b['bin']:<16} {n:>5} {pred:<6.3f} {actual:<7.3f} {err:<5.3f} {bar}"
        )
    lines.append("")
    # Add ideal line indicator
    ideal = "█" * max(1, int(0.5 * width)) + "░" * ((width - int(0.5 * width)))
    lines.append(f"{'Ideal (p=actual)':<39} {ideal}")
    lines.append("")
    return "\n".join(lines)


def _build_gate_mask(df: pd.DataFrame) -> np.ndarray:
    h_changed = df.get("home_qb_changed", pd.Series(0)).values.astype(float)
    a_changed = df.get("away_qb_changed", pd.Series(0)).values.astype(float)
    h_starts = df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    a_starts = df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    return (h_changed == 1) | (a_changed == 1) | (h_starts < 17) | (a_starts < 17)


# ── V3.0.0 incumbent probability (fitted on 2021-2024) ──


def _incumbent_probs(df: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    """Fit v3.0.0 on train_mask, return incumbent probability for all rows."""
    elo = df["elo_prob"].values.astype(float)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    y = df[TARGET_COLUMN].values.astype(float)
    has_feats = len(feat_cols) > 0

    if has_feats:
        x_tr = np.column_stack([elo[train_mask], df[feat_cols].values[train_mask]])
    else:
        x_tr = elo[train_mask].reshape(-1, 1)
    y_tr = y[train_mask].astype(int)

    pipe = _build_pipeline()
    pipe.fit(x_tr, y_tr)

    if has_feats:
        x_all = np.column_stack([elo, df[feat_cols].values])
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


# ── Per-fold validation calibration ──


def _fold_masks(df: pd.DataFrame):
    """Standard rolling-origin folds: train expands, val is next year."""
    folds = [
        ([2021], 2022),
        ([2021, 2022], 2023),
        ([2021, 2022, 2023], 2024),
    ]
    for train_seasons, val_season in folds:
        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values
        yield train_mask, val_mask, val_season


# ── Subset analysis ──


def _subset_analysis(y_v: np.ndarray, p_v: np.ndarray, label: str) -> Dict:
    """Run full calibration audit on a subset's y and p arrays."""
    n = len(y_v)
    if n == 0:
        return {"label": label, "n": 0}
    metrics = compute_classification_metrics(y_v, p_v)
    ece_result = ece_mce(y_v, p_v)
    brier_dec = brier_decomposition(y_v, p_v)
    oc = home_favorite_directional_error(y_v, p_v)
    sharp = sharpness_buckets(p_v)
    return {
        "label": label,
        "n": n,
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier_score"],
        "accuracy": metrics["accuracy"],
        "auc": metrics["roc_auc"],
        "ece": ece_result["ece"],
        "mce": ece_result["mce"],
        "uncertainty": brier_dec["uncertainty"],
        "resolution": brier_dec["resolution"],
        "reliability": brier_dec["reliability"],
        "overconfidence_rate": oc["overconfidence_rate"],
        "underconfidence_rate": oc["underconfidence_rate"],
        "sharpness_bins": sharp,
    }


def run_calibration_audit(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = "reports/experiments/calibration_audit.md",
) -> str:
    """Run comprehensive calibration and uncertainty audit.

    No network access. Uses existing feature table.
    """
    print("=== Calibration & Uncertainty Audit ===\n")

    # ── 1. Load data and build features ──
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

    y = df[TARGET_COLUMN].values.astype(float)

    # ── 2. Holdout (2025) calibration audit ──
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    train_mask_hold = df["season"].isin(HISTORICAL_SEASONS).values
    print(f"  Training: {HISTORICAL_SEASONS} ({int(train_mask_hold.sum())} games)")
    print(f"  Holdout:  {HOLDOUT_SEASON} ({int(hold_mask.sum())} games)")

    hold_probs = _incumbent_probs(df, train_mask_hold)
    hold_valid = ~np.isnan(y[hold_mask])
    hold_y = y[hold_mask][hold_valid]
    hold_p = hold_probs[hold_mask][hold_valid]

    # Overall holdout metrics
    hold_metrics = compute_classification_metrics(hold_y, hold_p)
    hold_ece = ece_mce(hold_y, hold_p)
    hold_brier = brier_decomposition(hold_y, hold_p)
    hold_oc = home_favorite_directional_error(hold_y, hold_p)
    hold_sharp = sharpness_buckets(hold_p)
    diagram = reliability_diagram_text(hold_ece["buckets"])

    print(f"\n  Holdout LL: {hold_metrics['log_loss']:.4f}")
    print(f"  Holdout Brier: {hold_metrics['brier_score']:.4f}")
    print(f"  ECE: {hold_ece['ece']:.4f}")
    print(f"  MCE: {hold_ece['mce']:.4f}")

    # ── 3. Subset analysis ──
    qb_h = df.loc[hold_mask, "home_qb_changed"].fillna(0).values.astype(bool)
    qb_a = df.loc[hold_mask, "away_qb_changed"].fillna(0).values.astype(bool)
    qb_change_mask = (qb_h | qb_a)
    early_mask = df.loc[hold_mask, "week"].values <= 4
    gate = _build_gate_mask(df)[hold_mask]

    high_conf_mask = hold_p >= 0.8
    low_conf_mask = hold_p <= 0.2
    mid_conf_mask = (hold_p >= 0.4) & (hold_p <= 0.6)

    subsets = {
        "All (2025 holdout)": slice(None),
        "QB-change games": qb_change_mask,
        "Non-QB-change games": ~qb_change_mask,
        "Early season (W1-4)": early_mask,
        "Mid-late season (W5+)": ~early_mask,
        "Gate active": gate,
        "Gate inactive": ~gate,
        "High confidence (>=0.8)": high_conf_mask,
        "Low confidence (<=0.2)": low_conf_mask,
        "Mid confidence (0.4-0.6)": mid_conf_mask,
    }

    subset_results = {}
    for label, sub_mask in subsets.items():
        if sub_mask is slice(None):
            s_y, s_p = hold_y, hold_p
        else:
            s_orig = sub_mask[hold_valid] if isinstance(sub_mask, np.ndarray) else sub_mask
            s_y = hold_y[s_orig]
            s_p = hold_p[s_orig]
        if len(s_y) < 3:
            continue
        subset_results[label] = _subset_analysis(s_y, s_p, label)

    # ── 4. Fold stability ──
    print("\n  Per-fold calibration stability...")
    fold_results = []
    for train_mask, val_mask, val_season in _fold_masks(df):
        fold_probs = _incumbent_probs(df, train_mask)
        val_y = y[val_mask][~np.isnan(y[val_mask])]
        val_p = fold_probs[val_mask][~np.isnan(y[val_mask])]
        val_ll = compute_classification_metrics(val_y, val_p)["log_loss"]
        val_ece = ece_mce(val_y, val_p)
        fold_results.append({
            "val_season": val_season,
            "n": len(val_y),
            "val_ll": round(val_ll, 4),
            "ece": val_ece["ece"],
            "mce": val_ece["mce"],
        })
        print(f"    {val_season}: LL={val_ll:.4f} ECE={val_ece['ece']:.4f} "
              f"MCE={val_ece['mce']:.4f}")

    ece_folds = [f["ece"] for f in fold_results]
    mce_folds = [f["mce"] for f in fold_results]

    # ── 5. Write report ──
    print(f"\n=== Writing report → {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# Calibration & Uncertainty Audit\n\n")
        _w(f"*Model: {INCUMBENT_VERSION} Frozen QB Overlay*\n")
        _w(f"*Holdout: {HOLDOUT_SEASON} season*\n\n")

        _w("## Summary\n\n")
        _w("| Metric | Value |\n")
        _w("|--------|-------|\n")
        _w(f"| Holdout LL | {hold_metrics['log_loss']:.4f} |\n")
        _w(f"| Holdout Brier | {hold_metrics['brier_score']:.4f} |\n")
        _w(f"| Accuracy | {hold_metrics['accuracy']:.4f} |\n")
        _w(f"| AUC | {hold_metrics['roc_auc']:.4f} |\n")
        _w(f"| **ECE** | **{hold_ece['ece']:.4f}** |\n")
        _w(f"| **MCE** | **{hold_ece['mce']:.4f}** |\n")
        _w(f"| N (holdout) | {hold_ece['n']} |\n\n")

        _w("## ECE & MCE\n\n")
        _w("Expected Calibration Error (ECE): weighted average of absolute difference ")
        _w("between mean predicted probability and observed frequency across 10 ")
        _w("equal-width bins. Maximum Calibration Error (MCE): max over bins.\n\n")
        _w(f"- **ECE** = {hold_ece['ece']:.4f}\n")
        _w(f"- **MCE** = {hold_ece['mce']:.4f}\n")
        _w(f"- Bins = {hold_ece['n_bins']}\n\n")
        _w("Interpretation:\n")
        _w("- ECE < 0.02: well-calibrated (avg within 2% of true frequency)\n")
        _w("- MCE > 0.10: some bins have meaningful miscalibration\n")
        _w("- Check reliability diagram and per-bucket table below for which bins.\n\n")

        _w("## Reliability Diagram\n\n")
        _w("```\n")
        _w(diagram)
        _w("```\n\n")

        _w("## Per-Bucket Calibration Table\n\n")
        _w("| Bucket | N | Mean Pred | Mean Actual | Cal Error |\n")
        _w("|--------|---|-----------|-------------|-----------|\n")
        for b in hold_ece["buckets"]:
            _w(f"| {b['bin']} | {b['n']} | {b['mean_pred']:.3f} | "
               f"{b['mean_actual']:.3f} | {b['cal_error']:.4f} |\n")

        _w("\n## Brier Score Decomposition\n\n")
        _w("| Component | Value | Description |\n")
        _w("|-----------|-------|-------------|\n")
        _w(f"| Brier score | {hold_brier['brier_score']:.4f} | Raw mean-squared error |\n")
        _w(f"| Uncertainty | {hold_brier['uncertainty']:.4f} | Base-rate variance "
           f"(p̄(1-p̄)); upper bound if always predicting 0.5 |\n")
        _w(f"| Resolution | {hold_brier['resolution']:.4f} | How much predictions "
           f"deviate from base rate by subgroup |\n")
        _w(f"| Reliability | {hold_brier['reliability']:.4f} | Calibration error "
           f"component; lower is better |\n")
        _w(f"| Decomposed Brier | {hold_brier['brier_decomposed']:.4f} | = Uncertainty "
           f"- Resolution + Reliability |\n\n")

        _w("## Sharpness (Confidence Distribution)\n\n")
        _w("How spread out are the predicted probabilities? ")
        _w("A well-sharpened model concentrates predictions away from 0.5.\n\n")
        _w("| Bin | Count | % of Predictions |\n")
        _w("|-----|-------|-----------------|\n")
        for b in hold_sharp["bins"]:
            bar = "█" * max(1, int(b["pct"] / 3))
            _w(f"| {b['bin']} | {b['n']} | {b['pct']:>5.1f}% {bar}\n")

        _w("\n## Home-Favorite Directional Error\n\n")
        _w("WARNING: This is NOT a general over/underconfidence metric.\n\n")
        _w("Only predictions > 0.5 (model favors home team) are counted.\n\n")
        _w("| Metric | Value |\n")
        _w("|--------|-------|\n")
        _w(f"| Overconfident (pred > actual) | {hold_oc['overconfident']} / {hold_oc['n']} "
           f"({hold_oc['overconfidence_rate']*100:.1f}%) |\n")
        _w(f"| Underconfident (pred < actual) | {hold_oc['underconfident']} / {hold_oc['n']} "
           f"({hold_oc['underconfidence_rate']*100:.1f}%) |\n")
        _w(f"| Correct | {hold_oc['correct']} / {hold_oc['n']} "
           f"({hold_oc['accuracy']*100:.1f}%) |\n\n")
        _w("## Subset-Specific Calibration\n\n")
        _w("| Subset | N | LL | Brier | Acc | AUC | ECE | MCE | Over% | Under% |\n")
        _w("|--------|---|----|-------|-----|-----|-----|-----|-------|--------|\n")
        for label, sr in subset_results.items():
            _w(f"| {label} | {sr['n']} | {sr['log_loss']:.4f} | {sr['brier']:.4f} "
               f"| {sr.get('accuracy', 0):.4f} | {sr.get('auc', 0) or 0:.4f} "
               f"| {sr['ece']:.4f} | {sr['mce']:.4f} "
               f"| {sr['overconfidence_rate']*100:.1f} | {sr['underconfidence_rate']*100:.1f} |\n")

        _w("\n## Fold Stability (Calibration)\n\n")
        _w("| Validation Season | N | Val LL | ECE | MCE |\n")
        _w("|-----------------|---|--------|-----|-----|\n")
        for fr in fold_results:
            _w(f"| {fr['val_season']} | {fr['n']} | {fr['val_ll']:.4f} "
               f"| {fr['ece']:.4f} | {fr['mce']:.4f} |\n")
        if len(ece_folds) >= 3:
            ece_std = round(float(np.std(ece_folds)), 4)
            mce_std = round(float(np.std(mce_folds)), 4)
            _w(f"\nECE across folds: mean={np.mean(ece_folds):.4f} "
               f"std={ece_std} range=[{min(ece_folds):.4f}, {max(ece_folds):.4f}]\n")
            _w(f"MCE across folds: mean={np.mean(mce_folds):.4f} "
               f"std={mce_std} range=[{min(mce_folds):.4f}, {max(mce_folds):.4f}]\n")

        _w("\n## Key Findings\n\n")
        _w("### Calibration Quality\n\n")
        if hold_ece["ece"] < 0.02:
            _w("- **ECE < 0.02**: model is well-calibrated overall.\n")
        else:
            _w(f"- **ECE = {hold_ece['ece']:.4f}**: moderate miscalibration detected.\n")
        if hold_ece["mce"] > 0.10:
            _w(f"- **MCE = {hold_ece['mce']:.4f}**: meaningful miscalibration in at "
               f"least one bin.\n")

        _w("\n### Sharpness\n\n")
        sharp_pct = {b["bin"]: b["pct"] for b in hold_sharp["bins"]}
        extreme_pct = sharp_pct.get("[0.0, 0.1)", 0) + sharp_pct.get("[0.9, 1.0)", 0)
        mid_pct = sharp_pct.get("[0.4, 0.5)", 0) + sharp_pct.get("[0.5, 0.6)", 0)
        _w(f"- Extreme predictions (<0.1 or >=0.9): {extreme_pct:.1f}% of predictions\n")
        _w(f"- Near-50/50 predictions (0.4-0.6): {mid_pct:.1f}% of predictions\n")

        _w("\n### Subset Gaps\n\n")
        for label, sr in sorted(subset_results.items(), key=lambda x: x[1]["ece"], reverse=True):
            if sr["ece"] > 0.05:
                _w(f"- **{label}**: ECE={sr['ece']:.4f} (LL={sr['log_loss']:.4f})\n")

        _w("\n### Fold Stability\n\n")
        if len(ece_folds) >= 2 and max(ece_folds) - min(ece_folds) > 0.03:
            _w(f"- ECE varies across folds (range {max(ece_folds)-min(ece_folds):.4f})")
            _w(" — calibration is not stable across seasons\n")
        else:
            _w(f"- ECE stable across folds (range {max(ece_folds)-min(ece_folds):.4f})")
            _w(" — calibration generalizes\n")

        _w("\n### Known Limitations\n\n")
        _w("- ASCII reliability diagram is text-based; no matplotlib dependency.\n")
        _w("- ECE/MCE use equal-width bins (10). Adaptive binning may give different results.\n")
        _w("- Subset analysis on small N (<50 games) may be noisy.\n")
        _w("- No isotonic or temperature-scaled comparison — this audit covers the ")
        _w("incumbent only.\n")

        _w("\n---\n")
        _w(f"*Report generated by `sportslab calibration-audit`. "
           f"Model: {INCUMBENT_VERSION}, Holdout: {HOLDOUT_SEASON}.*\n")

    print(f"\nReport: {rp}")
    return str(report_path)
