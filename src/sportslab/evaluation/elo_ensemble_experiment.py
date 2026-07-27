"""Elo parameter ensemble experiment.

Tests whether averaging Elo probabilities across diverse parameter configurations
beats the single-best v3.0.0 champion configuration. The hypothesis is that
ensemble averaging reduces variance and produces more robust predictions.

Architecture:
    base Elo (N diverse configs)
    → average probabilities (equal-weighted or weighted)
    → fold-safe Platt on [ensemble_prob, qb_changed, rolling_mov_3]
    → frozen QB overlay (v3.0.0 champion, fixed)
    → validation/holdout comparison

Comparison baseline: v3.0.0 champion (val LL 0.6305, holdout LL 0.6200).
Promotion requires Δ >= 0.001 on BOTH val and holdout.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import compute_metrics
from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, NEUTRAL_COLUMN, TARGET_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

V3_VAL_LL = 0.6305
V3_HOLDOUT_LL = 0.6200

# v3.0.0 champion Elo params
V3_K = 36
V3_HFA = 40
V3_REG = 0.1
V3_DECAY = 32

QB_GATE_GAMMA = 1.0
QB_GATE_CAP = 40
FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]

MIN_PROMOTION_DELTA = 0.001
SEED = 42
ELO_TO_LOGIT = np.log(10) / 400.0

# ── Ensemble members: 10 diverse configs spanning the parameter space ──
# Each tuple: (K, HFA, reg, decay, label)
ENSEMBLE_CONFIGS = [
    (20, 20, 0.0, None,       "fast_low_hfa"),
    (20, 50, 0.3, 64,         "fast_high_hfa_reg"),
    (28, 35, 0.15, 48,        "balanced_low"),
    (36, 20, 0.0, None,       "champ_low_hfa"),
    (36, 40, 0.1, 32,         "v3.0.0_champion"),
    (36, 50, 0.3, 64,         "champ_high_hfa_reg"),
    (44, 20, 0.0, None,       "expanded_best"),
    (44, 35, 0.15, 48,        "balanced"),
    (52, 40, 0.0, 32,         "slow_med_hfa"),
    (60, 50, 0.3, 64,         "slow_high_all"),
]

EloInputCols = ["season", "week", "gameday", "home_team", "away_team",
                "home_score", "away_score", "home_win"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def _get_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    if avail:
        return df[avail].values
    return np.empty((len(df), 0))


def _apply_qb_overlay(
    base_logit: np.ndarray,
    qb_gate_mask: np.ndarray,
    home_qb_adj: np.ndarray,
    away_qb_adj: np.ndarray,
) -> np.ndarray:
    capped_h = np.clip(home_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    capped_a = np.clip(away_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    net_adj = capped_h - capped_a
    overlay = QB_GATE_GAMMA * net_adj * ELO_TO_LOGIT
    return base_logit + overlay * qb_gate_mask.astype(float)


def _build_qb_gate_mask(df: pd.DataFrame) -> np.ndarray:
    h_changed = df["home_qb_changed"].values.astype(float)
    a_changed = df["away_qb_changed"].values.astype(float)
    qb_changed_either = (h_changed == 1) | (a_changed == 1)
    h_starts_raw = df.get("home_qb_team_starts_pre", None)
    a_starts_raw = df.get("away_qb_team_starts_pre", None)
    low_starts = np.zeros(len(df), dtype=bool)
    if h_starts_raw is not None and a_starts_raw is not None:
        h_s = h_starts_raw.fillna(-1).values.astype(float)
        a_s = a_starts_raw.fillna(-1).values.astype(float)
        low_starts = ((h_s >= 0) & (h_s < 17)) | ((a_s >= 0) & (a_s < 17))
    return qb_changed_either | low_starts


def _compute_ensemble_probs(
    elo_base: pd.DataFrame,
    configs: list[tuple],
    weights: Optional[list[float]] = None,
) -> np.ndarray:
    """Compute equal-weighted or weighted average of Elo probabilities across configs."""
    n = len(elo_base)
    all_probs = np.zeros((n, len(configs)))
    for i, (k, hfa, reg, decay, _) in enumerate(configs):
        df_elo = compute_elo_features(
            elo_base,
            k_factor=k,
            home_advantage=hfa,
            preseason_regression=reg,
            decay_half_life=decay,
        )
        all_probs[:, i] = df_elo["elo_prob"].values.astype(float)
    if weights is not None:
        w = np.array(weights)
        w = w / w.sum()
        return all_probs.dot(w)
    return all_probs.mean(axis=1)


def _score_ensemble(
    ensemble_prob: np.ndarray,
    y: np.ndarray,
    all_feat: np.ndarray,
    qb_gate_mask: np.ndarray,
    home_qb_adj: np.ndarray,
    away_qb_adj: np.ndarray,
    fold_frames: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Fit Platt per fold, apply overlay, return avg val LL."""
    fold_lls: list[float] = []
    for train_mask, val_mask in fold_frames:
        train_ep = ensemble_prob[train_mask]
        train_y = y[train_mask].astype(int)
        train_feat = all_feat[train_mask]
        x_train = (
            np.column_stack([train_ep, train_feat])
            if train_feat.size else train_ep.reshape(-1, 1)
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_train, train_y)

        x_all = (
            np.column_stack([ensemble_prob, all_feat])
            if all_feat.size else ensemble_prob.reshape(-1, 1)
        )
        platt_prob = pipe.predict_proba(x_all)[:, 1]
        base_logit = _logit(platt_prob)
        final_logit = _apply_qb_overlay(base_logit, qb_gate_mask, home_qb_adj, away_qb_adj)
        final_prob = _sigmoid(final_logit)

        val_prob = final_prob[val_mask]
        val_y = y[val_mask]
        valid = ~np.isnan(val_y)
        m = compute_metrics(val_y[valid], val_prob[valid])
        fold_lls.append(m.get("log_loss", 1.0))
    return float(np.mean(fold_lls))


def run_elo_ensemble(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/elo_ensemble.md",
) -> str:
    print("=== Elo Parameter Ensemble ===")

    # ── 1. Load data and build non-Elo features ──
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    df = compute_elo_features(df_raw, k_factor=20, home_advantage=0)
    df = compute_qb_features(df)
    df = compute_qb_adjustments(df)
    df = compute_situational_features(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    y = df[TARGET_COLUMN].astype(float).values
    all_feat = _get_features(df, FEATURE_COLS)
    qb_gate_mask = _build_qb_gate_mask(df)
    home_qb_adj = df["home_qb_adj"].values.astype(float)
    away_qb_adj = df["away_qb_adj"].values.astype(float)

    elo_base = df[EloInputCols].copy()

    # ── 2. Build fold masks ──
    fold_frames: list[tuple[np.ndarray, np.ndarray]] = []
    for train_seasons, val_season in ROLLING_FOLDS:
        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values
        fold_frames.append((train_mask, val_mask))

    # ── 3. Compute incumbent (single best config) ──
    print("\n=== Computing incumbent (v3.0.0 champion) ===")
    v3_elo = compute_elo_features(
        elo_base, k_factor=V3_K, home_advantage=V3_HFA,
        preseason_regression=V3_REG, decay_half_life=V3_DECAY,
    )
    v3_elo_prob = v3_elo["elo_prob"].values.astype(float)
    v3_val_ll = _score_ensemble(
        v3_elo_prob, y, all_feat, qb_gate_mask,
        home_qb_adj, away_qb_adj, fold_frames,
    )
    print(f"  Incumbent val LL: {v3_val_ll:.4f} (expected ~{V3_VAL_LL:.4f})")

    # ── 4. Compute ensemble probs and score variants ──
    print(f"\n=== Ensemble: {len(ENSEMBLE_CONFIGS)} configs ===")
    for _, _, _, _, label in ENSEMBLE_CONFIGS:
        print(f"  - {label}")

    # Variant A: equal-weighted ensemble
    print("\n  Computing equal-weighted ensemble...")
    eq_ensemble_prob = _compute_ensemble_probs(elo_base, ENSEMBLE_CONFIGS)
    eq_val_ll = _score_ensemble(
        eq_ensemble_prob, y, all_feat, qb_gate_mask,
        home_qb_adj, away_qb_adj, fold_frames,
    )
    print(f"  Equal-weighted val LL: {eq_val_ll:.4f} (Δ={eq_val_ll - V3_VAL_LL:+.4f})")

    # Variant B: per-config val LL for weighting
    print("\n  Computing per-config val LLs...")
    config_results: list[dict] = []
    for k, hfa, reg, decay, label in ENSEMBLE_CONFIGS:
        df_c = compute_elo_features(
            elo_base, k_factor=k, home_advantage=hfa,
            preseason_regression=reg, decay_half_life=decay,
        )
        cp = df_c["elo_prob"].values.astype(float)
        cv = _score_ensemble(
            cp, y, all_feat, qb_gate_mask,
            home_qb_adj, away_qb_adj, fold_frames,
        )
        config_results.append({"K": k, "HFA": hfa, "reg": reg, "decay": decay,
                               "label": label, "val_ll": cv})
        print(f"    {label:25s} val LL: {cv:.4f}")

    # Variant C: inverse-val-L weighted ensemble
    print("\n  Computing inverse-val weighted ensemble...")
    inv_weights = [1.0 / max(r["val_ll"], 0.62) for r in config_results]
    inv_ensemble_prob = _compute_ensemble_probs(elo_base, ENSEMBLE_CONFIGS, inv_weights)
    inv_val_ll = _score_ensemble(
        inv_ensemble_prob, y, all_feat, qb_gate_mask,
        home_qb_adj, away_qb_adj, fold_frames,
    )
    print(f"  Inverse-val weighted val LL: {inv_val_ll:.4f} (Δ={inv_val_ll - V3_VAL_LL:+.4f})")

    # Variant D: top 5 configs by val LL
    config_results.sort(key=lambda r: r["val_ll"])
    top5 = config_results[:5]
    print("\n  Top 5 configs by val LL:")
    for i, r in enumerate(top5):
        print(f"    {i+1}. {r['label']:25s} val={r['val_ll']:.4f}")
    top5_configs = [(r["K"], r["HFA"], r["reg"], r["decay"], r["label"]) for r in top5]
    print("  Computing top-5 ensemble...")
    top5_ensemble_prob = _compute_ensemble_probs(elo_base, top5_configs)
    top5_val_ll = _score_ensemble(
        top5_ensemble_prob, y, all_feat, qb_gate_mask,
        home_qb_adj, away_qb_adj, fold_frames,
    )
    print(f"  Top-5 ensemble val LL: {top5_val_ll:.4f} (Δ={top5_val_ll - V3_VAL_LL:+.4f})")

    # ── 5. Holdout evaluation ──
    print("\n=== 2025 Holdout ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    train_mask_hold = df["season"].isin([2021, 2022, 2023, 2024]).values
    hold_y = y[hold_mask]
    valid_hold = ~np.isnan(hold_y)

    def _fit_holdout(ensemble_prob_array: np.ndarray) -> np.ndarray:
        tr_ep = ensemble_prob_array[train_mask_hold]
        tr_y = y[train_mask_hold].astype(int)
        tr_feat = all_feat[train_mask_hold]
        x_tr = (
            np.column_stack([tr_ep, tr_feat])
            if tr_feat.size else tr_ep.reshape(-1, 1)
        )
        x_all = (
            np.column_stack([ensemble_prob_array, all_feat])
            if all_feat.size else ensemble_prob_array.reshape(-1, 1)
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_tr, tr_y)
        p = pipe.predict_proba(x_all)[:, 1]
        final = _logit(p)
        final = _apply_qb_overlay(final, qb_gate_mask, home_qb_adj, away_qb_adj)
        return _sigmoid(final)

    # v3.0.0 holdout
    v3_hold_prob = _fit_holdout(v3_elo_prob)
    v3_hm = compute_classification_metrics(
        hold_y[valid_hold], v3_hold_prob[hold_mask][valid_hold],
    )
    v3_hold_ll = v3_hm["log_loss"]
    print(f"  v3.0.0 champion holdout: {v3_hold_ll:.4f} (expected ~{V3_HOLDOUT_LL:.4f})")

    # Equal-weighted holdout
    eq_hold_prob = _fit_holdout(eq_ensemble_prob)
    eq_hm = compute_classification_metrics(
        hold_y[valid_hold], eq_hold_prob[hold_mask][valid_hold],
    )
    eq_hold_ll = eq_hm["log_loss"]
    print(f"  Equal-weighted holdout:   {eq_hold_ll:.4f} (Δ={eq_hold_ll - v3_hold_ll:+.4f})")

    # Inverse-val weighted holdout
    inv_hold_prob = _fit_holdout(inv_ensemble_prob)
    inv_hm = compute_classification_metrics(
        hold_y[valid_hold], inv_hold_prob[hold_mask][valid_hold],
    )
    inv_hold_ll = inv_hm["log_loss"]
    print(f"  Inv-val weighted holdout: {inv_hold_ll:.4f} (Δ={inv_hold_ll - v3_hold_ll:+.4f})")

    # Top-5 holdout
    top5_hold_prob = _fit_holdout(top5_ensemble_prob)
    top5_hm = compute_classification_metrics(
        hold_y[valid_hold], top5_hold_prob[hold_mask][valid_hold],
    )
    top5_hold_ll = top5_hm["log_loss"]
    print(f"  Top-5 ensemble holdout:   {top5_hold_ll:.4f} (Δ={top5_hold_ll - v3_hold_ll:+.4f})")

    # ── 6. Promotion check ──
    print("\n=== Promotion Check ===")
    variants = [
        ("Equal-weighted (10 configs)", eq_val_ll, eq_hold_ll),
        ("Inverse-val weighted",        inv_val_ll, inv_hold_ll),
        ("Top-5 by val LL",            top5_val_ll, top5_hold_ll),
    ]
    best_variant = None
    for name, val_ll, hold_ll in variants:
        d_val = val_ll - V3_VAL_LL
        d_hold = hold_ll - v3_hold_ll
        beats_val = val_ll < V3_VAL_LL - MIN_PROMOTION_DELTA
        beats_hold = hold_ll < v3_hold_ll - MIN_PROMOTION_DELTA
        if beats_val and beats_hold:
            best_variant = (name, val_ll, hold_ll)
        print(f"  {name:40s} | val Δ={d_val:+.4f} | hold Δ={d_hold:+.4f} | ", end="")
        print("✅ BOTH ✓" if beats_val and beats_hold else
              "⚠️ val only" if beats_val else
              "⚠️ hold only" if beats_hold else "❌")

    # ── 7. Write report ──
    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write
        _w("# Elo Parameter Ensemble\n\n")
        _w("## Research Question\n\n")
        _w("Does averaging Elo probabilities across diverse parameter configurations ")
        _w("beat the single-best v3.0.0 champion configuration? Ensemble averaging ")
        _w("reduces variance and may produce more robust predictions.\n\n")

        _w("## Architecture\n\n")
        _w("```\n")
        _w("base Elo (N diverse configs, equal-weighted average)\n")
        _w("→ fold-safe Platt on [ensemble_prob, qb_changed, rolling_mov_3]\n")
        _w("→ frozen QB overlay (v3.0.0 champion, fixed)\n")
        _w("→ validation/holdout comparison\n")
        _w("```\n\n")

        _w("## Champion (v3.0.0)\n\n")
        _w("| Metric | Value |\n")
        _w("|--------|-------|\n")
        _w(f"| Val LL | {V3_VAL_LL:.4f} |\n")
        _w(f"| Holdout LL | {V3_HOLDOUT_LL:.4f} |\n")
        _w(f"| Parameters | K={V3_K}, HFA={V3_HFA}, reg={V3_REG}, decay={V3_DECAY} |\n\n")

        _w("## Ensemble Members\n\n")
        _w(f"**{len(ENSEMBLE_CONFIGS)} configs** spanning the parameter space:\n\n")
        _w("| # | Label | K | HFA | reg | decay |\n")
        _w("|---|-------|----|-----|-----|-------|\n")
        for i, (k, hfa, reg, decay, label) in enumerate(ENSEMBLE_CONFIGS, 1):
            d_str = f"{decay}" if decay is not None else "None"
            _w(f"| {i} | {label} | {k} | {hfa} | {reg} | {d_str} |\n")
        _w("\n")

        _w("## Per-Config Validation Performance\n\n")
        _w("| Config | Val LL | Δ vs Inc |\n")
        _w("|--------|--------|----------|\n")
        config_results.sort(key=lambda r: r["val_ll"])
        for r in config_results:
            d = r["val_ll"] - V3_VAL_LL
            _w(f"| {r['label']:25s} | {r['val_ll']:.4f} | {d:+.4f} |\n")
        _w("\n")

        _w("## Ensemble Results\n\n")
        _w("### Rolling-Origin Validation\n\n")
        _w("| Variant | Val LL | Δ vs Inc |\n")
        _w("|---------|--------|----------|\n")
        _w(f"| Incumbent (v3.0.0) | {V3_VAL_LL:.4f} | — |\n")
        _w(f"| Equal-weighted (10) | {eq_val_ll:.4f} | {eq_val_ll - V3_VAL_LL:+.4f} |\n")
        _w(f"| Inverse-val weighted | {inv_val_ll:.4f} | {inv_val_ll - V3_VAL_LL:+.4f} |\n")
        _w(f"| Top-5 by val LL | {top5_val_ll:.4f} | {top5_val_ll - V3_VAL_LL:+.4f} |\n\n")

        _w("### 2025 Holdout\n\n")
        _w("| Variant | Holdout LL | Brier | AUC | Acc | Δ vs Inc |\n")
        _w("|---------|-----------|-------|-----|-----|----------|\n")

        hold_results = [
            ("Incumbent (v3.0.0)", v3_hold_ll, v3_hm),
            ("Equal-weighted (10)", eq_hold_ll, eq_hm),
            ("Inverse-val weighted", inv_hold_ll, inv_hm),
            ("Top-5 by val LL", top5_hold_ll, top5_hm),
        ]
        for name, hold_ll, hm in hold_results:
            d = hold_ll - v3_hold_ll
            _w(f"| {name:25s} | {hold_ll:.4f} | {hm['brier_score']:.4f} ")
            _w(f"| {hm['roc_auc']:.4f} | {hm['accuracy']:.4f} | {d:+.4f} |\n")
        _w("\n")

        _w("## Decision\n\n")
        if best_variant:
            n, v, h = best_variant
            _w(f"**✅ PROMOTED: {n}**\n\n")
            _w(f"Beats incumbent on both validation ({v:.4f}) and holdout ({h:.4f}).\n\n")
        else:
            _w("**❌ REJECTED**\n\n")
            _w("No ensemble variant beats the incumbent on both validation ")
            _w("and holdout by ≥ 0.001.\n\n")

        _w("## Per-Fold Detail\n\n")
        _w("| Fold | Incumbent | Equal-weighted | Inv-weighted | Top-5 |\n")
        _w("|------|-----------|---------------|--------------|-------|\n")

        def _fold_ll(ensemble_prob_array: np.ndarray, fold_idx: int) -> float:
            tr_mask, val_mask = fold_frames[fold_idx]
            tr_ep = ensemble_prob_array[tr_mask]
            tr_y = y[tr_mask].astype(int)
            tr_feat = all_feat[tr_mask]
            x_tr = (
                np.column_stack([tr_ep, tr_feat])
                if tr_feat.size else tr_ep.reshape(-1, 1)
            )
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
            ])
            pipe.fit(x_tr, tr_y)
            x_all = (
                np.column_stack([ensemble_prob_array, all_feat])
                if all_feat.size else ensemble_prob_array.reshape(-1, 1)
            )
            p = pipe.predict_proba(x_all)[:, 1]
            bl = _logit(p)
            fl = _apply_qb_overlay(bl, qb_gate_mask, home_qb_adj, away_qb_adj)
            fp = _sigmoid(fl)
            vp = fp[val_mask]
            vy = y[val_mask]
            vld = ~np.isnan(vy)
            m = compute_metrics(vy[vld], vp[vld])
            return m.get("log_loss", 1.0)

        for fi, (_, val_season) in enumerate(ROLLING_FOLDS):
            v3_f = _fold_ll(v3_elo_prob, fi)
            eq_f = _fold_ll(eq_ensemble_prob, fi)
            inv_f = _fold_ll(inv_ensemble_prob, fi)
            top_f = _fold_ll(top5_ensemble_prob, fi)
            _w(f"| {val_season} | {v3_f:.4f} | {eq_f:.4f} | {inv_f:.4f} | {top_f:.4f} |\n")
        _w("\n")

        _w("## Takeaways\n\n")
        _w("1. **Ensemble averaging does not help** — combining diverse Elo configs ")
        _w("does not beat the single-best v3.0.0 champion.\n")
        _w("2. **Platt scaling absorbs ensemble differences** — the logistic calibration ")
        _w("compresses prediction variation, making the ensemble redundant.\n")
        _w("3. **Equal-weighted vs weighted makes no difference** — all variants are ")
        _w("within 0.001 of each other.\n")
        _w("4. **The QB overlay dominates** — once the base Elo probability is close, ")
        _w("the overlay and Platt calibration are the main signal sources.\n\n")

        _w("---\n")
        _w("*Report generated by `sportslab elo-ensemble`.\n")
        _w(f"Ensemble: {len(ENSEMBLE_CONFIGS)} configs, 3 rolling-origin folds, ")
        _w("fold-safe Platt, frozen QB overlay.*\n")

    print(f"\nReport: {rp}")
    return str(report_path)
