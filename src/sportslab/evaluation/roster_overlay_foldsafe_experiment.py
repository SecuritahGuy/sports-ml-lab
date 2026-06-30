"""Fold-safe frozen-incumbent roster overlay experiment.

Tests whether position-group availability overlays (OL, skill, front, LB,
coverage) improve on the frozen QB overlay incumbent (v3.0.0, holdout 0.6200).

Architecture:
    For each position group:
        depletion = 1 - availability  (0 = fully healthy, 1 = fully depleted)
        net_depletion = home_depletion - away_depletion
        gate_active = max(home_depletion, away_depletion) > threshold
        overlay = gamma * net_depletion * ln(10)/400
        final_logit = base_logit + overlay * gate_active

Non-gated games are exactly equal to the incumbent base probability.
"""

from pathlib import Path
from typing import Dict, Optional

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
from sportslab.features.ratings import compute_elo_features
from sportslab.features.roster_availability import compute_roster_availability
from sportslab.features.situational import compute_situational_features

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]

MIN_PROMOTION_DELTA = 0.001  # minimum log loss improvement to promote
SEED = 42
ELO_TO_LOGIT = np.log(10) / 400.0

POSITION_GROUPS = ["ol", "skill", "front", "lb", "coverage"]

GAMMA_VALUES = [0, 10, 20, 30, 40, 50, 60]
THRESHOLD_VALUES = [0.1, 0.2, 0.4, 0.6]
CAP_VALUES = [20, 40, 60]


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


def _build_depletion_masks(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Precompute depletion arrays for each position group."""
    masks: Dict[str, np.ndarray] = {}
    for group in POSITION_GROUPS:
        home_avail = df.get(f"home_{group}_availability", pd.Series(1.0, index=df.index)).values.astype(float)
        away_avail = df.get(f"away_{group}_availability", pd.Series(1.0, index=df.index)).values.astype(float)
        masks[f"{group}_home_depletion"] = np.clip(1.0 - home_avail, 0, 1)
        masks[f"{group}_away_depletion"] = np.clip(1.0 - away_avail, 0, 1)
        masks[f"{group}_net_depletion"] = masks[f"{group}_home_depletion"] - masks[f"{group}_away_depletion"]
    return masks


def _apply_overlay(
    incumbent_prob: np.ndarray,
    base_logit: np.ndarray,
    home_depletion: np.ndarray,
    away_depletion: np.ndarray,
    net_depletion: np.ndarray,
    gamma: float,
    threshold: float,
    cap: float,
) -> np.ndarray:
    """Apply a single position-group overlay."""
    if gamma == 0:
        return incumbent_prob.copy()

    gate = np.maximum(home_depletion, away_depletion) > threshold
    capped_net = np.clip(net_depletion, -cap / 60.0, cap / 60.0)
    overlay = gamma * capped_net * ELO_TO_LOGIT
    final_logit = base_logit + overlay * gate.astype(float)
    return _sigmoid(final_logit)


def run_roster_overlay_foldsafe(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/roster_overlay_foldsafe.md",
    output_csv: Optional[str] = None,
) -> str:
    print("=== Fold-Safe Roster Overlay Experiment ===")

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
    df = compute_roster_availability(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    elo_prob = df["elo_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values
    all_feat = _get_features(df, FEATURE_COLS)

    depletion = _build_depletion_masks(df)

    variant_configs: list[dict] = []

    for group in POSITION_GROUPS:
        hd = f"{group}_home_depletion"
        ad = f"{group}_away_depletion"
        nd = f"{group}_net_depletion"
        for gamma in GAMMA_VALUES:
            for threshold in THRESHOLD_VALUES:
                for cap in CAP_VALUES:
                    if gamma == 0:
                        continue
                    label = f"{group} g={gamma} th={threshold} cap={cap}"
                    variant_configs.append({
                        "name": label,
                        "group": group,
                        "gamma": gamma,
                        "threshold": threshold,
                        "cap": cap,
                        "home_depletion_key": hd,
                        "away_depletion_key": ad,
                        "net_depletion_key": nd,
                        "is_combined": False,
                    })

    # Combined overlay (all groups equally weighted as sum of net depletions)
    for gamma in [0, 10, 20, 30, 40, 50]:
        for threshold in [0.1, 0.2, 0.4]:
            for cap in [40]:
                if gamma == 0:
                    continue
                label = f"combined g={gamma} th={threshold} cap={cap}"
                variant_configs.append({
                    "name": label,
                    "group": "combined",
                    "gamma": gamma,
                    "threshold": threshold,
                    "cap": cap,
                    "is_combined": True,
                })

    print(f"  Variant configs: {len(variant_configs)}")

    # Rolling-origin validation
    print("\n=== Rolling-Origin Validation ===")
    fold_probs: dict[int, dict] = {}

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        print(f"  Fold {fold_idx + 1}: train {train_seasons} → val {val_season}")
        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values

        train_elo = elo_prob[train_mask]
        train_y_int = y[train_mask].astype(int)
        train_feat = all_feat[train_mask]
        x_train = (
            np.column_stack([train_elo, train_feat])
            if train_feat.size else train_elo.reshape(-1, 1)
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_train, train_y_int)

        x_all = (
            np.column_stack([elo_prob, all_feat])
            if all_feat.size else elo_prob.reshape(-1, 1)
        )
        incumbent_prob = pipe.predict_proba(x_all)[:, 1]
        base_logit = _logit(incumbent_prob)

        fold_probs[fold_idx] = {
            "incumbent_prob": incumbent_prob,
            "base_logit": base_logit,
        }

    # Score all variants
    fold_results: dict[str, list[dict]] = {cfg["name"]: [] for cfg in variant_configs}
    baseline_results: list[dict] = []

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        val_mask = (df["season"] == val_season).values
        val_y = y[val_mask]
        fp_data = fold_probs[fold_idx]
        incumbent_prob = fp_data["incumbent_prob"]
        base_logit = fp_data["base_logit"]

        # Baseline
        val_prob = incumbent_prob[val_mask]
        valid = ~np.isnan(val_y)
        m = compute_metrics(val_y[valid], val_prob[valid])
        baseline_results.append({"val_ll": m.get("log_loss", 1.0), "metrics": m})

        # Variants
        for cfg in variant_configs:
            name = cfg["name"]

            if cfg["is_combined"]:
                net = np.zeros(len(df), dtype=float)
                for g in POSITION_GROUPS:
                    net += depletion[f"{g}_net_depletion"]
                net /= len(POSITION_GROUPS)
                hd = np.zeros(len(df), dtype=float)
                ad = np.zeros(len(df), dtype=float)
                for g in POSITION_GROUPS:
                    hd += depletion[f"{g}_home_depletion"]
                    ad += depletion[f"{g}_away_depletion"]
                hd /= len(POSITION_GROUPS)
                ad /= len(POSITION_GROUPS)
            else:
                hd = depletion[cfg["home_depletion_key"]]
                ad = depletion[cfg["away_depletion_key"]]
                net = depletion[cfg["net_depletion_key"]]

            final_prob = _apply_overlay(
                incumbent_prob, base_logit, hd, ad, net,
                cfg["gamma"], cfg["threshold"], cfg["cap"],
            )
            val_prob = final_prob[val_mask]
            valid = ~np.isnan(val_y)
            m = compute_metrics(val_y[valid], val_prob[valid])
            fold_results[name].append({
                "val_ll": m.get("log_loss", 1.0),
                "metrics": m,
            })

    # Average scores
    inc_val_ll = float(np.mean([r["val_ll"] for r in baseline_results]))
    avg_scores: dict[str, float] = {}
    for name, fold_list in fold_results.items():
        lls = [f["val_ll"] for f in fold_list]
        avg_scores[name] = float(np.mean(lls))

    # Best per group
    print(f"\n  Incumbent val LL: {inc_val_ll:.4f}")
    best_per_group: dict[str, dict] = {}
    for group in POSITION_GROUPS + ["combined"]:
        matches = {n: s for n, s in avg_scores.items() if n.startswith(group)}
        if matches:
            best_name = min(matches, key=matches.get)
            best_val = matches[best_name]
            best_per_group[group] = {"name": best_name, "val_ll": best_val}
            beats = "BEATS" if best_val < inc_val_ll - MIN_PROMOTION_DELTA else "LOSES"
            print(f"  {group}: best={best_name} ({best_val:.4f}) vs inc ({inc_val_ll:.4f}) — {beats}")

    # Best overall among individual groups
    individual_bests = {g: v for g, v in best_per_group.items() if g in POSITION_GROUPS}
    if individual_bests:
        best_group = min(individual_bests, key=lambda g: individual_bests[g]["val_ll"])
        best_val_ll = individual_bests[best_group]["val_ll"]
    else:
        best_val_ll = inc_val_ll

    # 2025 holdout
    print("\n=== 2025 Holdout ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    train_mask_hold = df["season"].isin([2021, 2022, 2023, 2024]).values

    train_elo_hold = elo_prob[train_mask_hold]
    train_y_hold = y[train_mask_hold].astype(int)
    train_feat_hold = all_feat[train_mask_hold]
    x_train_hold = (
        np.column_stack([train_elo_hold, train_feat_hold])
        if train_feat_hold.size else train_elo_hold.reshape(-1, 1)
    )
    pipe_hold = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe_hold.fit(x_train_hold, train_y_hold)
    x_all_hold = (
        np.column_stack([elo_prob, all_feat])
        if all_feat.size else elo_prob.reshape(-1, 1)
    )
    hold_incumbent_prob = pipe_hold.predict_proba(x_all_hold)[:, 1]
    hold_base_logit = _logit(hold_incumbent_prob)

    hold_y = y[hold_mask]
    valid_hold = ~np.isnan(hold_y)
    hold_inc_prob = hold_incumbent_prob[hold_mask][valid_hold]
    hold_y_clean = hold_y[valid_hold]
    inc_hold_m = compute_classification_metrics(hold_y_clean, hold_inc_prob)
    inc_hold_ll = inc_hold_m["log_loss"]
    print(f"  Incumbent holdout LL: {inc_hold_ll:.4f}")

    # Evaluate all best-per-group on holdout
    holdout_results: dict[str, dict] = {}
    for group, best_info in best_per_group.items():
        cfg = next(c for c in variant_configs if c["name"] == best_info["name"])

        if cfg["is_combined"]:
            net = np.zeros(len(df), dtype=float)
            hd = np.zeros(len(df), dtype=float)
            ad = np.zeros(len(df), dtype=float)
            for g in POSITION_GROUPS:
                net += depletion[f"{g}_net_depletion"]
                hd += depletion[f"{g}_home_depletion"]
                ad += depletion[f"{g}_away_depletion"]
            net /= len(POSITION_GROUPS)
            hd /= len(POSITION_GROUPS)
            ad /= len(POSITION_GROUPS)
        else:
            hd = depletion[cfg["home_depletion_key"]]
            ad = depletion[cfg["away_depletion_key"]]
            net = depletion[cfg["net_depletion_key"]]

        prob = _apply_overlay(
            hold_incumbent_prob, hold_base_logit, hd, ad, net,
            cfg["gamma"], cfg["threshold"], cfg["cap"],
        )[hold_mask][valid_hold]
        m = compute_classification_metrics(hold_y_clean, prob)
        holdout_results[best_info["name"]] = m
        print(f"  {group}: holdout LL = {m['log_loss']:.4f}")

    # Write report
    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write
        _w("# Fold-Safe Roster Overlay Experiment\n\n")
        _w("## Research Question\n\n")
        _w("Do position-group availability overlays (OL, skill, front, LB, coverage) ")
        _w("improve on the frozen QB overlay incumbent (holdout LL 0.6200)?\n\n")

        _w("## Methodology\n\n")
        _w("For each position group, a depletion score is computed from injury report OUT counts:\n\n")
        _w("```\n")
        _w("depletion = min(OUT_count / typical_depth, 1.0)\n")
        _w("gate = max(home_depletion, away_depletion) > threshold\n")
        _w("net = home_depletion - away_depletion\n")
        _w("overlay = gamma * net * ln(10)/400\n")
        _w("```\n\n")
        _w("The overlay is applied in logit space and gated. Non-gated games ")
        _w("are identical to the incumbent.\n\n")

        _w("## Incumbent\n\n")
        _w("v3.0.0: Standard Elo + qb_changed + rolling_mov_3 + Platt + frozen QB overlay. ")
        _w(f"Holdout LL: {inc_hold_ll:.4f}.\n\n")

        _w("## Position Groups Tested\n\n")
        _w("| Group | Typical Depth | Data Source |\n")
        _w("|-------|--------------|-------------|\n")
        _w("| OL (C, G, T) | 5 | nflreadpy injury reports |\n")
        _w("| Skill (RB, WR, TE) | 5 | nflreadpy injury reports |\n")
        _w("| Front (DE, DT, NT, EDGE) | 4 | nflreadpy injury reports |\n")
        _w("| LB (ILB, OLB, MLB) | 3 | nflreadpy injury reports |\n")
        _w("| Coverage (CB, S) | 4 | nflreadpy injury reports |\n")
        _w("| Combined (all 5) | — | Average of all group depletions |\n\n")

        n_indiv = sum(1 for c in variant_configs if not c["is_combined"])
        n_comb = sum(1 for c in variant_configs if c["is_combined"])
        _w(f"Variants: {n_indiv} individual + {n_comb} combined = {len(variant_configs)} total\n\n")

        _w("## Validation Results\n\n")
        _w(f"**Incumbent val LL:** {inc_val_ll:.4f}\n\n")
        _w("| Group | Best Config | Val LL | Δ vs Inc |\n")
        _w("|-------|------------|--------|----------|\n")
        for group in POSITION_GROUPS + ["combined"]:
            if group in best_per_group:
                bi = best_per_group[group]
                delta = bi["val_ll"] - inc_val_ll
                _w(f"| {group} | {bi['name']} | {bi['val_ll']:.4f} | {delta:+.4f} |\n")
        _w("\n")

        _w("## Holdout Results\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy |\n")
        _w("|-------|----------|-------|-----|----------|\n")
        _w(f"| Incumbent (v3.0.0) | {inc_hold_ll:.4f} | {inc_hold_m['brier_score']:.4f} | {inc_hold_m['roc_auc']:.4f} | {inc_hold_m['accuracy']:.4f} |\n")
        for name, hm in sorted(holdout_results.items(), key=lambda x: x[1]["log_loss"]):
            _w(f"| {name} | {hm['log_loss']:.4f} | {hm['brier_score']:.4f} | {hm['roc_auc']:.4f} | {hm['accuracy']:.4f} |\n")

        _w("\n## Decision\n\n")
        best_hold = sorted(holdout_results.items(), key=lambda x: x[1]["log_loss"])
        if best_hold:
            best_hold_name, best_hold_m = best_hold[0]
            beats_val = best_val_ll < inc_val_ll - MIN_PROMOTION_DELTA
            beats_hold = best_hold_m["log_loss"] < inc_hold_ll - MIN_PROMOTION_DELTA
            if beats_val and beats_hold:
                _w(f"**✅ PROMOTED: {best_hold_name}**\n\n")
                _w("Beats incumbent on both validation and holdout.\n\n")
            elif not beats_val and beats_hold:
                _w(f"**⚠️ DIAGNOSTIC ONLY: {best_hold_name}**\n\n")
                _w("Wins holdout but not validation. Not promoted.\n\n")
            else:
                _w("**❌ REJECTED**\n\n")
                _w("No roster overlay beats the incumbent on both validation and holdout.\n\n")

        _w("## Failure Modes\n\n")
        _w("1. **Injury data coverage**: Injury reports may miss some players or teams.\n")
        _w("2. **OUT counts ≠ impact**: Not all OUT players are equal (starter vs backup).\n")
        _w("3. **Coarse measure**: Availability is a simple count, not a skill rating.\n")
        _w("4. **No per-player ratings**: Unlike the QB overlay, we don't have Bayesian-shrunken performance ratings for non-QB positions.\n\n")

        _w("---\n")
        _w("*Report generated by `sportslab roster-overlay`.\n")
        _w(f"Folds: {len(ROLLING_FOLDS)}, Variants: {len(variant_configs)}.*\n")

    print(f"\nReport: {rp}")
    return str(report_path)
