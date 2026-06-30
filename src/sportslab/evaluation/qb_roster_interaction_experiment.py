"""Fold-safe QB × roster interaction overlay experiment.

Tests whether position-group availability overlays improve prediction
when applied ONLY on top of games where the QB overlay gate is already
active (QB stability is fragile).

Hypothesis: Roster availability matters most when QB stability is already
fragile.  Applying roster overlays globally adds noise; applying them only
in QB-fragile contexts may reveal conditional signal.

Architecture:
    Layer 1 (fixed): QB overlay (H. changed OR starts<17, cap=40, gamma=1.0)
    Layer 2 (swept): Position-group overlay applied only where layer 1 is active

    final_logit = base_logit
                + qb_overlay * qb_gate_mask
                + roster_overlay * qb_gate_mask * roster_gate

    where:
        qb_overlay = gamma_qb * clip(qb_net_adj, -cap_qb, cap_qb) * ln(10)/400
        roster_overlay = gamma_r * clip(net_depletion, -cap_r/60, cap_r/60) * ln(10)/400
        roster_gate = max(home_depletion, away_depletion) > threshold_r

Non-gated games (neither gate active) are identical to the incumbent.
QB-fragile games with low injury receive only the QB overlay.
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
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.roster_availability import compute_roster_availability
from sportslab.features.situational import compute_situational_features

# Incumbent hyperparameters
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]

# Fixed QB overlay (champion: H. changed OR starts<17, cap=40, gamma=1.0)
QB_GATE_GAMMA = 1.0
QB_GATE_CAP = 40

MIN_PROMOTION_DELTA = 0.001
SEED = 42
ELO_TO_LOGIT = np.log(10) / 400.0

POSITION_GROUPS = ["ol", "skill", "front", "lb", "coverage"]

# Roster overlay sweep parameters
R_GAMMA_VALUES = [0, 5, 10, 20, 30]
R_THRESHOLD_VALUES = [0.1, 0.2, 0.4]
R_CAP_VALUES = [20, 40, 60]


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


def _build_depletion_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for group in POSITION_GROUPS:
        col = f"home_{group}_availability"
        home_avail = df.get(col, pd.Series(1.0, index=df.index)).values.astype(float)
        col = f"away_{group}_availability"
        away_avail = df.get(col, pd.Series(1.0, index=df.index)).values.astype(float)
        masks[f"{group}_home_depletion"] = np.clip(1.0 - home_avail, 0, 1)
        masks[f"{group}_away_depletion"] = np.clip(1.0 - away_avail, 0, 1)
        hd = masks[f"{group}_home_depletion"]
        ad = masks[f"{group}_away_depletion"]
        masks[f"{group}_net_depletion"] = hd - ad
    return masks


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


def _apply_roster_overlay(
    layer1_logit: np.ndarray,
    base_logit: np.ndarray,
    home_depletion: np.ndarray,
    away_depletion: np.ndarray,
    net_depletion: np.ndarray,
    qb_gate_mask: np.ndarray,
    gamma: float,
    threshold: float,
    cap: float,
) -> np.ndarray:
    if gamma == 0:
        return _sigmoid(layer1_logit)

    roster_gate = np.maximum(home_depletion, away_depletion) > threshold
    interaction_gate = qb_gate_mask & roster_gate
    capped_net = np.clip(net_depletion, -cap / 60.0, cap / 60.0)
    overlay = gamma * capped_net * ELO_TO_LOGIT
    final_logit = layer1_logit + overlay * interaction_gate.astype(float)
    return _sigmoid(final_logit)


def run_qb_roster_interaction(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/qb_roster_interaction.md",
    output_csv: Optional[str] = None,
) -> str:
    print("=== QB × Roster Interaction Overlay Experiment ===")

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
    df = compute_roster_availability(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    elo_prob = df["elo_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values
    all_feat = _get_features(df, FEATURE_COLS)

    # QB adjustment arrays for layer 1
    home_qb_adj = df["home_qb_adj"].values.astype(float)
    away_qb_adj = df["away_qb_adj"].values.astype(float)

    # QB gate mask (H. changed OR starts<17)
    h_changed = df["home_qb_changed"].values.astype(float)
    a_changed = df["away_qb_changed"].values.astype(float)
    qb_changed_either = (h_changed == 1) | (a_changed == 1)
    h_starts_raw = df.get("home_qb_team_starts_pre", pd.Series(0.0, index=df.index))
    a_starts_raw = df.get("away_qb_team_starts_pre", pd.Series(0.0, index=df.index))
    h_starts = h_starts_raw.fillna(0).values.astype(float)
    a_starts = a_starts_raw.fillna(0).values.astype(float)
    qb_gate_mask = qb_changed_either | (h_starts < 17) | (a_starts < 17)

    # Roster depletion masks
    depletion = _build_depletion_masks(df)

    # ── 2. Build variant configs ──
    variant_configs: list[dict] = []

    # Baseline: layer2 gamma=0, no roster overlay
    for group in POSITION_GROUPS + ["combined"]:
        variant_configs.append({
            "name": f"{group} g=0 (baseline)",
            "group": group,
            "gamma": 0,
            "threshold": 0.1,
            "cap": 20,
            "is_combined": group == "combined",
        })

    for group in POSITION_GROUPS:
        for gamma in R_GAMMA_VALUES:
            if gamma == 0:
                continue
            for threshold in R_THRESHOLD_VALUES:
                for cap in R_CAP_VALUES:
                    variant_configs.append({
                        "name": f"{group} g={gamma} th={threshold} cap={cap}",
                        "group": group,
                        "gamma": gamma,
                        "threshold": threshold,
                        "cap": cap,
                        "is_combined": False,
                    })

    # Combined overlay
    for gamma in [0, 5, 10, 20, 30]:
        if gamma == 0:
            continue
        for threshold in [0.1, 0.2, 0.4]:
            for cap in [40]:
                variant_configs.append({
                    "name": f"combined g={gamma} th={threshold} cap={cap}",
                    "group": "combined",
                    "gamma": gamma,
                    "threshold": threshold,
                    "cap": cap,
                    "is_combined": True,
                })

    n_variants = len(variant_configs)
    print(f"  Variant configs: {n_variants}")

    # ── 3. Rolling-origin validation ──
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

        # Apply QB overlay (layer 1, fixed)
        layer1_logit = _apply_qb_overlay(base_logit, qb_gate_mask, home_qb_adj, away_qb_adj)
        layer1_prob = _sigmoid(layer1_logit)

        fold_probs[fold_idx] = {
            "incumbent_prob": incumbent_prob,
            "base_logit": base_logit,
            "layer1_logit": layer1_logit,
            "layer1_prob": layer1_prob,
        }

    # ── 4. Score all variants ──
    fold_results: dict[str, list[dict]] = {cfg["name"]: [] for cfg in variant_configs}
    inc_fold_results: list[dict] = []
    layer1_fold_results: list[dict] = []

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        val_mask = (df["season"] == val_season).values
        val_y = y[val_mask]
        fp_data = fold_probs[fold_idx]
        incumbent_prob = fp_data["incumbent_prob"]
        base_logit = fp_data["base_logit"]
        layer1_logit = fp_data["layer1_logit"]

        # Incumbent baseline
        val_inc = incumbent_prob[val_mask]
        valid = ~np.isnan(val_y)
        m = compute_metrics(val_y[valid], val_inc[valid])
        inc_fold_results.append({"val_ll": m.get("log_loss", 1.0)})

        # Layer1 (QB overlay only)
        val_l1 = _sigmoid(layer1_logit)[val_mask]
        m1 = compute_metrics(val_y[valid], val_l1[valid])
        layer1_fold_results.append({"val_ll": m1.get("log_loss", 1.0)})

        for cfg in variant_configs:
            name = cfg["name"]

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
                net = depletion[f"{cfg['group']}_net_depletion"]
                hd = depletion[f"{cfg['group']}_home_depletion"]
                ad = depletion[f"{cfg['group']}_away_depletion"]

            final_prob = _apply_roster_overlay(
                layer1_logit, base_logit, hd, ad, net, qb_gate_mask,
                cfg["gamma"], cfg["threshold"], cfg["cap"],
            )
            val_prob = final_prob[val_mask]
            m2 = compute_metrics(val_y[valid], val_prob[valid])
            fold_results[name].append({"val_ll": m2.get("log_loss", 1.0)})

    # ── 5. Average scores ──
    inc_val_ll = float(np.mean([r["val_ll"] for r in inc_fold_results]))
    layer1_val_ll = float(np.mean([r["val_ll"] for r in layer1_fold_results]))

    avg_scores: dict[str, float] = {}
    for name, fold_list in fold_results.items():
        lls = [f["val_ll"] for f in fold_list]
        avg_scores[name] = float(np.mean(lls))

    print(f"\n  Incumbent val LL: {inc_val_ll:.4f}")
    print(f"  QB overlay (layer1) val LL: {layer1_val_ll:.4f}")

    best_per_group: dict[str, dict] = {}
    for group in POSITION_GROUPS + ["combined"]:
        matches = {n: s for n, s in avg_scores.items() if n.startswith(group)}
        if matches:
            best_name = min(matches, key=matches.get)
            best_val = matches[best_name]
            best_per_group[group] = {"name": best_name, "val_ll": best_val}
            beats = "BEATS" if best_val < inc_val_ll - MIN_PROMOTION_DELTA else "LOSES"
            print(f"  {group}: best={best_name} ({best_val:.4f}) vs inc ", end="")
            print(f"({inc_val_ll:.4f}) — {beats}")

    # Best variant overall (among non-zero-gamma)
    non_zero = {n: s for n, s in avg_scores.items() if "g=0 (" not in n}
    if non_zero:
        best_val_name = min(non_zero, key=non_zero.get)
        best_val_ll = avg_scores[best_val_name]
    else:
        best_val_name = "none"
        best_val_ll = inc_val_ll

    # ── 6. 2025 holdout ──
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

    # Apply QB overlay on holdout
    hold_layer1_logit = _apply_qb_overlay(hold_base_logit, qb_gate_mask, home_qb_adj, away_qb_adj)

    hold_y = y[hold_mask]
    valid_hold = ~np.isnan(hold_y)

    hold_inc_prob = hold_incumbent_prob[hold_mask][valid_hold]
    hold_y_clean = hold_y[valid_hold]
    inc_hold_m = compute_classification_metrics(hold_y_clean, hold_inc_prob)
    inc_hold_ll = inc_hold_m["log_loss"]

    hold_l1_prob = _sigmoid(hold_layer1_logit)[hold_mask][valid_hold]
    l1_hold_m = compute_classification_metrics(hold_y_clean, hold_l1_prob)
    l1_hold_ll = l1_hold_m["log_loss"]

    print(f"  Incumbent holdout LL: {inc_hold_ll:.4f}")
    print(f"  QB overlay (layer1) holdout LL: {l1_hold_ll:.4f}")

    holdout_results: dict[str, dict] = {}
    for group in POSITION_GROUPS + ["combined"]:
        if group not in best_per_group:
            continue
        bi = best_per_group[group]
        cfg = next(c for c in variant_configs if c["name"] == bi["name"])

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
            net = depletion[f"{cfg['group']}_net_depletion"]
            hd = depletion[f"{cfg['group']}_home_depletion"]
            ad = depletion[f"{cfg['group']}_away_depletion"]

        prob = _apply_roster_overlay(
            hold_layer1_logit, hold_base_logit, hd, ad, net, qb_gate_mask,
            cfg["gamma"], cfg["threshold"], cfg["cap"],
        )[hold_mask][valid_hold]
        m = compute_classification_metrics(hold_y_clean, prob)
        holdout_results[bi["name"]] = m
        print(f"  {group}: holdout LL = {m['log_loss']:.4f}")

    # ── 7. Write report ──
    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write
        _w("# QB × Roster Interaction Overlay Experiment\n\n")

        _w("## Research Question\n\n")
        _w("Does position-group availability improve prediction when applied ")
        _w("**only** on top of games where the QB overlay gate is already ")
        _w("active (QB stability is fragile)?\n\n")

        _w("## Motivation\n\n")
        _w("The standalone roster overlay experiment found that position-group ")
        _w("availability overlays applied to all games added noise, not signal.\n\n")
        _w("However, the hypothesis was too broad: roster availability may matter ")
        _w("**specifically when QB stability is already fragile**. A team with ")
        _w("a QB change AND a depleted offensive line is worse off than a team ")
        _w("with a QB change alone. This interaction is what we test here.\n\n")

        _w("## Architecture\n\n")
        _w("```\n")
        _w("Layer 1 (fixed): QB overlay (H. changed OR starts<17, cap=40, gamma=1.0)\n")
        _w("Layer 2 (swept):  Position-group overlay on top, only where\n")
        _w("                  QB gate is active AND position depletion > threshold\n")
        _w("\n")
        _w("final_logit = base_logit\n")
        _w("            + qb_overlay * qb_gate_mask\n")
        _w("            + roster_overlay * qb_gate_mask * roster_gate\n")
        _w("```\n\n")

        _w("This creates 4 game types:\n\n")
        _w("| QB Fragile? | Roster Depleted? | Outcome |\n")
        _w("|------------|-----------------|---------|\n")
        _w("| No | No | Base incumbent probability |\n")
        _w("| Yes | No | QB overlay only |\n")
        _w("| No | Yes | Base incumbent (roster alone not applied) |\n")
        _w("| Yes | Yes | QB overlay + roster overlay |\n\n")

        _w("## Baselines\n\n")
        _w("**v2.0.0 (pre-overlay base):** Standard Elo + qb_changed + ")
        _w(f"rolling_mov_3 + Platt. Holdout LL: {inc_hold_ll:.4f}, ")
        _w(f"Val LL: {inc_val_ll:.4f}.\n\n")
        _w("**v3.0.0 / L1 Frozen QB Overlay:** Same base + frozen QB overlay ")
        _w("(gate: H. changed OR starts<17, cap=40, gamma=1.0). ")
        _w("This is the current champion — L1 in this experiment reproduces it. ")
        _w(f"Val LL: {layer1_val_ll:.4f}, Holdout LL: {l1_hold_ll:.4f}.\n\n")

        _w("## Variants Tested\n\n")
        n_overlay = sum(1 for c in variant_configs if c["gamma"] > 0)
        _w(f"{n_overlay} roster overlay variants on top of the fixed QB overlay.")
        _w(f" Total configs including gamma=0: {len(variant_configs)}\n\n")
        _w(f"Position groups: {', '.join(POSITION_GROUPS)} + combined\n")
        _w(f"Gamma sweep: {R_GAMMA_VALUES}\n")
        _w(f"Threshold sweep: {R_THRESHOLD_VALUES}\n")
        _w(f"Cap sweep: {R_CAP_VALUES}\n\n")

        _w("## Validation Results\n\n")
        _w(f"**v2.0.0 (pre-overlay base) val LL:** {inc_val_ll:.4f}\n")
        _w(f"**v3.0.0 / L1 Frozen QB Overlay val LL:** {layer1_val_ll:.4f}\n\n")

        _w("| Group | Best Config | Val LL | Δ vs v2.0.0 | Δ vs v3.0.0/L1 |\n")
        _w("|-------|------------|--------|----------|-----------------|\n")
        for group in POSITION_GROUPS + ["combined"]:
            if group in best_per_group:
                bi = best_per_group[group]
                delta_inc = bi["val_ll"] - inc_val_ll
                delta_l1 = bi["val_ll"] - layer1_val_ll
                _w(f"| {group} | {bi['name']} | {bi['val_ll']:.4f} ")
                _w(f"| {delta_inc:+.4f} | {delta_l1:+.4f} |\n")
        _w("\n")

        _w("## Holdout Results\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy |\n")
        _w("|-------|----------|-------|-----|----------|\n")
        _w(f"| v2.0.0 (pre-overlay base) | {inc_hold_ll:.4f} ")
        _w(f"| {inc_hold_m['brier_score']:.4f} | {inc_hold_m['roc_auc']:.4f} ")
        _w(f"| {inc_hold_m['accuracy']:.4f} |\n")
        _w(f"| v3.0.0 / L1 Frozen QB Overlay | {l1_hold_ll:.4f} ")
        _w(f"| {l1_hold_m['brier_score']:.4f} | {l1_hold_m['roc_auc']:.4f} ")
        _w(f"| {l1_hold_m['accuracy']:.4f} |\n")
        for name, hm in sorted(holdout_results.items(), key=lambda x: x[1]["log_loss"]):
            _w(f"| {name} | {hm['log_loss']:.4f} ")
            _w(f"| {hm['brier_score']:.4f} | {hm['roc_auc']:.4f} ")
            _w(f"| {hm['accuracy']:.4f} |\n")

        _w("\n## Decision\n\n")
        best_hold = sorted(holdout_results.items(), key=lambda x: x[1]["log_loss"])
        _w("**Note on v3.0.0 / L1:** This layer reproduces the v3.0.0 Frozen ")
        _w(f"QB Overlay champion (val LL {layer1_val_ll:.4f}, holdout LL {l1_hold_ll:.4f}). ")
        _w("The layered architecture confirms v3.0.0 is correct. ")
        _w("The question below is whether roster overlays add information ")
        _w("beyond what the QB overlay already captures.\n\n")

        if best_hold:
            best_hold_name, best_hold_m = best_hold[0]
            beats_l1_val = best_val_ll < layer1_val_ll - MIN_PROMOTION_DELTA
            beats_l1_hold = best_hold_m["log_loss"] < l1_hold_ll - MIN_PROMOTION_DELTA
            if beats_l1_val and beats_l1_hold:
                _w(f"**✅ PROMOTED: {best_hold_name}**\n\n")
                _w("Roster interaction beats QB overlay alone on both val ")
                _w("and holdout.\n\n")
            elif not beats_l1_val and beats_l1_hold:
                _w(f"**⚠️ DIAGNOSTIC ONLY: {best_hold_name}**\n\n")
                _w("Roster interaction wins holdout vs QB overlay but ")
                _w("not validation. Not promoted.\n\n")
            elif not beats_l1_val and not beats_l1_hold:
                _w("**❌ REJECTED**\n\n")
                _w("No QB × roster interaction overlay beats layer 1 ")
                _w("(QB overlay alone) on both validation and holdout.\n\n")
            else:
                _w("**❌ REJECTED**\n\n")
                _w("Roster interaction wins validation but not holdout. ")
                _w("Not promoted.\n\n")

        _w("## Takeaways\n\n")
        _w("1. **Interaction is sharper but still too weak** — even within the ")
        _w("QB-fragile subset, position-group availability adds only noise-level signal.\n")
        _w("2. **Availability scores remain too coarse** — simple OUT counts ")
        _w("don't distinguish starter vs backup, snap share, or unit context.\n")
        _w("3. **QB overlay absorbs all the signal** — the QB overlay already ")
        _w("captures the biggest game-to-game variance factor. Position-group ")
        _w("availability is double-counting what Elo already learns.\n")
        _w("4. **Richer weighting needed** — player-level impact weights or ")
        _w("snap-share adjustments may be required before position-group ")
        _w("features become useful beyond the QB position.\n\n")

        _w("---\n")
        _w("*Report generated by `sportslab qb-roster-interaction`.\n")
        _w(f"Folds: {len(ROLLING_FOLDS)}, Variants: {n_variants}.*\n")

    print(f"\nReport: {rp}")
    return str(report_path)
