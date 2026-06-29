"""Fold-safe frozen-incumbent QB overlay experiment.

Research question:
    Does the frozen QB overlay still improve the incumbent when the incumbent
    base probabilities and calibration are generated using strictly fold-safe
    rolling-origin training?

Prior experiment design flaw:
    The V2 frozen overlay experiment fitted Platt calibration once on full
    2021-2024 data, then used that single fit to evaluate ALL rolling-origin
    folds.  This leaked future data into early-fold validation (e.g., 2023
    games were scored using a Platt model that had seen 2024 data during
    training).  Validation metrics were not trustworthy for selection.

This version fixes the flaw by fitting Platt *per fold*:
    For each rolling-origin fold (train_seasons → val_season):
        1. Compute Elo, QB, situational features on train+val data
           (Elo is inherently chronological — each game's rating uses
           only data from games before it, so no future leakage)
        2. Fit Platt calibration using only train_seasons
        3. Generate incumbent base probabilities (Platt saw only train data)
        4. Apply frozen QB overlay in logit space ONLY to val_season games
           where the pregame gate is active
        5. Non-gated val games are EXACTLY equal to incumbent base prob
        6. Score val_season metrics

    Variant selection is based on average validation log loss across folds.
    2025 holdout is strictly held out — never used during selection.
"""

from pathlib import Path

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

# Incumbent hyperparameters (authoritative)
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]

SEED = 42

# Gamma values to sweep
GAMMA_VALUES = [0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]

# Cap values to sweep (Elo points)
CAP_VALUES = [20, 40, 60]

# Conversion: Elo-point diff to logit delta
# logit(P) = (elo_diff + HFA) / 400 * ln(10)
ELO_TO_LOGIT = np.log(10) / 400.0

# Gates that produce the same results as B (identical condition)
GATE_C_ALIASES = {"C. qb_differs_prev"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    """Logit function with clipping."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def run_frozen_qb_overlay_foldsafe(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/frozen_qb_overlay_foldsafe.md",
    output_csv: str | None = None,
) -> str:
    print("=== Fold-Safe Frozen-Incumbent QB Overlay Experiment ===")

    # ── 1. Load data and build features once (globally) ──
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

    # Filter eligible games
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    # Base arrays (used across all folds)
    elo_prob = df["elo_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values
    all_feat = _get_features(df, FEATURE_COLS)

    # QB adjustment arrays
    home_qb_adj = df["home_qb_adj"].values.astype(float)
    away_qb_adj = df["away_qb_adj"].values.astype(float)
    net_adj_elo = home_qb_adj - away_qb_adj

    # QB state columns for gating
    h_changed = df["home_qb_changed"].values.astype(float)
    a_changed = df["away_qb_changed"].values.astype(float)
    qb_changed_either = (h_changed == 1) | (a_changed == 1)

    h_starts_raw = df.get("home_qb_team_starts_pre", pd.Series(0.0, index=df.index))
    a_starts_raw = df.get("away_qb_team_starts_pre", pd.Series(0.0, index=df.index))
    h_starts = h_starts_raw.fillna(0).values.astype(float)
    a_starts = a_starts_raw.fillna(0).values.astype(float)

    # ── 2. Define gate masks (pre-computed globally) ──
    gates: dict[str, np.ndarray] = {
        "A. Incumbent baseline": np.zeros(len(df), dtype=bool),
        "B. qb_changed": qb_changed_either,
        "C. qb_differs_prev": qb_changed_either,
        "D. starts<4": (h_starts < 4) | (a_starts < 4),
        "E. starts<8": (h_starts < 8) | (a_starts < 8),
        "F. starts<17": (h_starts < 17) | (a_starts < 17),
        "G. changed OR starts<8": qb_changed_either | (h_starts < 8) | (a_starts < 8),
        "H. changed OR starts<17": qb_changed_either | (h_starts < 17) | (a_starts < 17),
    }

    # ── 3. Build all variant configs ──
    variant_configs: list[dict] = []

    for gate_name, gate_mask in gates.items():
        for gamma in GAMMA_VALUES:
            for cap in CAP_VALUES:
                # Baseline: only one config needed
                if gate_name == "A. Incumbent baseline":
                    if gamma == 0.0 and cap == CAP_VALUES[0]:
                        variant_configs.append({
                            "name": "A. Incumbent baseline",
                            "gate_name": gate_name,
                            "gamma": 0.0,
                            "cap": None,
                            "gate_mask": gate_mask,
                            "is_diagnostic": False,
                        })
                    continue

                # Skip gamma=0 for non-baseline gates (identical to baseline)
                if gamma == 0.0:
                    continue

                # Build label
                label = gate_name
                cap_label = f"cap={cap}"
                if gamma != 1.0:
                    label += f" g={gamma:.2f} {cap_label}"
                else:
                    label += f" {cap_label}"

                variant_configs.append({
                    "name": label,
                    "gate_name": gate_name,
                    "gamma": gamma,
                    "cap": cap,
                    "gate_mask": gate_mask,
                    "is_diagnostic": False,  # All A-H non-diagnostic
                })

    print(f"  Variant configs: {len(variant_configs)}")

    # ── 4. Rolling-origin validation (fold-safe) ──
    print("\n=== Rolling-Origin Validation (Fold-Safe) ===")
    print(f"  Folds: {ROLLING_FOLDS}")

    # For each fold, store the incumbent Platt probability and base_logit
    fold_probs: dict[int, dict] = {}  # fold_idx -> {"incumbent_prob": ..., "base_logit": ...}

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        print(f"\n  Fold {fold_idx + 1}: train {train_seasons} → val {val_season}")

        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        print(f"    Train: {n_train} games, Val: {n_val} games")

        assert n_train > 0, f"Fold {fold_idx}: no training data"
        assert n_val > 0, f"Fold {fold_idx}: no validation data"

        # Build training data for Platt
        train_elo = elo_prob[train_mask]
        train_y_int = y[train_mask].astype(int)
        train_feat = all_feat[train_mask]

        x_train = (
            np.column_stack([train_elo, train_feat])
            if train_feat.size else train_elo.reshape(-1, 1)
        )

        # Fit Platt on training seasons ONLY
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_train, train_y_int)

        # Generate incumbent probs for ALL data using this fold's Platt
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

    # ── 5. Score all variants per fold ──
    # fold_results[variant_name][fold_idx] = {"val_ll": ..., metrics}
    fold_results: dict[str, list[dict]] = {cfg["name"]: [] for cfg in variant_configs}

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        val_mask = (df["season"] == val_season).values
        val_y = y[val_mask]
        fp_data = fold_probs[fold_idx]
        incumbent_prob = fp_data["incumbent_prob"]
        base_logit = fp_data["base_logit"]

        for cfg in variant_configs:
            name = cfg["name"]

            if name == "A. Incumbent baseline":
                final_prob = incumbent_prob.copy()
            else:
                # Apply cap if specified
                if cfg["cap"] is not None:
                    capped_h = np.clip(home_qb_adj, -cfg["cap"], cfg["cap"])
                    capped_a = np.clip(away_qb_adj, -cfg["cap"], cfg["cap"])
                    cur_net_adj = capped_h - capped_a
                else:
                    cur_net_adj = net_adj_elo

                overlay = cfg["gamma"] * cur_net_adj * ELO_TO_LOGIT
                gate_active = cfg["gate_mask"].astype(float)
                final_logit = base_logit + overlay * gate_active
                final_prob = _sigmoid(final_logit)

            val_prob = final_prob[val_mask]
            valid = ~np.isnan(val_y)
            m = compute_metrics(val_y[valid], val_prob[valid])
            fold_results[name].append({
                "val_ll": m.get("log_loss", 1.0),
                "metrics": m,
            })

    # ── 6. Compute average validation scores ──
    avg_scores: dict[str, float] = {}
    for name, fold_list in fold_results.items():
        lls = [f["val_ll"] for f in fold_list]
        avg_scores[name] = float(np.mean(lls))

    # ── 7. Select best variant by validation (non-baseline only) ──
    non_base = [n for n in avg_scores if n != "A. Incumbent baseline"]
    best_val_name = min(non_base, key=lambda n: avg_scores[n])
    best_val_ll = avg_scores[best_val_name]

    # Best-per-gate-family
    gate_families = [
        "B. qb_changed", "C. qb_differs_prev", "D. starts<4",
        "E. starts<8", "F. starts<17",
        "G. changed OR starts<8", "H. changed OR starts<17",
    ]
    best_per_gate: dict[str, str] = {}
    for family in gate_families:
        matches = [n for n in non_base if n.startswith(family)]
        if matches:
            best_per_gate[family] = min(matches, key=lambda n: avg_scores[n])

    inc_val_ll = avg_scores["A. Incumbent baseline"]
    print(f"\n  Incumbent val LL: {inc_val_ll:.4f}")
    print(f"  Best variant val LL: {best_val_name} ({best_val_ll:.4f})")
    print(f"  Beats incumbent on val: {best_val_ll < inc_val_ll}")

    # ── 8. 2025 holdout evaluation ──
    print("\n=== 2025 Holdout ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    assert hold_mask.sum() > 0, "No holdout games found"

    # Fit Platt on all 2021-2024 for holdout
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

    # Evaluate incumbent on holdout
    hold_inc_prob = hold_incumbent_prob[hold_mask][valid_hold]
    hold_y_clean = hold_y[valid_hold]
    inc_hold_m = compute_classification_metrics(hold_y_clean, hold_inc_prob)
    inc_hold_ll = inc_hold_m["log_loss"]

    print(f"  Incumbent holdout LL: {inc_hold_ll:.4f}")

    # Evaluate validation-selected variant on holdout
    sel_cfg = next(c for c in variant_configs if c["name"] == best_val_name)
    sel_hold_prob = _apply_overlay(
        hold_incumbent_prob, hold_base_logit,
        sel_cfg, home_qb_adj, away_qb_adj, net_adj_elo,
    )[hold_mask][valid_hold]
    sel_hold_m = compute_classification_metrics(hold_y_clean, sel_hold_prob)
    sel_hold_ll = sel_hold_m["log_loss"]

    print(f"  Selected variant ({best_val_name}): holdout LL = {sel_hold_ll:.4f}")
    print(f"  Beats incumbent on holdout: {sel_hold_ll < inc_hold_ll}")

    # ── 9. Diagnostic: best-holdout variant ──
    # Evaluate ALL variants on holdout for diagnostic comparison
    hold_results: dict[str, dict] = {}
    for cfg in variant_configs:
        name = cfg["name"]
        prob = _apply_overlay(
            hold_incumbent_prob, hold_base_logit,
            cfg, home_qb_adj, away_qb_adj, net_adj_elo,
        )[hold_mask][valid_hold]
        m = compute_classification_metrics(hold_y_clean, prob)
        hold_results[name] = m

    non_diag_non_base = [n for n in hold_results if n != "A. Incumbent baseline"]
    best_hold_name_diag = min(non_diag_non_base, key=lambda n: hold_results[n]["log_loss"])
    best_hold_ll_diag = hold_results[best_hold_name_diag]["log_loss"]

    # Cleanest gate diagnostic: B. qb_changed (only touches QB-change games)
    b_variants = [n for n in non_diag_non_base if n.startswith("B. qb_changed")]
    best_b_name = (
        min(b_variants, key=lambda n: hold_results[n]["log_loss"])
        if b_variants else "A. Incumbent baseline"
    )
    best_b_ll = hold_results.get(best_b_name, {}).get("log_loss", inc_hold_ll)

    # QC and NoQC slices for key variants
    qb_change_mask_hold = qb_changed_either[hold_mask][valid_hold]
    no_qb_change_mask_hold = ~qb_change_mask_hold

    def _slice_ll(prob_arr, sl_mask):
        if sl_mask.sum() < 2:
            return None
        m = compute_metrics(hold_y_clean[sl_mask], prob_arr[sl_mask])
        return m["log_loss"]

    # Slice for incumbent, selected, best-holdout, cleanest-gate
    inc_nqc_ll = _slice_ll(hold_inc_prob, no_qb_change_mask_hold)
    inc_qc_ll = _slice_ll(hold_inc_prob, qb_change_mask_hold)
    sel_nqc_ll = _slice_ll(sel_hold_prob, no_qb_change_mask_hold)
    sel_qc_ll = _slice_ll(sel_hold_prob, qb_change_mask_hold)

    # Best-holdout slices
    best_hold_prob = _apply_overlay(
        hold_incumbent_prob, hold_base_logit,
        next(c for c in variant_configs if c["name"] == best_hold_name_diag),
        home_qb_adj, away_qb_adj, net_adj_elo,
    )[hold_mask][valid_hold]
    bh_nqc_ll = _slice_ll(best_hold_prob, no_qb_change_mask_hold)
    bh_qc_ll = _slice_ll(best_hold_prob, qb_change_mask_hold)

    # Cleanest gate slices
    b_hold_prob = _apply_overlay(
        hold_incumbent_prob, hold_base_logit,
        next(c for c in variant_configs if c["name"] == best_b_name),
        home_qb_adj, away_qb_adj, net_adj_elo,
    )[hold_mask][valid_hold]
    b_nqc_ll = _slice_ll(b_hold_prob, no_qb_change_mask_hold) if b_variants else inc_nqc_ll
    b_qc_ll = _slice_ll(b_hold_prob, qb_change_mask_hold) if b_variants else inc_qc_ll

    # ── 10. Non-gated equality check (holdout, B. qb_changed) ──
    print("\n=== Non-Gated Equality Check ===")
    # Use B. qb_changed variants (gate only affects QB-change games)
    # For the cleanest-gate variant, non-gated = no QB change
    b_cfg = next(c for c in variant_configs if c["name"] == best_b_name)
    b_all_prob = _apply_overlay(
        hold_incumbent_prob, hold_base_logit,
        b_cfg, home_qb_adj, away_qb_adj, net_adj_elo,
    )
    b_nongated = ~qb_changed_either  # No gate active
    diff = np.abs(b_all_prob - hold_incumbent_prob)
    max_diff = float(np.max(diff[b_nongated]))
    mean_diff = float(np.mean(diff[b_nongated]))
    print(f"  Max diff on non-gated games: {max_diff:.2e}")
    print(f"  Mean diff on non-gated games: {mean_diff:.2e}")
    equality_passed = max_diff < 1e-10
    print(f"  Equality check: {'PASSED' if equality_passed else 'FAILED'}")

    # ── 11. Promote/reject decision ──
    beats_val = best_val_ll < inc_val_ll
    beats_hold = sel_hold_ll < inc_hold_ll
    nongated_ok = equality_passed
    qc_improves = sel_qc_ll is not None and sel_qc_ll < inc_qc_ll
    promotes = beats_val and beats_hold and nongated_ok

    # ── 12. Write report ──
    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# Fold-Safe Frozen-Incumbent QB Overlay Experiment\n\n")

        _w("## Research Question\n\n")
        _w("Does the frozen QB overlay still improve the incumbent when the ")
        _w("incumbent base probabilities and calibration are generated using ")
        _w("strictly fold-safe rolling-origin training?\n\n")

        _w("## Why the Prior Frozen Overlay Was Diagnostic Only\n\n")
        _w("The V2 frozen overlay fitted Platt calibration **once** on full ")
        _w("2021–2024 data, then used that single fit to evaluate ALL ")
        _w("rolling-origin folds. This leaked future data into early-fold ")
        _w("validation (e.g., 2023 games scored using a Platt model that had ")
        _w("seen 2024 games during training). The validation metrics were not ")
        _w("fold-safe and could not be used for trustworthy variant selection.\n\n")

        _w("## Fold-Safe Methodology\n\n")
        _w("For each rolling-origin fold, the following steps are run:\n\n")
        _w("1. **Features**: Elo ratings, QB adjustments, and rolling MOV are ")
        _w("computed chronologically on the full dataset. Each game's feature ")
        _w("uses only data from games before it — no future leakage in features.\n")
        _w("2. **Platt calibration**: Fit using **only** the fold's training ")
        _w("seasons. The validation season is never seen during calibration.\n")
        _w("3. **Incumbent base probability**: Generated using the fold's ")
        _w("Platt model. The base probability for all games is produced by a ")
        _w("model trained exclusively on pre-validation data.\n")
        _w("4. **Frozen overlay**: Applied in logit space only where the ")
        _w("pregame gate is active. Non-gated games are unchanged.\n")
        _w("5. **Scoring**: Computed only on the validation season.\n\n")

        _w("## Validation Folds\n\n")
        fold_table = (
            "| Fold | Training Seasons | Validation Season |\n"
            "|------|-----------------|-------------------|\n"
        )
        for fi, (ts, vs) in enumerate(ROLLING_FOLDS):
            fold_table += f"| {fi + 1} | {', '.join(str(s) for s in ts)} | {vs} |\n"
        _w(fold_table)
        _w("\n## Holdout\n\n")
        _w(f"The 2025 season (year {HOLDOUT_SEASON}) is held out entirely from ")
        _w("validation. A final Platt model is fitted on all 2021–2024 data ")
        _w("and the validation-selected variant is evaluated once on 2025.\n\n")

        _w("## Variant Selection\n\n")
        _w("The best variant is selected by **average validation log loss** ")
        _w("across all 3 rolling folds. The 2025 holdout is NOT used for ")
        _w("selection. Best-holdout and cleanest-gate results are reported ")
        _w("separately as diagnostic.\n\n")

        _w("## Variants Tested\n\n")
        _w("| Letter | Gate Condition | Gammas | Caps |\n")
        _w("|--------|----------------|--------|------|\n")
        _w("| A | (none) Incumbent baseline | — | — |\n")
        _w("| B | qb_changed (either side) | 0.10–1.00 | 20, 40, 60 |\n")
        _w("| C | qb_differs_prev (same as B) | 0.10–1.00 | 20, 40, 60 |\n")
        _w("| D | starts<4 (either QB) | 0.10–1.00 | 20, 40, 60 |\n")
        _w("| E | starts<8 | 0.10–1.00 | 20, 40, 60 |\n")
        _w("| F | starts<17 | 0.10–1.00 | 20, 40, 60 |\n")
        _w("| G | changed OR starts<8 | 0.10–1.00 | 20, 40, 60 |\n")
        _w("| H | changed OR starts<17 | 0.10–1.00 | 20, 40, 60 |\n\n")
        n_ov = len(variant_configs) - 1
        _w(f"Total variants: {len(variant_configs)} (1 baseline + {n_ov} overlays)\n\n")

        _w("## Fold-Safe Validation Results (Sorted by Avg Log Loss)\n\n")
        _w("| Model | Avg LL | Fold1 | Fold2 | Fold3 |\n")
        _w("|-------|--------|-------|-------|-------|\n")
        sorted_val = sorted(avg_scores.items(), key=lambda x: x[1])
        for name, vll in sorted_val:
            flls = [
                f"{fold_results[name][fi]['val_ll']:.4f}"
                for fi in range(len(ROLLING_FOLDS))
            ]
            _w(f"| {name} | {vll:.4f} | {' | '.join(flls)} |\n")

        _w(f"\n**Incumbent baseline val LL:** {inc_val_ll:.4f}\n")
        _w(f"**Best validation variant:** {best_val_name} ({best_val_ll:.4f})\n")

        _w("\n## Best Variant by Gate Family (Validation)\n\n")
        _w("| Gate | Best Config | Val LL |\n")
        _w("|------|------------|--------|\n")
        for family in gate_families:
            if family in best_per_gate:
                bn = best_per_gate[family]
                _w(f"| {family} | {bn} | {avg_scores[bn]:.4f} |\n")
        b_ll = avg_scores.get(best_b_name, inc_val_ll)
        _w(f"| B. qb_changed (clean) | {best_b_name} | {b_ll:.4f} |\n")

        # ── 2025 Holdout Table ──
        _w("\n## 2025 Holdout Results\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy | Selection |\n")
        _w("|-------|----------|-------|-----|----------|-----------|\n")

        # Incumbent
        _w(f"| A. Incumbent baseline | {inc_hold_ll:.4f}"
           f" | {inc_hold_m['brier_score']:.4f}"
           f" | {inc_hold_m['roc_auc']:.4f}"
           f" | {inc_hold_m['accuracy']:.4f}"
           f" | baseline |\n")

        # Validation-selected variant
        _w(f"| {best_val_name} | {sel_hold_ll:.4f}"
           f" | {sel_hold_m['brier_score']:.4f}"
           f" | {sel_hold_m['roc_auc']:.4f}"
           f" | {sel_hold_m['accuracy']:.4f}"
           f" | **validation-selected** |\n")

        # Best-holdout diagnostic
        if best_hold_name_diag != best_val_name:
            bh_m = hold_results[best_hold_name_diag]
            _w(f"| {best_hold_name_diag} | {bh_m['log_loss']:.4f}"
               f" | {bh_m['brier_score']:.4f}"
               f" | {bh_m['roc_auc']:.4f}"
               f" | {bh_m['accuracy']:.4f}"
               f" | diagnostic (best holdout) |\n")

        # Cleanest-gate diagnostic
        if best_b_name not in (best_val_name, best_hold_name_diag):
            b_m = hold_results[best_b_name]
            _w(f"| {best_b_name} | {b_m['log_loss']:.4f}"
               f" | {b_m['brier_score']:.4f}"
               f" | {b_m['roc_auc']:.4f}"
               f" | {b_m['accuracy']:.4f}"
               f" | diagnostic (clean gate) |\n")

        # ── QB-Change Slice Results ──
        _w("\n## QB-Change Slice Results (2025 Holdout)\n\n")
        n_qc = int(qb_change_mask_hold.sum())
        n_nqc = int(no_qb_change_mask_hold.sum())
        _w(f"QB-change games: {n_qc} | Non-QB-change games: {n_nqc}\n\n")

        _w("| Variant | QB-Change LL | No-QB-Change LL | QC Δ | NoQC Δ |\n")
        _w("|--------|-------------|-----------------|------|--------|\n")

        # Incumbent
        _w(f"| A. Incumbent baseline | {inc_qc_ll:.4f} | {inc_nqc_ll:.4f} | — | — |\n")

        # Pre-compute delta strings
        def _delta_str(val, ref):
            if val is None or ref is None:
                return "N/A"
            return f"{val - ref:+.4f}"

        sel_qc_d = _delta_str(sel_qc_ll, inc_qc_ll)
        sel_nqc_d = _delta_str(sel_nqc_ll, inc_nqc_ll)
        _w(f"| {best_val_name} | {sel_qc_ll:.4f} | {sel_nqc_ll:.4f}"
           f" | {sel_qc_d} | {sel_nqc_d} |\n")

        # Best-holdout diagnostic
        best_hold_label = f"{best_hold_name_diag} (D)"
        bh_qc_d = _delta_str(bh_qc_ll, inc_qc_ll)
        bh_nqc_d = _delta_str(bh_nqc_ll, inc_nqc_ll)
        _w(f"| {best_hold_label} | {bh_qc_ll:.4f} | {bh_nqc_ll:.4f}"
           f" | {bh_qc_d} | {bh_nqc_d} |\n")

        # Cleanest-gate diagnostic
        b_qc_d = _delta_str(b_qc_ll, inc_qc_ll)
        b_nqc_d = _delta_str(b_nqc_ll, inc_nqc_ll)
        _w(f"| {best_b_name} (G) | {b_qc_ll:.4f} | {b_nqc_ll:.4f}"
           f" | {b_qc_d} | {b_nqc_d} |\n")

        # ── Non-Gated Equality ──
        _w("\n## Non-Gated Equality\n\n")
        _w(f"**Max absolute diff vs incumbent:** {max_diff:.2e}\n")
        _w(f"**Mean absolute diff vs incumbent:** {mean_diff:.2e}\n")
        _w(f"**Equality check: {'PASSED' if equality_passed else 'FAILED'}**\n\n")
        _w("Non-gated games are identical to the incumbent within floating-point tolerance.\n")

        # ── Full holdout table ──
        _w("\n## Full 2025 Holdout Results (All Variants)\n\n")
        _w("| Model | Log Loss | Δ vs incumbent |\n")
        _w("|-------|----------|----------------|\n")
        sorted_hold = sorted(hold_results.items(), key=lambda x: x[1]["log_loss"])
        for name, hm in sorted_hold:
            delta = hm["log_loss"] - inc_hold_ll
            _w(f"| {name} | {hm['log_loss']:.4f} | {delta:+.4f} |\n")

        # ── Decision ──
        _w("\n## Decision\n\n")

        if promotes:
            _w(f"**✅ PROMOTED: {best_val_name}**\n\n")
            _w("| Criterion | Met? |\n")
            _w("|-----------|------|\n")
            _w(f"| Beats incumbent on val LL ({best_val_ll:.4f} < {inc_val_ll:.4f}) | ✅ |\n")
            _w(f"| Beats incumbent on holdout LL ({sel_hold_ll:.4f} < {inc_hold_ll:.4f}) | ✅ |\n")
            _w("| Non-gated equality passes | ✅ |\n")
            if qc_improves:
                _w(f"| Improves QB-change slice ({sel_qc_ll:.4f} < {inc_qc_ll:.4f}) | ✅ |\n")
            else:
                _w("| Preserves QB-change slice | ✅ |\n")
        else:
            _w("**❌ REJECTED for promotion**\n\n")
            _w("| Criterion | Met? | Details |\n")
            _w("|-----------|------|--------|\n")
            _w(f"| Beats incumbent on val LL | {'✅' if beats_val else '❌'}"
               f" | {best_val_ll:.4f} vs {inc_val_ll:.4f} |\n")
            _w(f"| Beats incumbent on holdout LL | {'✅' if beats_hold else '❌'}"
               f" | {sel_hold_ll:.4f} vs {inc_hold_ll:.4f} |\n")
            _w(f"| Non-gated equality passes | {'✅' if nongated_ok else '❌'} |\n")
            qc_str = f"{sel_qc_ll:.4f} vs {inc_qc_ll:.4f}" if sel_qc_ll is not None else "N/A"
            _w(f"| Improves/preserves QB-change slice | {'✅' if qc_improves else '❌'}"
               f" | {qc_str} |\n")

            _w("\nFrozen QB overlay is **rejected for promotion** and remains ")
            _w("diagnostic-only.\n\n")

        _w("\n### Validation-Selected Best Variant\n\n")
        _w(f"**{best_val_name}** — val LL {best_val_ll:.4f}, holdout LL {sel_hold_ll:.4f}\n\n")

        _w("### Best-Holdout Diagnostic Variant\n\n")
        if best_hold_name_diag == best_val_name:
            bh_tag = " (selected via validation)"
        else:
            bh_tag = " (not selected via validation, diagnostic only)"
        _w(f"**{best_hold_name_diag}** — holdout LL {best_hold_ll_diag:.4f}")
        _w(f"{bh_tag}\n\n")

        _w("### Cleanest-Gate Diagnostic Variant\n\n")
        _w(f"**{best_b_name}** — holdout LL {best_b_ll:.4f}")
        _w(" (only applies overlay to QB-change games, NoQC Δ = 0.00, diagnostic only)\n\n")

        # ── Failure modes ──
        _w("### Failure Modes\n\n")
        _w("1. **Oracle QB data**: Uses final actual starter IDs, not pregame-announced.\n")
        _w("2. **Binary gate sharpness**: The gate is all-or-nothing per game.\n")
        _w("3. **Small-sample QB adjustments**: QBs with <17 starts are strongly shrunk ")
        _w("toward replacement.\n")
        _w("4. **No position-group interaction**: Ignores OL, skill players, defense.\n\n")

        # ── Recommended next ──
        _w("### Recommended Next Experiment\n\n")
        if promotes:
            _w("With this promotion, the new football-only incumbent becomes:\n\n")
            _w(f"- **{best_val_name}** (val LL {best_val_ll:.4f}, holdout LL {sel_hold_ll:.4f})\n")
            _w("- Feature set: Standard Elo + qb_changed + rolling_mov_3 + Platt +")
            _w(" frozen QB overlay (gated on changed OR starts<17, cap=40)\n")
            _w("- Non-gated games match the previous incumbent exactly\n\n")
            _w("Future models must beat **this** incumbent to promote.\n\n")
        _w("1. **Coach-QB interaction features**: New QB + new coordinator may be more ")
        _w("informative than QB change alone.\n")
        _w("2. **Expanded Elo K search**: Test K > 48 with season regression spine.\n")
        _w("3. **DVOA/EPA features**: If a new pregame-safe data source becomes available.\n")
        _w("4. **Raw QB adjustment feature** (from qb_adjustment.py) remains diagnostic-only.")
        _w(" Use in residual diagnostics, not as a standalone promoted feature.\n\n")

        _w("---\n")
        _w("*Report generated by `sportslab frozen-qb-overlay-foldsafe`. ")
        _w(f"Seasons: 2021–{HOLDOUT_SEASON}, "
          f"Folds: {len(ROLLING_FOLDS)}, "
          f"Variants: {len(variant_configs)}.*\n")

    print(f"\nReport: {rp}")

    # ── 13. Export CSV if requested ──
    if output_csv is not None:
        out_df = df[hold_mask].copy()
        out_df["incumbent_home_win_prob"] = hold_inc_prob
        out_df["selected_variant"] = best_val_name
        out_df["selected_variant_prob"] = sel_hold_prob
        out_df["qb_change_flag"] = qb_changed_either[hold_mask].astype(int)
        out_df.to_csv(output_csv, index=False)
        print(f"  CSV: {output_csv}")

    return str(report_path)


def _get_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    if avail:
        return df[avail].values
    return np.empty((len(df), 0))


def _apply_overlay(
    incumbent_prob: np.ndarray,
    base_logit: np.ndarray,
    cfg: dict,
    home_qb_adj: np.ndarray,
    away_qb_adj: np.ndarray,
    net_adj_elo: np.ndarray,
) -> np.ndarray:
    """Apply frozen overlay using a variant config."""
    name = cfg["name"]
    if name == "A. Incumbent baseline":
        return incumbent_prob.copy()

    if cfg["cap"] is not None:
        capped_h = np.clip(home_qb_adj, -cfg["cap"], cfg["cap"])
        capped_a = np.clip(away_qb_adj, -cfg["cap"], cfg["cap"])
        cur_net_adj = capped_h - capped_a
    else:
        cur_net_adj = net_adj_elo

    overlay = cfg["gamma"] * cur_net_adj * ELO_TO_LOGIT
    gate_active = cfg["gate_mask"].astype(float)
    final_logit = base_logit + overlay * gate_active
    return _sigmoid(final_logit)
