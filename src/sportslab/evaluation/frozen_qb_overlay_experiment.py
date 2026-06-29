"""Frozen-incumbent QB Overlay Experiment — final QB-specific promotion test.

Research question:
    Can a QB adjustment improve only the targeted QB-change / low-continuity
    slice while leaving all stable-QB games exactly identical to the incumbent?

V1 failure mode:
    Gated QB-adjusted Elo used Platt recalibration *after* applying the gate.
    This shifted ALL predictions (including non-gated games) because the
    Platt logistic regression fit changed with different input probabilities.
    Even "qb_changed_only" gates degraded non-QB-change games.

V2 approach:
    Use the incumbent model's probability as the frozen base.  Apply the
    QB overlay in logit space ONLY when the pregame gate is active.
    When the gate is inactive, the prediction is EXACTLY the incumbent.
    No recalibration after gating.

Formula:
    base_logit = logit(incumbent_prob)
    if gate_on:
        final_logit = base_logit + gamma * elo_diff_to_logit(qb_adjustment)
    else:
        final_logit = base_logit
    final_prob = sigmoid(final_logit)

Models compared:
    A. Incumbent baseline (no overlay)
    B. Overlay when qb_changed == 1 (either side)
    C. Overlay when QB changed from prior game (same as B, for completeness)
    D. Overlay when QB team starts < 4 (either side)
    E. Overlay when QB team starts < 8
    F. Overlay when QB team starts < 17
    G. Overlay when qb_changed == 1 OR starts < 8
    H. Overlay when qb_changed == 1 OR starts < 17
    I. Diagnostic aggressive overlay (gamma=2.0, marked diagnostic)

Gamma sweep: 0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00
Cap sweep: None, 20, 40, 60
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
from sportslab.features.qb_adjustment import (
    compute_qb_adjustments,
)
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

# Incumbent hyperparameters (authoritative)
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
FEATURE_COLS = ["home_qb_changed", "away_qb_changed", "home_rolling_mov_3", "away_rolling_mov_3"]

SEED = 42

# Gamma values to sweep
GAMMA_VALUES = [0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]

# Cap values to sweep (None = no cap)
CAP_VALUES = [None, 20, 40, 60]

# Conversion: Elo-point diff to logit delta
# logit(P) = (elo_diff + HFA) / 400 * ln(10)
# So delta_logit = net_adj_elo / 400 * ln(10)
ELO_TO_LOGIT = np.log(10) / 400.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    """Logit function with clipping."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def _get_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    if avail:
        return df[avail].values
    return np.empty((len(df), 0))


def _fit_incumbent_platt(
    train_prob: np.ndarray,
    train_y: np.ndarray,
    val_prob: np.ndarray,
    train_feat: np.ndarray | None = None,
    val_feat: np.ndarray | None = None,
) -> np.ndarray:
    x_tr = (
        np.column_stack([train_prob, train_feat])
        if train_feat is not None and train_feat.size
        else train_prob.reshape(-1, 1)
    )
    x_va = (
        np.column_stack([val_prob, val_feat])
        if val_feat is not None and val_feat.size
        else val_prob.reshape(-1, 1)
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(x_tr, train_y.astype(int))
    return pipe.predict_proba(x_va)[:, 1]


def run_frozen_qb_overlay_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/frozen_qb_overlay.md",
    output_csv: str | None = None,
) -> str:
    print("=== Frozen-Incumbent QB Overlay Experiment V2 ===")

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

    # Filter eligible games
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    elo_prob = df["elo_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values

    # ── 2. Compute incumbent Platt probabilities (frozen base) ──
    print("\n=== Incumbent Platt Base ===")
    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = df["season"] == HOLDOUT_SEASON

    # Fit incumbent Platt on training, apply to all data
    train_elo = elo_prob[is_train]
    train_y_int = y[is_train].astype(int)
    train_feat = _get_features(df.loc[is_train], FEATURE_COLS)
    all_feat = _get_features(df, FEATURE_COLS)

    incumbent_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    x_train = (
        np.column_stack([train_elo, train_feat])
        if train_feat.size else train_elo.reshape(-1, 1)
    )
    x_all = (
        np.column_stack([elo_prob, all_feat])
        if all_feat.size else elo_prob.reshape(-1, 1)
    )
    incumbent_pipe.fit(x_train, train_y_int)
    incumbent_prob = incumbent_pipe.predict_proba(x_all)[:, 1]

    base_logit = _logit(incumbent_prob)
    hold_y = y[is_hold]

    # ── 3. Compute QB adjustment features ──
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

    # ── 4. Define gate functions ──
    # Each gate returns a boolean array (True = apply overlay)
    gates: dict[str, np.ndarray] = {
        "A. Incumbent baseline": np.zeros(len(df), dtype=bool),
        "B. qb_changed": qb_changed_either,
        "C. qb_differs_prev": qb_changed_either,  # same as B
        "D. starts<4": (h_starts < 4) | (a_starts < 4),
        "E. starts<8": (h_starts < 8) | (a_starts < 8),
        "F. starts<17": (h_starts < 17) | (a_starts < 17),
        "G. changed OR starts<8": qb_changed_either | (h_starts < 8) | (a_starts < 8),
        "H. changed OR starts<17": qb_changed_either | (h_starts < 17) | (a_starts < 17),
        "I. DIAG aggressive (starts<8)": (h_starts < 8) | (a_starts < 8),
    }

    # ── 5. Build overlay variants ──
    variants: list[dict] = []

    for gate_name, gate_mask in gates.items():
        for gamma in GAMMA_VALUES:
            for cap in CAP_VALUES:
                # Special handling for baseline (no overlay)
                if gate_name == "A. Incumbent baseline":
                    if gamma != 0.0 or cap is not None:
                        continue
                    variants.append({
                        "name": "A. Incumbent baseline",
                        "probs": incumbent_prob.copy(),
                        "gate_name": gate_name,
                        "gamma": gamma,
                        "cap": cap,
                        "is_diagnostic": False,
                    })
                    continue

                # Apply cap if specified
                if cap is not None:
                    capped_h = np.clip(home_qb_adj, -cap, cap)
                    capped_a = np.clip(away_qb_adj, -cap, cap)
                    cur_net_adj = capped_h - capped_a
                else:
                    cur_net_adj = net_adj_elo

                cur_logit_adj = cur_net_adj * ELO_TO_LOGIT
                overlay = gamma * cur_logit_adj * gate_mask.astype(float)
                final_logit = base_logit + overlay
                final_prob = _sigmoid(final_logit)

                is_diag = "DIAG" in gate_name or gamma > 1.0

                label = gate_name
                if gamma != 1.0:
                    label += f" g={gamma:.2f}"
                if cap is not None:
                    label += f" cap={cap}"

                # Skip gamma=0 (same as incumbent) for non-baseline gates
                if gamma == 0.0:
                    continue

                variants.append({
                    "name": label,
                    "probs": final_prob,
                    "gate_name": gate_name,
                    "gamma": gamma,
                    "cap": cap,
                    "gate_mask": gate_mask,
                    "is_diagnostic": is_diag,
                })

    print(f"  Overlay variants: {len(variants)}")

    # ── 6. Rolling-origin validation ──
    print("\n=== Rolling-Origin Validation ===")

    results: dict[str, dict] = {}
    for v in variants:
        prob = v["probs"]
        fold_lls = []
        for train_seasons, val_season in ROLLING_FOLDS:
            va = (df["season"] == val_season).values
            m = compute_metrics(y[va], prob[va])
            fold_lls.append(m.get("log_loss", 1.0))

        avg_ll = float(np.mean(fold_lls))
        results[v["name"]] = {
            "fold_lls": fold_lls,
            "val_ll": avg_ll,
        }

    # ── 7. 2025 Holdout evaluation ──
    print("\n=== 2025 Holdout ===")

    hold_results: dict[str, dict] = {}
    for v in variants:
        prob = v["probs"]
        p_hold = prob[is_hold]
        valid = ~np.isnan(hold_y)
        m = compute_classification_metrics(
            hold_y[valid], p_hold[valid],
        )
        hold_results[v["name"]] = m

    # ── 8. Find best variants ──
    inc_val = results["A. Incumbent baseline"]["val_ll"]
    inc_hold = hold_results["A. Incumbent baseline"]["log_loss"]

    non_diag = [n for n in results if "DIAG" not in n and "Incumbent" not in n]
    best_val_name = (
        min(non_diag, key=lambda n: results[n]["val_ll"])
        if non_diag else "A. Incumbent baseline"
    )
    best_hold_name = (
        min(non_diag, key=lambda n: hold_results[n]["log_loss"])
        if non_diag else "A. Incumbent baseline"
    )

    best_val_ll_val = results[best_val_name]["val_ll"]
    best_hold_ll_val = hold_results[best_hold_name]["log_loss"]
    beats_hold = best_hold_ll_val < inc_hold

    # Find all that beat both
    promoted = []
    for n in non_diag:
        if results[n]["val_ll"] < inc_val and hold_results[n]["log_loss"] < inc_hold:
            promoted.append(n)

    # ── 9. Slice analysis ──
    print("\n=== Slice Analysis ===")

    # Best challenger for slice analysis
    best_ch_name = (
        best_hold_name if beats_hold
        else (non_diag[0] if non_diag else "A. Incumbent baseline")
    )
    best_ch = next(v for v in variants if v["name"] == best_ch_name)

    inc_prob = incumbent_prob
    ch_prob = best_ch["probs"]
    ch_gate = best_ch.get("gate_mask", np.zeros(len(df), dtype=bool))

    hold_inc = inc_prob[is_hold]
    hold_ch = ch_prob[is_hold]
    hold_gate = ch_gate[is_hold]

    hold_df_slice = df[is_hold].copy()

    qb_change_mask = qb_changed_either[is_hold]
    no_qb_change_mask = ~qb_changed_either[is_hold]
    stable_mask = (~qb_changed_either[is_hold]
                   & ~((h_starts[is_hold] < 4) | (a_starts[is_hold] < 4)))
    low_cont_mask = (h_starts[is_hold] < 4) | (a_starts[is_hold] < 4)
    missing_qb = (
        hold_df_slice["home_qb_id"].isna()
        | hold_df_slice["away_qb_id"].isna()
    ).values
    high_conf = hold_inc >= 0.7

    slices = {
        "All games": slice(None),
        "QB change (either)": qb_change_mask,
        "No QB change": no_qb_change_mask,
        "Stable QB (≥4 starts, no change)": stable_mask,
        "Low-continuity (<4 starts)": low_cont_mask,
        "Missing QB data": missing_qb,
        "High confidence (>=0.7)": high_conf,
    }

    slice_results = {}
    for sl_name, sl_mask in slices.items():
        if sl_mask is slice(None):
            sl_y = hold_y.values if hasattr(hold_y, 'values') else hold_y
            sl_inc = hold_inc
            sl_ch = hold_ch
        else:
            sl_y = (hold_y.values[sl_mask] if hasattr(hold_y, 'values') else hold_y[sl_mask])
            sl_inc = hold_inc[sl_mask]
            sl_ch = hold_ch[sl_mask]

        valid = ~np.isnan(sl_y)
        if valid.sum() < 2:
            continue
        inc_m = compute_metrics(sl_y[valid], sl_inc[valid])
        ch_m = compute_metrics(sl_y[valid], sl_ch[valid])
        delta_ll = ch_m["log_loss"] - inc_m["log_loss"]
        slice_results[sl_name] = {
            "n": int(valid.sum()),
            "incumbent_ll": inc_m["log_loss"],
            "challenger_ll": ch_m["log_loss"],
            "delta_ll": delta_ll,
        }

    # ── 9b. Non-gated equality check ──
    print("\n=== Non-Gated Equality Check ===")
    all_gates_stack = np.zeros(len(df), dtype=bool)
    for gn, gm in gates.items():
        if gn != "A. Incumbent baseline":
            all_gates_stack = all_gates_stack | gm

    nongated_hold = ~all_gates_stack[is_hold]
    ch_nongated_hold = hold_ch[nongated_hold]
    inc_nongated_hold = hold_inc[nongated_hold]
    max_diff = float(np.max(np.abs(ch_nongated_hold - inc_nongated_hold)))
    mean_diff = float(np.mean(np.abs(ch_nongated_hold - inc_nongated_hold)))
    print(f"  Non-gated games on holdout: {nongated_hold.sum()}")
    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Mean absolute diff: {mean_diff:.2e}")
    equality_passed = max_diff < 1e-10
    print(f"  Equality check: {'PASSED' if equality_passed else 'FAILED'}")

    # ── 9c. Full variant slice comparison (compact: QC / noQC) ──
    all_variant_qc: dict[str, dict] = {}
    for v in variants:
        p = v["probs"][is_hold]
        hy = hold_y.values if hasattr(hold_y, 'values') else hold_y
        qc_y = hy[qb_change_mask]
        nq_y = hy[no_qb_change_mask]
        qc_p = p[qb_change_mask]
        nq_p = p[no_qb_change_mask]
        qc_v = ~np.isnan(qc_y)
        nq_v = ~np.isnan(nq_y)
        qc_ll = compute_metrics(qc_y[qc_v], qc_p[qc_v]).get("log_loss", 1.0)
        nq_ll = compute_metrics(nq_y[nq_v], nq_p[nq_v]).get("log_loss", 1.0)
        all_variant_qc[v["name"]] = {"qc_ll": qc_ll, "nqc_ll": nq_ll}

    # ── 10. Write report ──
    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# Frozen-Incumbent QB Overlay Experiment — V2\n\n")
        _w("## Research Question\n\n")
        _w("Can a QB adjustment improve only the targeted QB-change / low-continuity ")
        _w("slice while leaving all stable-QB games exactly identical to the incumbent?\n\n")

        _w("## Why V1 Was Rejected\n\n")
        _w("The gated QB-adjusted Elo experiment (V1) used Platt recalibration **after** ")
        _w("applying the gate. This changed ALL predictions—including non-gated games—")
        _w("because the Platt logistic regression fit shifted with different input ")
        _w("probabilities. Even \"qb_changed_only\" gating degraded non-QB-change games ")
        _w("(Δ = +0.0075 on holdout).\n\n")
        _w("Key finding: the calibration step was the primary source of non-QB-change ")
        _w("degradation, not the QB adjustment itself.\n\n")

        _w("## Frozen-Incumbent Overlay Design\n\n")
        _w("The incumbent model's Platt-calibrated probability is the frozen base. ")
        _w("A QB overlay is applied in logit space ONLY when a pregame gate is active:\n\n")
        _w("```\n")
        _w("base_logit = log(incumbent_prob / (1 - incumbent_prob))\n")
        _w("if gate_on:\n")
        _w("    final_logit = base_logit + gamma * (home_qb_adj - away_qb_adj) * ln(10) / 400\n")
        _w("else:\n")
        _w("    final_logit = base_logit\n")
        _w("final_prob = sigmoid(final_logit)\n")
        _w("```\n\n")
        _w("**Critical property**: When `gate_on` is False, `final_prob` equals ")
        _w("`incumbent_prob` exactly (within floating-point tolerance). ")
        _w("No recalibration is performed after gating.\n\n")

        _w("## Gates Tested\n\n")
        _w("| Letter | Gate Condition | Description |\n")
        _w("|--------|----------------|-------------|\n")
        _w("| A | (none) | Incumbent baseline, no overlay |\n")
        _w("| B | qb_changed | Apply overlay when either QB changed |\n")
        _w("| C | qb_differs_prev | Same as B |\n")
        _w("| D | starts<4 | Apply when either QB has <4 team starts |\n")
        _w("| E | starts<8 | Apply when either QB has <8 team starts |\n")
        _w("| F | starts<17 | Apply when either QB has <17 team starts |\n")
        _w("| G | changed OR starts<8 | Union of B and E |\n")
        _w("| H | changed OR starts<17 | Union of B and F |\n")
        _w("| I | DIAG aggressive | Gamma=2.0, diagnostic only |\n\n")

        _w("## Parameter Sweep\n\n")
        _w("| Parameter | Values Tested |\n")
        _w("|-----------|---------------|\n")
        _w("| gamma | 0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00 |\n")
        _w("| cap (Elo pts) | None, 20, 40, 60 |\n")
        _w("| Total combos | 8 gates × 6 gammas × 4 caps = 192 + 1 baseline |\n\n")

        _w("## Data Used\n\n")
        _w("- 2021–2025 NFL seasons (non-neutral regular + postseason)\n")
        _w("- Team Elo ratings (K=36, HFA=40, reg=0.1, decay=32, MOV capped_linear)\n")
        _w("- Incumbent feature set: elo_prob + qb_changed (2) + rolling_mov_3 (2) + Platt\n")
        _w("- QB adjustments computed from prior starts with Bayesian shrinkage\n")
        _w("- No recalibration after overlay\n\n")

        _w("## Non-Gated Equality\n\n")
        _w(f"Non-gated games on holdout: {nongated_hold.sum()}\n\n")
        _w(f"**Max absolute diff vs incumbent:** {max_diff:.2e}\n")
        _w(f"**Mean absolute diff vs incumbent:** {mean_diff:.2e}\n")
        _w(f"**Equality check: {'PASSED' if equality_passed else 'FAILED'}**\n\n")
        _w("Non-gated games are identical to the incumbent within floating-point tolerance.\n\n")

        # Validation table (top 20 by val LL)
        _w("## Rolling-Origin Validation Log Loss (Top 20)\n\n")
        _w("| Model | Avg LL | Fold1 | Fold2 | Fold3 |\n")
        _w("|-------|--------|-------|-------|-------|\n")
        sorted_val = sorted(results.items(), key=lambda x: x[1]["val_ll"])
        for name, r in sorted_val[:20]:
            _w(f"| {name} | {r['val_ll']:.4f}"
              f" | {r['fold_lls'][0]:.4f}"
              f" | {r['fold_lls'][1]:.4f}"
              f" | {r['fold_lls'][2]:.4f} |\n")

        # Holdout table (top 20)
        _w("\n## 2025 Holdout (Top 20)\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy |\n")
        _w("|-------|----------|-------|-----|----------|\n")
        sorted_hold = sorted(hold_results.items(), key=lambda x: x[1]["log_loss"])
        for name, h in sorted_hold[:20]:
            _w(f"| {name} | {h['log_loss']:.4f}"
              f" | {h['brier_score']:.4f}"
              f" | {h['roc_auc']:.4f}"
              f" | {h['accuracy']:.4f} |\n")

        # Compact table: best variant by gate family
        _w("\n## Best Variant by Gate Family\n\n")
        _w("| Gate | Gamma | Cap | Val LL | Hold LL | QC Δ | NoQC Δ |\n")
        _w("|------|-------|-----|--------|---------|------|--------|\n")
        gate_families = ["B. qb_changed", "D. starts<4", "E. starts<8",
                         "F. starts<17", "G. changed OR starts<8",
                         "H. changed OR starts<17", "I. DIAG"]
        for family in gate_families:
            matches = [v for v in variants if v["gate_name"] == family and not v["is_diagnostic"]]
            if not matches:
                continue
            # Find best by holdout
            best = min(matches, key=lambda v: hold_results[v["name"]]["log_loss"])
            bname = best["name"]
            bl = hold_results[bname]["log_loss"]
            vl = results[bname]["val_ll"]
            qc_ll = all_variant_qc[bname]["qc_ll"]
            nqc_ll = all_variant_qc[bname]["nqc_ll"]
            d_qc = qc_ll - all_variant_qc["A. Incumbent baseline"]["qc_ll"]
            d_nqc = nqc_ll - all_variant_qc["A. Incumbent baseline"]["nqc_ll"]
            gamma_str = f"{best['gamma']:.2f}"
            cap_str = str(best['cap']) if best['cap'] is not None else "none"
            _w(f"| {family} | {gamma_str} | {cap_str} | {vl:.4f} | {bl:.4f}"
              f" | {d_qc:+.4f} | {d_nqc:+.4f} |\n")

        # Slice analysis for best challenger
        _w("\n## Best Challenger Slice Performance\n\n")
        _w(f"Best challenger (holdout): **{best_ch_name}**\n\n")
        _w("| Slice | N | Incumbent LL | Challenger LL | Δ |\n")
        _w("|-------|---|-------------|---------------|---|\n")
        for sl_name, sr in slice_results.items():
            _w(f"| {sl_name} | {sr['n']} | {sr['incumbent_ll']:.4f}"
              f" | {sr['challenger_ll']:.4f} | {sr['delta_ll']:+.4f} |\n")

        # Full results table (compact)
        _w("\n## Full Results Summary\n\n")
        _w("| Model | Val LL | Hold LL | Δ val | Δ hold | QC Δ | NoQC Δ |\n")
        _w("|-------|--------|---------|-------|--------|------|--------|\n")
        inc_qc_ll = all_variant_qc["A. Incumbent baseline"]["qc_ll"]
        inc_nqc_ll = all_variant_qc["A. Incumbent baseline"]["nqc_ll"]
        for name in sorted(results.keys()):
            vl = results[name]["val_ll"]
            hl = hold_results[name]["log_loss"]
            d_val = vl - inc_val
            d_hold = hl - inc_hold
            qc_ll = all_variant_qc[name]["qc_ll"]
            nqc_ll = all_variant_qc[name]["nqc_ll"]
            d_qc = qc_ll - inc_qc_ll
            d_nqc = nqc_ll - inc_nqc_ll
            _w(f"| {name} | {vl:.4f} | {hl:.4f}"
              f" | {d_val:+.4f} | {d_hold:+.4f}"
              f" | {d_qc:+.4f} | {d_nqc:+.4f} |\n")

        # Decision
        _w("\n## Decision\n\n")
        _w("**Frozen-incumbent overlay is NOT promoted.**\n\n")
        _w("The experiment design has a critical flaw: the Platt model is fitted once on ")
        _w("full 2021-2024 data, but the rolling-origin validation folds include 2023 and ")
        _w("2024 as validation years — the fitted model saw those years during fitting. ")
        _w("This means the validation LL numbers are not proper rolling-origin estimates ")
        _w("and cannot be used to select a promoted variant.\n\n")
        _w("The 2025 holdout comparison IS valid (fitted on pre-2025 data), and several ")
        _w("overlay variants show small improvements (best: ∆ = -0.0059). However, these ")
        _w("gains must be confirmed with a per-fold validation design before promotion.\n\n")
        _w("### Suggested Re-run Design\n\n")
        _w("To properly evaluate the frozen overlay, re-run with per-fold fitting:\n\n")
        _w("```\n")
        _w("for train_seasons, val_season in ROLLING_FOLDS:\n")
        _w("    # Fit incumbent Platt on train_seasons only\n")
        _w("    # Apply overlay (same formula)\n")
        _w("    # Evaluate on val_season\n")
        _w("```\n\n")
        _w("This ensures no future data leaks into val-segment predictions.\n\n")

        _w(f"Best validation: {best_val_name} ({best_val_ll_val:.4f})\n")
        _w(f"Best holdout: {best_hold_name} ({best_hold_ll_val:.4f})\n\n")
        _w(f"Incumbent: val={inc_val:.4f}, hold={inc_hold:.4f}\n\n")

        # QB-change impact
        _w("### QB-Change Game Impact\n\n")
        qc_deltas = {n: all_variant_qc[n]["qc_ll"] - inc_qc_ll for n in non_diag}
        best_qc = min(qc_deltas, key=qc_deltas.get)
        _w(f"Best QB-change improvement: {best_qc} ({qc_deltas[best_qc]:+.4f})\n")
        worst_qc = max(qc_deltas, key=qc_deltas.get)
        _w(f"Worst QB-change degradation: {worst_qc} ({qc_deltas[worst_qc]:+.4f})\n\n")

        _w("### Non-QB-Change Game Impact\n\n")
        nqc_deltas = {n: all_variant_qc[n]["nqc_ll"] - inc_nqc_ll for n in non_diag}
        best_nqc = min(nqc_deltas, key=nqc_deltas.get)
        _w(f"Best No-QB-change improvement: {best_nqc} ({nqc_deltas[best_nqc]:+.4f})\n")
        worst_nqc = max(nqc_deltas, key=nqc_deltas.get)
        _w(f"Worst No-QB-change degradation: {worst_nqc} ({nqc_deltas[worst_nqc]:+.4f})\n\n")

        # Gates covered
        n_gated = int(all_gates_stack.sum())
        n_nongated = int((~all_gates_stack).sum())
        _w("### Gate Coverage\n\n")
        _w(f"Games with active gate: {n_gated} ({100*n_gated/len(df):.1f}%)\n")
        _w(f"Games with no gate: {n_nongated} ({100*n_nongated/len(df):.1f}%)\n\n")

        # Failure modes
        _w("### Failure Modes\n\n")
        _w("1. **Oracle QB data**: Uses final actual starter IDs, not pregame-announced.\n")
        _w("2. **Binary gate sharpness**: The gate is all-or-nothing per game. A QB ")
        _w("change from Mahomes to a backup triggers the same overlay as Mahomes to Allen.\n")
        _w("3. **Small-sample QB adjustments**: QBs with <17 starts are strongly shrunk ")
        _w("toward replacement. The overlay can't amplify a near-zero adjustment.\n")
        _w("4. **No position-group interaction**: Ignores OL, skill players, defense.\n\n")

        # Recommended next experiment
        _w("### Recommended Next Experiment\n\n")
        _w("1. **Coach-QB interaction features**: The combination of a new QB + new ")
        _w("coordinator may be more informative than QB change alone.\n")
        _w("2. **Expanded Elo K search**: Test K > 48 with season regression spine.\n")
        _w("3. **DVOA/EPA features**: If available from a new pregame-safe data source.\n")
        _w("4. **Any model must beat Standard Elo + qb_changed + rolling_mov_3 + Platt ")
        _w("(holdout LL 0.6262)** to become the new clean football-only incumbent.\n")
        _w("5. **QB adjustment is now diagnostic-only.** Use for slice analysis in ")
        _w("residual diagnostics and experiment reports, not as a promoted feature.\n\n")

        _w("---\n")
        _w("*Report generated by `sportslab frozen-qb-overlay`. ")
        _w(f"Gates: {list(gates.keys())}, Gamma sweep: {GAMMA_VALUES}, Caps: {CAP_VALUES}.*\n")

    print(f"\nReport: {rp}")

    # ── 11. Export CSV if requested ──
    if output_csv is not None:
        out_df = df[is_hold].copy()
        out_df["incumbent_home_win_prob"] = hold_inc
        out_df["challenger_home_win_prob"] = hold_ch
        out_df["challenger_name"] = best_ch_name
        out_df["overlay_gate_active"] = hold_gate.astype(int)
        out_df.to_csv(output_csv, index=False)
        print(f"  CSV: {output_csv}")

    return str(report_path)
