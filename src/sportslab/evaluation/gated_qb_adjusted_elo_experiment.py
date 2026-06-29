"""Gated QB-Adjusted Elo Experiment — V1: pregame-safe gating.

Research question:
    Can QB-adjusted Elo improve games with meaningful QB uncertainty
    or QB changes without degrading games where the starting QB
    situation is stable?

V0 finding:
    QB adjustment helped QB-change games (Δ = -0.0035 holdout) but
    hurt non-QB-change games (Δ = +0.0046), causing overall holdout
    drag (+0.0030).  The adjustment adds noise for stable-QB games.

V1 approach:
    Gate the QB adjustment so it only affects games where the QB
    situation has recently changed or is uncertain.  Test 7+ gating
    variants with hyperparameter sweeps.

Models compared:
    A. Incumbent baseline (Elo + qb_changed + rolling_mov_3 + Platt)
    B. Full QB adjustment (V0)
    C. QB adjustment only when qb_changed == 1
    D. QB adjustment only when low continuity (starts < threshold)
    E. Shrunk QB adjustment for stable starters, full for changed
    F. Capped QB adjustment (lower max)
    G. Diagnostic aggressive gated (2x changed, 0x stable)
    H. Recency-weighted QB adjustment (decayed older games)
    I. Low continuity + capped combined

Slices:
    - QB-change games
    - non-QB-change games
    - Stable QB games
    - Low-sample QB games
    - Missing QB data games
    - High-confidence predictions
    - Home favorites / away underdogs
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
    GATE_MODES,
    compute_gated_qb_adjustments,
    compute_qb_adjusted_elo_prob,
    compute_recency_weighted_qb_adjustments,
)
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

# Incumbent matching the authoritative benchmark
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
FEATURE_COLS = ["home_qb_changed", "away_qb_changed", "home_rolling_mov_3", "away_rolling_mov_3"]

SEED = 42


def _get_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    if avail:
        return df[avail].values
    return np.empty((len(df), 0))


def _gate_name(name: str, params: dict | None = None) -> str:
    base = name.strip()
    if params:
        parts = []
        for k, v in sorted(params.items()):
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            else:
                parts.append(f"{k}={v}")
        return f"{base} ({', '.join(parts)})"
    return base


def _fit_incumbent_platt(
    train_prob: np.ndarray,
    train_feat: np.ndarray | None,
    train_y: np.ndarray,
    val_prob: np.ndarray,
    val_feat: np.ndarray | None,
) -> np.ndarray:
    """Fit Platt-scaled model (incumbent style)."""
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


def run_gated_qb_adjusted_elo_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/gated_qb_adjusted_elo.md",
    output_csv: str | None = None,
) -> str:
    print("=== Gated QB-Adjusted Elo Experiment V1 ===")

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
    # QB features FIRST (needed for gating)
    df = compute_qb_features(df)
    # Then full adjustments (needed as base for gating)
    df_full = compute_gated_qb_adjustments(df, gate_mode="full")
    df = compute_situational_features(df)

    # Merge QB adjustment columns from the full run
    for col in ["home_qb_adj", "away_qb_adj", "home_qb_starts", "away_qb_starts"]:
        df[col] = df_full[col].values

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    elo_prob = df["elo_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values

    # ── 2. Define challenger variants ──
    qb_adj_full = compute_qb_adjusted_elo_prob(
        df["home_elo_pre"].values, df["away_elo_pre"].values,
        df["home_qb_adj"].values, df["away_qb_adj"].values,
        hfa=BEST_HFA,
    )

    challengers: list[dict] = [
        {
            "name": "A. Incumbent baseline",
            "prob": elo_prob,
            "use_platt": True,
            "feats": FEATURE_COLS,
            "group": "baseline",
        },
        {
            "name": "B. Full QB adj (V0)",
            "prob": qb_adj_full,
            "use_platt": True,
            "feats": [],
            "group": "full",
        },
    ]

    # Build gated variants
    h_adj = df["home_qb_adj"].values.copy()
    a_adj = df["away_qb_adj"].values.copy()

    from sportslab.features.qb_adjustment import apply_qb_adjustment_gate

    gate_sweeps = [
        # C: qb_changed_only
        {"gate_mode": "qb_changed_only", "name_suffix": ""},
        # D: low_continuity with different thresholds
        {"gate_mode": "low_continuity", "min_starts_for_stable": 4, "name_suffix": "starts<4"},
        {"gate_mode": "low_continuity", "min_starts_for_stable": 8, "name_suffix": "starts<8"},
        {"gate_mode": "low_continuity", "min_starts_for_stable": 17, "name_suffix": "starts<17"},
        # E: shrunk_stable with different shrink values
        {"gate_mode": "shrunk_stable", "stable_shrink": 0.1, "name_suffix": "shrink=0.1"},
        {"gate_mode": "shrunk_stable", "stable_shrink": 0.3, "name_suffix": "shrink=0.3"},
        {"gate_mode": "shrunk_stable", "stable_shrink": 0.5, "name_suffix": "shrink=0.5"},
        # F: capped_only
        {"gate_mode": "capped_only", "max_adj_cap": 40, "name_suffix": "cap=40"},
        {"gate_mode": "capped_only", "max_adj_cap": 60, "name_suffix": "cap=60"},
        {"gate_mode": "capped_only", "max_adj_cap": 80, "name_suffix": "cap=80"},
    ]

    h_starts_arr = (
        df["home_qb_team_starts_pre"].values
        if "home_qb_team_starts_pre" in df.columns else None
    )
    a_starts_arr = (
        df["away_qb_team_starts_pre"].values
        if "away_qb_team_starts_pre" in df.columns else None
    )

    for gate_def in gate_sweeps:
        g_h, g_a = apply_qb_adjustment_gate(
            h_adj, a_adj,
            df["home_qb_changed"].values, df["away_qb_changed"].values,
            home_qb_team_starts_pre=h_starts_arr,
            away_qb_team_starts_pre=a_starts_arr,
            gate_mode=gate_def["gate_mode"],
            stable_shrink=gate_def.get("stable_shrink", 0.3),
            min_starts_for_stable=gate_def.get("min_starts_for_stable", 4),
            max_adj_cap=gate_def.get("max_adj_cap", None),
        )
        gated_prob = compute_qb_adjusted_elo_prob(
            df["home_elo_pre"].values, df["away_elo_pre"].values,
            g_h, g_a, hfa=BEST_HFA,
        )

        suffix = gate_def.get("name_suffix", "")
        label = gate_def["gate_mode"]
        if suffix:
            label = f"{label} ({suffix})"

        # Use group naming: first letter or category
        group_map = {
            "qb_changed_only": "C",
            "low_continuity": "D",
            "shrunk_stable": "E",
            "capped_only": "F",
        }
        letter = group_map.get(gate_def["gate_mode"], "X")

        challengers.append({
            "name": f"{letter}. Gated QB ({label})",
            "prob": gated_prob,
            "use_platt": True,
            "feats": [],
            "group": gate_def["gate_mode"],
        })

    # H. Diagnostic aggressive gated
    g_h_diag, g_a_diag = apply_qb_adjustment_gate(
        h_adj, a_adj,
        df["home_qb_changed"].values, df["away_qb_changed"].values,
        gate_mode="aggressive_diagnostic",
    )
    diag_prob = compute_qb_adjusted_elo_prob(
        df["home_elo_pre"].values, df["away_elo_pre"].values,
        g_h_diag, g_a_diag, hfa=BEST_HFA,
    )
    challengers.append({
        "name": "H. Diagnostic: aggressive gated",
        "prob": diag_prob,
        "use_platt": True,
        "feats": [],
        "group": "diagnostic",
    })

    # I. Recency-weighted QB adjustment
    df_recency = compute_recency_weighted_qb_adjustments(df, decay_half_life=32.0)
    recency_prob = compute_qb_adjusted_elo_prob(
        df["home_elo_pre"].values, df["away_elo_pre"].values,
        df_recency["home_qb_adj"].values,
        df_recency["away_qb_adj"].values,
        hfa=BEST_HFA,
    )
    challengers.append({
        "name": "I. Recency-weighted QB adj",
        "prob": recency_prob,
        "use_platt": True,
        "feats": [],
        "group": "recency",
    })

    # J. Low continuity + capped combined
    g_h_comb, g_a_comb = apply_qb_adjustment_gate(
        h_adj, a_adj,
        df["home_qb_changed"].values, df["away_qb_changed"].values,
        home_qb_team_starts_pre=h_starts_arr,
        away_qb_team_starts_pre=a_starts_arr,
        gate_mode="low_continuity",
        min_starts_for_stable=8,
    )
    g_h_comb = np.clip(g_h_comb, -60, 60)
    g_a_comb = np.clip(g_a_comb, -60, 60)
    comb_prob = compute_qb_adjusted_elo_prob(
        df["home_elo_pre"].values, df["away_elo_pre"].values,
        g_h_comb, g_a_comb, hfa=BEST_HFA,
    )
    challengers.append({
        "name": "J. Combined (low cont+cap=60)",
        "prob": comb_prob,
        "use_platt": True,
        "feats": [],
        "group": "combined",
    })

    print(f"  Variants to evaluate: {len(challengers)}")

    # ── 3. Rolling-origin validation ──
    print("\n=== Rolling-Origin Validation ===")

    results: dict[str, dict] = {}
    for ch in challengers:
        prob = ch["prob"]
        fold_lls = []
        fold_briers = []
        for train_seasons, val_season in ROLLING_FOLDS:
            tr = df["season"].isin(train_seasons).values
            va = (df["season"] == val_season).values

            if ch["use_platt"]:
                tr_feat = _get_features(df.loc[tr], ch["feats"])
                va_feat = _get_features(df.loc[va], ch["feats"])
                p = _fit_incumbent_platt(
                    prob[tr], tr_feat, y[tr],
                    prob[va], va_feat,
                )
            else:
                p = prob[va].copy()

            m = compute_metrics(y[va], p)
            fold_lls.append(m.get("log_loss", 1.0))
            fold_briers.append(m.get("brier", 1.0))

        avg_ll = float(np.mean(fold_lls))
        avg_brier = float(np.mean(fold_briers))
        results[ch["name"]] = {
            "fold_lls": fold_lls,
            "val_ll": avg_ll,
            "val_brier": avg_brier,
        }
        print(f"  {ch['name']}: {avg_ll:.4f}  "
              f"(folds: {fold_lls[0]:.4f}, {fold_lls[1]:.4f}, {fold_lls[2]:.4f})")

    # ── 4. 2025 Holdout evaluation ──
    print("\n=== 2025 Holdout ===")

    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = df["season"] == HOLDOUT_SEASON
    train_y = y[is_train].astype(int)
    hold_y = y[is_hold]

    hold_results: dict[str, dict] = {}
    for ch in challengers:
        prob = ch["prob"]

        if ch["use_platt"]:
            tr_feat = _get_features(df.loc[is_train], ch["feats"])
            ho_feat = _get_features(df.loc[is_hold], ch["feats"])
            p_hold = _fit_incumbent_platt(
                prob[is_train], tr_feat, train_y,
                prob[is_hold], ho_feat,
            )
        else:
            p_hold = prob[is_hold].copy()

        m = compute_classification_metrics(
            hold_y[~np.isnan(hold_y)], p_hold[~np.isnan(hold_y)]
        )
        hold_results[ch["name"]] = m

        print(f"  {ch['name']}: LL={m['log_loss']:.4f}, "
              f"Brier={m['brier_score']:.4f}, "
              f"AUC={m['roc_auc']:.4f}, "
              f"Acc={m['accuracy']:.4f}")

    # ── 5. Slice analysis (2025 holdout) ──
    print("\n=== Slice Analysis (2025 Holdout) ===")

    # Find best gated variant by holdout LL (non-diagnostic, non-baseline)
    non_diag_names = [
        n for n in hold_results
        if "diagnostic" not in n.lower()
    ]
    # Best challenger = min holdout LL among non-incumbent, non-diagnostic
    challenger_candidates = [
        n for n in non_diag_names
        if "incumbent" not in n.lower()
    ]
    best_challenger_name = min(
        challenger_candidates, key=lambda n: hold_results[n]["log_loss"]
    )

    inc_hold_prob = elo_prob[is_hold]

    # Fit Platt for champion challenger on holdout
    challenger_ch = next(c for c in challengers if c["name"] == best_challenger_name)
    ch_tr_feat = _get_features(df.loc[is_train], challenger_ch["feats"])
    ch_ho_feat = _get_features(df.loc[is_hold], challenger_ch["feats"])
    ch_hold_prob_cal = _fit_incumbent_platt(
        challenger_ch["prob"][is_train], ch_tr_feat, train_y,
        challenger_ch["prob"][is_hold], ch_ho_feat,
    )

    hold_df = df[is_hold].copy()
    hold_df["incumbent_prob"] = inc_hold_prob
    hold_df["challenger_prob"] = ch_hold_prob_cal

    # Determine qb_changed for either side
    qb_changed_either = (
        hold_df["home_qb_changed"].astype(bool).values
        | hold_df["away_qb_changed"].astype(bool).values
    )

    # Low sample: either QB has < 4 starts with team
    low_sample = (
        hold_df["home_qb_team_starts_pre"].fillna(0).values < 4
    ) | (
        hold_df["away_qb_team_starts_pre"].fillna(0).values < 4
    )

    # Stable: both QBs have >= 4 starts, no change, not missing data
    stable = (~qb_changed_either) & (~low_sample)

    slices = {
        "All games": slice(None),
        "QB change (either)": qb_changed_either,
        "No QB change": ~qb_changed_either,
        "Stable QB (≥4 starts, no change)": stable,
        "Low-sample QB (<4 starts)": low_sample,
        "High confidence (>=0.7)": inc_hold_prob >= 0.7,
        "Medium confidence (0.5-0.7)": (inc_hold_prob >= 0.5) & (inc_hold_prob < 0.7),
        "Home favorite (>=0.5)": inc_hold_prob >= 0.5,
        "Away underdog (<0.5)": inc_hold_prob < 0.5,
    }

    slice_results = {}
    for sl_name, sl_mask in slices.items():
        if sl_mask is slice(None):
            sl_y = hold_y.values if hasattr(hold_y, 'values') else hold_y
            sl_inc = inc_hold_prob
            sl_chal = ch_hold_prob_cal
        else:
            sl_y = (
                hold_y.values[sl_mask]
                if hasattr(hold_y, 'values') else hold_y[sl_mask]
            )
            sl_inc = inc_hold_prob[sl_mask]
            sl_chal = ch_hold_prob_cal[sl_mask]

        valid = ~np.isnan(sl_y)
        if valid.sum() < 2:
            continue
        inc_m = compute_metrics(sl_y[valid], sl_inc[valid])
        chal_m = compute_metrics(sl_y[valid], sl_chal[valid])
        delta_ll = chal_m["log_loss"] - inc_m["log_loss"]
        slice_results[sl_name] = {
            "n": int(valid.sum()),
            "incumbent_ll": inc_m["log_loss"],
            "challenger_ll": chal_m["log_loss"],
            "delta_ll": delta_ll,
        }
        print(f"  {sl_name}: n={valid.sum()}, inc={inc_m['log_loss']:.4f}, "
              f"chal={chal_m['log_loss']:.4f}, Δ={delta_ll:+.4f}")

    # Also compare best challenger vs incumbent for key slices
    print(f"\n  Best challenger: {best_challenger_name}")

    # ── 5b. Slice table for ALL variants (compact: only QB-change and no-QB-change) ──
    qb_change_mask = qb_changed_either
    no_qb_change_mask = ~qb_changed_either

    all_variant_slices: dict[str, dict] = {}
    for ch in challengers:
        prob = ch["prob"]
        ch_name = ch["name"]

        if ch["use_platt"]:
            tr_feat = _get_features(df.loc[is_train], ch["feats"])
            ho_feat = _get_features(df.loc[is_hold], ch["feats"])
            p_hold = _fit_incumbent_platt(
                prob[is_train], tr_feat, train_y,
                prob[is_hold], ho_feat,
            )
        else:
            p_hold = prob[is_hold].copy()

        hv = hold_y.values if hasattr(hold_y, 'values') else hold_y

        # QB change slice
        qc_y = hv[qb_change_mask]
        qc_p = p_hold[qb_change_mask]
        qc_valid = ~np.isnan(qc_y)
        qc_ll = compute_metrics(qc_y[qc_valid], qc_p[qc_valid]).get("log_loss", 1.0)

        # No QB change slice
        nq_y = hv[no_qb_change_mask]
        nq_p = p_hold[no_qb_change_mask]
        nq_valid = ~np.isnan(nq_y)
        nq_ll = compute_metrics(nq_y[nq_valid], nq_p[nq_valid]).get("log_loss", 1.0)

        all_variant_slices[ch_name] = {
            "qb_change_ll": qc_ll,
            "no_qb_change_ll": nq_ll,
        }

    # ── 6. Write report ──
    print(f"\n=== Writing report -> {report_path} ==")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    inc_val = results["A. Incumbent baseline"]["val_ll"]
    inc_hold = hold_results["A. Incumbent baseline"]["log_loss"]

    # Determine promote/reject
    promoted = []
    for ch in challengers:
        n = ch["name"]
        if n == "A. Incumbent baseline":
            continue
        if "diagnostic" in n.lower():
            continue
        if n not in results or n not in hold_results:
            continue
        if results[n]["val_ll"] < inc_val and hold_results[n]["log_loss"] < inc_hold:
            promoted.append(n)

    with open(rp, "w") as f:
        _w = f.write

        _w("# Gated QB-Adjusted Elo Experiment — V1\n\n")
        _w("## Research Question\n\n")
        _w("Can QB-adjusted Elo improve games with meaningful QB uncertainty ")
        _w("or QB changes without degrading games where the starting QB ")
        _w("situation is stable?\n\n")

        _w("## Why V0 Was Rejected\n\n")
        _w("The first QB-adjusted Elo experiment (V0) showed that QB adjustments ")
        _w("helped QB-change games (holdout Δ = −0.0035) but hurt non-QB-change ")
        _w("games (Δ = +0.0046). The overall holdout was worse (0.6376 vs ")
        _w("incumbent 0.6259). The adjustment introduces noise for stable-QB ")
        _w("situations where the Elo rating already captures team quality.\n\n")
        _w("Key insight: the adjustment is useful specifically when the QB ")
        _w("situation is uncertain (change, low continuity, rookie/backup), ")
        _w("not for established starters who have a long track record.\n\n")

        _w("## Gating Strategy\n\n")
        _w("Instead of applying the QB adjustment uniformly to all games, a ")
        _w("pregame-safe **gate** determines the adjustment multiplier for ")
        _w("each side of each game:\n\n")
        _w("```\n")
        _w("gate_multiplier = f(qb_changed, qb_team_starts, games_since_change)\n")
        _w("gated_adj = full_qb_adj * gate_multiplier\n")
        _w("```\n\n")
        _w("All gating features (`qb_changed`, `qb_team_starts_pre`, ")
        _w("`games_since_qb_change`) are computed chronologically from ")
        _w("games before the current game — no leakage.\n\n")
        _w("### Gating Variants Tested\n\n")
        _w("| Letter | Gate Mode | Description |\n")
        _w("|--------|-----------|-------------|\n")
        _w("| B | full | V0: no gating, full adjustment |\n")
        _w("| C | qb_changed_only | Adjust only when QB changed (gate=0 otherwise) |\n")
        _w("| D | low_continuity | Adjust when QB changed OR <N starts with team |\n")
        _w("| E | shrunk_stable | Full adjust for changed, scaled (0.1-0.5x) for stable |\n")
        _w("| F | capped_only | Same as V0 but lower max cap (40-80) |\n")
        _w("| G | aggressive_diagnostic | DIAGNOSTIC: 2x changed, 0x stable |\n")
        _w("| H | recency_weighted | Decayed older games (HL=32), standard shrinkage |\n")
        _w("| I | combined | Low continuity (starts<8) + cap=60 |\n\n")

        _w("## Hyperparameters Tested\n\n")
        _w("| Parameter | Values Tested |\n")
        _w("|-----------|---------------|\n")
        _w("| gate_mode | full, qb_changed_only, low_continuity, shrunk_stable, capped_only |\n")
        _w("| stable_shrink | 0.1, 0.3, 0.5 |\n")
        _w("| min_starts_for_stable | 4, 8, 17 |\n")
        _w("| max_adj_cap | 40, 60, 80 |\n")
        _w("| decay_half_life | 32 (recency only) |\n\n")

        _w("## Data Used\n\n")
        _w("- 2021–2025 NFL seasons (non-neutral regular + postseason)\n")
        _w("- QB starter IDs from nflreadpy (`home_qb_id`, `away_qb_id` GSIS IDs)\n")
        _w("- Team Elo ratings (K=36, HFA=40, reg=0.1, decay=32, MOV capped_linear)\n")
        _w("- Incumbent feature set: elo_prob + qb_changed (2) + rolling_mov_3 (2) + Platt\n")
        _w("- Gated variants: same Elo spine, QB adj prob + Platt\n\n")

        _w("## Data Excluded\n\n")
        _w("- Seasons before 2021\n")
        _w("- Neutral-site games\n")
        _w("- Ties and games with missing scores\n")
        _w("- 2026 games (no scores)\n")
        _w("- Post-game stats, final scores, market data\n\n")

        _w("## Leakage Safeguards\n\n")
        _w("1. QB adjustments computed chronologically — only prior starts used\n")
        _w("2. Gate conditions (qb_changed, starts, continuity) are pregame only\n")
        _w("3. No post-game data used in any rating or gate computation\n")
        _w("4. Season-boundary reset for rolling features\n")
        _w("5. Holdout (2025) untouched during training/validation\n")
        _w("6. Bayesian shrinkage prevents small-sample overfitting\n")
        _w("7. Missing QB IDs assigned 0.0 adjustment (no signal)\n\n")

        _w("## Backtest Setup\n\n")
        _w(f"- Rolling-origin folds: {ROLLING_FOLDS}\n")
        _w(f"- Holdout: {HOLDOUT_SEASON}\n")
        _w("- Platt: StandardScaler + LogisticRegression (C=1.0, lbfgs)\n")
        _w("- Incumbent feature set: elo_prob + qb_changed (2) + rolling_mov_3 (2)\n")
        _w("- Gated variants: qb-adjusted prob + Platt (no extra features)\n\n")

        # Validation table
        _w("## Rolling-Origin Validation Log Loss\n\n")
        _w("| Model | Avg LL | Fold1 | Fold2 | Fold3 |\n")
        _w("|-------|--------|-------|-------|-------|\n")
        for name, r in sorted(results.items(), key=lambda x: x[1]["val_ll"]):
            _w(f"| {name} | {r['val_ll']:.4f}"
              f" | {r['fold_lls'][0]:.4f}"
              f" | {r['fold_lls'][1]:.4f}"
              f" | {r['fold_lls'][2]:.4f} |\n")

        # Holdout table
        _w("\n## 2025 Holdout\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy |\n")
        _w("|-------|----------|-------|-----|----------|\n")
        for name in sorted(hold_results, key=lambda n: hold_results[n]["log_loss"]):
            h = hold_results[name]
            _w(f"| {name} | {h['log_loss']:.4f}"
              f" | {h['brier_score']:.4f}"
              f" | {h['roc_auc']:.4f}"
              f" | {h['accuracy']:.4f} |\n")

        # Slice vs variant table (compact)
        _w("\n## Slice Performance by Variant (2025 Holdout)\n\n")
        _w("| Model | QB-change LL | No-QB-change LL | Δ vs inc (QC) | Δ vs inc (noQC) |\n")
        _w("|-------|-------------|-----------------|---------------|-----------------|\n")

        inc_qc = all_variant_slices["A. Incumbent baseline"]["qb_change_ll"]
        inc_nqc = all_variant_slices["A. Incumbent baseline"]["no_qb_change_ll"]
        for name in sorted(all_variant_slices, key=lambda n: all_variant_slices[n]["qb_change_ll"]):
            avs = all_variant_slices[name]
            d_qc = avs["qb_change_ll"] - inc_qc
            d_nqc = avs["no_qb_change_ll"] - inc_nqc
            _w(f"| {name} | {avs['qb_change_ll']:.4f}"
              f" | {avs['no_qb_change_ll']:.4f}"
              f" | {d_qc:+.4f}"
              f" | {d_nqc:+.4f} |\n")

        # Best challenger slice analysis
        _w("\n## Best Challenger Slice Performance\n\n")
        _w(f"Best gated challenger: **{best_challenger_name}**\n\n")
        _w("| Slice | N | Incumbent LL | Challenger LL | Δ |\n")
        _w("|-------|---|-------------|---------------|---|\n")
        for sl_name, sr in slice_results.items():
            _w(f"| {sl_name} | {sr['n']} | {sr['incumbent_ll']:.4f}"
              f" | {sr['challenger_ll']:.4f} | {sr['delta_ll']:+.4f} |\n")

        # Summary table of all variants
        _w("\n## Full Results Summary\n\n")
        _w("| Model | Val LL | Hold LL | Δ vs inc (val) | Δ vs inc (hold) | QC Δ | NoQC Δ |\n")
        _w("|-------|--------|---------|----------------|-----------------|------|--------|\n")
        for name in sorted(results.keys()):
            vl = results[name]["val_ll"]
            hl = hold_results[name]["log_loss"]
            d_val = vl - inc_val
            d_hold = hl - inc_hold
            qc_ll = all_variant_slices[name]["qb_change_ll"]
            nqc_ll = all_variant_slices[name]["no_qb_change_ll"]
            d_qc = qc_ll - inc_qc
            d_nqc = nqc_ll - inc_nqc
            _w(f"| {name} | {vl:.4f} | {hl:.4f}"
              f" | {d_val:+.4f} | {d_hold:+.4f}"
              f" | {d_qc:+.4f} | {d_nqc:+.4f} |\n")

        # Decision
        _w("\n## Decision\n\n")
        if promoted:
            _w(f"**Promoted: {', '.join(promoted)}**\n\n")
        else:
            _w("**No gated variant beats incumbent on both validation and holdout.**\n\n")
            _w("All gated variants are either rejected or marked diagnostic only.\n\n")

        best_val_name = min(results, key=lambda n: results[n]["val_ll"])
        best_hold_name = min(hold_results, key=lambda n: hold_results[n]["log_loss"])
        _w(f"Best validation: {best_val_name} ({results[best_val_name]['val_ll']:.4f})\n")
        _w(f"Best holdout: {best_hold_name} ({hold_results[best_hold_name]['log_loss']:.4f})\n\n")

        # QB change game impact
        _w("### QB-Change Game Impact\n\n")
        qc_deltas = {}
        for name in challenger_candidates:
            qc_deltas[name] = all_variant_slices[name]["qb_change_ll"] - inc_qc

        best_qc_name = min(qc_deltas, key=qc_deltas.get)
        worst_qc_name = max(qc_deltas, key=qc_deltas.get)
        _w(f"Best QB-change improvement: {best_qc_name} ({qc_deltas[best_qc_name]:+.4f})\n")
        _w(f"Worst QB-change degradation: {worst_qc_name} ({qc_deltas[worst_qc_name]:+.4f})\n\n")

        _w("### Non-QB-Change Game Impact\n\n")
        nqc_deltas = {}
        for name in challenger_candidates:
            nqc_deltas[name] = all_variant_slices[name]["no_qb_change_ll"] - inc_nqc

        best_nqc_name = min(nqc_deltas, key=nqc_deltas.get)
        worst_nqc_name = max(nqc_deltas, key=nqc_deltas.get)
        _w(f"Best non-QB-change improvement: {best_nqc_name}"
           f" ({nqc_deltas[best_nqc_name]:+.4f})\n")
        _w(f"Worst non-QB-change degradation: {worst_nqc_name}"
           f" ({nqc_deltas[worst_nqc_name]:+.4f})\n\n")

        # Failure modes
        _w("### Failure Modes\n\n")
        _w("1. **Gating sharpness**: The qb_changed flag is coarse — a QB change ")
        _w("from one elite starter to another elite starter triggers the same gate ")
        _w("as a change to a backup.\n")
        _w("2. **Small-sample QBs**: QBs with <17 starts are strongly shrunk toward ")
        _w("replacement. Gating can't fix the fact that tiny-sample adjustments ")
        _w("are inherently noisy.\n")
        _w("3. **Oracle QB data**: Uses final actual starter IDs, not pregame-announced. ")
        _w("Live-pregame requires `--qb-input CSV`.\n")
        _w("4. **No position-group interaction**: The adjustment ignores offensive line, ")
        _w("skill-position talent, and defensive support.\n\n")

        # Recommended next experiment
        _w("### Recommended Next Experiment\n\n")

        if promoted:
            _w("The gated QB adjustment has been promoted. Consider:\n")
            _w("1. **Integrating into build_features.py** as a default pipeline option\n")
            _w("2. **Testing with season-expanded data** (if more seasons accumulate)\n")
            _w("3. **Position-group roster strength (V1)**: Extend to OL, skill, DL\n\n")
        else:
            _w("1. **QB-specific position-group ratings**: Instead of a global QB rating, ")
            _w("separate QB rating into passing/timing/decision-making components\n")
            _w("2. **Coach-QB interaction features**: The combination of a new QB")
            _w(" and new coordinator may be more informative than QB change alone\n")
            _w("3. **Expanded Elo K search**: Test K > 48 with the season regression spine\n")
            _w("4. **Any model must beat Standard Elo + qb_changed + rolling_mov_3 + Platt ")
            _w("(holdout LL 0.6262)** to become the new clean football-only incumbent\n\n")

        _w("---\n*Report generated by `sportslab gated-qb-elo`. ")
        _w(f"GATE_MODES tested: {list(GATE_MODES.keys())}.*\n")

    print(f"\nReport: {rp}")

    # ── 7. Export CSV if requested ──
    if output_csv is not None:
        out_df = df[is_hold].copy()
        out_df["incumbent_home_win_prob"] = inc_hold_prob
        out_df["best_challenger_prob"] = ch_hold_prob_cal
        out_df["best_challenger_name"] = best_challenger_name
        out_df["qb_change_either"] = qb_changed_either.astype(int)
        out_df.to_csv(output_csv, index=False)
        print(f"  CSV: {output_csv}")

    return str(report_path)
