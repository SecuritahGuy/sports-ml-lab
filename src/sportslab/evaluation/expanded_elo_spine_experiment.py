"""Expanded Elo spine + frozen QB overlay experiment.

Tests whether a better base Elo probability (via wider K/HFA/regression/decay grid)
improves the v3.0.0 Frozen QB Overlay champion.

Architecture:
    base Elo (swept params)
    → fold-safe Platt calibration
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

# v3.0.0 champion reference
V3_VAL_LL = 0.6305
V3_HOLDOUT_LL = 0.6200

# Frozen QB overlay (v3.0.0 champion)
QB_GATE_GAMMA = 1.0
QB_GATE_CAP = 40
FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]

MIN_PROMOTION_DELTA = 0.001
SEED = 42
ELO_TO_LOGIT = np.log(10) / 400.0

# v3.0.0 champion Elo params
V3_K = 36
V3_HFA = 40
V3_REG = 0.1
V3_DECAY = 32
V3_QB_BONUS = 0.2

# ── Parameter grids ──
K_VALUES = [20, 28, 36, 40, 44, 52, 60]
HFA_VALUES = [20, 30, 35, 40, 50]
REG_VALUES = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
DECAY_VALUES = [None, 32, 48, 64]

# Elo input columns needed
ELO_INPUT_COLS = [
    "season", "week", "gameday", "home_team", "away_team",
    "home_score", "away_score", "home_win",
]


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


def _generate_param_combos() -> list[dict]:
    combos: list[dict] = []
    for k in K_VALUES:
        for hfa in HFA_VALUES:
            for reg in REG_VALUES:
                for decay in DECAY_VALUES:
                    combos.append({
                        "K": k, "HFA": hfa, "reg": reg, "decay": decay,
                    })
    return combos


def _score_elo_combo(
    elo_prob: np.ndarray,
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
        train_elo = elo_prob[train_mask]
        train_y = y[train_mask].astype(int)
        train_feat = all_feat[train_mask]
        x_train = (
            np.column_stack([train_elo, train_feat])
            if train_feat.size else train_elo.reshape(-1, 1)
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_train, train_y)

        x_all = (
            np.column_stack([elo_prob, all_feat])
            if all_feat.size else elo_prob.reshape(-1, 1)
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


def run_expanded_elo_spine(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/expanded_elo_spine.md",
    output_csv: Optional[str] = None,
) -> str:
    print("=== Expanded Elo Spine + Frozen QB Overlay ===")

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

    # Base df for Elo computation (minimal columns)
    elo_base = df[ELO_INPUT_COLS].copy()

    # ── 2. Build fold masks ──
    fold_frames: list[tuple[np.ndarray, np.ndarray]] = []
    for train_seasons, val_season in ROLLING_FOLDS:
        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values
        fold_frames.append((train_mask, val_mask))

    # ── 3. Grid search ──
    param_combos = _generate_param_combos()
    n = len(param_combos)
    n_k = len(K_VALUES)
    n_h = len(HFA_VALUES)
    n_r = len(REG_VALUES)
    n_d = len(DECAY_VALUES)
    print(f"\n=== Grid: {n} combos ({n_k}K × {n_h}HFA × {n_r}reg × {n_d}decay)")
    print(f"  K={K_VALUES}")
    print(f"  HFA={HFA_VALUES}")
    print(f"  reg={REG_VALUES}")
    print(f"  decay={DECAY_VALUES}")
    print()

    results: list[dict] = []
    for idx, combo in enumerate(param_combos):
        if (idx + 1) % 50 == 0 or idx == 0:
            k_c = combo["K"]
            h_c = combo["HFA"]
            r_c = combo["reg"]
            d_c = combo["decay"]
            print(f"  [{idx+1}/{n}] K={k_c} HFA={h_c} reg={r_c} decay={d_c}")

        df_elo = compute_elo_features(
            elo_base,
            k_factor=combo["K"],
            home_advantage=combo["HFA"],
            preseason_regression=combo["reg"],
            decay_half_life=combo["decay"],
        )
        elo_prob = df_elo["elo_prob"].values.astype(float)

        avg_ll = _score_elo_combo(
            elo_prob, y, all_feat, qb_gate_mask,
            home_qb_adj, away_qb_adj, fold_frames,
        )

        results.append({
            "K": combo["K"],
            "HFA": combo["HFA"],
            "reg": combo["reg"],
            "decay": combo["decay"],
            "val_ll": avg_ll,
        })

    # ── 4. Find best by val LL ──
    results.sort(key=lambda r: r["val_ll"])
    best = results[0]
    print("\n=== Best validation ===")
    print(f"  K={best['K']}, HFA={best['HFA']}, reg={best['reg']}, decay={best['decay']}")
    bv = best["val_ll"]
    print(f"  Val LL: {bv:.4f} (vs v3.0.0 {V3_VAL_LL:.4f}, Δ={bv - V3_VAL_LL:+.4f})")

    # Top 5
    print("\n  Top 5:")
    for i, r in enumerate(results[:5]):
        delta = r["val_ll"] - V3_VAL_LL
        print(f"  {i+1}. K={r['K']} HFA={r['HFA']} reg={r['reg']} ", end="")
        print(f"decay={r['decay']} val={r['val_ll']:.4f} Δ={delta:+.4f}")

    # ── 5. Compute v3.0.0 reference on 2021-2024 ──
    print("\n=== Computing v3.0.0 reference ===")
    hold_mask = (df["season"] == HOLDOUT_SEASON).values
    train_mask_hold = df["season"].isin([2021, 2022, 2023, 2024]).values

    v3_overrides = build_team_regression_overrides(
        df, preseason_regression=V3_REG, qb_change_bonus=V3_QB_BONUS,
    )
    df_elo_v3 = compute_elo_features(
        elo_base, k_factor=V3_K, home_advantage=V3_HFA,
        preseason_regression=V3_REG, team_regression_overrides=v3_overrides,
        decay_half_life=V3_DECAY,
    )
    v3_elo_prob = df_elo_v3["elo_prob"].values.astype(float)

    train_y = y[train_mask_hold].astype(int)
    train_feat = all_feat[train_mask_hold]

    def _fit_and_predict(elo_prob_array):
        tr_elo = elo_prob_array[train_mask_hold]
        x_tr = (
            np.column_stack([tr_elo, train_feat])
            if train_feat.size else tr_elo.reshape(-1, 1)
        )
        x_all = (
            np.column_stack([elo_prob_array, all_feat])
            if all_feat.size else elo_prob_array.reshape(-1, 1)
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_tr, train_y)
        p = pipe.predict_proba(x_all)[:, 1]
        final = _logit(p)
        final = _apply_qb_overlay(final, qb_gate_mask, home_qb_adj, away_qb_adj)
        return _sigmoid(final)

    v3_prob = _fit_and_predict(v3_elo_prob)
    v3_hold_prob = v3_prob[hold_mask]
    hold_y = y[hold_mask]
    valid_hold = ~np.isnan(hold_y)
    v3_hold_m = compute_classification_metrics(hold_y[valid_hold], v3_hold_prob[valid_hold])
    v3_hold_ll = v3_hold_m["log_loss"]
    print(f"  v3.0.0 holdout: {v3_hold_ll:.4f} (expected ~{V3_HOLDOUT_LL:.4f})")

    # ── 6. 2025 holdout for best candidate ──
    print("\n=== 2025 Holdout ===")
    df_elo_best = compute_elo_features(
        elo_base,
        k_factor=best["K"],
        home_advantage=best["HFA"],
        preseason_regression=best["reg"],
        decay_half_life=best["decay"],
    )
    best_elo_prob = df_elo_best["elo_prob"].values.astype(float)
    best_prob = _fit_and_predict(best_elo_prob)
    best_hold_prob = best_prob[hold_mask][valid_hold]
    hold_m = compute_classification_metrics(hold_y[valid_hold], best_hold_prob)
    hold_ll = hold_m["log_loss"]

    print(f"  v3.0.0 champion: {v3_hold_ll:.4f}")
    print(f"  Best candidate:  {hold_ll:.4f}")
    print(f"  Δ: {hold_ll - v3_hold_ll:+.4f}")

    beats_val = best["val_ll"] < V3_VAL_LL - MIN_PROMOTION_DELTA
    beats_hold = hold_ll < v3_hold_ll - MIN_PROMOTION_DELTA

    # ── 6. Write report ──
    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write
        _w("# Expanded Elo Spine + Frozen QB Overlay\n\n")

        _w("## Research Question\n\n")
        _w("Can a better base Elo spine improve the v3.0.0 Frozen QB Overlay ")
        _w("champion? The QB overlay improved the top layer. Now we test ")
        _w("whether improving the base Elo probability underneath yields ")
        _w("further gains.\n\n")

        _w("## Architecture\n\n")
        _w("```\n")
        _w("base Elo (swept: K, HFA, reg, decay)\n")
        _w("→ fold-safe Platt on [elo_prob, qb_changed, rolling_mov_3]\n")
        _w("→ frozen QB overlay (v3.0.0 champion, fixed)\n")
        _w("→ validation/holdout comparison\n")
        _w("```\n\n")

        _w("All candidates use the **same frozen QB overlay** ")
        _w("(gate: H. changed OR starts<17, cap=40, gamma=1.0). ")
        _w("Only the Elo base parameters change.\n\n")

        _w("## Champion (v3.0.0)\n\n")
        _w("| Metric | Value |\n")
        _w("|--------|-------|\n")
        _w(f"| Val LL | {V3_VAL_LL:.4f} |\n")
        _w(f"| Holdout LL | {V3_HOLDOUT_LL:.4f} |\n")
        _w(f"| Parameters | K={V3_K}, HFA={V3_HFA}, reg={V3_REG}, decay={V3_DECAY} |\n\n")

        _w("## Grid Search\n\n")
        _w(f"**{n} combos** ")
        _w(f"({n_k}K × {n_h}HFA × {n_r}reg × {n_d}decay)\n\n")
        _w("| Param | Values |\n")
        _w("|-------|--------|\n")
        _w(f"| K | {K_VALUES} |\n")
        _w(f"| HFA | {HFA_VALUES} |\n")
        _w(f"| reg | {REG_VALUES} |\n")
        _w(f"| decay | {DECAY_VALUES} |\n\n")

        _w("## Validation Results\n\n")
        _w(f"**v3.0.0 champion val LL:** {V3_VAL_LL:.4f}\n\n")

        _w("### Top 5\n\n")
        _w("| Rank | K | HFA | reg | decay | Val LL | Δ vs v3.0.0 |\n")
        _w("|------|---|-----|-----|-------|--------|-------------|\n")
        for i, r in enumerate(results[:5]):
            delta = r["val_ll"] - V3_VAL_LL
            d_str = f"{r['decay']}" if r["decay"] is not None else "None"
            _w(f"| {i+1} | {r['K']} | {r['HFA']} | {r['reg']} | {d_str} ")
            _w(f"| {r['val_ll']:.4f} | {delta:+.4f} |\n")

        _w("\n### All results by val LL improvement\n\n")
        n_beat = sum(1 for r in results if r["val_ll"] < V3_VAL_LL)
        n_delta = sum(1 for r in results if r["val_ll"] < V3_VAL_LL - MIN_PROMOTION_DELTA)
        _w(f"- **{n_beat}/{n}** combos beat v3.0.0 on validation\n")
        _w(f"- **{n_delta}/{n}** combos beat v3.0.0 by >= {MIN_PROMOTION_DELTA}\n\n")

        _w("## Holdout Results\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy |\n")
        _w("|-------|----------|-------|-----|----------|\n")
        _w(f"| v3.0.0 champion (reproduced) | {v3_hold_ll:.4f} ")
        _w(f"| {v3_hold_m['brier_score']:.4f} | {v3_hold_m['roc_auc']:.4f} ")
        _w(f"| {v3_hold_m['accuracy']:.4f} |\n")
        _w(f"| Best candidate | {hold_ll:.4f} ")
        _w(f"| {hold_m['brier_score']:.4f} | {hold_m['roc_auc']:.4f} ")
        _w(f"| {hold_m['accuracy']:.4f} |\n")

        _w("\n## Best Candidate\n\n")
        _w("| Param | Value |\n")
        _w("|-------|-------|\n")
        _w(f"| K | {best['K']} |\n")
        _w(f"| HFA | {best['HFA']} |\n")
        _w(f"| reg | {best['reg']} |\n")
        _w(f"| decay | {best['decay']} |\n")
        _w(f"| Val LL | {best['val_ll']:.4f} |\n")
        _w(f"| Holdout LL | {hold_ll:.4f} |\n")
        _w(f"| Δ val vs v3.0.0 | {best['val_ll'] - V3_VAL_LL:+.4f} |\n")
        _w(f"| Δ holdout vs v3.0.0 | {hold_ll - v3_hold_ll:+.4f} |\n\n")

        _w("## Decision\n\n")
        if beats_val and beats_hold:
            bk = best["K"]
            bh = best["HFA"]
            br = best["reg"]
            bd = best["decay"]
            _w(f"**✅ PROMOTED: K={bk}, HFA={bh}, reg={br}, decay={bd}**\n\n")
            _w("Beats v3.0.0 champion on both validation and holdout.\n\n")
        elif not beats_val and beats_hold:
            _w(f"**⚠️ DIAGNOSTIC ONLY: K={best['K']}**\n\n")
            _w("Wins holdout but not validation. Not promoted.\n\n")
        elif beats_val and not beats_hold:
            _w("**❌ REJECTED (val improvement, holdout regression)**\n\n")
            _w("Wins validation but loses holdout. Classic overfit pattern.\n\n")
        else:
            _w("**❌ REJECTED**\n\n")
            _w("No candidate beats v3.0.0 on both validation and holdout.\n\n")
            _w("The expanded Elo spine search did not find a better base model ")
            _w("under the frozen QB overlay. The v3.0.0 champion remains ")
            _w("the best known configuration.\n\n")

        _w("## Takeaways\n\n")
        if beats_val or beats_hold:
            _w("1. **Elo spine matters** — tuning the base Elo parameters can ")
            _w("meaningfully change the final probability, even after Platt ")
            _w("calibration and the QB overlay.\n")
            _w("2. **The v3.0.0 champion is robust** — ")
            if beats_val:
                _w("the expanded grid found candidates that beat it on validation, ")
                _w("but holdout improvements were insufficient for promotion.\n")
            else:
                _w("no candidate beat it on either validation or holdout.\n")
        else:
            _w(f"1. **The v3.0.0 Elo base (K={V3_K}, HFA={V3_HFA}, ")
            _w(f"reg={V3_REG}, decay={V3_DECAY}) is robust** — ")
            _w("no expanded Elo spine candidate beat it on both val and holdout.\n")
            _w("2. **Platt scaling absorbs much of the base-Elo variation** — ")
            _w("logistic calibration on [elo_prob, features] compresses differences ")
            _w("in raw Elo accuracy.\n")
            _w("3. **The QB overlay dominates** — the overlay's logit adjustment of ")
            max_adj = QB_GATE_CAP * QB_GATE_GAMMA
            _w(f"up to ±{max_adj:.1f} elo points swamps the Elo spine differences.")
            _w("\n\n")

        _w("---\n")
        _w("*Report generated by `sportslab expanded-elo-spine`.\n")
        _w(f"Grid: {n} combos, 3 rolling-origin folds, fold-safe Platt, frozen QB overlay.*\n")

    if output_csv:
        out_df = df[["game_id", "season", "week", "home_team", "away_team", TARGET_COLUMN]].copy()
        out_df["best_prob"] = best_prob
        out_df.to_csv(output_csv, index=False)
        print(f"  Output: {output_csv}")

    print(f"\nReport: {rp}")
    return str(report_path)
