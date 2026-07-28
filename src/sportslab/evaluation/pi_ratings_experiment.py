"""Pi-Ratings experiment — coupled home/away ratings with nonlinear updates.

Design:
  Each team has a single pi rating. Two innovations vs standard Elo:
    1. Power-law MOV:  mov = |margin|^alpha  (alpha != 1 means nonlinear)
    2. Asymmetric K:   k_home = base_k * hk_ratio,
                       k_away = base_k * (2 - hk_ratio)
                       (hk_ratio != 1 creates home/away coupling)

  When alpha=1 and hk_ratio=1, this is standard capped_linear Elo.

Grid: 144 combos (4 alpha × 3 base_k × 3 hk_ratio × 2 HFA × 2 reg).
Bounded experiment — no large parameter hunt.

Comparison: v3.0.0 champion (val LL 0.6305, holdout LL 0.6200).
Promotion requires Δ >= 0.001 on BOTH val and holdout.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import compute_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_pi_ratings_features
from sportslab.features.situational import compute_situational_features

FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"

V3_VAL_LL = 0.6305
V3_HOLDOUT_LL = 0.6200

QB_GATE_GAMMA = 1.0
QB_GATE_CAP = 40
FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]
MIN_PROMOTION_DELTA = 0.001
SEED = 42
ELO_TO_LOGIT = np.log(10) / 400.0

# Pi-Ratings grid — bounded: 4 × 3 × 3 × 2 × 2 = 144 combos
PI_ALPHAS = [0.5, 0.75, 1.0, 1.25]
PI_BASE_KS = [28, 36, 44]
PI_HK_RATIOS = [0.75, 1.0, 1.25]
PI_HFAS = [30, 40]
PI_REGS = [0.0, 0.1]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p + 1e-15))


def _build_qb_gate_mask(df: pd.DataFrame) -> np.ndarray:
    """Build QB gate mask: True when QB is changed or has < 17 career starts."""
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


def _score_pi_model(
    pi_prob: np.ndarray, y: np.ndarray,
    all_feat: np.ndarray,
    qb_gate_mask: np.ndarray,
    home_qb_adj: np.ndarray, away_qb_adj: np.ndarray,
    fold_frames: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Fit Platt per fold, apply QB overlay, return avg val LL."""
    fold_lls: list[float] = []
    for train_mask, val_mask in fold_frames:
        train_pp = pi_prob[train_mask]
        train_y = y[train_mask].astype(int)
        train_feat = all_feat[train_mask]
        x_train = (
            np.column_stack([train_pp, train_feat])
            if train_feat.size else train_pp.reshape(-1, 1)
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_train, train_y)

        x_all = (
            np.column_stack([pi_prob, all_feat])
            if all_feat.size else pi_prob.reshape(-1, 1)
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


def _score_v3_incumbent(
    elo_prob: np.ndarray, y: np.ndarray,
    all_feat: np.ndarray,
    qb_gate_mask: np.ndarray,
    home_qb_adj: np.ndarray, away_qb_adj: np.ndarray,
    fold_frames: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Score the v3.0.0 champion (standard Elo) for comparison."""
    return _score_pi_model(
        elo_prob, y, all_feat,
        qb_gate_mask, home_qb_adj, away_qb_adj, fold_frames,
    )


def _compute_holdout_ll(
    pi_prob: np.ndarray, y: np.ndarray,
    all_feat: np.ndarray,
    qb_gate_mask: np.ndarray,
    home_qb_adj: np.ndarray, away_qb_adj: np.ndarray,
    fold_frames: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Compute holdout LL using average of per-fold Platt models + QB overlay."""
    hold_logits: list[np.ndarray] = []
    for train_mask, val_mask in fold_frames:
        train_pp = pi_prob[train_mask]
        train_y = y[train_mask].astype(int)
        train_feat = all_feat[train_mask]
        x_train = (
            np.column_stack([train_pp, train_feat])
            if train_feat.size else train_pp.reshape(-1, 1)
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_train, train_y)

        x_all = (
            np.column_stack([pi_prob, all_feat])
            if all_feat.size else pi_prob.reshape(-1, 1)
        )
        platt_prob = pipe.predict_proba(x_all)[:, 1]
        base_logit = _logit(platt_prob)

        # Fold sees all training data, apply overlay on all data
        final_logit = _apply_qb_overlay(base_logit, qb_gate_mask, home_qb_adj, away_qb_adj)
        hold_logits.append(final_logit)

    # Average logits across folds
    avg_logit = np.mean(hold_logits, axis=0)
    final_prob = _sigmoid(avg_logit)

    holdout_mask = fold_frames[-1][1]  # last fold's val mask
    hold_prob = final_prob[holdout_mask]
    hold_y = y[holdout_mask]
    valid = ~np.isnan(hold_y)
    m = compute_metrics(hold_y[valid], hold_prob[valid])
    return m.get("log_loss", 1.0)


def run_pi_ratings_experiment(
    ft_path: Optional[str] = None,
    report_path: str = "reports/experiments/pi_ratings.md",
) -> str:
    """Run Pi-Ratings bounded experiment, write report."""
    print("=== Pi-Ratings Experiment ===")

    fp = Path(ft_path or FEATURE_TABLE_PATH)
    df_raw = pd.read_parquet(fp)

    # Build QB features, adjustments, and situational features once
    from sportslab.features.ratings import compute_elo_features
    df = compute_elo_features(df_raw, k_factor=20, home_advantage=0)
    df = compute_qb_features(df)
    df = compute_qb_adjustments(df)
    df = compute_situational_features(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    y = df[TARGET_COLUMN].astype(float).values
    all_feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    all_feat = df[all_feat_cols].values.astype(float) if all_feat_cols else np.empty((len(df), 0))
    qb_gate_mask = _build_qb_gate_mask(df)
    home_qb_adj = df["home_qb_adj"].values.astype(float)
    away_qb_adj = df["away_qb_adj"].values.astype(float)

    # Build fold frames using ROLLING_FOLDS
    fold_frames: list[tuple[np.ndarray, np.ndarray]] = []
    for train_seasons, val_season in ROLLING_FOLDS:
        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values
        fold_frames.append((train_mask, val_mask))

    # ── Compute v3.0.0 incumbent ──
    print("  Computing incumbent (v3.0.0 champion)...")
    v3_elo = compute_elo_features(
        df, k_factor=36, home_advantage=40,
        preseason_regression=0.1, decay_half_life=32,
    )
    v3_val_ll = _score_v3_incumbent(
        v3_elo["elo_prob"].values.astype(float), y, all_feat,
        qb_gate_mask, home_qb_adj, away_qb_adj, fold_frames,
    )
    print(f"  V3 incumbent val LL: {v3_val_ll:.4f}")

    # Compute v3 holdout
    v3_hold_ll = _compute_holdout_ll(
        v3_elo["elo_prob"].values.astype(float), y, all_feat,
        qb_gate_mask, home_qb_adj, away_qb_adj, fold_frames,
    )
    print(f"  V3 incumbent hold LL: {v3_hold_ll:.4f}")

    # ── Pi-Ratings grid ──
    results = []
    total = len(PI_ALPHAS) * len(PI_BASE_KS) * len(PI_HK_RATIOS) * len(PI_HFAS) * len(PI_REGS)
    count = 0

    elo_base = df.copy()  # features without Elo columns

    for alpha in PI_ALPHAS:
        for base_k in PI_BASE_KS:
            for hk_ratio in PI_HK_RATIOS:
                for hfa in PI_HFAS:
                    for reg in PI_REGS:
                        count += 1
                        if count % 24 == 0:
                            print(f"  Progress: {count}/{total}")

                        pi = compute_pi_ratings_features(
                            elo_base,
                            base_k=base_k, home_advantage=hfa,
                            preseason_regression=reg,
                            alpha=alpha, hk_ratio=hk_ratio,
                        )
                        pi_prob = pi["pi_prob"].values.astype(float)

                        val_ll = _score_pi_model(
                            pi_prob, y, all_feat,
                            qb_gate_mask, home_qb_adj, away_qb_adj,
                            fold_frames,
                        )
                        hold_ll = _compute_holdout_ll(
                            pi_prob, y, all_feat,
                            qb_gate_mask, home_qb_adj, away_qb_adj,
                            fold_frames,
                        )
                        results.append({
                            "alpha": alpha,
                            "base_k": base_k,
                            "hk_ratio": hk_ratio,
                            "hfa": hfa,
                            "reg": reg,
                            "avg_val_ll": val_ll,
                            "hold_ll": hold_ll,
                        })

    print(f"  Progress: {count}/{total} — complete")
    print(f"  Total candidates: {len(results)}")

    rdf = pd.DataFrame(results)
    rdf["delta_val"] = rdf["avg_val_ll"] - v3_val_ll
    rdf["delta_hold"] = rdf["hold_ll"] - v3_hold_ll

    best_val = rdf.loc[rdf["avg_val_ll"].idxmin()]
    best_hold = rdf.loc[rdf["hold_ll"].idxmin()]

    promoted = rdf[
        (rdf["delta_val"] <= -MIN_PROMOTION_DELTA)
        & (rdf["delta_hold"] <= -MIN_PROMOTION_DELTA)
    ]

    print(f"  V3 incumbent:       val={v3_val_ll:.4f}  hold={v3_hold_ll:.4f}")
    print(f"  Best val:           {best_val['avg_val_ll']:.4f}  (Δ={best_val['delta_val']:+.4f})  "
          f"hold={best_val['hold_ll']:.4f}")
    print(f"  Best hold:          {best_hold['hold_ll']:.4f}  (Δ={best_hold['delta_hold']:+.4f})  "
          f"val={best_hold['avg_val_ll']:.4f}")
    print(f"  Candidates beating v3 on both: {len(promoted)}")

    # ── Write report ──
    report_lines = [
        "# Pi-Ratings Experiment",
        "",
        "Coupled home/away rating with nonlinear score-error updates.",
        "",
        "**Design:**",
        "- Power-law MOV: `mov = |margin|^alpha`",
        "- Asymmetric K: `k_home = base_k * hk_ratio`, `k_away = base_k * (2 - hk_ratio)`",
        "- alpha=1, hk_ratio=1 = standard capped_linear Elo",
        "",
        f"**Grid:** {total} combos (alpha∈{PI_ALPHAS} × base_k∈{PI_BASE_KS} × "
        f"hk_ratio∈{PI_HK_RATIOS} × HFA∈{PI_HFAS} × reg∈{PI_REGS})",
        "",
        f"**V3 incumbent:** val={v3_val_ll:.4f}  hold={v3_hold_ll:.4f}",
        "",
    ]

    header = "| alpha | base_k | hk_ratio | HFA | reg | Avg Val LL | Δval | Hold LL | Δhold |"
    sep = "|-------|--------|----------|-----|-----|-----------|------|---------|-------|"

    report_lines.append("### Top 10 by Validation Log Loss")
    report_lines.append("")
    report_lines.append(header)
    report_lines.append(sep)
    top10 = rdf.sort_values("avg_val_ll").head(10)
    for _, r in top10.iterrows():
        report_lines.append(
            f"| {r['alpha']} | {r['base_k']} | {r['hk_ratio']} | {r['hfa']} | {r['reg']} | "
            f"{r['avg_val_ll']:.4f} | {r['delta_val']:+.4f} | "
            f"{r['hold_ll']:.4f} | {r['delta_hold']:+.4f} |"
        )
    report_lines.append("")

    # Best candidate detail
    report_lines.append("### Best Candidate (by val)")
    report_lines.append("")
    bc = best_val
    report_lines.append(("| Param | Value |"))
    report_lines.append(("|-------|-------|"))
    for p in ["alpha", "base_k", "hk_ratio", "hfa", "reg"]:
        report_lines.append(f"| {p} | {bc[p]} |")
    report_lines.append(f"| Avg Val LL | {bc['avg_val_ll']:.4f} |")
    report_lines.append(f"| Δval | {bc['delta_val']:+.4f} |")
    report_lines.append(f"| Hold LL | {bc['hold_ll']:.4f} |")
    report_lines.append(f"| Δhold | {bc['delta_hold']:+.4f} |")
    report_lines.append("")

    if len(promoted) > 0:
        report_lines.append("## ✅ PROMOTED")
        report_lines.append("")
        report_lines.append(f"**{len(promoted)}** candidate(s) beat v3 on both val and holdout.")
        report_lines.append("")
        report_lines.append("| alpha | base_k | hk_ratio | HFA | reg | Δval | Δhold |")
        report_lines.append("|-------|--------|----------|-----|-----|------|-------|")
        for _, r in promoted.sort_values("delta_val").iterrows():
            report_lines.append(
                f"| {r['alpha']} | {r['base_k']} | {r['hk_ratio']} | {r['hfa']} | {r['reg']} | "
                f"{r['delta_val']:+.4f} | {r['delta_hold']:+.4f} |"
            )
    else:
        report_lines.append("## ❌ NOT PROMOTED")
        report_lines.append("")
        report_lines.append("No candidate beats v3 on BOTH val and holdout with Δ ≥ 0.001.")
        report_lines.append("")
        report_lines.append(
            f"Best val (Δ={best_val['delta_val']:+.4f}) vs "
            f"Best hold (Δ={best_hold['delta_hold']:+.4f})."
        )
        report_lines.append("")

    # All results
    report_lines.append("### All Results")
    report_lines.append("")
    report_lines.append(header)
    report_lines.append(sep)
    for _, r in rdf.sort_values(["avg_val_ll", "hold_ll"]).iterrows():
        report_lines.append(
            f"| {r['alpha']} | {r['base_k']} | {r['hk_ratio']} | {r['hfa']} | {r['reg']} | "
            f"{r['avg_val_ll']:.4f} | {r['delta_val']:+.4f} | "
            f"{r['hold_ll']:.4f} | {r['delta_hold']:+.4f} |"
        )

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("Report: pi_ratings_experiment.py")

    report = "\n".join(report_lines)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(report)
    print(f"\n  Report: {report_path}")

    return report_path
