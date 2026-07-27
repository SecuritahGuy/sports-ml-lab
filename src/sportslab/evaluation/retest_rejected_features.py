"""Re-test weather, scheduling, and injury features on the modern Elo spine.

The original experiments used an older Elo spine (K=40, reg=0.25, no MOV,
no season regression, no QB overlay). Re-run on the current incumbent spine
(K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2) with qb_changed + rolling_mov_3
+ QB overlay to see if modern infrastructure changes the outcome.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.injuries import INJURY_FEATURE_COLUMNS, compute_injury_features
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.scheduling import SCHEDULING_FEATURE_COLUMNS, compute_scheduling_features
from sportslab.features.situational import compute_situational_features
from sportslab.features.weather import WEATHER_FEATURE_COLUMNS, compute_weather_features

HOLDOUT_SEASON = 2025
FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]

BEST_K, BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 36, 40, 0.1, 32, 0.2
INCUMBENT_FEATURES = ["home_qb_changed", "away_qb_changed",
                      "home_rolling_mov_3", "away_rolling_mov_3"]

ELO_TO_LOGIT = np.log(10) / 400.0
OVERLAY_GAMMA = 1.0
OVERLAY_CAP = 40


def _logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _build_gate_mask(df):
    h_changed = df.get("home_qb_changed", pd.Series(0)).fillna(0).values.astype(float)
    a_changed = df.get("away_qb_changed", pd.Series(0)).fillna(0).values.astype(float)
    h_starts = df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    a_starts = df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    return (h_changed == 1) | (a_changed == 1) | (h_starts < 17) | (a_starts < 17)


def _build_base_model(df, extra_feat_cols, gate_mask, y, skip_elo=False):
    """Fit Platt + extra features, apply QB overlay, return (fold_lls, hold_metrics).

    If skip_elo=True, uses only extra features (no elo_prob, no incumbent features).
    """
    elo_prob = df["elo_prob"].values
    if skip_elo:
        feat_cols = [c for c in extra_feat_cols if c in df.columns]
    else:
        feat_cols = INCUMBENT_FEATURES + [c for c in extra_feat_cols if c in df.columns]
    print(f"    Features ({len(feat_cols)}): {feat_cols[:5]}...")

    home_qb_adj = df.get("home_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)
    away_qb_adj = df.get("away_qb_adj", pd.Series(0.0)).fillna(0).values.astype(float)

    def _build_design(mask):
        """Build design matrix for given mask."""
        parts = []
        if not skip_elo:
            parts.append(elo_prob[mask].reshape(-1, 1))
        for c in feat_cols:
            parts.append(df.loc[mask, c].values.reshape(-1, 1))
        return np.column_stack(parts)

    fold_lls = []
    for train_s, val_s in FOLDS:
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values
        x_tr, x_va = _build_design(tr), _build_design(va)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(x_tr, y[tr].astype(int))
        base_prob = pipe.predict_proba(x_va)[:, 1]

        if skip_elo:
            final_prob = base_prob
        else:
            base_logit = _logit(base_prob)
            capped_h = np.clip(home_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
            capped_a = np.clip(away_qb_adj[va], -OVERLAY_CAP, OVERLAY_CAP)
            overlay = OVERLAY_GAMMA * (capped_h - capped_a) * ELO_TO_LOGIT
            final_prob = _sigmoid(base_logit + overlay * gate_mask[va].astype(float))

        fold_lls.append(compute_classification_metrics(y[va], final_prob)["log_loss"])

    # Holdout
    is_train = df["season"] != HOLDOUT_SEASON
    is_hold = df["season"] == HOLDOUT_SEASON
    x_tr, x_hold = _build_design(is_train), _build_design(is_hold)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_tr, y[is_train].astype(int))
    base_prob = pipe.predict_proba(x_hold)[:, 1]

    if skip_elo:
        final_prob = base_prob
    else:
        base_logit = _logit(base_prob)
        capped_h = np.clip(home_qb_adj[is_hold], -OVERLAY_CAP, OVERLAY_CAP)
        capped_a = np.clip(away_qb_adj[is_hold], -OVERLAY_CAP, OVERLAY_CAP)
        overlay = OVERLAY_GAMMA * (capped_h - capped_a) * ELO_TO_LOGIT
        final_prob = _sigmoid(base_logit + overlay * gate_mask[is_hold].astype(float))

    hold_metrics = compute_classification_metrics(y[is_hold], final_prob)
    return fold_lls, hold_metrics


def run_retest_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/retest_rejected_features.md",
) -> str:
    rp = Path(report_path)

    print("=== Loading + building feature pipeline ===")
    df_raw = pd.read_parquet(ft_path)
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
    df = compute_scheduling_features(df)
    df = compute_injury_features(df)
    df = compute_weather_features(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy().reset_index(drop=True)
    y = df[TARGET_COLUMN].values
    gate_mask = _build_gate_mask(df)
    print(f"  {len(df)} eligible games, {gate_mask.sum()} with active overlay")

    # Define model variants (filter out non-numeric columns)
    weather_cols = [c for c in WEATHER_FEATURE_COLUMNS if c != "weather_source" and c in df.columns]
    sched_cols = [c for c in SCHEDULING_FEATURE_COLUMNS if c in df.columns]
    injury_cols = [c for c in INJURY_FEATURE_COLUMNS if c in df.columns]
    all_three_cols = weather_cols + sched_cols + injury_cols
    models = {
        "Incumbent (Platt)": [],
        "Incumbent + Weather": weather_cols,
        "Incumbent + Scheduling": sched_cols,
        "Incumbent + Injury": injury_cols,
        "Incumbent + All three": all_three_cols,
        "Weather only (no Elo)": weather_cols,
    }

    results = {}
    print("\n=== Rolling-origin validation ===")
    for name, extra_cols in models.items():
        skip = "no Elo" in name

        fold_lls, hold_metrics = _build_base_model(df, extra_cols, gate_mask, y, skip_elo=skip)

        n_feat = len([c for c in extra_cols if c in df.columns])
        avg_ll = float(np.mean(fold_lls))
        results[name] = {
            "fold_ll": fold_lls,
            "avg_val_ll": avg_ll,
            "holdout_ll": hold_metrics["log_loss"],
            "holdout_brier": hold_metrics["brier_score"],
            "holdout_auc": hold_metrics["roc_auc"],
            "holdout_acc": hold_metrics["accuracy"],
            "n_feat": n_feat,
        }
        print(f"  {name:<35s} val={avg_ll:.4f} hold={hold_metrics['log_loss']:.4f} "
              f"({n_feat} feat)")

    # Report
    inc = results.get("Incumbent (Platt)", {})
    print(f"\n=== Report: {rp} ===")
    with open(rp, "w") as f:
        f.write("# Re-test: Weather, Scheduling, Injury Features\n\n")
        f.write("Original tests used an older Elo spine (K=40, reg=0.25, no MOV, ")
        f.write("no season regression, no QB overlay). ")
        f.write("Re-running on the current incumbent spine:\n")
        f.write(f"- K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG}, ")
        f.write(f"decay={BEST_DECAY}, qb_bonus={BEST_QB_BONUS}\n")
        f.write(f"- {len(FOLDS)}-fold rolling-origin + {HOLDOUT_SEASON} holdout\n")
        f.write("- All models include QB overlay in logit space\n\n")

        f.write("| Model | Val LL | Fold1 | Fold2 | Fold3 | Hold LL | Brier | AUC | Acc | Feat |\n")
        f.write("|-------|--------|-------|-------|-------|---------|-------|-----|-----|------|\n")
        for name, r in results.items():
            fls = [f"{x:.4f}" for x in r['fold_ll']]
            while len(fls) < 3:
                fls.append("-")
            delta = r['avg_val_ll'] - inc.get('avg_val_ll', 0)
            d = f" (Δ={delta:+.4f})" if name != "Incumbent (Platt)" else ""
            f.write(f"| {name} | {r['avg_val_ll']:.4f}{d} "
                    f"| {fls[0]} | {fls[1]} | {fls[2]} "
                    f"| {r['holdout_ll']:.4f} | {r['holdout_brier']:.4f} "
                    f"| {r['holdout_auc']:.3f} | {r['holdout_acc']:.3f} | {r['n_feat']} |\n")

        promoted = False
        for name, r in results.items():
            if name == "Incumbent (Platt)":
                continue
            vd = r['avg_val_ll'] - inc['avg_val_ll']
            hd = r['holdout_ll'] - inc['holdout_ll']
            if vd <= -0.001 and hd <= -0.001:
                f.write(f"\n✅ **{name} beats incumbent on BOTH** "
                        f"(Δval={vd:.4f}, Δhold={hd:.4f})\n")
                promoted = True

        if not promoted:
            f.write("\n**No model beats incumbent on both val and holdout by ≥ 0.001.**\n")

        f.write("\n---\nReport: retest_rejected_features.py\n")

    print(f"  Report written to {rp}")
    return str(rp)
