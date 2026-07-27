"""Rolling-origin experiment: test rolling-window StatSpace components.

Compares 5 model variants:
  1. Platt (incumbent) — Elo prob + qb_changed + rolling_mov_3
  2. Incumbent + rolling StatSpace (window=3)
  3. Incumbent + rolling StatSpace (window=5)
  4. Rolling StatSpace only (no Elo, window=3)
  5. Rolling StatSpace all windows combined
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.predict_incumbent import (
    BEST_DECAY,
    BEST_HFA,
    BEST_K,
    BEST_QB_BONUS,
    BEST_REG,
    FEATURE_COLS,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.epa import load_pbp_data
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.rolling_statspace import (
    ROLLING_WINDOWS,
    compute_rolling_composites,
)
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]


def _rolling_cols(window):
    """Generate composite column names for a given rolling window."""
    cols = []
    for side in ("home", "away"):
        for metric in ("doba_composite", "chaos_composite"):
            cols.append(f"{side}_{metric}_{window}")
    return cols


def _fit_and_eval(x_tr, y_tr, x_va, y_va):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_tr, y_tr.astype(int))
    proba = pipe.predict_proba(x_va)[:, 1]
    return compute_classification_metrics(y_va, proba)


def run_rolling_statspace_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/rolling_statspace.md",
) -> str:
    rp = Path(report_path)

    print("=== Loading feature table ===")
    df_raw = pd.read_parquet(ft_path)
    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus= BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw, k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG, team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy().reset_index(drop=True)
    y = df[TARGET_COLUMN].values
    print(f"  {len(df)} eligible games")

    print("\n=== Loading PBP data ===")
    seasons = sorted(s for s in df["season"].unique() if s <= 2025)
    pbp = load_pbp_data(seasons)
    print(f"  {len(pbp)} plays loaded")

    print("\n=== Computing rolling StatSpace composites ===")
    df_rs = compute_rolling_composites(df, pbp=pbp, windows=[3, 5])
    for w in ROLLING_WINDOWS:
        cols = _rolling_cols(w)
        present = [c for c in cols if c in df_rs.columns]
        print(f"  Window {w}: {len(present)} composite features available")

    is_train = df["season"] != HOLDOUT_SEASON
    is_hold = df["season"] == HOLDOUT_SEASON

    # --- Define model variants ---
    models = {
        "Platt (incumbent)": {
            "feat_cols": FEATURE_COLS,
            "use_elo": True,
        },
        "Incumbent + RS (w=3)": {
            "feat_cols": FEATURE_COLS + _rolling_cols(3),
            "use_elo": True,
        },
        "Incumbent + RS (w=5)": {
            "feat_cols": FEATURE_COLS + _rolling_cols(5),
            "use_elo": True,
        },
        "Incumbent + RS (w=3+5)": {
            "feat_cols": FEATURE_COLS + _rolling_cols(3) + _rolling_cols(5),
            "use_elo": True,
        },
        "RS only (w=3)": {
            "feat_cols": _rolling_cols(3),
            "use_elo": False,
        },
    }

    results = {}
    print("\n=== Rolling-origin validation ===")
    for name, cfg in models.items():
        feat_cols = [c for c in cfg["feat_cols"] if c in df_rs.columns]
        if not feat_cols and not cfg["use_elo"]:
            print(f"  {name}: no features, skipping")
            continue

        folds_ll = []
        for train_s, val_s in ROLLING_FOLDS:
            tr = df["season"].isin(train_s).values
            va = (df["season"] == val_s).values
            parts, parts_va = [], []
            if cfg["use_elo"]:
                parts.append(df_rs.loc[tr, "elo_prob"].values.reshape(-1, 1))
                parts_va.append(df_rs.loc[va, "elo_prob"].values.reshape(-1, 1))
            for c in feat_cols:
                parts.append(df_rs.loc[tr, c].values.reshape(-1, 1))
                parts_va.append(df_rs.loc[va, c].values.reshape(-1, 1))
            x_tr = np.column_stack(parts)
            x_va = np.column_stack(parts_va) if parts_va else np.empty((va.sum(), 0))
            if x_va.shape[1] == 0:
                folds_ll.append(np.nan)
                continue
            metrics = _fit_and_eval(x_tr, y[tr], x_va, y[va])
            folds_ll.append(metrics["log_loss"])

        tr_parts, va_parts = [], []
        if cfg["use_elo"]:
            tr_parts.append(df_rs.loc[is_train, "elo_prob"].values.reshape(-1, 1))
            va_parts.append(df_rs.loc[is_hold, "elo_prob"].values.reshape(-1, 1))
        for c in feat_cols:
            tr_parts.append(df_rs.loc[is_train, c].values.reshape(-1, 1))
            va_parts.append(df_rs.loc[is_hold, c].values.reshape(-1, 1))
        x_tr = np.column_stack(tr_parts)
        x_va = np.column_stack(va_parts) if va_parts else np.empty((len(y[is_hold]), 0))

        if x_va.shape[1] == 0:
            hold_metrics = {
                "log_loss": np.nan, "brier_score": np.nan,
                "roc_auc": np.nan, "accuracy": np.nan,
            }
        else:
            hold_metrics = _fit_and_eval(x_tr, y[is_train], x_va, y[is_hold])
        avg_ll = float(np.nanmean(folds_ll)) if any(not np.isnan(f) for f in folds_ll) else np.nan
        results[name] = {
            "fold_ll": folds_ll,
            "avg_val_ll": avg_ll,
            "holdout_ll": hold_metrics["log_loss"],
            "holdout_brier": hold_metrics["brier_score"],
            "holdout_auc": hold_metrics["roc_auc"],
            "holdout_acc": hold_metrics["accuracy"],
            "n_feat": len(feat_cols),
        }
        print(f"  {name:<30s} val={avg_ll:.4f} "
              f"hold={hold_metrics['log_loss']:.4f} "
              f"({len(feat_cols)} feat)")

    # --- Report ---
    incumbent_name = "Platt (incumbent)"
    inc = results.get(incumbent_name, {})
    print(f"\n=== Report: {rp} ===")
    with open(rp, "w") as f:
        f.write("# Rolling StatSpace Experiment\n\n")
        f.write(f"Test whether rolling-window StatSpace composite metrics "
                f"(DOBA + Chaos Rate from per-game PBP, z-scored, averaged over "
                f"{ROLLING_WINDOWS}-game windows) improve on the incumbent.\n\n")

        f.write("| Model | Val LL | Fold1 | Fold2 | Fold3 | Hold LL | Brier | AUC | Acc | Feat |\n")
        f.write("|------|--------|-------|-------|-------|---------|-------|-----|-----|------|\n")
        for name, r in results.items():
            if pd.isna(r['avg_val_ll']) or pd.isna(r['holdout_ll']):
                continue
            fls = r['fold_ll']
            fl = f"{fls[0]:.4f}" if len(fls) > 0 and not pd.isna(fls[0]) else "-"
            fl2 = f"{fls[1]:.4f}" if len(fls) > 1 and not pd.isna(fls[1]) else "-"
            fl3 = f"{fls[2]:.4f}" if len(fls) > 2 and not pd.isna(fls[2]) else "-"
            delta = r['avg_val_ll'] - inc.get('avg_val_ll', 0)
            delta_str = f" (Δ={delta:+.4f})" if name != incumbent_name else ""
            f.write(f"| {name} | {r['avg_val_ll']:.4f}{delta_str} | {fl} | {fl2} | {fl3} "
                    f"| {r['holdout_ll']:.4f} | {r['holdout_brier']:.4f} "
                    f"| {r['holdout_auc']:.3f} | {r['holdout_acc']:.3f} | {r['n_feat']} |\n")

        f.write("\n### Winners\n\n")
        best_val = min(results.items(), key=lambda x: x[1]["avg_val_ll"])
        best_hold = min(results.items(), key=lambda x: x[1]["holdout_ll"])
        f.write(f"- **Best val LL**: {best_val[0]} ({best_val[1]['avg_val_ll']:.4f})\n")
        f.write(f"- **Best hold LL**: {best_hold[0]} ({best_hold[1]['holdout_ll']:.4f})\n")

        any_promoted = False
        for name, r in results.items():
            if name == incumbent_name:
                continue
            val_delta = r['avg_val_ll'] - inc['avg_val_ll']
            hold_delta = r['holdout_ll'] - inc['holdout_ll']
            if val_delta <= -0.001 and hold_delta <= -0.001:
                f.write(f"\n✅ **{name} beats incumbent on BOTH** "
                        f"(Δval={val_delta:.4f}, Δhold={hold_delta:.4f})\n")
                any_promoted = True

        if not any_promoted:
            f.write("\n**No model beats incumbent on both val and holdout by ≥ 0.001.**\n")

        f.write("\n---\nReport: rolling_statspace_experiment.py\n")

    print(f"  Report written to {rp}")
    return str(rp)
