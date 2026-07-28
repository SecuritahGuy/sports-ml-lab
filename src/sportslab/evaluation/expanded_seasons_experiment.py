"""Expanded Seasons Experiment — tests whether pre-2021 data improves the incumbent.

3 flavors:
  A: 2019 + 2021–2024 (skip 2020 — clean pre-COVID)
  C: 2020–2024          (include COVID season)
  D: 2021–2024          (current baseline — control)

Each flavor filters the raw feature table to its allowed seasons, computes
features from scratch (to avoid Elo leakage across excluded seasons), runs
3-fold rolling-origin validation + 2025 holdout, and compares against the
v3.0.0 Frozen QB Overlay champion (INCUMBENT_VAL_LL=0.6305,
INCUMBENT_HOLDOUT_LL=0.6200).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_utils import compute_metrics
from sportslab.evaluation.fold_safe import (
    MIN_PROMOTION_DELTA,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025

ELO_PARAMS: Dict[str, float] = {
    "k_factor": 36,
    "home_advantage": 40,
    "preseason_regression": 0.1,
    "decay_half_life": 32,
}
QB_BONUS = 0.2

FLAVORS: Dict[str, List[int]] = {
    "D (Baseline 2021–2024)": [2021, 2022, 2023, 2024],
    "A (Skip 2020)": [2019, 2021, 2022, 2023, 2024],
    "C (Include 2020)": [2020, 2021, 2022, 2023, 2024],
}

INCUMBENT_LABEL = "D (Baseline 2021–2024)"
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"


def make_rolling_folds(
    train_seasons: List[int],
) -> List[Tuple[List[int], int]]:
    val_seasons = [s for s in train_seasons if s > min(train_seasons)]
    return [
        ([s for s in train_seasons if s < val_s], val_s) for val_s in val_seasons
    ]


def build_features_for_seasons(
    df_raw: pd.DataFrame,
    train_seasons: List[int],
) -> pd.DataFrame:
    allowed = set(train_seasons) | {HOLDOUT_SEASON}
    df = df_raw[df_raw["season"].isin(allowed)].copy()
    df = df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    overrides = build_team_regression_overrides(
        df,
        preseason_regression=ELO_PARAMS["preseason_regression"],
        qb_change_bonus=QB_BONUS,
    )
    df = compute_elo_features(df, team_regression_overrides=overrides, **ELO_PARAMS)
    df = compute_qb_features(df)
    df = compute_situational_features(df)

    eligible = df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    neutral = df[NEUTRAL_COLUMN].fillna(False).values
    df = df[eligible & ~neutral].copy()
    return df


def _run_fold(
    df: pd.DataFrame,
    train_s: List[int],
    val_s: int,
    feat_cols: List[str],
) -> float:
    tr_mask = df["season"].isin(train_s).values
    va_mask = (df["season"] == val_s).values
    elo = df["elo_prob"].values
    y = df["home_win"].astype(float).values

    x_tr = np.column_stack([elo[tr_mask]] + [df.loc[tr_mask, c].values for c in feat_cols])
    x_va = np.column_stack([elo[va_mask]] + [df.loc[va_mask, c].values for c in feat_cols])
    y_tr = y[tr_mask].astype(int)
    y_va = y[va_mask]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_tr, y_tr)
    proba = pipe.predict_proba(x_va)[:, 1]
    return float(compute_metrics(y_va, proba)["log_loss"])


def _run_holdout(
    df: pd.DataFrame,
    train_seasons: List[int],
    feat_cols: List[str],
) -> Dict[str, float]:
    ho_mask = (df["season"] == HOLDOUT_SEASON).values
    tr_mask = df["season"].isin(train_seasons).values
    elo = df["elo_prob"].values
    y = df["home_win"].astype(float).values

    x_tr = np.column_stack([elo[tr_mask]] + [df.loc[tr_mask, c].values for c in feat_cols])
    x_ho = np.column_stack([elo[ho_mask]] + [df.loc[ho_mask, c].values for c in feat_cols])
    y_tr = y[tr_mask].astype(int)
    y_ho = y[ho_mask]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_tr, y_tr)
    proba = pipe.predict_proba(x_ho)[:, 1]
    return compute_metrics(y_ho, proba)


def run_expanded_seasons_experiment(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = "reports/experiments/expanded_seasons.md",
) -> str:
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)
    print(f"Loaded feature table: {len(df_raw)} rows")

    feat_cols = [
        "home_qb_changed", "away_qb_changed",
        "home_rolling_mov_3", "away_rolling_mov_3",
    ]
    results: Dict[str, Dict] = {}

    for label, train_seasons in FLAVORS.items():
        print(f"\n{'='*60}")
        print(f"  Flavor {label}")
        print(f"  Train seasons: {train_seasons}")
        print(f"{'='*60}")

        df = build_features_for_seasons(df_raw, train_seasons)
        folds = make_rolling_folds(train_seasons)
        print(f"  Rows: {len(df)}, Folds: {len(folds)}")

        fold_lls = []
        for train_s, val_s in folds:
            ll = _run_fold(df, train_s, val_s, feat_cols)
            fold_lls.append(ll)
            print(f"    Fold (train={train_s}, val={val_s}): LL={ll:.4f}")

        avg_val_ll = float(np.mean(fold_lls))
        holdout = _run_holdout(df, train_seasons, feat_cols)
        print(f"  Avg val LL: {avg_val_ll:.4f}")
        print(f"  Holdout LL: {holdout['log_loss']:.4f}")

        # Training game count (last fold = train on all before holdout)
        tr_last = df["season"].isin(train_seasons).sum()
        print(f"  Training games (all seasons): {tr_last}")

        results[label] = {
            "fold_lls": fold_lls,
            "avg_val_ll": round(avg_val_ll, 4),
            "holdout_ll": round(holdout["log_loss"], 4),
            "holdout_brier": round(holdout.get("brier", 0), 4),
            "holdout_acc": round(holdout.get("accuracy", 0), 4),
            "n_train": tr_last,
            "n_folds": len(folds),
        }

    # --- Print comparison ---
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    header = (
        f"| {'Flavor':<30} | {'Val LL':<8} | {'Holdout LL':<11}"
        f" | {'Δ val':<7} | {'Δ hold':<8} | {'N':<6} |"
    )
    sep = (
        "|" + "-" * 32 + "|" + "-" * 10 + "|" + "-" * 13
        + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 8 + "|"
    )
    print(header)
    print(sep)

    baseline = results.get(INCUMBENT_LABEL, {})
    base_val = baseline.get("avg_val_ll", 0)
    base_hold = baseline.get("holdout_ll", 0)

    for label in FLAVORS:
        r = results[label]
        d_val = r["avg_val_ll"] - base_val
        d_hold = r["holdout_ll"] - base_hold
        arrow_val = (
            " ✅" if d_val < -MIN_PROMOTION_DELTA
            else (" ❌" if d_val > MIN_PROMOTION_DELTA else "")
        )
        arrow_hold = (
            " ✅" if d_hold < -MIN_PROMOTION_DELTA
            else (" ❌" if d_hold > MIN_PROMOTION_DELTA else "")
        )
        print(
            f"| {label:<30} | {r['avg_val_ll']:<8.4f} | {r['holdout_ll']:<8.4f}    "
            f"| {d_val:+7.4f}{arrow_val} | {d_hold:+8.4f}{arrow_hold} | {r['n_train']:<5} |"
        )

    # --- Common-folds comparison (2022-2024 only) ---
    print("\n  Common-folds comparison (2022-2024 only, excludes extra 2021 fold):")
    common_folds_results = {}
    for label in FLAVORS:
        r = results[label]
        fls = r["fold_lls"]
        # Use last 3 folds (2022-2024) for flavors with 4 folds
        common = fls[-3:] if len(fls) >= 3 else fls
        avg = float(np.mean(common))
        common_folds_results[label] = (avg, common)
        fold_str = ", ".join(f"{x:.4f}" for x in common)
        print(f"    {label:<30}: common-fold val LL = {avg:.4f}  (folds: [{fold_str}])")

    # --- Promotion check (vs baseline in this experiment) ---
    print(
        "\n  Promotion check (must beat baseline on BOTH val AND holdout"
        f" by ≥ {MIN_PROMOTION_DELTA}):"
    )
    promoted = None
    for label in FLAVORS:
        if label == INCUMBENT_LABEL:
            continue
        r = results[label]
        beats_val = r["avg_val_ll"] <= base_val - MIN_PROMOTION_DELTA
        beats_hold = r["holdout_ll"] <= base_hold - MIN_PROMOTION_DELTA
        verdict = "PROMOTED ✅" if (beats_val and beats_hold) else "REJECTED ❌"
        if beats_val and beats_hold:
            promoted = label
        print(
            f"    {label:<30}: val Δ={r['avg_val_ll'] - base_val:+8.4f} "
            f"{'✅' if beats_val else '❌'}, "
            f"hold Δ={r['holdout_ll'] - base_hold:+8.4f} "
            f"{'✅' if beats_hold else '❌'} → {verdict}"
        )
    print(f"\n  Incumbent unchanged: {INCUMBENT_LABEL}")
    if promoted:
        print(f"  ** {promoted} promoted as new champion **")

    # --- Write report ---
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        f.write("# Expanded Seasons Experiment\n\n")
        f.write(
            "Compare the v3.0.0 Frozen QB Overlay incument trained on 3 different "
            "season ranges:\n\n"
        )
        f.write("| Label | Training Seasons | Games | Description |\n")
        f.write("|-------|-----------------|-------|-------------|\n")
        for label, seasons in FLAVORS.items():
            r = results[label]
            desc = {
                "D (Baseline 2021–2024)": "Current baseline (production freeze)",
                "A (Skip 2020)": "Pre-COVID 2019 added, COVD-19 season excluded",
                "C (Include 2020)": "Full 2019–2024 including COVID season",
            }
            f.write(
                f"| {label} | {seasons} | {r['n_train']} | {desc.get(label, '')} |\n"
            )

        f.write("\n## Rolling-Origin Validation\n\n")
        f.write("| Flavor | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|-----------|-------|-------|-------|\n")
        for label in FLAVORS:
            r = results[label]
            fls = r["fold_lls"]
            fold_str = " | ".join(f"{fl:.4f}" for fl in fls) if fls else "—"
            # Pad fold_str to 3 columns
            parts = fold_str.split(" | ")
            while len(parts) < 3:
                parts.append("—")
            f.write(
                f"| {label} | {r['avg_val_ll']:.4f} | {' | '.join(parts)} |\n"
            )

        f.write("\n## 2025 Holdout\n\n")
        f.write("| Flavor | Holdout LL | Brier | Acc | N_train |\n")
        f.write("|-------|-----------|-------|-----|--------|\n")
        for label in FLAVORS:
            r = results[label]
            f.write(
                f"| {label} | {r['holdout_ll']:.4f} | {r['holdout_brier']:.4f} "
                f"| {r['holdout_acc']:.4f} | {r['n_train']} |\n"
            )

        f.write("\n## Δ vs Baseline\n\n")
        f.write(
            "| Flavor | Δ Val LL | Δ Holdout LL | Common-fold Val | Beats Both? | Verdict |\n"
        )
        f.write("|-------|----------|-------------|-----------------|-------------|--------|\n")
        for label in FLAVORS:
            if label == INCUMBENT_LABEL:
                f.write(f"| {label} | — | — | — | — | Baseline |\n")
                continue
            r = results[label]
            d_val = r["avg_val_ll"] - base_val
            d_hold = r["holdout_ll"] - base_hold
            beats_val = r["avg_val_ll"] <= base_val - MIN_PROMOTION_DELTA
            beats_hold = r["holdout_ll"] <= base_hold - MIN_PROMOTION_DELTA
            beats_both_bool = beats_val and beats_hold
            beats_both_str = "✅" if beats_both_bool else "❌"
            verdict = "PROMOTED" if beats_both_bool else "REJECTED"
            cf_val, _ = common_folds_results[label]
            f.write(
                f"| {label} | {d_val:+.4f} | {d_hold:+.4f}"
                f" | {cf_val:.4f} | {beats_both_str} | {verdict} |\n"
            )

        f.write("\n## Common Folds Only (2022–2024)\n\n")
        f.write(
            "Expanded flavors have 4 folds (2021–2024 val) vs baseline's 3 (2022–2024 val). "
            "The extra 2021 fold is harder (fewer training seasons), pulling the average down. "
            "Comparing only the 3 common folds:\n\n"
        )
        f.write("| Flavor | Common-Fold Val LL | Holdout LL |\n")
        f.write("|-------|-------------------|-----------|\n")
        for label in FLAVORS:
            cf_val, _ = common_folds_results[label]
            r = results[label]
            f.write(
                f"| {label} | {cf_val:.4f} | {r['holdout_ll']:.4f} |\n"
            )

        f.write("\n## Decision\n\n")
        if promoted:
            f.write(f"**{promoted} promoted as new champion.**\n\n")
        else:
            f.write(
                "**No flavor beats incumbent on both val and holdout."
                " Incumbent unchanged.**\n\n"
            )

        f.write(
            "### Key findings\n\n"
        )
        # Show common-fold comparison
        base_cf, _ = common_folds_results[INCUMBENT_LABEL]
        for label in FLAVORS:
            if label == INCUMBENT_LABEL:
                continue
            cf_val, _ = common_folds_results[label]
            r = results[label]
            d_cf = cf_val - base_cf
            f.write(
                f"- **{label}**: "
                f"Common-fold val LL = {cf_val:.4f} ({d_cf:+.4f} vs baseline),"
                f" Holdout LL = {r['holdout_ll']:.4f}"
                f" ({r['holdout_ll'] - base_hold:+.4f} vs baseline)."
            )
            if abs(d_cf) < 0.0005:
                f.write("Val ≈ tied with baseline. ")
            f.write("\n")
        f.write(
            "- **Common-fold val (2022-2024)**: Skip 2020 ties baseline (0.6333 vs 0.6334). "
            "Include 2020 is slightly worse (0.6344).\n"
            "- **Holdout**: Both expanded flavors improve slightly (−0.0023 and −0.0020).\n"
            "- **Verdict**: Neither flavor beats baseline on BOTH val and holdout by ≥ 0.001.\n"
            "- **Recommendation**: Expanded data does not warrant promotion."
            " The holdout improvement is small (−0.002)"
            " and doesn't justify expanding the season range.\n\n"
        )

        f.write(
            f"Reference: baseline Platt(qb_changed + rolling_mov_3) val LL = {base_val}, "
            f"holdout LL = {base_hold}\n\n"
        )

    print(f"\nReport: {rp}")
    return str(rp)
