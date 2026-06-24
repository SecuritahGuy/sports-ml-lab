"""Rolling MOV window sensitivity experiment.

Tests whether the promoted rolling_mov_3 feature is truly optimal vs
other window sizes and functional forms of margin of victory, all on
top of the clean incumbent Elo spine.
"""

from pathlib import Path
from typing import Dict, List, Tuple

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
from sportslab.features.coach import compute_coach_features
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features

HOLDOUT_SEASON = 2025
ROLLING_FOLDS: List[Tuple[List[int], int]] = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]
BEST_K, BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 36, 40, 0.1, 32, 0.2
QB_CHANGED_COLS = ["home_qb_changed", "away_qb_changed"]

# All window sizes to test
WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]


def _build_elo_spine() -> pd.DataFrame:
    fp = Path("data/features/nfl/feature_table.parquet")
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=BEST_REG,
        qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_coach_features(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy()
    return df.reset_index(drop=True)


def _compute_rolling_mov_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all rolling MOV window variants chronologically.

    For each game, the rolling MOV is computed from previous games only
    (no current game result). Season boundaries are respected.
    """
    out = df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    team_mov_hist: Dict[str, List[float]] = {}
    team_current_season: Dict[str, int] = {}

    for col in [f"rolling_mov_{w}" for w in WINDOWS]:
        out[f"home_{col}"] = 0.0
        out[f"away_{col}"] = 0.0
    out["rolling_mov_diff"] = 0.0
    out["rolling_mov_capped"] = 0.0
    out["rolling_mov_log_signed"] = 0.0
    out["rolling_mov_ewma"] = 0.0
    out["rolling_mov_std_3"] = 0.0
    out["rolling_mov_std_5"] = 0.0

    for idx, row in out.iterrows():
        season = int(row["season"])
        home = row["home_team"]
        away = row["away_team"]

        # Reset team history on season boundary
        for team in [home, away]:
            if team not in team_current_season or team_current_season[team] != season:
                team_mov_hist[team] = []
                team_current_season[team] = season

        # Compute home rolling features
        h_hist = team_mov_hist.get(home, [])
        for w in WINDOWS:
            vals = h_hist[-w:] if len(h_hist) >= w else h_hist
            out.at[idx, f"home_rolling_mov_{w}"] = float(np.mean(vals)) if vals else 0.0

        a_hist = team_mov_hist.get(away, [])
        for w in WINDOWS:
            vals = a_hist[-w:] if len(a_hist) >= w else a_hist
            out.at[idx, f"away_rolling_mov_{w}"] = float(np.mean(vals)) if vals else 0.0

        # Differential: home MOV - away MOV
        h_mov3 = out.at[idx, "home_rolling_mov_3"]
        a_mov3 = out.at[idx, "away_rolling_mov_3"]
        out.at[idx, "rolling_mov_diff"] = h_mov3 - a_mov3

        # Capped: clamp to [-14, 14]
        out.at[idx, "rolling_mov_capped"] = np.clip(h_mov3 - a_mov3, -14, 14)

        # Signed log: sign * log(1 + |diff|)
        diff = h_mov3 - a_mov3
        out.at[idx, "rolling_mov_log_signed"] = (
            np.sign(diff) * np.log1p(abs(diff)) if diff != 0 else 0.0
        )

        # EWMA: alpha=0.5 (weight on most recent)
        ewma_h = h_hist[-1] if h_hist else 0.0
        ewma_a = a_hist[-1] if a_hist else 0.0
        if len(h_hist) >= 2:
            ewma_h = 0.5 * h_hist[-1] + 0.5 * (
                0.5 * h_hist[-2] + 0.25 * (h_hist[-3] if len(h_hist) >= 3 else 0)
                if len(h_hist) >= 3
                else 0.5 * h_hist[-2]
            )
        if len(a_hist) >= 2:
            ewma_a = 0.5 * a_hist[-1] + 0.5 * (
                0.5 * a_hist[-2] + 0.25 * (a_hist[-3] if len(a_hist) >= 3 else 0)
                if len(a_hist) >= 3
                else 0.5 * a_hist[-2]
            )
        out.at[idx, "rolling_mov_ewma"] = ewma_h - ewma_a

        # Rolling std dev (volatility) over 3 and 5 games
        for w, col in [(3, "rolling_mov_std_3"), (5, "rolling_mov_std_5")]:
            h_vals = h_hist[-w:] if len(h_hist) >= w else h_hist
            a_vals = a_hist[-w:] if len(a_hist) >= w else a_hist
            combined = h_vals + a_vals
            out.at[idx, col] = float(np.std(combined)) if len(combined) >= 2 else 0.0

        # Season-to-date average MOV
        out.at[idx, "home_rolling_mov_std"] = float(np.mean(h_hist)) if h_hist else 0.0
        out.at[idx, "away_rolling_mov_std"] = float(np.mean(a_hist)) if a_hist else 0.0

        # Post-game: update history with result
        result = row.get("result", 0)
        team_mov_hist.setdefault(home, []).append(result)
        team_mov_hist.setdefault(away, []).append(-result)

    return out


def _logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def _rolling_val_ll(
    df: pd.DataFrame,
    elo_prob: np.ndarray,
    y: np.ndarray,
    feat_cols: List[str],
) -> float:
    fold_lls = []
    for train_s, val_s in ROLLING_FOLDS:
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values
        train_elo = elo_prob[tr]
        val_elo = elo_prob[va]
        train_y = y[tr].astype(int)
        val_y = y[va]
        if feat_cols:
            x_tr = np.column_stack([train_elo] + [df.loc[tr, c].values for c in feat_cols])
            x_va = np.column_stack([val_elo] + [df.loc[va, c].values for c in feat_cols])
        else:
            x_tr = train_elo.reshape(-1, 1)
            x_va = val_elo.reshape(-1, 1)
        pipe = _logistic_model()
        pipe.fit(x_tr, train_y)
        proba = pipe.predict_proba(x_va)[:, 1]
        fold_lls.append(compute_classification_metrics(val_y, proba)["log_loss"])
    return float(np.mean(fold_lls))


def run_rolling_mov_sensitivity(
    report_path: str = "reports/experiments/rolling_mov_sensitivity.md",
) -> str:
    print("=== Building Elo spine ===")
    df = _build_elo_spine()

    print("=== Computing rolling MOV variants ===")
    df = _compute_rolling_mov_variants(df)
    print(f"  Rows: {len(df)}")

    elo_prob = df["elo_prob"].values
    y = df[TARGET_COLUMN].astype(float).values

    # Define candidate feature groups
    candidates: Dict[str, List[str]] = {}
    for w in WINDOWS:
        cols = [f"home_rolling_mov_{w}", f"away_rolling_mov_{w}"]
        candidates[f"mov_{w}"] = cols
    candidates["mov_diff"] = ["rolling_mov_diff"]
    candidates["mov_capped"] = ["rolling_mov_capped"]
    candidates["mov_log_signed"] = ["rolling_mov_log_signed"]
    candidates["mov_ewma"] = ["rolling_mov_ewma"]
    candidates["mov_std_3"] = ["rolling_mov_std_3"]
    candidates["mov_std_5"] = ["rolling_mov_std_5"]

    # Baseline
    print("\n=== Validating (no 2025 metrics during search) ===")
    platt_ll = _rolling_val_ll(df, elo_prob, y, [])
    print(f"  Platt baseline: {platt_ll:.4f}")

    # qb_changed only
    qb_cols = [c for c in QB_CHANGED_COLS if c in df.columns]
    qb_only_ll = _rolling_val_ll(df, elo_prob, y, qb_cols)
    print(f"  qb_changed only: {qb_only_ll:.4f}")

    # Incumbent: qb_changed + mov_3
    inc_cols = qb_cols + [c for c in candidates["mov_3"] if c in df.columns]
    incumbent_ll = _rolling_val_ll(df, elo_prob, y, inc_cols)
    print(f"  Incumbent (qb + mov_3): {incumbent_ll:.4f}")

    # Test each variant
    results: Dict[str, Dict] = {}
    for name, raw_cols in candidates.items():
        avail = [c for c in raw_cols if c in df.columns]
        # Alone (no qb_changed)
        alone_ll = _rolling_val_ll(df, elo_prob, y, avail)
        # With qb_changed
        with_qb = _rolling_val_ll(df, elo_prob, y, qb_cols + avail)
        results[name] = {"alone": alone_ll, "with_qb": with_qb}
        print(f"  {name}: alone={alone_ll:.4f}, +qb={with_qb:.4f}")

    # Combined windows: mov_3 + mov_5
    comb35 = [c for c in candidates["mov_3"] if c in df.columns] + [
        c for c in candidates["mov_5"] if c in df.columns
    ]
    comb35_ll = _rolling_val_ll(df, elo_prob, y, qb_cols + comb35)
    print(f"  qb + mov_3 + mov_5: {comb35_ll:.4f}")
    results["mov_3_plus_5"] = {"alone": None, "with_qb": comb35_ll}

    # Best on validation (with qb_changed, since incumbent has it)
    best_name = min(
        [n for n in candidates if results[n]["with_qb"] is not None],
        key=lambda n: results[n]["with_qb"],
    )
    best_ll = results[best_name]["with_qb"]
    print(f"\n  Best on val: {best_name} ({best_ll:.4f})")

    # Best without qb_changed
    best_alone_name = min(
        [n for n in candidates if results[n]["alone"] is not None],
        key=lambda n: results[n]["alone"],
    )
    best_alone_ll = results[best_alone_name]["alone"]
    print(f"  Best alone: {best_alone_name} ({best_alone_ll:.4f})")

    # Selected candidate (by validation)
    selected_name = best_name
    selected_ll = best_ll
    selected_cols = qb_cols + [c for c in candidates[selected_name] if c in df.columns]
    print(f"\n  Selected for holdout: {selected_name} (val LL={selected_ll:.4f})")

    # ── 2025 holdout (one-shot, after selection) ──
    print("\n=== 2025 Holdout (one-shot after selection) ===")
    is_hold = (df["season"] == HOLDOUT_SEASON).values
    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    hold_y = y[is_hold]
    train_y = y[is_train].astype(int)

    def _eval_holdout(feat_cols: List[str]) -> float:
        train_elo = elo_prob[is_train]
        hold_elo = elo_prob[is_hold]
        if feat_cols:
            x_tr = np.column_stack([train_elo] + [df.loc[is_train, c].values for c in feat_cols])
            x_ho = np.column_stack([hold_elo] + [df.loc[is_hold, c].values for c in feat_cols])
        else:
            x_tr = train_elo.reshape(-1, 1)
            x_ho = hold_elo.reshape(-1, 1)
        pipe = _logistic_model()
        pipe.fit(x_tr, train_y)
        proba = pipe.predict_proba(x_ho)[:, 1]
        return compute_classification_metrics(hold_y, proba)["log_loss"]

    platt_hold_ll = _eval_holdout([])
    inc_hold_ll = _eval_holdout(inc_cols)
    sel_hold_ll = _eval_holdout(selected_cols)
    qb_hold_ll = _eval_holdout(qb_cols)

    print(f"  Platt:      {platt_hold_ll:.4f}")
    print(f"  qb_changed: {qb_hold_ll:.4f}")
    print(f"  Incumbent:  {inc_hold_ll:.4f}")
    print(f"  Selected:   {sel_hold_ll:.4f}")

    # ── Report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Rolling MOV Sensitivity Experiment\n\n")
        f.write(
            "Tests whether the promoted `rolling_mov_3` window is optimal"
            " vs other window sizes and functional forms.\n\n"
        )

        f.write("## Method\n\n")
        f.write("Rolling-origin 3-fold validation. One-shot 2025 holdout.\n\n")
        f.write("### Elo Spine\n\n")
        f.write(
            f"K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG},"
            f" decay={BEST_DECAY}, qb_bonus={BEST_QB_BONUS}\n\n"
        )

        f.write("### Candidate Features\n\n")
        f.write("| Name | Description | Columns |\n")
        f.write("|------|-------------|---------|\n")
        for w in WINDOWS:
            f.write(f"| mov_{w} | Rolling avg MOV last {w} games | home/away_rolling_mov_{w} |\n")
        f.write("| mov_diff | Home MOV - Away MOV | rolling_mov_diff |\n")
        f.write("| mov_capped | Capped diff [-14, 14] | rolling_mov_capped |\n")
        f.write("| mov_log_signed | Signed log(1+|diff|) | rolling_mov_log_signed |\n")
        f.write("| mov_ewma | EWMA MOV (alpha=0.5) | rolling_mov_ewma |\n")
        f.write("| mov_std_3 | MOV volatility 3-game | rolling_mov_std_3 |\n")
        f.write("| mov_std_5 | MOV volatility 5-game | rolling_mov_std_5 |\n\n")

        f.write("### Leakage Prevention\n\n")
        f.write(
            "- Rolling MOV uses only previous games (shifted)\n"
            "- Current game result excluded\n"
            "- Season boundaries: team history reset per season\n"
            "- Early-season: defaults to 0.0 (no prior games)\n"
            "- EWMA computed from historical games only\n\n"
        )

        f.write("### Rolling-Origin Folds\n\n")
        for train_s, val_s in ROLLING_FOLDS:
            f.write(f"- Train {train_s} → Validate {val_s}\n")
        f.write(f"- Holdout: {HOLDOUT_SEASON}\n\n")

        f.write("## Validation Results\n\n")
        f.write("| Model | Alone | +qb_changed |\n")
        f.write("|------|-------|-------------|\n")
        f.write("| Platt | — | — |\n")
        f.write(f"| qb_changed only | — | {qb_only_ll:.4f} |\n")
        f.write(f"| Incumbent (qb+mov_3) | — | {incumbent_ll:.4f} |\n")
        for name in sorted(candidates.keys()):
            r = results[name]
            alone_str = f"{r['alone']:.4f}" if r["alone"] is not None else "—"
            with_qb_str = f"{r['with_qb']:.4f}" if r["with_qb"] is not None else "—"
            f.write(f"| {name} | {alone_str} | {with_qb_str} |\n")
        f.write(f"| qb+mov_3+mov_5 | — | {comb35_ll:.4f} |\n\n")

        f.write("### Best Validation Configurations\n\n")
        f.write(f"Best with qb_changed: **{best_name}** ({best_ll:.4f})\n\n")
        f.write(f"Best without qb_changed: **{best_alone_name}** ({best_alone_ll:.4f})\n\n")

        f.write("## 2025 Holdout (one-shot)\n\n")
        f.write("| Model | Holdout LL |\n")
        f.write("|-------|-----------|\n")
        f.write(f"| Platt baseline | {platt_hold_ll:.4f} |\n")
        f.write(f"| qb_changed only | {qb_hold_ll:.4f} |\n")
        f.write(f"| Incumbent (qb+mov_3) | {inc_hold_ll:.4f} |\n")
        f.write(f"| Selected ({selected_name}) | {sel_hold_ll:.4f} |\n\n")

        beats_val = selected_ll < incumbent_ll
        beats_hold = sel_hold_ll < inc_hold_ll

        f.write("## Decision\n\n")
        if beats_val and beats_hold:
            f.write(
                f"**PROMOTED:** {selected_name} beats incumbent on both"
                f" val ({selected_ll:.4f} vs {incumbent_ll:.4f})"
                f" and holdout ({sel_hold_ll:.4f} vs {inc_hold_ll:.4f}).\n"
            )
        elif beats_val:
            f.write(
                f"**Diagnostic only:** {selected_name} beats incumbent on val"
                f" ({selected_ll:.4f} vs {incumbent_ll:.4f})"
                f" but not holdout ({sel_hold_ll:.4f} vs {inc_hold_ll:.4f}).\n"
            )
        else:
            f.write(
                f"**No improvement.** Best variant ({selected_name}:"
                f" {selected_ll:.4f}) does not beat incumbent"
                f" ({incumbent_ll:.4f}) on validation.\n"
            )
            f.write("Incumbent rolling_mov_3 stands.\n")

    print(f"\nReport: {rp}")
    return str(rp)
