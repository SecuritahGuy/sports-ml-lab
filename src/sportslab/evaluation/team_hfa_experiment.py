"""Team-specific HFA experiment — testing per-team home field advantages."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.hfa import compute_team_hfa, margin_to_elo_hfa
from sportslab.features.ratings import (
    MOV_CAPPED_LINEAR,
    compute_elo_features,
)

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

# Incumbent params (decayed Elo)
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.20
BEST_MOV_TYPE = MOV_CAPPED_LINEAR
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0
BEST_DECAY = 32


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _fit_platt(train_prob: np.ndarray, train_y: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(train_prob.reshape(-1, 1), train_y)
    return pipe


def run_team_hfa_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/team_hfa.md",
) -> str:
    """Rolling-origin experiment comparing global HFA vs per-team HFA.

    Per-team HFA is computed from training data only (no validation/holdout
    leakage).  The incumbent (decayed Elo + Platt) is the baseline.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Rolling-origin comparison: global HFA vs team HFA ──
    print("=== Team HFA Experiment ===")
    all_results: list[dict[str, Any]] = []

    # Test both global-only and team-specific HFA
    for use_team_hfa in [False, True]:
        label = "team HFA" if use_team_hfa else "global HFA"
        fold_lls: list[float] = []
        fold_platt_lls: list[float] = []

        for train_seasons, val_season in ROLLING_FOLDS:
            team_hfa_dict = None
            if use_team_hfa:
                # Compute per-team HFA from training data only
                raw_hfa = compute_team_hfa(df_raw, train_seasons)
                team_hfa_dict = {team: margin_to_elo_hfa(val) for team, val in raw_hfa.items()}

            edf = compute_elo_features(
                df_raw,
                k_factor=BEST_K,
                home_advantage=BEST_HFA,
                preseason_regression=BEST_REG,
                mov_type=BEST_MOV_TYPE,
                mov_scale=BEST_MOV_SCALE,
                mov_cap=BEST_MOV_CAP,
                decay_half_life=BEST_DECAY,
                team_hfa=team_hfa_dict,
            )
            edf = _filter_df(edf)

            is_train = edf["season"].isin(train_seasons).values
            is_val = (edf["season"] == val_season).values
            y = edf[TARGET_COLUMN].astype(float).values
            elo_prob = edf["elo_prob"].values

            # Raw Elo LL
            val_ll = float(log_loss(y[is_val], elo_prob[is_val]))
            fold_lls.append(val_ll)

            # Platt
            train_p = elo_prob[is_train]
            train_y_ = y[is_train].astype(int)
            val_p = elo_prob[is_val]
            val_y_ = y[is_val]
            if len(np.unique(train_y_)) > 1:
                platt = _fit_platt(train_p, train_y_)
                platt_val = platt.predict_proba(val_p.reshape(-1, 1))[:, 1]
                platt_ll = float(log_loss(val_y_, platt_val))
            else:
                platt_ll = val_ll
            fold_platt_lls.append(platt_ll)

        avg_ll = float(np.mean(fold_lls))
        avg_platt = float(np.mean(fold_platt_lls))
        print(f"  {label}: avg raw LL={avg_ll:.4f}, avg Platt LL={avg_platt:.4f}")
        all_results.append(
            {
                "use_team_hfa": use_team_hfa,
                "label": label,
                "avg_raw_ll": round(avg_ll, 5),
                "avg_platt_ll": round(avg_platt, 5),
                "fold_lls": fold_lls,
                "fold_platt_lls": fold_platt_lls,
            }
        )

    # ── Full 2021-2024 training → 2025 holdout ──
    print("\n=== 2025 Holdout ===")

    # Global HFA (incumbent)
    print("  Global HFA:")
    inc_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
    )
    inc_elo = _filter_df(inc_elo)
    is_hold = (inc_elo["season"] == HOLDOUT_SEASON).values
    hold_y = inc_elo.loc[is_hold, TARGET_COLUMN].astype(float).values
    hold_prob_inc = inc_elo.loc[is_hold, "elo_prob"].values
    inc_hold_raw = compute_classification_metrics(hold_y, hold_prob_inc)
    is_train_inc = inc_elo["season"].isin([2021, 2022, 2023, 2024]).values
    inc_platt = _fit_platt(
        inc_elo.loc[is_train_inc, "elo_prob"].values,
        inc_elo.loc[is_train_inc, TARGET_COLUMN].astype(int).values,
    )
    hold_platt_inc = inc_platt.predict_proba(hold_prob_inc.reshape(-1, 1))[:, 1]
    inc_hold_platt = compute_classification_metrics(hold_y, hold_platt_inc)
    print(f"    raw Elo holdout: {inc_hold_raw['log_loss']:.4f}")
    print(f"    + Platt holdout: {inc_hold_platt['log_loss']:.4f}")

    # Team HFA
    print("  Team HFA:")
    team_all = compute_team_hfa(df_raw, [2021, 2022, 2023, 2024])
    team_hfa_all = {t: margin_to_elo_hfa(v) for t, v in team_all.items()}
    thfa_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        team_hfa=team_hfa_all,
    )
    thfa_elo = _filter_df(thfa_elo)
    hold_prob_thfa = thfa_elo.loc[is_hold, "elo_prob"].values
    thfa_hold_raw = compute_classification_metrics(hold_y, hold_prob_thfa)
    is_train_thfa = thfa_elo["season"].isin([2021, 2022, 2023, 2024]).values
    thfa_platt = _fit_platt(
        thfa_elo.loc[is_train_thfa, "elo_prob"].values,
        thfa_elo.loc[is_train_thfa, TARGET_COLUMN].astype(int).values,
    )
    hold_platt_thfa = thfa_platt.predict_proba(hold_prob_thfa.reshape(-1, 1))[:, 1]
    thfa_hold_platt = compute_classification_metrics(hold_y, hold_platt_thfa)
    print(f"    raw Elo holdout: {thfa_hold_raw['log_loss']:.4f}")
    print(f"    + Platt holdout: {thfa_hold_platt['log_loss']:.4f}")

    # ── Write report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    glob = all_results[0]
    team = all_results[1]

    with open(rp, "w") as f:
        f.write("# Team-Specific HFA Experiment\n\n")
        f.write("*Testing per-team home field advantages vs global HFA.*\n\n")

        f.write("## Method\n\n")
        f.write("For each team, compute home margin advantage over away margin:\n")
        f.write("```\n")
        f.write("HFA_offset = mean(home_margin) - mean(away_margin)\n")
        f.write("→ scaled to Elo units (1 pt ≈ 25 Elo, capped at ±30)\n")
        f.write("effective_HFA = global_HFA + team_HFA_offset\n")
        f.write("```\n\n")
        f.write(f"Global HFA: {BEST_HFA}\n\n")

        f.write("## Rolling-Origin Validation\n\n")
        f.write("| Model | Avg Raw LL | Avg Platt LL |\n")
        f.write("|-------|-----------|-------------|\n")
        f.write(f"| Global HFA | {glob['avg_raw_ll']:.4f} | {glob['avg_platt_ll']:.4f} |\n")
        f.write(f"| Team HFA | {team['avg_raw_ll']:.4f} | {team['avg_platt_ll']:.4f} |\n\n")

        f.write("## 2025 Holdout\n\n")
        f.write("| Model | Raw LL | Brier | Acc | AUC |\n")
        f.write("|-------|--------|-------|-----|-----|\n")
        f.write(
            f"| Global HFA raw | {inc_hold_raw['log_loss']:.4f}"
            f" | {inc_hold_raw['brier_score']:.4f}"
            f" | {inc_hold_raw['accuracy']:.4f}"
            f" | {inc_hold_raw['roc_auc']:.4f} |\n"
        )
        f.write(
            f"| Global HFA + Platt | {inc_hold_platt['log_loss']:.4f}"
            f" | {inc_hold_platt['brier_score']:.4f}"
            f" | {inc_hold_platt['accuracy']:.4f}"
            f" | {inc_hold_platt['roc_auc']:.4f} |\n"
        )
        f.write(
            f"| Team HFA raw | {thfa_hold_raw['log_loss']:.4f}"
            f" | {thfa_hold_raw['brier_score']:.4f}"
            f" | {thfa_hold_raw['accuracy']:.4f}"
            f" | {thfa_hold_raw['roc_auc']:.4f} |\n"
        )
        f.write(
            f"| Team HFA + Platt | {thfa_hold_platt['log_loss']:.4f}"
            f" | {thfa_hold_platt['brier_score']:.4f}"
            f" | {thfa_hold_platt['accuracy']:.4f}"
            f" | {thfa_hold_platt['roc_auc']:.4f} |\n\n"
        )

        if thfa_hold_platt["log_loss"] < inc_hold_platt["log_loss"]:
            f.write("## Decision\n\n")
            f.write("✅ **Team-specific HFA beats the incumbent.**\n\n")
            f.write(
                f"Holdout log loss {thfa_hold_platt['log_loss']:.4f}"
                f" vs incumbent {inc_hold_platt['log_loss']:.4f}.\n\n"
            )
        else:
            f.write("## Decision\n\n")
            f.write("❌ **Team-specific HFA does not beat the incumbent.**\n\n")
            f.write(f"Team HFA + Platt holdout: {thfa_hold_platt['log_loss']:.4f}\n")
            f.write(f"Global HFA (incumbent) + Platt holdout: {inc_hold_platt['log_loss']:.4f}\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write("- Per-team HFA computed from training seasons only.\n")
        f.write("- No validation or holdout data used in HFA estimation.\n")
        f.write("- Rolling-origin folds enforce temporal split.\n\n")

        f.write("## Sample Team HFA Values\n\n")
        f.write("| Team | Margin Adv | Elo Offset |\n")
        f.write("|------|-----------|-----------|\n")
        for team in sorted(team_hfa_all, key=team_hfa_all.get, reverse=True)[:10]:
            margin_adv = team_all.get(team, 0)
            f.write(f"| {team} | {margin_adv:.2f} | {team_hfa_all[team]:.1f} |\n")
        f.write("\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
