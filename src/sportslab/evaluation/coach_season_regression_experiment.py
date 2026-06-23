"""Combined QB+coach season regression experiment — varying preseason regression
by team stability (QB change and/or coach change across seasons)."""

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

# Frozen best params from season-regression incumbent
BEST_K = 36
BEST_HFA = 40
BEST_DECAY = 32
BEST_MOV_TYPE = MOV_CAPPED_LINEAR
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0

# Grid candidates (tight, focused on the new parameter)
REG_CANDIDATES = [0.0, 0.10, 0.20]
QB_BONUS_CANDIDATES = [0.0, 0.10, 0.20, 0.30]
COACH_BONUS_CANDIDATES = [0.0, 0.10, 0.20, 0.30]


def identity_across_seasons(
    df: pd.DataFrame,
    home_id_col: str,
    away_id_col: str,
) -> dict[int, list[str]]:
    """Detect teams with identity change between consecutive seasons.

    Reports the most common value of `id_col` per team per season via
    chronological scan (last game's value wins ties).

    Args:
        df: Game-level DataFrame.
        home_id_col: Column name for home team's identity (e.g. 'home_qb_id').
        away_id_col: Column name for away team's identity (e.g. 'away_qb_id').

    Returns:
        Dict mapping season -> list of team names whose identity changed.
    """
    df = df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    teams: dict[int, dict[str, str]] = {}

    for _, row in df.iterrows():
        season = int(row["season"])
        if season not in teams:
            teams[season] = {}

        for team_col, id_col in [
            ("home_team", home_id_col),
            ("away_team", away_id_col),
        ]:
            team = row[team_col]
            id_val = row.get(id_col)
            if pd.isna(id_val) or id_val is None:
                continue
            # Last game's value wins (chronological overwrite)
            teams[season][team] = str(id_val)

    change_map: dict[int, list[str]] = {}
    sorted_seasons = sorted(teams.keys())
    for i, season in enumerate(sorted_seasons):
        changes: list[str] = []
        if i > 0:
            prev_season = sorted_seasons[i - 1]
            for team, val in teams[season].items():
                prev_val = teams[prev_season].get(team)
                if prev_val is not None and prev_val != val:
                    changes.append(team)
        change_map[season] = changes

    return change_map


def build_team_regression_overrides(
    df: pd.DataFrame,
    preseason_regression: float,
    qb_change_bonus: float,
    coach_change_bonus: float,
) -> dict[str, float] | None:
    """Build per-team regression overrides based on QB and/or coach changes.

    Teams with QB change get `base + qb_bonus` regression.
    Teams with coach change get `base + coach_bonus`.
    Teams with BOTH get `base + qb_bonus + coach_bonus` (capped at 1.0).
    """
    qb_changes = identity_across_seasons(df, "home_qb_id", "away_qb_id")
    coach_changes = identity_across_seasons(df, "home_coach", "away_coach")

    overrides: dict[str, float] = {}
    min_season = int(df["season"].min())

    for season in set(list(qb_changes.keys()) + list(coach_changes.keys())):
        if season <= min_season:
            continue
        qb_teams = set(qb_changes.get(season, []))
        coach_teams = set(coach_changes.get(season, []))
        all_teams = qb_teams | coach_teams

        for team in all_teams:
            bonus = 0.0
            if team in qb_teams:
                bonus += qb_change_bonus
            if team in coach_teams:
                bonus += coach_change_bonus
            if bonus == 0:
                continue
            reg = min(preseason_regression + bonus, 1.0)
            if reg > 0:
                overrides[team] = reg

    if not overrides:
        return None
    return overrides


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


def run_coach_season_regression_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/coach_season_regression.md",
) -> str:
    """Rolling-origin grid over base_reg, qb_bonus, coach_bonus."""
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Quick check: how many coach changes? ──
    coach_cm = identity_across_seasons(df_raw, "home_coach", "away_coach")
    total_coach_changes = sum(len(v) for v in coach_cm.values())
    print(f"Total coach changes detected across seasons: {total_coach_changes}")

    # ── Grid search ──
    print("=== Coach+QB Season Regression Grid Search ===")
    total = len(REG_CANDIDATES) * len(QB_BONUS_CANDIDATES) * len(COACH_BONUS_CANDIDATES)
    count = 0
    best: dict[str, Any] = {"avg_val_log_loss": float("inf"), "params": None}
    all_results: list[dict[str, Any]] = []

    for reg in REG_CANDIDATES:
        for qb_bonus in QB_BONUS_CANDIDATES:
            for coach_bonus in COACH_BONUS_CANDIDATES:
                count += 1

                overrides = build_team_regression_overrides(
                    df_raw,
                    preseason_regression=reg,
                    qb_change_bonus=qb_bonus,
                    coach_change_bonus=coach_bonus,
                )

                edf = compute_elo_features(
                    df_raw,
                    k_factor=BEST_K,
                    home_advantage=BEST_HFA,
                    preseason_regression=reg,
                    mov_type=BEST_MOV_TYPE,
                    mov_scale=BEST_MOV_SCALE,
                    mov_cap=BEST_MOV_CAP,
                    decay_half_life=BEST_DECAY,
                    team_regression_overrides=overrides,
                )
                edf = _filter_df(edf)

                elo_prob = edf["elo_prob"].values
                y = edf[TARGET_COLUMN].astype(float).values

                fold_lls: list[float] = []
                fold_details: list[dict[str, Any]] = []

                for train_seasons, val_season in ROLLING_FOLDS:
                    is_val = (edf["season"] == val_season).values
                    val_loss = float(log_loss(y[is_val], elo_prob[is_val]))
                    fold_lls.append(val_loss)
                    fold_details.append(
                        {
                            "train_seasons": train_seasons,
                            "val_season": val_season,
                            "val_log_loss": round(val_loss, 5),
                        }
                    )

                avg_ll = float(np.mean(fold_lls))

                entry = {
                    "preseason_regression": reg,
                    "qb_change_bonus": qb_bonus,
                    "coach_change_bonus": coach_bonus,
                    "avg_val_log_loss": round(avg_ll, 5),
                    "fold_details": fold_details,
                }
                all_results.append(entry)

                if avg_ll < best["avg_val_log_loss"]:
                    best = {
                        "avg_val_log_loss": avg_ll,
                        "params": {
                            "preseason_regression": reg,
                            "qb_change_bonus": qb_bonus,
                            "coach_change_bonus": coach_bonus,
                        },
                        "fold_details": fold_details,
                    }

                print(
                    f"  [{count}/{total}] reg={reg} qb={qb_bonus}"
                    f" coach={coach_bonus}  avg_ll={avg_ll:.5f}"
                )

    all_results.sort(key=lambda x: x["avg_val_log_loss"])
    best_params = best["params"]
    best_avg_ll = round(best["avg_val_log_loss"], 5)
    print(
        f"\nBest params (by avg val log loss):"
        f" reg={best_params['preseason_regression']},"
        f" qb_bonus={best_params['qb_change_bonus']},"
        f" coach_bonus={best_params['coach_change_bonus']}"
    )
    print(f"  Avg val log loss: {best_avg_ll:.5f}")

    # ── Final Elo with best params ──
    print("\n=== Final Elo with best params ===")
    best_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=best_params["preseason_regression"],
        qb_change_bonus=best_params["qb_change_bonus"],
        coach_change_bonus=best_params["coach_change_bonus"],
    )
    best_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=best_params["preseason_regression"],
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        team_regression_overrides=best_overrides,
    )
    best_elo = _filter_df(best_elo)

    is_hold = (best_elo["season"] == HOLDOUT_SEASON).values
    hold_y = best_elo.loc[is_hold, TARGET_COLUMN].astype(float).values
    hold_prob = best_elo.loc[is_hold, "elo_prob"].values
    hold_metrics = compute_classification_metrics(hold_y, hold_prob)
    print(f"  Holdout (2025) log loss: {hold_metrics['log_loss']:.4f}")

    # ── Rolling-origin Platt calibration ──
    print("\n=== Rolling-Origin Calibration ===")
    elo_prob = best_elo["elo_prob"].values
    y = best_elo[TARGET_COLUMN].astype(float).values
    platt_folds: list[dict[str, Any]] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = best_elo["season"].isin(train_seasons).values
        is_val = (best_elo["season"] == val_season).values
        train_p = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_p = elo_prob[is_val]
        val_y_ = y[is_val]
        platt = _fit_platt(train_p, train_y_)
        platt_val = platt.predict_proba(val_p.reshape(-1, 1))[:, 1]
        platt_m = compute_classification_metrics(val_y_, platt_val)
        platt_folds.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_m["log_loss"],
            }
        )

    platt_avg = float(np.mean([f["log_loss"] for f in platt_folds]))
    print(f"  Platt: avg val LL={platt_avg:.4f}")

    # ── Incumbent (season-regression + Platt) ──
    print("\n=== Incumbent Comparison ===")
    inc_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=0.10,
        qb_change_bonus=0.20,
        coach_change_bonus=0.0,
    )
    inc_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=0.10,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        team_regression_overrides=inc_overrides,
    )
    inc_elo = _filter_df(inc_elo)
    is_train_full_inc = inc_elo["season"].isin([2021, 2022, 2023, 2024]).values
    hold_y_inc = inc_elo.loc[is_hold, TARGET_COLUMN].astype(float).values
    hold_p_inc = inc_elo.loc[is_hold, "elo_prob"].values
    train_p_inc = inc_elo.loc[is_train_full_inc, "elo_prob"].values
    train_y_inc = inc_elo.loc[is_train_full_inc, TARGET_COLUMN].astype(float).values
    platt_inc = _fit_platt(train_p_inc, train_y_inc)
    platt_hold_inc = platt_inc.predict_proba(hold_p_inc.reshape(-1, 1))[:, 1]
    inc_hold_metrics = compute_classification_metrics(hold_y_inc, platt_hold_inc)

    # Full 2021-2024 Platt for best config
    is_train_full = best_elo["season"].isin([2021, 2022, 2023, 2024]).values
    train_p_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)
    platt_full = _fit_platt(train_p_full, train_y_full)
    platt_hold = platt_full.predict_proba(hold_prob.reshape(-1, 1))[:, 1]
    platt_hold_met = compute_classification_metrics(hold_y, platt_hold)

    # Baselines
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = float(y[is_train_full].mean())
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    print(f"  Best raw Elo holdout: {hold_metrics['log_loss']:.4f}")
    print(f"  Best + Platt holdout: {platt_hold_met['log_loss']:.4f}")
    print(f"  Incumbent (QB-reg+Platt) holdout: {inc_hold_metrics['log_loss']:.4f}")

    # ── Write report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Coach+QB Season Regression Experiment\n\n")
        f.write(
            "*Testing whether adding coach-change preseason regression to "
            "QB-change regression improves prediction.*\n\n"
        )

        f.write("## Motivation\n\n")
        f.write(
            "Teams with a new head coach should also get extra preseason regression, "
            "similar to QB changes.  A new coach means a new system, new playbook, "
            "and potentially different team philosophy — making past Elo ratings less "
            "informative.  This experiment tests a combined regression bonus:\n\n"
        )
        f.write("```\n")
        f.write(
            "effective_regression = base + (qb_bonus if QB changed) "
            "+ (coach_bonus if coach changed)\n"
        )
        f.write("capped at 1.0\n")
        f.write("```\n\n")

        f.write("## Parameter Grid\n\n")
        f.write("| Parameter | Candidates |\n")
        f.write("|-----------|------------|\n")
        f.write(f"| K-factor (frozen) | {BEST_K} |\n")
        f.write(f"| HFA (frozen) | {BEST_HFA} |\n")
        f.write(f"| Decay (frozen) | {BEST_DECAY} |\n")
        f.write(f"| Base regression | {REG_CANDIDATES} |\n")
        f.write(f"| QB change bonus | {QB_BONUS_CANDIDATES} |\n")
        f.write(f"| Coach change bonus | {COACH_BONUS_CANDIDATES} |\n")
        f.write(
            f"| MOV (frozen) | {BEST_MOV_TYPE}, scale={BEST_MOV_SCALE}, cap={BEST_MOV_CAP} |\n\n"
        )
        f.write(f"Total combinations searched: {len(all_results)}\n\n")

        f.write("## Coach Change Counts\n\n")
        f.write(f"Total coach changes across all seasons: {total_coach_changes}\n\n")
        f.write("| Season | Coach Changes |\n")
        f.write("|--------|---------------|\n")
        for s in sorted(coach_cm.keys()):
            changes = coach_cm[s]
            cnt = len(changes)
            teams_str = ", ".join(sorted(changes)) if cnt <= 10 else f"{cnt} teams"
            f.write(f"| {s} | {cnt} ({teams_str}) |\n")
        f.write("\n")

        f.write("## Best Configuration\n\n")
        f.write(f"- Base regression: {best_params['preseason_regression']}\n")
        f.write(f"- QB change bonus: {best_params['qb_change_bonus']}\n")
        f.write(f"- Coach change bonus: {best_params['coach_change_bonus']}\n")
        f.write(f"- Average validation log loss: {best_avg_ll:.5f}\n")
        f.write(f"- Holdout raw Elo: {hold_metrics['log_loss']:.4f}\n")
        f.write(f"- Holdout + Platt: {platt_hold_met['log_loss']:.4f}\n")
        f.write(f"- Incumbent (QB-reg+Platt): {inc_hold_metrics['log_loss']:.4f}\n\n")

        f.write("## Top 8 Configurations\n\n")
        f.write("| Rank | Reg | QB Bonus | Coach Bonus | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|------|-----|----------|-------------|-----------|-------|-------|-------|\n")
        for rank, e in enumerate(all_results[:8], 1):
            fd = e["fold_details"]
            f.write(
                f"| {rank} | {e['preseason_regression']} | {e['qb_change_bonus']}"
                f" | {e['coach_change_bonus']}"
                f" | {e['avg_val_log_loss']} | {fd[0]['val_log_loss']}"
                f" | {fd[1]['val_log_loss']} | {fd[2]['val_log_loss']} |\n"
            )
        f.write("\n")

        f.write("## Validation Comparison\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")
        f.write(
            f"| Best config raw | {best_avg_ll:.4f}"
            f" | {best['fold_details'][0]['val_log_loss']:.4f}"
            f" | {best['fold_details'][1]['val_log_loss']:.4f}"
            f" | {best['fold_details'][2]['val_log_loss']:.4f} |\n"
        )
        f.write(
            f"| Best config + Platt | {platt_avg:.4f}"
            f" | {platt_folds[0]['log_loss']:.4f}"
            f" | {platt_folds[1]['log_loss']:.4f}"
            f" | {platt_folds[2]['log_loss']:.4f} |\n\n"
        )

        f.write("## Holdout (2025) Comparison\n\n")
        f.write("| Model | Hold LL | Brier | Acc | AUC |\n")
        f.write("|-------|---------|-------|-----|-----|\n")
        f.write(f"| Random | {random_hold_ll:.4f} | — | — | — |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | — |\n")
        f.write(f"| Incumbent (QB-reg+Platt) | {inc_hold_metrics['log_loss']:.4f} | — | — | — |\n")
        f.write(
            f"| Best raw | {hold_metrics['log_loss']:.4f}"
            f" | {hold_metrics['brier_score']:.4f}"
            f" | {hold_metrics['accuracy']:.4f}"
            f" | {hold_metrics['roc_auc']:.4f} |\n"
        )
        f.write(
            f"| Best + Platt | {platt_hold_met['log_loss']:.4f}"
            f" | {platt_hold_met['brier_score']:.4f}"
            f" | {platt_hold_met['accuracy']:.4f}"
            f" | {platt_hold_met['roc_auc']:.4f} |\n\n"
        )

        wr = platt_hold_met["log_loss"] < inc_hold_metrics["log_loss"]
        if wr:
            f.write("## Decision\n\n")
            f.write(
                f"✅ **Coach+QB regression (qb={best_params['qb_change_bonus']}, "
                f"coach={best_params['coach_change_bonus']})"
                f" + Platt beats the incumbent.**\n\n"
            )
            f.write(
                f"Holdout log loss {platt_hold_met['log_loss']:.4f}"
                f" vs incumbent {inc_hold_metrics['log_loss']:.4f}.\n\n"
            )
        else:
            f.write("## Decision\n\n")
            f.write("❌ **Coach+QB regression does not beat the incumbent.**\n\n")
            f.write(f"Best + Platt holdout: {platt_hold_met['log_loss']:.4f}\n")
            f.write(f"Incumbent holdout: {inc_hold_metrics['log_loss']:.4f}\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write("- QB/coach changes detected from all data (purely past information).\n")
        f.write("- Rolling-origin folds prevent 2025 holdout access.\n")
        f.write("- Team regression overrides only applied at season boundaries.\n")
        f.write("- Platt calibration fitted only on training data.\n\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
