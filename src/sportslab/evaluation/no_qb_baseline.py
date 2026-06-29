"""No-QB live-safe baseline comparison experiment.

Compares the full incumbent (Elo + qb_changed + rolling_mov_3 + Platt)
against a no-QB variant (Elo + rolling_mov_3 + Platt) using the
2025 week-by-week simulation.

The no-QB variant represents what the model would predict with only
pregame-safe features, without relying on oracle QB data (starter
identity, QB-change flags) that requires final actual starter info.

Usage:
    sportslab no-qb-baseline
    make no-qb-baseline
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.predict_incumbent import (
    BEST_DECAY,
    BEST_HFA,
    BEST_K,
    BEST_QB_BONUS,
    BEST_REG,
    FEATURE_COLS,
    INCUMBENT_DATE,
    INCUMBENT_FEATURE_SET,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VERSION,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

SIMULATION_SEASON = 2025
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
TRAIN_SEASONS = [2021, 2022, 2023, 2024]
DEFAULT_REPORT = "reports/experiments/no_qb_live_baseline.md"

# Non-QB feature set: only rolling_mov_3 (no qb_changed)
NO_QB_FEATURE_COLS = [c for c in FEATURE_COLS
                       if "qb" not in c and "rolling_mov" in c]


def _build_no_qb_feature_set(df: pd.DataFrame) -> np.ndarray:
    """Build feature matrix using only Elo prob + non-QB features."""
    elo = df["elo_prob"].values
    feat_cols = [c for c in NO_QB_FEATURE_COLS if c in df.columns]
    if feat_cols:
        feat = df[feat_cols].values
        return np.column_stack([elo, feat])
    return elo.reshape(-1, 1)


def _build_incumbent_feature_set(df: pd.DataFrame) -> np.ndarray:
    """Build feature matrix using Elo prob + all incumbent features."""
    elo = df["elo_prob"].values
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    if feat_cols:
        feat = df[feat_cols].values
        return np.column_stack([elo, feat])
    return elo.reshape(-1, 1)


def _fit_platt(x: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, solver="lbfgs")),
    ])
    pipe.fit(x, y)
    return pipe


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    valid = ~np.isnan(y_true)
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]
    if len(y_true) == 0:
        return {}
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    ll = float(sk_log_loss(y_true, y_prob))
    brier = float(np.mean((y_true - y_prob) ** 2))
    acc = float(np.mean((y_prob >= 0.5) == y_true))
    return {"log_loss": round(ll, 4), "brier": round(brier, 4), "accuracy": round(acc, 4)}


def _run_simulation(
    df_raw: pd.DataFrame,
    weeks: list[int],
    use_qb_features: bool,
) -> Dict:
    """Run week-by-week simulation with or without QB features.

    Args:
        df_raw: Full feature table.
        weeks: 2025 weeks to simulate.
        use_qb_features: If True, uses full incumbent (with qb_changed).
                         If False, uses only rolling_mov_3 (no QB data).

    Returns:
        Dict with overall metrics and per-week metrics.
    """
    label = "incumbent" if use_qb_features else "no_qb"
    all_preds = []
    week_metrics = []

    for week in weeks:
        train_mask = (
            (df_raw["season"].isin(TRAIN_SEASONS)) |
            ((df_raw["season"] == SIMULATION_SEASON) & (df_raw["week"] < week))
        ) & df_raw[MODEL_ELIGIBLE_COLUMN]

        pred_mask = (
            (df_raw["season"] == SIMULATION_SEASON)
            & (df_raw["week"] == week)
            & df_raw[MODEL_ELIGIBLE_COLUMN]
            & (~df_raw.get("is_neutral", pd.Series(False)).fillna(False))
        )

        if pred_mask.sum() == 0:
            continue

        df_train = df_raw[train_mask].copy()
        df_pred = df_raw[pred_mask].copy()
        pred_game_ids = set(df_pred["game_id"].values)
        pred_actuals = df_pred.set_index("game_id")["home_win"].copy()
        df_pred["home_win"] = np.nan
        df_pred["home_win"] = df_pred["home_win"].astype(float)
        df_combined = pd.concat([df_train, df_pred], ignore_index=True)
        df_combined = df_combined.sort_values(
            ["season", "week", "gameday"]
        ).reset_index(drop=True)
        df_combined["_is_pred"] = df_combined["game_id"].isin(pred_game_ids)

        overrides = build_team_regression_overrides(
            df_combined, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
        )
        df_feat = compute_elo_features(
            df_combined,
            k_factor=BEST_K, home_advantage=BEST_HFA,
            preseason_regression=BEST_REG,
            team_regression_overrides=overrides,
            decay_half_life=BEST_DECAY,
        )
        if use_qb_features:
            df_feat = compute_qb_features(df_feat)
        df_feat = compute_situational_features(df_feat)

        is_pred = df_feat["_is_pred"].values
        is_train = ~is_pred

        train_y = df_feat.loc[is_train, "home_win"].astype(int).values
        if use_qb_features:
            x_train = _build_incumbent_feature_set(df_feat.loc[is_train])
            x_pred = _build_incumbent_feature_set(df_feat.loc[is_pred])
        else:
            x_train = _build_no_qb_feature_set(df_feat.loc[is_train])
            x_pred = _build_no_qb_feature_set(df_feat.loc[is_pred])

        pipe = _fit_platt(x_train, train_y)
        prob = pipe.predict_proba(x_pred)[:, 1]

        pred_ids = df_feat.loc[is_pred, "game_id"].values
        actual = np.array([
            pred_actuals.get(gid, np.nan)
            if pd.notna(pred_actuals.get(gid, np.nan)) else np.nan
            for gid in pred_ids
        ], dtype=float)

        df_week = pd.DataFrame({
            "game_id": df_feat.loc[is_pred, "game_id"].values,
            "season": SIMULATION_SEASON,
            "week": week,
            "away_team": df_feat.loc[is_pred, "away_team"].values,
            "home_team": df_feat.loc[is_pred, "home_team"].values,
            "home_win_actual": actual,
            f"{label}_prob": prob.round(4),
        })

        metrics = _compute_metrics(actual, prob)
        week_metrics.append({"week": week, **metrics})
        all_preds.append(df_week)

    if not all_preds:
        return {"overall": {}, "weekly": []}

    df_all = pd.concat(all_preds, ignore_index=True)
    overall_prob = df_all[f"{label}_prob"].values
    actual = df_all["home_win_actual"].astype(int).values
    overall = _compute_metrics(actual, overall_prob)

    return {"overall": overall, "weekly": week_metrics, "num_games": len(df_all)}


def run_no_qb_baseline() -> Dict[str, str]:
    """Run no-QB vs incumbent comparison on 2025 simulation."""
    print("=== No-QB Live-Safe Baseline Comparison ===")
    fp = Path(FEATURE_TABLE_PATH)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")
    df_raw = pd.read_parquet(fp)

    season_mask = (
        (df_raw["season"] == SIMULATION_SEASON)
        & df_raw.get(MODEL_ELIGIBLE_COLUMN, pd.Series(True))
    )
    weeks = sorted(df_raw[season_mask]["week"].unique())
    total_season_games = int(season_mask.sum())
    w0, w1 = weeks[0], weeks[-1]
    print(f"  2025 eligible games: {total_season_games} across "
          f"{len(weeks)} weeks ({w0}–{w1})")

    print("  Running incumbent simulation (with QB features)...")
    inc_result = _run_simulation(df_raw, weeks, use_qb_features=True)
    print("  Running no-QB simulation (live-safe features only)...")
    no_qb_result = _run_simulation(df_raw, weeks, use_qb_features=False)

    overall_inc = inc_result["overall"]
    overall_noqb = no_qb_result["overall"]

    # Generate report
    report_path = Path(DEFAULT_REPORT)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# No-QB Live-Safe Baseline Comparison\n\n")
        f.write(
            f"*Generated by `sportslab no-qb-baseline`"
            f" ({INCUMBENT_VERSION}, {INCUMBENT_DATE})*\n\n"
        )

        f.write("## Overview\n\n")
        f.write(
            "Compares the full incumbent (with QB-change features from oracle data)\n"
            "against a no-QB variant using only pregame-safe features\n"
            "(Elo prob + rolling_mov_3). The no-QB variant is the model's\n"
            "performance without any starter identity or QB-change signal.\n\n"
        )

        f.write("## Models Compared\n\n")
        f.write("| Model | Features | QB Dependency |\n")
        f.write("|-------|----------|---------------|\n")
        f.write(
            f"| Incumbent | {INCUMBENT_FEATURE_SET} | Oracle qb_changed |\n"
        )
        f.write(
            "| No-QB Live-Safe | rolling_mov_3 only | None (pregame-safe) |\n\n"
        )

        f.write("## Overall 2025 Simulation\n\n")
        f.write("| Metric | Incumbent | No-QB Live-Safe | Δ |\n")
        f.write("|--------|-----------|-----------------|---|\n")

        inc_games = inc_result.get("num_games", 0)
        noqb_games = no_qb_result.get("num_games", 0)
        metrics_order = ["log_loss", "brier", "accuracy"]
        for key in metrics_order:
            inc_val = overall_inc.get(key, "N/A")
            noqb_val = overall_noqb.get(key, "N/A")
            if isinstance(inc_val, float) and isinstance(noqb_val, float):
                delta = f"{noqb_val - inc_val:+.4f}"
            else:
                delta = "—"
            f.write(f"| {key} | {inc_val} | {noqb_val} | {delta} |\n")
        f.write(f"| games | {inc_games} | {noqb_games} | — |\n\n")

        f.write("## Weekly Comparison\n\n")
        f.write("| Week | Incumbent LL | No-QB LL | Δ |\n")
        f.write("|------|-------------|----------|---|\n")

        week_map = {m["week"]: m for m in inc_result["weekly"]}
        for wm in no_qb_result["weekly"]:
            wk = wm["week"]
            inc_w = week_map.get(wk, {})
            inc_ll = inc_w.get("log_loss", "N/A")
            noqb_ll = wm.get("log_loss", "N/A")
            if isinstance(inc_ll, float) and isinstance(noqb_ll, float):
                delta = f"{noqb_ll - inc_ll:+.4f}"
            else:
                delta = "—"
            f.write(f"| {wk} | {inc_ll} | {noqb_ll} | {delta} |\n")
        f.write("\n")

        f.write("## Conclusion\n\n")
        if overall_inc.get("log_loss", 1) < overall_noqb.get("log_loss", 0):
            diff = overall_noqb["log_loss"] - overall_inc["log_loss"]
            f.write(
                f"The incumbent (with QB features) outperforms the no-QB baseline"
                f" by **{diff:.4f}** log loss ({overall_inc['log_loss']:.4f} vs"
                f" {overall_noqb['log_loss']:.4f}).\n\n"
            )
            f.write(
                "The qb_changed feature adds measurable predictive value, but depends"
                " on oracle starter data. For live-pregame prediction, the no-QB"
                " baseline is the safer choice when starter info is uncertain.\n\n"
            )
        else:
            diff = overall_inc["log_loss"] - overall_noqb["log_loss"]
            f.write(
                f"The no-QB baseline matches or beats the incumbent"
                f" (Δ {diff:+.4f}).\n\n"
            )

        f.write("---\n")
        f.write(
            f"*Report generated by `sportslab no-qb-baseline`."
            f" Incumbent: {INCUMBENT_VERSION}, {INCUMBENT_HOLDOUT_LL} holdout LL.*\n"
        )

    inc_games = inc_result.get("num_games", 0)
    noqb_games = no_qb_result.get("num_games", 0)
    inc_ll = overall_inc.get("log_loss", "N/A")
    noqb_ll = overall_noqb.get("log_loss", "N/A")
    print(f"\nNo-QB baseline comparison: {report_path}")
    print(f"  Incumbent LL:     {inc_ll} ({inc_games} games)")
    print(f"  No-QB LL:         {noqb_ll} ({noqb_games} games)")

    return {"report": str(report_path)}
