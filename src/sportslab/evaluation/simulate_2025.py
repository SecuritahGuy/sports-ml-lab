"""Historical as-if-future simulation for 2025.

Iterates week-by-week through the 2025 season, fitting Elo on all
available data up to each week, then predicting that week's games
using the incumbent model (Elo + qb_changed + rolling_mov_3 + Platt).

Two modes:
  - oracle: Uses final actual QB starter data (backtest-safe).
  - live_pregame: Overrides QB data with user-supplied CSV.

Produces a detailed CSV of per-game predictions and a metrics summary.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss

from sportslab.evaluation.predict_incumbent import (
    FEATURE_COLS,
    _assign_confidence_bucket,
    _build_pipeline,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_input import apply_qb_input, parse_qb_input_csv
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

SIMULATION_SEASON = 2025
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
DEFAULT_OUTPUT = "reports/simulations/simulate_2025_results.csv"
DEFAULT_REPORT = "reports/simulations/simulate_2025_report.md"
TRAIN_SEASONS = [2021, 2022, 2023, 2024]

QB_MODE_ORACLE = "oracle"
QB_MODE_LIVE = "live_pregame"


def _load_feature_table() -> pd.DataFrame:
    fp = Path(FEATURE_TABLE_PATH)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")
    return pd.read_parquet(fp)


def _get_2025_weeks(df: pd.DataFrame) -> list[int]:
    games = df[(df["season"] == SIMULATION_SEASON) & df[MODEL_ELIGIBLE_COLUMN]]
    return sorted(games["week"].unique())


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    valid = ~np.isnan(y_true)
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]
    if len(y_true) == 0:
        return {}
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    ll = sk_log_loss(y_true, y_prob)
    brier = float(np.mean((y_true - y_prob) ** 2))
    acc = float(np.mean((y_prob >= 0.5) == y_true))
    return {"log_loss": round(ll, 4), "brier": round(brier, 4), "accuracy": round(acc, 4)}


def simulate_2025(
    qb_input_path: Optional[str] = None,
    output_path: str = DEFAULT_OUTPUT,
    report_path: str = DEFAULT_REPORT,
) -> Dict[str, str]:
    """Run week-by-week 2025 simulation.

    Args:
        qb_input_path: Optional CSV with game_id,home_qb_id,away_qb_id
            for live-safe QB overrides.
        output_path: Where to save the per-game prediction CSV.
        report_path: Where to save the Markdown summary report.

    Returns:
        Dict with paths to output files.
    """
    qb_source = QB_MODE_LIVE if qb_input_path else QB_MODE_ORACLE
    print(f"=== 2025 Simulation (QB source: {qb_source}) ===")
    df_raw = _load_feature_table()
    weeks = _get_2025_weeks(df_raw)
    print(f"  2025 weeks to simulate: {len(weeks)} ({weeks[0]}–{weeks[-1]})")

    # Pre-load QB input if provided
    qb_input_df = None
    if qb_input_path:
        qb_input_df = parse_qb_input_csv(qb_input_path)

    all_preds: list[pd.DataFrame] = []
    week_metrics: list[dict] = []

    for week in weeks:
        # Build training+prediction data: training games before this week
        train_mask = (
            (df_raw["season"].isin(TRAIN_SEASONS)) |
            ((df_raw["season"] == SIMULATION_SEASON) & (df_raw["week"] < week))
        ) & df_raw[MODEL_ELIGIBLE_COLUMN]

        # Build prediction data: this week's games
        pred_mask = (
            (df_raw["season"] == SIMULATION_SEASON)
            & (df_raw["week"] == week)
            & df_raw[MODEL_ELIGIBLE_COLUMN]
            & (~df_raw.get("is_neutral", pd.Series(False)).fillna(False))
        )

        if pred_mask.sum() == 0:
            print(f"  Week {week}: no eligible games — skipping")
            continue

        # Combine training + prediction rows; prediction rows have future home_win
        df_train = df_raw[train_mask].copy()
        df_pred = df_raw[pred_mask].copy()
        pred_game_ids = set(df_pred["game_id"].values)
        # Save actual results before clearing home_win
        pred_actuals = df_pred.set_index("game_id")["home_win"].copy()
        # Clear home_win on prediction rows so they are treated as future
        df_pred["home_win"] = pd.NA
        df_combined = pd.concat([df_train, df_pred], ignore_index=True)
        df_combined = df_combined.sort_values(
            ["season", "week", "gameday"]
        ).reset_index(drop=True)

        # Mark prediction rows in the combined DataFrame
        df_combined["_is_pred"] = df_combined["game_id"].isin(pred_game_ids)

        # Apply QB input override
        if qb_input_df is not None:
            df_combined = apply_qb_input(df_combined, qb_input_df)

        # Build features
        overrides = build_team_regression_overrides(
            df_combined, preseason_regression=0.1, qb_change_bonus=0.2,
        )
        df_feat = compute_elo_features(
            df_combined,
            k_factor=36, home_advantage=40,
            preseason_regression=0.1,
            team_regression_overrides=overrides,
            decay_half_life=32,
        )
        df_feat = compute_qb_features(df_feat)
        df_feat = compute_situational_features(df_feat)

        # Fit Platt on training rows only
        is_pred = df_feat["_is_pred"].values
        is_train = ~is_pred

        feat_cols = [c for c in FEATURE_COLS if c in df_feat.columns]
        train_elo = df_feat.loc[is_train, "elo_prob"].values
        n_train = is_train.sum()
        train_feat = (
            df_feat.loc[is_train, feat_cols].values
            if feat_cols
            else np.empty((n_train, 0))
        )
        train_y = df_feat.loc[is_train, "home_win"].astype(int).values
        x_train = (
            np.column_stack([train_elo, train_feat])
            if train_feat.size
            else train_elo.reshape(-1, 1)
        )

        pipe = _build_pipeline()
        pipe.fit(x_train, train_y)

        # Predict
        pred_elo = df_feat.loc[is_pred, "elo_prob"].values
        n_pred = is_pred.sum()
        pred_feat = (
            df_feat.loc[is_pred, feat_cols].values
            if feat_cols
            else np.empty((n_pred, 0))
        )
        x_pred = (
            np.column_stack([pred_elo, pred_feat])
            if pred_feat.size
            else pred_elo.reshape(-1, 1)
        )
        prob = pipe.predict_proba(x_pred)[:, 1]

        # Actual result for metric computation
        pred_ids = df_feat.loc[is_pred, "game_id"].values
        actual = np.array([
            pred_actuals.get(gid, np.nan) if pd.notna(pred_actuals.get(gid, np.nan)) else np.nan
            for gid in pred_ids
        ], dtype=float)

        # Build results
        df_week = pd.DataFrame({
            "game_id": df_feat.loc[is_pred, "game_id"].values,
            "season": SIMULATION_SEASON,
            "week": week,
            "gameday": df_feat.loc[is_pred, "gameday"].values,
            "away_team": df_feat.loc[is_pred, "away_team"].values,
            "home_team": df_feat.loc[is_pred, "home_team"].values,
            "home_score": pd.to_numeric(df_feat.loc[is_pred, "home_score"], errors="coerce").values,
            "away_score": pd.to_numeric(df_feat.loc[is_pred, "away_score"], errors="coerce").values,
            "home_win_actual": actual,
            "incumbent_home_win_prob": prob.round(4),
            "predicted_winner": np.where(prob >= 0.5,
                                         df_feat.loc[is_pred, "home_team"].values,
                                         df_feat.loc[is_pred, "away_team"].values),
            "confidence_bucket": [_assign_confidence_bucket(p) for p in prob],
            "qb_source": qb_source,
        })

        metrics = _compute_metrics(actual, prob)
        week_metrics.append({"week": week, "n_games": len(actual), **metrics})
        print(f"  Week {week}: {len(df_week)} games, LL={metrics.get('log_loss', 'N/A')}")
        all_preds.append(df_week)

    if not all_preds:
        print("  No games predicted — nothing to report.")
        return {}

    df_all = pd.concat(all_preds, ignore_index=True)

    # Clean up temporary marker columns
    for c in ["_is_pred"]:
        if c in df_all.columns:
            df_all.drop(columns=[c], inplace=True)

    # Save per-game CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)

    # Generate report
    report_path_obj = Path(report_path)
    report_path_obj.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path_obj, df_all, week_metrics, qb_source)
    print(f"\nSimulation report: {report_path_obj}")

    return {"predictions": str(out_path), "report": str(report_path_obj)}


def _write_report(
    path: Path,
    df_all: pd.DataFrame,
    week_metrics: list[dict],
    qb_source: str,
) -> None:
    overall_prob = df_all["incumbent_home_win_prob"].values
    actual = df_all["home_win_actual"].astype(int).values
    overall = _compute_metrics(actual, overall_prob)

    with open(path, "w") as f:
        f.write("# 2025 Simulation Report\n\n")
        f.write(f"*QB source: {qb_source}*\n\n")
        f.write("## Overall\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for k, v in overall.items():
            f.write(f"| {k} | {v} |\n")
        f.write(f"| games | {len(df_all)} |\n")
        f.write("\n")

        f.write("## By Week\n\n")
        f.write("| Week | Games | Log Loss | Brier | Accuracy |\n")
        f.write("|------|-------|----------|-------|----------|\n")
        for m in week_metrics:
            f.write(
                f"| {m['week']} | {m.get('n_games', '')}"
                f" | {m.get('log_loss', 'N/A')}"
                f" | {m.get('brier', 'N/A')}"
                f" | {m.get('accuracy', 'N/A')} |\n"
            )
        f.write("\n")

        f.write("## Per-Game Predictions\n\n")
        f.write("| Week | Away | Home | Score | Prob | Predicted | Actual |\n")
        f.write("|------|------|------|-------|------|-----------|--------|\n")
        for _, row in df_all.iterrows():
            actual_str = "H" if row["home_win_actual"] == 1 else "A"
            f.write(
                f"| {row['week']} | {row['away_team']}"
                f" | {row['home_team']}"
                f" | {row['away_score']}–{row['home_score']}"
                f" | {row['incumbent_home_win_prob']:.3f}"
                f" | {row['predicted_winner']} | {actual_str} |\n"
            )
