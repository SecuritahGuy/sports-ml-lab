"""Historical rehearsal — replays a completed season through the weekly pipeline.

Produces isolated outputs under reports/predictions/rehearsal/ with:
  - Timestamped snapshots per week
  - Rehearsal-scoped manifest and history
  - Season report
  - Prediction audit (calibration, confidence buckets, worst predictions)

All outputs are clearly labeled as rehearsal/historical simulation.
Does not modify live prediction artifacts or the incumbent model.
"""

import contextlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sportslab.evaluation.predict_incumbent import (
    BEST_DECAY,
    BEST_HFA,
    BEST_K,
    BEST_QB_BONUS,
    BEST_REG,
    FEATURE_COLS,
    INCUMBENT_DATE,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VAL_LL,
    INCUMBENT_VERSION,
    _assign_confidence_bucket,
    _build_pipeline,
)
from sportslab.evaluation.prediction_audit import run_prediction_audit
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.evaluation.weekly_pipeline import (
    _register_snapshot,
    grade_week,
    season_report,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

REHEARSAL_BASE = Path("reports/predictions/rehearsal")
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024]


# ── Path context manager ──


@contextlib.contextmanager
def rehearsal_paths():
    """Context manager redirecting weekly_pipeline paths to rehearsal directory.

    Also redirects prediction_audit output paths so audit reports
    stay in the rehearsal namespace.
    """
    import sportslab.evaluation.prediction_audit as pa
    import sportslab.evaluation.weekly_pipeline as wp

    rehearsal_dir = REHEARSAL_BASE
    rehearsal_dir.mkdir(parents=True, exist_ok=True)
    (rehearsal_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    # Save originals
    orig = {
        "wp_manifest": wp.MANIFEST_PATH,
        "wp_history": wp.HISTORY_PATH,
        "wp_snapshot_dir": wp.SNAPSHOT_DIR,
        "wp_report_dir": wp.REPORT_DIR,
        "pa_report_dir": pa.REPORT_DIR,
        "pa_docs_dir": pa.DOCS_DIR,
    }

    # Redirect to rehearsal
    wp.MANIFEST_PATH = rehearsal_dir / "manifest.json"
    wp.HISTORY_PATH = rehearsal_dir / "prediction_history.csv"
    wp.SNAPSHOT_DIR = rehearsal_dir / "snapshots"
    wp.REPORT_DIR = rehearsal_dir
    pa.REPORT_DIR = rehearsal_dir
    pa.DOCS_DIR = rehearsal_dir  # No docs/ contamination for rehearsal

    # Initialize clean manifest and empty history
    wp._write_manifest({"manifest_version": 1, "snapshots": []})
    empty_history = pd.DataFrame(columns=[
        "season", "week", "n", "log_loss", "brier",
        "accuracy", "auc", "model_version", "snapshot", "graded_at",
    ])
    empty_history.to_csv(wp.HISTORY_PATH, index=False)

    try:
        yield rehearsal_dir
    finally:
        wp.MANIFEST_PATH = orig["wp_manifest"]
        wp.HISTORY_PATH = orig["wp_history"]
        wp.SNAPSHOT_DIR = orig["wp_snapshot_dir"]
        wp.REPORT_DIR = orig["wp_report_dir"]
        pa.REPORT_DIR = orig["pa_report_dir"]
        pa.DOCS_DIR = orig["pa_docs_dir"]


# ── Data loading ──


def _load_feature_table() -> pd.DataFrame:
    fp = Path(FEATURE_TABLE_PATH)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURE_TABLE_PATH}")
    return pd.read_parquet(fp)


def get_season_weeks(df: Optional[pd.DataFrame] = None, season: int = 2025) -> List[int]:
    """Get sorted list of weeks with eligible games for a season."""
    if df is None:
        df = _load_feature_table()
    games = df[(df["season"] == season) & df[MODEL_ELIGIBLE_COLUMN]]
    return sorted(games["week"].unique())


# ── Rehearsal command ──


def rehearse_season(
    season: int = 2025,
    qb_input_path: Optional[str] = None,
) -> Dict:
    """Run full historical rehearsal for a completed season.

    Iterates week-by-week, fitting Elo on all available data before
    each week, predicting that week's games using the incumbent model,
    saving immutable snapshots, grading, and producing reports.

    All outputs are isolated to ``reports/predictions/rehearsal/``.
    Does not modify live manifest, history, or prediction artifacts.

    Args:
        season: Season year to rehearse.
        qb_input_path: Optional CSV for live-safe QB overrides.

    Returns:
        Dict with rehearsal_dir, manifest, history paths and overall_metrics.
    """
    qb_source = "oracle" if qb_input_path is None else "live_pregame"

    print(f"\n{'=' * 60}")
    print(f"  Historical Rehearsal: {season} Season")
    print(f"  QB source:            {qb_source}")
    print(f"  Model:                {INCUMBENT_VERSION}")
    print(f"{'=' * 60}\n")

    df_raw = _load_feature_table()
    weeks = get_season_weeks(df_raw, season)
    print(f"  Weeks to simulate: {weeks[0]}–{weeks[-1]} ({len(weeks)} total)")

    # Pre-load QB input
    qb_input_df = None
    if qb_input_path:
        from sportslab.features.qb_input import parse_qb_input_csv
        qb_input_df = parse_qb_input_csv(qb_input_path)

    overall_metrics = {}
    week_metrics_list = []

    with rehearsal_paths() as rehearsal_dir:
        print(f"  Rehearsal directory: {rehearsal_dir}\n")

        for week_num in weeks:
            result = _rehearse_week(
                df_raw, season, week_num, qb_source, qb_input_df,
            )
            if result is None:
                print(f"  Week {week_num}: no eligible games — skipping")
                continue

            df_week, metrics = result
            week_metrics_list.append(metrics)
            print(
                f"  Week {week_num}: {len(df_week)} games,"
                f" LL={metrics['log_loss']},"
                f" Brier={metrics['brier']},"
                f" Acc={metrics['accuracy']}"
            )

        # Season report
        print("\n  Generating season report...")
        try:
            season_report(season=season)
        except Exception as e:
            print(f"  Season report warning: {e}")

        # Audit
        print("  Generating prediction audit...")
        try:
            run_prediction_audit(season=season, mode="rehearsal")
        except Exception as e:
            print(f"  Audit warning: {e}")

        # Summary
        print(f"\n{'=' * 60}")
        print(f"  Rehearsal Complete: {season} Season")
        print(f"  Weeks simulated: {len(week_metrics_list)}")
        print(f"  Path:             {rehearsal_dir}")
        if week_metrics_list:
            avg_ll = round(
                float(np.mean([m["log_loss"] for m in week_metrics_list
                               if m["log_loss"] is not None])), 4)
            avg_br = round(
                float(np.mean([m["brier"] for m in week_metrics_list
                               if m["brier"] is not None])), 4)
            avg_acc = round(
                float(np.mean([m["accuracy"] for m in week_metrics_list])), 4)
            total_games = sum(m["n"] for m in week_metrics_list)
            print(f"  Games:            {total_games}")
            print(f"  Avg weekly LL:    {avg_ll}")
            print(f"  Avg weekly Brier: {avg_br}")
            print(f"  Avg weekly Acc:   {avg_acc}")
            overall_metrics = {
                "n": total_games,
                "log_loss": avg_ll,
                "brier": avg_br,
                "accuracy": avg_acc,
            }
        print(f"{'=' * 60}\n")

    return {
        "rehearsal_dir": str(REHEARSAL_BASE),
        "manifest": str(REHEARSAL_BASE / "manifest.json"),
        "history": str(REHEARSAL_BASE / "prediction_history.csv"),
        "overall_metrics": overall_metrics,
    }


def _rehearse_week(
    df_raw: pd.DataFrame,
    season: int,
    week_num: int,
    qb_source: str,
    qb_input_df: Optional[pd.DataFrame],
) -> Optional[tuple]:
    """Run prediction + grading for a single rehearsal week.

    Args:
        df_raw: Full feature table.
        season: Season year.
        week_num: Week to rehearse.
        qb_source: "oracle" or "live_pregame".
        qb_input_df: Optional QB override DataFrame.

    Returns:
        Tuple of (df_week_snapshot, metrics) or None if no eligible games.
    """
    # Training data: historical seasons + previous weeks of this season
    train_mask = (
        (df_raw["season"].isin(HISTORICAL_SEASONS))
        | ((df_raw["season"] == season) & (df_raw["week"] < week_num))
    ) & df_raw[MODEL_ELIGIBLE_COLUMN]

    # Prediction data: this week's eligible non-neutral games
    pred_mask = (
        (df_raw["season"] == season)
        & (df_raw["week"] == week_num)
        & df_raw[MODEL_ELIGIBLE_COLUMN]
        & (~df_raw.get("is_neutral", pd.Series(False)).fillna(False))
    )

    if pred_mask.sum() == 0:
        return None

    df_train = df_raw[train_mask].copy()
    df_pred = df_raw[pred_mask].copy()
    pred_game_ids = set(df_pred["game_id"].values)
    # Save actual results before clearing home_win for prediction
    pred_actuals = df_pred.set_index("game_id")["home_win"].copy()
    df_pred["home_win"] = pd.NA

    df_combined = pd.concat([df_train, df_pred], ignore_index=True)
    df_combined = df_combined.sort_values(
        ["season", "week", "gameday"],
    ).reset_index(drop=True)
    df_combined["_is_pred"] = df_combined["game_id"].isin(pred_game_ids)

    # Apply QB input
    if qb_input_df is not None:
        from sportslab.features.qb_input import apply_qb_input
        df_combined = apply_qb_input(df_combined, qb_input_df)

    # Build features
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
    df_feat = compute_qb_features(df_feat)
    df_feat = compute_situational_features(df_feat)

    # Train Platt on historical rows only
    is_pred = df_feat["_is_pred"].values
    is_train = ~is_pred

    cols = [c for c in FEATURE_COLS if c in df_feat.columns]
    train_elo = df_feat.loc[is_train, "elo_prob"].values
    n_train = is_train.sum()
    train_feat = (
        df_feat.loc[is_train, cols].values
        if cols else np.empty((n_train, 0))
    )
    train_y = df_feat.loc[is_train, "home_win"].astype(int).values
    x_train = (
        np.column_stack([train_elo, train_feat])
        if train_feat.size else train_elo.reshape(-1, 1)
    )

    pipe = _build_pipeline()
    pipe.fit(x_train, train_y)

    # Predict
    pred_elo = df_feat.loc[is_pred, "elo_prob"].values
    n_pred = is_pred.sum()
    pred_feat = (
        df_feat.loc[is_pred, cols].values
        if cols else np.empty((n_pred, 0))
    )
    x_pred = (
        np.column_stack([pred_elo, pred_feat])
        if pred_feat.size else pred_elo.reshape(-1, 1)
    )
    prob = pipe.predict_proba(x_pred)[:, 1]

    # Get actual results
    pred_ids = df_feat.loc[is_pred, "game_id"].values
    actual = np.array([
        float(pred_actuals.get(gid, np.nan))
        if pd.notna(pred_actuals.get(gid, np.nan)) else np.nan
        for gid in pred_ids
    ])

    # Build snapshot DataFrame
    df_week = pd.DataFrame({
        "game_id": df_feat.loc[is_pred, "game_id"].values,
        "season": season,
        "week": week_num,
        "gameday": df_feat.loc[is_pred, "gameday"].values,
        "away_team": df_feat.loc[is_pred, "away_team"].values,
        "home_team": df_feat.loc[is_pred, "home_team"].values,
        "incumbent_home_win_prob": prob.round(6),
        "predicted_winner": np.where(
            prob >= 0.5,
            df_feat.loc[is_pred, "home_team"].values,
            df_feat.loc[is_pred, "away_team"].values,
        ),
        "confidence_bucket": [_assign_confidence_bucket(p) for p in prob],
        "model_version": INCUMBENT_VERSION,
        "model_date": INCUMBENT_DATE,
        "training_seasons": "2021-2024",
        "feature_set": "qb_changed + rolling_mov_3",
        "calibration_method": "Platt (logistic on Elo prob + features)",
        "model_val_ll": INCUMBENT_VAL_LL,
        "model_holdout_ll": INCUMBENT_HOLDOUT_LL,
        "elo_k": BEST_K,
        "elo_hfa": BEST_HFA,
        "elo_reg": BEST_REG,
        "elo_decay": BEST_DECAY,
        "elo_qb_bonus": BEST_QB_BONUS,
        "qb_source": qb_source,
        "home_qb_id": df_feat.loc[is_pred, "home_qb_id"].values,
        "away_qb_id": df_feat.loc[is_pred, "away_qb_id"].values,
        "actual_home_win": actual,
    })

    # Caution flags
    qb_caught = df_feat.loc[is_pred, ["home_qb_changed", "away_qb_changed"]].any(axis=1)
    df_week["caution_qb_change"] = qb_caught.astype(int).values
    df_week["caution_early_season"] = (week_num <= 4).astype(int)

    snap_path = _save_snapshot(df_week, season, week_num)

    # Register in manifest (cast to Python int for JSON-safe serialization)
    _register_snapshot(snap_path, int(season), int(week_num), qb_source, int(len(df_week)))

    # Grade (actuals already in snapshot)
    grade_result = grade_week(season=season, week=week_num, snapshot=str(snap_path))
    metrics = grade_result["metrics"]

    return df_week, metrics


def _save_snapshot(df: pd.DataFrame, season: int, week: int) -> Path:
    """Save a rehearsal snapshot CSV with timestamp."""
    from datetime import datetime, timezone

    import sportslab.evaluation.weekly_pipeline as wp

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snap_dir = wp.SNAPSHOT_DIR
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"week_{season}_{week:02d}_{ts}_rehearsal.csv"
    df.to_csv(snap_path, index=False)
    return snap_path
