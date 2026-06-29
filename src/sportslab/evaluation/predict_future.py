"""Future-prediction mode for the incumbent model.

Generates pregame predictions for games without requiring:
  - home_score
  - away_score
  - home_win
  - result

How it works:
  1. Loads feature table (historical games with scores for Elo fitting)
  2. Separates known games (with scores) from prediction games (without scores)
  3. Fits Elo on known games only, chronologically
  4. For each prediction game, emits pregame features (elo_prob, qb_changed,
     rolling_mov_3) without updating Elo (since no result is known)
  5. Applies incumbent Platt calibration to produce final probabilities
  6. Saves prediction CSV

Two QB modes:
  - oracle: Uses final actual starter data from nflreadpy schedules (backtest
    only — NOT fully live-pregame-safe, because the schedule's qb_id is set
    after injury reports are finalized).
  - live_pregame: Uses user-supplied QB starter info via --qb-input CSV.
    Overrides oracle data with pregame-announced starters.

Usage:
    sportslab predict-future
    sportslab predict-future --input predictions_to_make.csv --output predictions.csv
    sportslab predict-future --qb-input qb_starters.csv --season 2025 --week 1
    sportslab predict-future --qb-input qb_starters.csv --season 2025
"""

from pathlib import Path
from typing import Dict, Optional

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
    INCUMBENT_FEATURE_SET,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VAL_LL,
    INCUMBENT_VERSION,
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

HISTORICAL_SEASONS = [2021, 2022, 2023, 2024]
DEFAULT_OUTPUT = "reports/predictions/future_predictions.csv"
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"


def _load_historical_and_future(
    input_path: Optional[str] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
) -> pd.DataFrame:
    """Load feature table and optionally merge future games.

    If input_path is provided, loads those rows and merges with
    the feature table's pregame columns. Otherwise, uses the
    full feature table but only predicts on rows without home_win
    (i.e., future games in the table).

    When season and/or week are specified, filters the prediction
    scope to those games only.

    Returns:
        DataFrame with historical games (home_win not null) and
        future games (home_win is null) merged.
    """
    if not Path(FEATURE_TABLE_PATH).exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURE_TABLE_PATH}")

    df = pd.read_parquet(FEATURE_TABLE_PATH)

    if input_path:
        future = pd.read_csv(input_path)
        # Merge future rows with feature table pregame columns
        id_cols = [c for c in ["game_id", "season", "week", "gameday",
                                "home_team", "away_team"] if c in future.columns]
        if not id_cols:
            raise ValueError("Input CSV must contain game_id, season, week, home_team, away_team")

        merge_cols = [c for c in df.columns if c not in
                      ["home_score", "away_score", "home_win", "result",
                       "is_tie", MODEL_ELIGIBLE_COLUMN]]
        merged = future.merge(df[merge_cols], on=id_cols, how="left")
        # Future games have no score/target
        merged["home_win"] = pd.NA
        # Append to historical
        df_hist = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
        df_out = pd.concat([df_hist, merged], ignore_index=True)
        df_out = df_out.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
        return df_out

    # Use all rows in feature table; future games have no home_win
    df_out = df.copy()
    if season is not None:
        df_out = df_out[df_out["season"] == season].copy()
    if week is not None:
        df_out = df_out[df_out["week"] == week].copy()
    if season is not None or week is not None:
        # Still need historical data for Elo fitting
        df_hist = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
        df_out = pd.concat([df_hist, df_out], ignore_index=True)
        df_out = df_out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
        df_out = df_out.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    return df_out


def _split_by_availability(df: pd.DataFrame):
    """Split into known (has home_win) and unknown (no home_win) games."""
    has_result = df["home_win"].notna().values
    df_known = df[has_result].copy().reset_index(drop=True)
    df_future = df[~has_result].copy().reset_index(drop=True)
    return df_known, df_future, has_result


LIVE_MODES = ["live"]


def predict_future(
    input_path: Optional[str] = None,
    output_path: str = DEFAULT_OUTPUT,
    qb_input_path: Optional[str] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
    mode: str = "live",
) -> Dict[str, str]:
    """Generate predictions for future games using the incumbent model.

    Args:
        input_path: Optional CSV with game_id, season, week, home_team, away_team.
        output_path: Where to save the prediction CSV.
        qb_input_path: Optional CSV with game_id, home_qb_id, away_qb_id.
            Overrides oracle QB data with live-safe pregame starter info.
        season: Optional season to filter future predictions.
        week: Optional week to filter future predictions.
        mode: Snapshot mode — 'live', 'dry_run', or 'rehearsal'. In live mode,
            oracle QB data is blocked.

    Returns:
        Dict with paths to output files.
    """
    from sportslab.evaluation.weekly_pipeline import _validate_mode
    _validate_mode(mode)

    # Live mode: block oracle QB data
    if mode in LIVE_MODES and not qb_input_path:
        raise ValueError(
            f"Oracle QB data not allowed in live mode ({mode}). "
            f"Provide qb_input_path with live-safe pregame QB starters. "
            f"Use mode='dry_run' or mode='rehearsal' for oracle-QB predictions."
        )

    print(f"=== Future Prediction Mode ({mode}) ===")

    # Load data
    df_all = _load_historical_and_future(input_path, season=season, week=week)
    df_known, df_future, has_result = _split_by_availability(df_all)

    if len(df_future) == 0:
        print("  No future games found to predict (all games have results).")
        return {}

    print(f"  Historical games (with scores): {len(df_known)}")
    print(f"  Future games (to predict):       {len(df_future)}")

    # Apply live-safe QB input override if provided
    qb_source = "oracle"
    if qb_input_path:
        qb_input_df = parse_qb_input_csv(qb_input_path)
        df_all = apply_qb_input(df_all, qb_input_df)
        qb_source = "live_pregame"
        print(f"  QB source: live_pregame (overrode oracle data for {len(qb_input_df)} games)")

    # Build features on full sorted dataset
    overrides = build_team_regression_overrides(
        df_all,
        preseason_regression=0.1,
        qb_change_bonus=0.2,
    )
    df_feat = compute_elo_features(
        df_all,
        k_factor=36,
        home_advantage=40,
        preseason_regression=0.1,
        team_regression_overrides=overrides,
        decay_half_life=32,
    )
    df_feat = compute_qb_features(df_feat)
    df_feat = compute_situational_features(df_feat)

    # Feature columns for Platt
    feat_cols = [c for c in FEATURE_COLS if c in df_feat.columns]

    # Fit Platt on known games only
    known_mask = df_feat["home_win"].notna().values
    elo_prob_all = df_feat["elo_prob"].values
    feat_all = df_feat[feat_cols].values if feat_cols else np.empty((len(df_feat), 0))

    has_feats = len(feat_cols) > 0
    x_known = np.column_stack(
        [elo_prob_all[known_mask],
         feat_all[known_mask]] if has_feats else [elo_prob_all[known_mask]]
    )
    y_known = df_feat.loc[known_mask, "home_win"].astype(int).values

    pipe = _build_pipeline()
    pipe.fit(x_known, y_known)
    print("  Platt calibration fitted on historical games")

    # Predict on future games
    future_mask = ~known_mask & ~df_feat["is_neutral"].fillna(False).values
    x_future = np.column_stack(
        [elo_prob_all[future_mask],
         feat_all[future_mask]] if has_feats else [elo_prob_all[future_mask]]
    )
    prob = pipe.predict_proba(x_future)[:, 1]
    pred_winner = np.where(prob >= 0.5,
                           df_feat.loc[future_mask, "home_team"].values,
                           df_feat.loc[future_mask, "away_team"].values)

    # Build output DataFrame
    df_out = pd.DataFrame({
        "game_id": df_feat.loc[future_mask, "game_id"].values,
        "season": df_feat.loc[future_mask, "season"].values,
        "week": df_feat.loc[future_mask, "week"].values,
        "gameday": df_feat.loc[future_mask, "gameday"].values,
        "away_team": df_feat.loc[future_mask, "away_team"].values,
        "home_team": df_feat.loc[future_mask, "home_team"].values,
        "incumbent_home_win_prob": prob,
        "predicted_winner": pred_winner,
        "confidence_bucket": [_assign_confidence_bucket(p) for p in prob],
        "model_version": INCUMBENT_VERSION,
        "model_date": INCUMBENT_DATE,
        "training_seasons": "2021-2024",
        "feature_set": INCUMBENT_FEATURE_SET,
        "calibration_method": "Platt (logistic on Elo prob + features)",
        "model_val_ll": INCUMBENT_VAL_LL,
        "model_holdout_ll": INCUMBENT_HOLDOUT_LL,
        "elo_k": BEST_K,
        "elo_hfa": BEST_HFA,
        "elo_reg": BEST_REG,
        "elo_decay": BEST_DECAY,
        "elo_qb_bonus": BEST_QB_BONUS,
        "qb_source": qb_source,
        "home_qb_id": df_feat.loc[future_mask, "home_qb_id"].values,
        "away_qb_id": df_feat.loc[future_mask, "away_qb_id"].values,
    })

    # Add caution flags
    qb_cols = [c for c in ["home_qb_changed", "away_qb_changed"] if c in df_feat.columns]
    if qb_cols:
        df_out["caution_qb_change"] = (
            df_feat.loc[future_mask, qb_cols].any(axis=1).astype(int).values
        )
    else:
        df_out["caution_qb_change"] = 0

    weeks = df_feat.loc[future_mask, "week"].values
    df_out["caution_early_season"] = (weeks <= 4).astype(int)

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nFuture predictions saved: {out_path}")
    print(f"  {len(df_out)} games predicted")
    print(f"  QB source: {qb_source}")

    return {"predictions": str(out_path)}


def run_predict_future(
    input_path: Optional[str] = None,
    output: Optional[str] = None,
    qb_input: Optional[str] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
    mode: str = "live",
) -> Dict[str, str]:
    """CLI entry point for future prediction."""
    if qb_input is not None and not Path(qb_input).exists():
        raise FileNotFoundError(f"QB input file not found: {qb_input}")
    return predict_future(
        input_path=input_path,
        output_path=output or DEFAULT_OUTPUT,
        qb_input_path=qb_input,
        season=season,
        week=week,
        mode=mode,
    )
