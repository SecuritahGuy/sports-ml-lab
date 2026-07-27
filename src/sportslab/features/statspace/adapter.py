"""Adapter layer: wraps StatSpace R&D metrics to work with our feature pipeline.

All 5 metrics take nflverse PBP + optional schedule/elo data and return
per-team-season DataFrames. This module provides:

1. schedule_to_nfl_historical_games(schedule) — convert our schedule df
   into NFLHistoricalGame objects for the FDR metric
2. compute_statspace_*(...) — one wrapper per metric
3. merge_team_season_metrics(features, metric_df, prefix) — attach
   team-season values into our game-level feature table
"""

from __future__ import annotations

import logging

import pandas as pd

from .nfl_branded_stats import (
    FraudDetectorWeights,
    build_fraud_detector_rating,
)
from .nfl_chaos_rate import build_chaos_rate
from .nfl_coward_tax import build_coward_tax
from .nfl_doba import build_doba
from .nfl_elo import NFLEloEngine
from .nfl_qb_lift_index import build_qb_lift_index
from .nfl_schedule_loader import NFLHistoricalGame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Schedule → NFLHistoricalGame converter
# ---------------------------------------------------------------------------

# Columns our schedule df must have for conversion
SCHEDULE_REQUIRED_FIELDS = [
    "game_id", "season", "week", "gameday",
    "away_team", "home_team",
    "away_score", "home_score",
]

# Optional QB columns our schedule may have
QB_SCHEDULE_FIELDS = [
    "away_qb_id", "home_qb_id",
    "away_qb_name", "home_qb_name",
    "away_qb_attempts", "home_qb_attempts",
    "away_qb_completions", "home_qb_completions",
    "away_qb_passing_yards", "home_qb_passing_yards",
    "away_qb_passing_tds", "home_qb_passing_tds",
    "away_qb_interceptions", "home_qb_interceptions",
    "away_qb_sacks", "home_qb_sacks",
    "away_qb_passing_epa", "home_qb_passing_epa",
]


def schedule_to_nfl_historical_games(
    schedule: pd.DataFrame,
) -> list[NFLHistoricalGame]:
    """Convert our schedule/feature DataFrame into NFLHistoricalGame objects.

    Parameters
    ----------
    schedule : pd.DataFrame
        Must at minimum contain SCHEDULE_REQUIRED_FIELDS.
        May optionally contain QB_SCHEDULE_FIELDS for QB-aware metrics.

    Returns
    -------
    list[NFLHistoricalGame]
    """
    missing = [c for c in SCHEDULE_REQUIRED_FIELDS if c not in schedule.columns]
    if missing:
        raise ValueError(
            f"schedule missing required columns: {missing}"
        )

    available_qb = [c for c in QB_SCHEDULE_FIELDS if c in schedule.columns]

    games: list[NFLHistoricalGame] = []
    for _, row in schedule.sort_values(["season", "week", "gameday", "game_id"]).iterrows():
        game_id = str(row["game_id"])
        home = int(row["home_score"]) if pd.notna(row["home_score"]) else 0
        away = int(row["away_score"]) if pd.notna(row["away_score"]) else 0
        completed = pd.notna(row["home_score"]) and pd.notna(row["away_score"])
        status = "completed" if completed else "scheduled"

        kwargs = {
            c.replace("home_", "").replace("away_", ""): None
            for c in QB_SCHEDULE_FIELDS
        }
        kwargs = {
            "away_qb_id": None,
            "home_qb_id": None,
            "away_qb_name": None,
            "home_qb_name": None,
            "away_qb_attempts": None,
            "home_qb_attempts": None,
            "away_qb_completions": None,
            "home_qb_completions": None,
            "away_qb_passing_yards": None,
            "home_qb_passing_yards": None,
            "away_qb_passing_tds": None,
            "home_qb_passing_tds": None,
            "away_qb_interceptions": None,
            "home_qb_interceptions": None,
            "away_qb_sacks": None,
            "home_qb_sacks": None,
            "away_qb_passing_epa": None,
            "home_qb_passing_epa": None,
        }
        for col in available_qb:
            val = row.get(col)
            kwargs[col] = float(val) if isinstance(val, (int, float)) and pd.notna(val) else (
                str(val) if pd.notna(val) else None
            )

        games.append(NFLHistoricalGame(
            game_id=game_id,
            date=str(row.get("gameday", "")),
            season=int(row["season"]),
            week=int(row["week"]),
            game_type=str(row.get("game_type", "REG")),
            away_team=str(row["away_team"]),
            home_team=str(row["home_team"]),
            away_score=away,
            home_score=home,
            completed=completed,
            status=status,
            **kwargs,
        ))
    return games


# ---------------------------------------------------------------------------
# 2. Metric wrappers
# ---------------------------------------------------------------------------

STATSPACE_FEATURE_COLUMNS: list[str] = []


def compute_statspace_coward_tax(
    pbp_df: pd.DataFrame,
    *,
    season: int | None = None,
    through_week: int | None = None,
    min_games: int = 4,
) -> pd.DataFrame:
    """Compute Coward Tax — 4th-down aggressiveness score per team-season.

    Positive = aggressive (good), negative = cowardly (bad).
    Returns team-season DataFrame with columns including:
      team, season, games_played, coward_tax_per_game, aggression_score, ...
    """
    result, _metadata = build_coward_tax(
        pbp_df, season=season, through_week=through_week, min_games=min_games,
    )
    return result


def compute_statspace_doba(
    pbp_df: pd.DataFrame,
    *,
    season: int | None = None,
    through_week: int | None = None,
    min_games: int = 4,
) -> pd.DataFrame:
    """Compute DOBA — sustainable offensive efficiency composite.

    Higher = better offensive sustainability.
    """
    result, _metadata = build_doba(
        pbp_df, season=season, through_week=through_week, min_games=min_games,
    )
    return result


def compute_statspace_chaos_rate(
    pbp_df: pd.DataFrame,
    *,
    season: int | None = None,
    through_week: int | None = None,
    min_games: int = 4,
) -> pd.DataFrame:
    """Compute Chaos Rate — defensive disruption composite.

    Higher = more disruptive defense.
    """
    result, _metadata = build_chaos_rate(
        pbp_df, season=season, through_week=through_week, min_games=min_games,
    )
    return result


def compute_statspace_qb_lift(
    pbp_df: pd.DataFrame,
    *,
    season: int | None = None,
    through_week: int | None = None,
    min_dropbacks: int = 100,
    opponent_adjusted: bool = False,
) -> pd.DataFrame:
    """Compute QB Lift Index — QB value beyond supporting cast.

    Returns per-QB DataFrame. Use qb_lift_index (opponent-adjusted if
    available) as the primary signal. For team-level analysis, aggregate
    by taking the primary QB's value per team.
    """
    result, _metadata = build_qb_lift_index(
        pbp_df,
        season=season,
        through_week=through_week,
        min_dropbacks=min_dropbacks,
        opponent_adjusted=opponent_adjusted,
    )
    return result


def compute_statspace_fdr(
    games: list[NFLHistoricalGame],
    *,
    pbp_df: pd.DataFrame | None = None,
    season: int | None = None,
    through_week: int | None = None,
) -> pd.DataFrame:
    """Compute FDR — Fraud Detector Rating.

    Positive FDR = record inflated vs underlying quality (overachiever risk).
    Negative FDR = record worse than underlying quality (underachiever upside).
    Uses NFLEloEngine internally with default config (K=20, HFA=65, carryover=0.67).
    """
    result, _metadata = build_fraud_detector_rating(
        games,
        pbp_df=pbp_df,
        season=season,
        through_week=through_week,
        elo_engine=NFLEloEngine(),
        weights=FraudDetectorWeights(),
    )
    return result


# ---------------------------------------------------------------------------
# 3. Merge helper
# ---------------------------------------------------------------------------

def merge_team_season_metrics(
    features: pd.DataFrame,
    metric_df: pd.DataFrame,
    prefix: str = "",
    *,
    value_columns: list[str] | None = None,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    season_col: str = "season",
    metric_team_col: str = "team",
    metric_season_col: str = "season",
) -> pd.DataFrame:
    """Attach team-season metric values to a game-level feature table.

    For each game, creates home_{prefix}_{col} and away_{prefix}_{col}.

    Parameters
    ----------
    features : pd.DataFrame
        Game-level feature table (must have home_team, away_team, season).
    metric_df : pd.DataFrame
        Team-season metrics (must have team, season, and value columns).
    prefix : str
        Prefix for output columns (e.g. "coward_tax" → home_coward_tax_score).
    value_columns : list[str] | None
        Which metric columns to merge. If None, uses all columns
        except rank, through_week, games_played, label, why_flagged, etc.
    home_team_col, away_team_col, season_col : str
        Column names in features.
    metric_team_col, metric_season_col : str
        Column names in metric_df.

    Returns
    -------
    pd.DataFrame
        features with new columns.
    """
    skip_patterns = ["rank", "through_week", "games_played", "label", "why_flagged",
                     "public_tier", "formula_version"]
    if value_columns is None:
        value_columns = [
            c for c in metric_df.columns
            if c not in [metric_team_col, metric_season_col]
            and not any(c.startswith(p) or c == p for p in skip_patterns)
        ]

    home = metric_df[[metric_team_col, metric_season_col] + value_columns].copy()
    home.columns = [home_team_col, season_col] + [
        f"home_{prefix}_{c}" if prefix else f"home_{c}" for c in value_columns
    ]
    away = metric_df[[metric_team_col, metric_season_col] + value_columns].copy()
    away.columns = [away_team_col, season_col] + [
        f"away_{prefix}_{c}" if prefix else f"away_{c}" for c in value_columns
    ]

    result = features.merge(home, on=[home_team_col, season_col], how="left")
    result = result.merge(away, on=[away_team_col, season_col], how="left")
    return result
