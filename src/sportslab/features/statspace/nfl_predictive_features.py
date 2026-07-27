"""Leakage-safe NFL predictive feature builder for benchmark experiments."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .nfl_elo import NFLEloEngine
from .nfl_schedule_loader import NFLHistoricalGame

PYTHAGOREAN_EXPONENT = 2.37


def pythagorean_expected_wins(
    points_for: float,
    points_against: float,
    games_played: int,
    *,
    exponent: float = PYTHAGOREAN_EXPONENT,
) -> float:
    """Return expected wins from pregame scoring margin totals."""
    if games_played <= 0:
        return 0.0
    pf = max(float(points_for), 1e-9)
    pa = max(float(points_against), 1e-9)
    win_pct = (pf**exponent) / ((pf**exponent) + (pa**exponent))
    return float(win_pct * games_played)


def aggregate_team_game_pbp(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-by-play into one team-game offense/defense row."""
    pbp = pbp_df.copy()
    if pbp.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "team",
                "off_epa_per_play",
                "off_success_rate",
                "def_epa_per_play_allowed",
                "def_success_rate_allowed",
            ]
        )

    pbp = pbp[(pbp["posteam"].notna()) & (pbp["defteam"].notna())].copy()
    pbp["qb_dropback"] = pbp["qb_dropback"].fillna(0.0)
    pbp["rush_attempt"] = pbp["rush_attempt"].fillna(0.0)
    pbp["is_scrimmage"] = (pbp["qb_dropback"] == 1.0) | (
        pbp["rush_attempt"] == 1.0
    )
    pbp = pbp[pbp["is_scrimmage"]].copy()
    pbp["success"] = (pbp["epa"] > 0.0).astype(float)

    off = (
        pbp.groupby(["game_id", "season", "week", "posteam"])
        .agg(
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    defense = (
        pbp.groupby(["game_id", "season", "week", "defteam"])
        .agg(
            def_epa_per_play_allowed=("epa", "mean"),
            def_success_rate_allowed=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return off.merge(
        defense,
        on=["game_id", "season", "week", "team"],
        how="outer",
    )


def build_team_pregame_rolling_features(
    games: Iterable[NFLHistoricalGame],
    pbp_team_games: pd.DataFrame,
    *,
    ema_span: int = 5,
    exponent: float = PYTHAGOREAN_EXPONENT,
) -> pd.DataFrame:
    """Build one pregame feature row per team-game."""
    ordered_games = sorted(
        games, key=lambda item: (item.date, item.week, item.game_id)
    )
    team_rows: list[dict[str, object]] = []
    for game in ordered_games:
        team_rows.extend(
            [
                {
                    "game_id": game.game_id,
                    "date": game.date,
                    "season": game.season,
                    "week": game.week,
                    "team": game.home_team,
                    "opponent": game.away_team,
                    "side": "home",
                    "points_for": game.home_score,
                    "points_against": game.away_score,
                    "win": 1 if game.home_score > game.away_score else 0,
                },
                {
                    "game_id": game.game_id,
                    "date": game.date,
                    "season": game.season,
                    "week": game.week,
                    "team": game.away_team,
                    "opponent": game.home_team,
                    "side": "away",
                    "points_for": game.away_score,
                    "points_against": game.home_score,
                    "win": 1 if game.away_score > game.home_score else 0,
                },
            ]
        )

    features = pd.DataFrame(team_rows)
    if features.empty:
        return features

    features = features.sort_values(
        ["team", "season", "date", "week", "game_id"]
    ).reset_index(drop=True)
    features["cum_points_for_pre"] = (
        features.groupby(["season", "team"])["points_for"].cumsum()
        - features["points_for"]
    )
    features["cum_points_against_pre"] = (
        features.groupby(["season", "team"])["points_against"].cumsum()
        - features["points_against"]
    )
    features["actual_wins_pre"] = (
        features.groupby(["season", "team"])["win"].cumsum() - features["win"]
    )
    features["games_played_pre"] = features.groupby(
        ["season", "team"]
    ).cumcount()
    features["expected_wins_pre"] = features.apply(
        lambda row: pythagorean_expected_wins(
            row["cum_points_for_pre"],
            row["cum_points_against_pre"],
            int(row["games_played_pre"]),
            exponent=exponent,
        ),
        axis=1,
    )
    features["pyth_delta_pre"] = (
        features["actual_wins_pre"] - features["expected_wins_pre"]
    )

    merged = features.merge(
        pbp_team_games,
        on=["game_id", "season", "week", "team"],
        how="left",
    )
    merged = merged.sort_values(
        ["team", "season", "date", "week", "game_id"]
    ).reset_index(drop=True)

    for column in [
        "off_epa_per_play",
        "off_success_rate",
        "def_epa_per_play_allowed",
        "def_success_rate_allowed",
    ]:
        merged[f"{column}_ema{ema_span}"] = merged.groupby(["season", "team"])[
            column
        ].transform(
            lambda series: series.shift(1)
            .ewm(span=ema_span, adjust=False)
            .mean()
        )

    return merged


def build_game_level_predictive_features(
    *,
    games: Iterable[NFLHistoricalGame],
    schedules_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    elo_engine: Optional[NFLEloEngine] = None,
    ema_span: int = 5,
    exponent: float = PYTHAGOREAN_EXPONENT,
) -> pd.DataFrame:
    """Return deterministic pregame feature rows keyed by game_id."""
    ordered_games = sorted(
        games, key=lambda item: (item.date, item.week, item.game_id)
    )
    pbp_team_games = aggregate_team_game_pbp(pbp_df)
    team_features = build_team_pregame_rolling_features(
        ordered_games,
        pbp_team_games,
        ema_span=ema_span,
        exponent=exponent,
    )

    engine = elo_engine or NFLEloEngine()
    elo_features = engine.build_game_features(ordered_games)

    games_frame = pd.DataFrame(
        [game.to_dict() for game in ordered_games]
    ).sort_values(["date", "week", "game_id"])
    if games_frame.empty:
        return games_frame

    home_team_features = team_features[team_features["side"] == "home"][
        [
            "game_id",
            "team",
            "off_epa_per_play_ema5",
            "off_success_rate_ema5",
            "def_epa_per_play_allowed_ema5",
            "def_success_rate_allowed_ema5",
            "pyth_delta_pre",
        ]
    ].rename(
        columns={
            "team": "home_team",
            "off_epa_per_play_ema5": "home_off_epa_per_play_ema5",
            "off_success_rate_ema5": "home_off_success_rate_ema5",
            "def_epa_per_play_allowed_ema5": (
                "home_def_epa_per_play_allowed_ema5"
            ),
            "def_success_rate_allowed_ema5": (
                "home_def_success_rate_allowed_ema5"
            ),
            "pyth_delta_pre": "home_pyth_delta_pre",
        }
    )
    away_team_features = team_features[team_features["side"] == "away"][
        [
            "game_id",
            "team",
            "off_epa_per_play_ema5",
            "off_success_rate_ema5",
            "def_epa_per_play_allowed_ema5",
            "def_success_rate_allowed_ema5",
            "pyth_delta_pre",
        ]
    ].rename(
        columns={
            "team": "away_team",
            "off_epa_per_play_ema5": "away_off_epa_per_play_ema5",
            "off_success_rate_ema5": "away_off_success_rate_ema5",
            "def_epa_per_play_allowed_ema5": (
                "away_def_epa_per_play_allowed_ema5"
            ),
            "def_success_rate_allowed_ema5": (
                "away_def_success_rate_allowed_ema5"
            ),
            "pyth_delta_pre": "away_pyth_delta_pre",
        }
    )

    schedule_rest = schedules_df.copy()
    schedule_rest = schedule_rest.where(pd.notna(schedule_rest), None)
    available_columns = [
        column
        for column in ["game_id", "home_rest", "away_rest"]
        if column in schedule_rest.columns
    ]
    schedule_rest = schedule_rest[available_columns].drop_duplicates("game_id")

    frame = games_frame.merge(
        elo_features,
        on=["game_id", "date", "season", "week", "home_team", "away_team"],
        how="left",
    )
    frame = frame.merge(
        home_team_features, on=["game_id", "home_team"], how="left"
    )
    frame = frame.merge(
        away_team_features, on=["game_id", "away_team"], how="left"
    )
    frame = frame.merge(schedule_rest, on="game_id", how="left")

    frame["rest_diff"] = frame["home_rest"].fillna(0.0) - frame[
        "away_rest"
    ].fillna(0.0)
    frame = frame.sort_values(["date", "week", "game_id"]).reset_index(
        drop=True
    )
    return frame
