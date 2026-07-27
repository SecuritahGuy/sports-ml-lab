"""Experimental StatSpace NFL branded-stat calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd

from .nfl_elo import NFLEloEngine
from .nfl_predictive_features import (
    PYTHAGOREAN_EXPONENT,
    aggregate_team_game_pbp,
    pythagorean_expected_wins,
)
from .nfl_schedule_loader import NFLHistoricalGame

FDR_FORMULA_VERSION = 2


@dataclass(frozen=True)
class FraudDetectorWeights:
    """Weights for the public-data Fraud Detector Rating formula."""

    record_strength: float = 0.35
    underlying_quality: float = -0.85
    luck_gap: float = 0.85
    close_game_luck: float = 0.5
    turnover_luck: float = 0.35
    schedule_suspicion: float = 0.35


@dataclass(frozen=True)
class FraudDetectorMetadata:
    """Machine-readable metadata for one FDR build."""

    metric: str
    formula_version: int
    season: Optional[int]
    through_week: Optional[int]
    game_count: int
    team_count: int
    weights: dict[str, float]
    built_at: str
    notes: list[str]


def build_fraud_detector_rating(
    games: Iterable[NFLHistoricalGame],
    *,
    pbp_df: Optional[pd.DataFrame] = None,
    season: Optional[int] = None,
    through_week: Optional[int] = None,
    elo_engine: Optional[NFLEloEngine] = None,
    weights: FraudDetectorWeights | None = None,
) -> tuple[pd.DataFrame, FraudDetectorMetadata]:
    """Build a season-to-date Fraud Detector Rating table.

    Positive FDR means a team's record is running ahead of its underlying
    profile. Negative FDR means the underlying profile is better than the
    record.
    """
    resolved_weights = weights or FraudDetectorWeights()
    ordered_games = _filter_completed_games(
        games,
        season=season,
        through_week=through_week,
    )
    notes: list[str] = []
    if not ordered_games:
        metadata = _metadata(
            season=season,
            through_week=through_week,
            game_count=0,
            team_count=0,
            weights=resolved_weights,
            notes=["No completed games matched the requested filter."],
        )
        return _empty_fdr_frame(), metadata

    team_frame = _build_team_results(ordered_games)
    team_frame = _add_pbp_components(team_frame, ordered_games, pbp_df, notes)
    team_frame = _add_elo_components(
        team_frame,
        ordered_games,
        elo_engine or NFLEloEngine(),
    )
    team_frame = _add_fdr_components(team_frame, resolved_weights)

    team_frame = team_frame.sort_values(
        ["fraud_detector_rating", "win_pct", "team"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    team_frame["rank"] = team_frame.index + 1
    ordered_columns = [
        "rank",
        "team",
        "season",
        "through_week",
        "games_played",
        "wins",
        "losses",
        "ties",
        "win_pct",
        "point_diff",
        "point_diff_per_game",
        "pythagorean_win_pct",
        "actual_minus_pyth_wins",
        "one_score_games",
        "one_score_luck",
        "turnover_giveaways_per_game",
        "turnover_takeaways_per_game",
        "turnover_margin_per_game",
        "epa_diff_per_play",
        "success_rate_diff",
        "current_elo",
        "avg_opponent_elo_pre",
        "record_strength_z",
        "underlying_quality_z",
        "luck_gap_z",
        "close_game_luck_z",
        "turnover_luck_z",
        "schedule_suspicion_z",
        "fraud_detector_rating",
        "label",
        "why_flagged",
    ]
    team_frame = team_frame[ordered_columns]
    metadata = _metadata(
        season=season,
        through_week=through_week,
        game_count=len(ordered_games),
        team_count=len(team_frame),
        weights=resolved_weights,
        notes=notes,
    )
    return team_frame, metadata


def _filter_completed_games(
    games: Iterable[NFLHistoricalGame],
    *,
    season: Optional[int],
    through_week: Optional[int],
) -> list[NFLHistoricalGame]:
    filtered = [
        game
        for game in games
        if game.completed
        and (season is None or game.season == season)
        and (through_week is None or game.week <= through_week)
    ]
    return sorted(filtered, key=lambda game: (game.date, game.week, game.game_id))


def _build_team_results(games: list[NFLHistoricalGame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game in games:
        margin = game.home_score - game.away_score
        rows.append(
            _team_game_result(
                game=game,
                team=game.home_team,
                opponent=game.away_team,
                points_for=game.home_score,
                points_against=game.away_score,
                margin=margin,
                is_home=True,
            )
        )
        rows.append(
            _team_game_result(
                game=game,
                team=game.away_team,
                opponent=game.home_team,
                points_for=game.away_score,
                points_against=game.home_score,
                margin=-margin,
                is_home=False,
            )
        )

    team_games = pd.DataFrame(rows)
    grouped = team_games.groupby(["season", "team"], as_index=False).agg(
        through_week=("week", "max"),
        games_played=("game_id", "count"),
        wins=("win", "sum"),
        losses=("loss", "sum"),
        ties=("tie", "sum"),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
        point_diff=("margin", "sum"),
        one_score_games=("one_score_game", "sum"),
        one_score_wins=("one_score_win", "sum"),
        one_score_losses=("one_score_loss", "sum"),
    )
    grouped["win_pct"] = (grouped["wins"] + (0.5 * grouped["ties"])) / grouped[
        "games_played"
    ]
    grouped["point_diff_per_game"] = grouped["point_diff"] / grouped["games_played"]
    grouped["expected_wins"] = grouped.apply(
        lambda row: pythagorean_expected_wins(
            row["points_for"],
            row["points_against"],
            int(row["games_played"]),
            exponent=PYTHAGOREAN_EXPONENT,
        ),
        axis=1,
    )
    grouped["pythagorean_win_pct"] = grouped["expected_wins"] / grouped["games_played"]
    grouped["actual_wins"] = grouped["wins"] + (0.5 * grouped["ties"])
    grouped["actual_minus_pyth_wins"] = (
        grouped["actual_wins"] - grouped["expected_wins"]
    )
    grouped["one_score_luck"] = grouped.apply(
        lambda row: _ratio_or_zero(
            row["one_score_wins"] - row["one_score_losses"],
            row["one_score_games"],
        ),
        axis=1,
    )
    return grouped


def _team_game_result(
    *,
    game: NFLHistoricalGame,
    team: str,
    opponent: str,
    points_for: int,
    points_against: int,
    margin: int,
    is_home: bool,
) -> dict[str, object]:
    return {
        "game_id": game.game_id,
        "season": game.season,
        "week": game.week,
        "team": team,
        "opponent": opponent,
        "is_home": is_home,
        "points_for": points_for,
        "points_against": points_against,
        "margin": margin,
        "win": 1.0 if margin > 0 else 0.0,
        "loss": 1.0 if margin < 0 else 0.0,
        "tie": 1.0 if margin == 0 else 0.0,
        "one_score_game": 1.0 if abs(margin) <= 8 else 0.0,
        "one_score_win": 1.0 if 0 < margin <= 8 else 0.0,
        "one_score_loss": 1.0 if -8 <= margin < 0 else 0.0,
    }


def _add_pbp_components(
    team_frame: pd.DataFrame,
    games: list[NFLHistoricalGame],
    pbp_df: Optional[pd.DataFrame],
    notes: list[str],
) -> pd.DataFrame:
    result = team_frame.copy()
    result["epa_diff_per_play"] = 0.0
    result["success_rate_diff"] = 0.0
    result["turnover_giveaways_per_game"] = 0.0
    result["turnover_takeaways_per_game"] = 0.0
    result["turnover_margin_per_game"] = 0.0
    if pbp_df is None or pbp_df.empty:
        notes.append(
            "PBP data missing; EPA, success, and turnover components defaulted to zero."
        )
        return result

    game_ids = {game.game_id for game in games}
    pbp = pbp_df[pbp_df["game_id"].isin(game_ids)].copy()
    if pbp.empty:
        notes.append(
            "PBP data had no matching game IDs; "
            "EPA, success, and turnover components defaulted to zero."
        )
        return result

    team_games = aggregate_team_game_pbp(pbp)
    if team_games.empty:
        notes.append(
            "PBP aggregation was empty; EPA and success components defaulted to zero."
        )
    else:
        team_components = team_games.groupby(["season", "team"], as_index=False).agg(
            off_epa_per_play=("off_epa_per_play", "mean"),
            off_success_rate=("off_success_rate", "mean"),
            def_epa_per_play_allowed=("def_epa_per_play_allowed", "mean"),
            def_success_rate_allowed=("def_success_rate_allowed", "mean"),
        )
        team_components["epa_diff_per_play"] = team_components[
            "off_epa_per_play"
        ].fillna(0.0) - team_components["def_epa_per_play_allowed"].fillna(0.0)
        team_components["success_rate_diff"] = team_components[
            "off_success_rate"
        ].fillna(0.0) - team_components["def_success_rate_allowed"].fillna(0.0)
        result = result.drop(columns=["epa_diff_per_play", "success_rate_diff"]).merge(
            team_components[
                [
                    "season",
                    "team",
                    "epa_diff_per_play",
                    "success_rate_diff",
                ]
            ],
            on=["season", "team"],
            how="left",
        )
        result["epa_diff_per_play"] = result["epa_diff_per_play"].fillna(0.0)
        result["success_rate_diff"] = result["success_rate_diff"].fillna(0.0)

    turnover_components = _turnover_components(pbp)
    if turnover_components.empty:
        notes.append(
            "Turnover columns missing from PBP; turnover component defaulted to zero."
        )
        return result

    result = result.drop(
        columns=[
            "turnover_giveaways_per_game",
            "turnover_takeaways_per_game",
            "turnover_margin_per_game",
        ]
    ).merge(
        turnover_components,
        on=["season", "team"],
        how="left",
    )
    result["turnover_margin_per_game"] = result["turnover_margin_per_game"].fillna(0.0)
    return result


def _turnover_components(pbp: pd.DataFrame) -> pd.DataFrame:
    if not {"posteam", "defteam", "season"}.issubset(pbp.columns):
        return pd.DataFrame()

    turnover_columns = [
        column for column in ["interception", "fumble_lost"] if column in pbp.columns
    ]
    if not turnover_columns:
        return pd.DataFrame()

    turnover_frame = pbp.copy()
    turnover_frame["turnover"] = (
        turnover_frame[turnover_columns].fillna(0.0).max(axis=1)
    )

    giveaways = (
        turnover_frame.groupby(["season", "posteam"], as_index=False)["turnover"]
        .sum()
        .rename(columns={"posteam": "team", "turnover": "giveaways"})
    )
    takeaways = (
        turnover_frame.groupby(["season", "defteam"], as_index=False)["turnover"]
        .sum()
        .rename(columns={"defteam": "team", "turnover": "takeaways"})
    )
    margins = giveaways.merge(takeaways, on=["season", "team"], how="outer")
    margins["giveaways"] = margins["giveaways"].fillna(0.0)
    margins["takeaways"] = margins["takeaways"].fillna(0.0)
    margins["turnover_margin"] = margins["takeaways"] - margins["giveaways"]
    game_counts = _team_game_counts_from_pbp(pbp)
    margins = margins.merge(game_counts, on=["season", "team"], how="left")
    margins["turnover_giveaways_per_game"] = margins["giveaways"] / margins[
        "games_played"
    ].replace(0, pd.NA)
    margins["turnover_takeaways_per_game"] = margins["takeaways"] / margins[
        "games_played"
    ].replace(0, pd.NA)
    margins["turnover_margin_per_game"] = margins["turnover_margin"] / margins[
        "games_played"
    ].replace(0, pd.NA)
    return margins[
        [
            "season",
            "team",
            "turnover_giveaways_per_game",
            "turnover_takeaways_per_game",
            "turnover_margin_per_game",
        ]
    ]


def _team_game_counts_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    offense = pbp[["season", "game_id", "posteam"]].rename(columns={"posteam": "team"})
    defense = pbp[["season", "game_id", "defteam"]].rename(columns={"defteam": "team"})
    teams = pd.concat([offense, defense], ignore_index=True).dropna()
    return (
        teams.drop_duplicates(["season", "game_id", "team"])
        .groupby(["season", "team"], as_index=False)
        .agg(games_played=("game_id", "count"))
    )


def _add_elo_components(
    team_frame: pd.DataFrame,
    games: list[NFLHistoricalGame],
    elo_engine: NFLEloEngine,
) -> pd.DataFrame:
    result = team_frame.copy()
    elo_features = elo_engine.build_game_features(games, reset=True)
    opponent_rows: list[dict[str, object]] = []
    for row in elo_features.itertuples(index=False):
        opponent_rows.append(
            {
                "season": row.season,
                "team": row.home_team,
                "opponent_elo_pre": float(row.away_elo_pre),
            }
        )
        opponent_rows.append(
            {
                "season": row.season,
                "team": row.away_team,
                "opponent_elo_pre": float(row.home_elo_pre),
            }
        )
    opponent_frame = pd.DataFrame(opponent_rows)
    avg_opponents = opponent_frame.groupby(["season", "team"], as_index=False).agg(
        avg_opponent_elo_pre=("opponent_elo_pre", "mean")
    )

    final_ratings = elo_engine.process_games(games)["team_ratings"]
    current_elo = pd.DataFrame(
        [
            {"team": team, "current_elo": rating}
            for team, rating in final_ratings.items()
        ]
    )
    result = result.merge(avg_opponents, on=["season", "team"], how="left")
    result = result.merge(current_elo, on="team", how="left")
    default_elo = elo_engine.config.initial_elo
    result["avg_opponent_elo_pre"] = result["avg_opponent_elo_pre"].fillna(default_elo)
    result["current_elo"] = result["current_elo"].fillna(default_elo)
    return result


def _add_fdr_components(
    team_frame: pd.DataFrame,
    weights: FraudDetectorWeights,
) -> pd.DataFrame:
    result = team_frame.copy()
    result["record_strength_z"] = _zscore(result["win_pct"])
    result["point_diff_z"] = _zscore(result["point_diff_per_game"])
    result["pyth_win_pct_z"] = _zscore(result["pythagorean_win_pct"])
    result["epa_diff_z"] = _zscore(result["epa_diff_per_play"])
    result["success_diff_z"] = _zscore(result["success_rate_diff"])
    result["elo_z"] = _zscore(result["current_elo"])
    result["underlying_quality_z"] = result[
        [
            "point_diff_z",
            "pyth_win_pct_z",
            "epa_diff_z",
            "success_diff_z",
            "elo_z",
        ]
    ].mean(axis=1)
    result["luck_gap_z"] = _zscore(result["actual_minus_pyth_wins"])
    result["close_game_luck_z"] = _zscore(result["one_score_luck"])
    result["turnover_luck_z"] = _zscore(result["turnover_margin_per_game"])
    result["schedule_suspicion_z"] = -_zscore(result["avg_opponent_elo_pre"])
    result["fraud_detector_rating"] = (
        weights.record_strength * result["record_strength_z"]
        + weights.underlying_quality * result["underlying_quality_z"]
        + weights.luck_gap * result["luck_gap_z"]
        + weights.close_game_luck * result["close_game_luck_z"]
        + weights.turnover_luck * result["turnover_luck_z"]
        + weights.schedule_suspicion * result["schedule_suspicion_z"]
    )
    result["label"] = result.apply(_fdr_label, axis=1)
    result["why_flagged"] = result.apply(_why_flagged, axis=1)
    return result


def _fdr_label(row: pd.Series) -> str:
    rating = float(row["fraud_detector_rating"])
    quality = float(row["underlying_quality_z"])
    win_pct = float(row["win_pct"])
    if rating >= 1.50 and quality < 0.50 and win_pct >= 0.500:
        return "Full Fraud Alert"
    if rating >= 0.75 and quality >= 0.50 and win_pct >= 0.500:
        return "Inflated Contender"
    if rating >= 0.50:
        return "Paper Tiger Watch"
    if rating <= -0.50:
        if quality < 0.0:
            return "Bad, But Unlucky"
        return "Better Than Record"
    if quality >= 0.50:
        return "Legit Contender"
    return "Properly Priced"


def _why_flagged(row: pd.Series) -> str:
    label = str(row["label"])
    drivers = _fdr_driver_phrases(row)
    if drivers:
        return ", ".join(drivers[:2])

    if label == "Legit Contender":
        return "Strong record backed by strong underlying quality"
    if label == "Inflated Contender":
        return "Real contender, but record may overstate dominance"
    if label == "Better Than Record":
        return "Underlying profile is stronger than the record"
    if label == "Bad, But Unlucky":
        return "Weak team, but results may still be harsher than deserved"
    return "Record broadly matches the underlying profile"


def _fdr_driver_phrases(row: pd.Series) -> list[str]:
    rating = float(row["fraud_detector_rating"])
    quality = float(row["underlying_quality_z"])
    phrases: list[str] = []

    if rating >= 0.50:
        if float(row["luck_gap_z"]) >= 0.75:
            phrases.append("record running ahead of Pythagorean expectation")
        if float(row["close_game_luck_z"]) >= 0.75:
            phrases.append("one-score-heavy profile")
        if float(row["turnover_luck_z"]) >= 0.75:
            phrases.append("turnover-driven results")
        if float(row["schedule_suspicion_z"]) >= 0.75:
            phrases.append("lighter schedule profile")
        if quality < 0.50:
            phrases.append("modest underlying quality")
        elif quality >= 0.50:
            phrases.append("strong team, but record may overstate dominance")
        return phrases

    if rating <= -0.50:
        if float(row["luck_gap_z"]) <= -0.75:
            phrases.append("record trails Pythagorean expectation")
        if float(row["close_game_luck_z"]) <= -0.75:
            phrases.append("brutal close-game luck")
        if float(row["turnover_luck_z"]) <= -0.75:
            phrases.append("turnover margin suppressing results")
        if quality >= 0.50:
            phrases.append("strong underlying quality")
        elif quality >= 0.0:
            phrases.append("positive underlying profile")
        else:
            phrases.append("still weak by underlying quality")
        return phrases

    if quality >= 0.50:
        phrases.append("record backed by strong underlying quality")
    elif quality <= -0.50:
        phrases.append("record matches a weak underlying profile")
    return phrases


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(numeric.std(ddof=0))
    if std == 0.0:
        return pd.Series(0.0, index=series.index)
    return (numeric - float(numeric.mean())) / std


def _ratio_or_zero(numerator: object, denominator: object) -> float:
    denom = float(denominator)
    if denom == 0.0:
        return 0.0
    return float(numerator) / denom


def _metadata(
    *,
    season: Optional[int],
    through_week: Optional[int],
    game_count: int,
    team_count: int,
    weights: FraudDetectorWeights,
    notes: list[str],
) -> FraudDetectorMetadata:
    return FraudDetectorMetadata(
        metric="fraud_detector_rating",
        formula_version=FDR_FORMULA_VERSION,
        season=season,
        through_week=through_week,
        game_count=game_count,
        team_count=team_count,
        weights=asdict(weights),
        built_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        notes=notes,
    )


def _empty_fdr_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rank",
            "team",
            "season",
            "through_week",
            "games_played",
            "wins",
            "losses",
            "ties",
            "win_pct",
            "point_diff",
            "point_diff_per_game",
            "pythagorean_win_pct",
            "actual_minus_pyth_wins",
            "one_score_games",
            "one_score_luck",
            "turnover_giveaways_per_game",
            "turnover_takeaways_per_game",
            "turnover_margin_per_game",
            "epa_diff_per_play",
            "success_rate_diff",
            "current_elo",
            "avg_opponent_elo_pre",
            "record_strength_z",
            "underlying_quality_z",
            "luck_gap_z",
            "close_game_luck_z",
            "turnover_luck_z",
            "schedule_suspicion_z",
            "fraud_detector_rating",
            "label",
            "why_flagged",
        ]
    )
