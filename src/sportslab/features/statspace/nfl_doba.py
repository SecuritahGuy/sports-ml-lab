"""Experimental StatSpace NFL DOBA calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

DOBA_FORMULA_VERSION = 1


@dataclass(frozen=True)
class DOBAWeights:
    """Weights for Deserved Offensive Balance Added."""

    offensive_epa_per_play: float = 0.30
    offensive_success_rate: float = 0.25
    early_down_efficiency: float = 0.20
    explosive_rate: float = 0.15
    red_zone_efficiency: float = 0.15
    third_fourth_down_efficiency: float = 0.10
    negative_play_rate: float = -0.15
    turnover_rate: float = -0.10
    dependency_penalty: float = -0.10


@dataclass(frozen=True)
class DOBAMetadata:
    """Machine-readable metadata for one DOBA build."""

    metric: str
    formula_version: int
    season: Optional[int]
    through_week: Optional[int]
    min_games: int
    qualified_team_count: int
    weights: dict[str, float]
    built_at: str
    notes: list[str]


def build_doba(
    pbp_df: pd.DataFrame,
    *,
    season: Optional[int] = None,
    through_week: Optional[int] = None,
    min_games: int = 4,
    weights: DOBAWeights | None = None,
) -> tuple[pd.DataFrame, DOBAMetadata]:
    """Build season-to-date Deserved Offensive Balance Added rankings."""
    resolved_weights = weights or DOBAWeights()
    notes: list[str] = []
    if pbp_df.empty:
        return _empty_doba_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_games=min_games,
            qualified_team_count=0,
            weights=resolved_weights,
            notes=["PBP data was empty."],
        )

    pbp = _filter_pbp(pbp_df, season=season, through_week=through_week)
    if pbp.empty:
        return _empty_doba_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_games=min_games,
            qualified_team_count=0,
            weights=resolved_weights,
            notes=["No offensive PBP rows matched the requested filter."],
        )

    _ensure_optional_columns(pbp, notes)
    components = _build_team_components(pbp)
    qualified = components[components["games_played"] >= int(min_games)].copy()
    if qualified.empty:
        return _empty_doba_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_games=min_games,
            qualified_team_count=0,
            weights=resolved_weights,
            notes=[f"No teams reached the minimum games threshold ({min_games})."],
        )

    scored = _add_doba_scores(qualified, resolved_weights)
    scored = scored.sort_values(
        ["doba_score", "offensive_success_rate", "team"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    scored["formula_version"] = DOBA_FORMULA_VERSION
    scored = scored[_doba_columns()]
    return scored, _metadata(
        season=season,
        through_week=through_week,
        min_games=min_games,
        qualified_team_count=len(scored),
        weights=resolved_weights,
        notes=notes,
    )


def _filter_pbp(
    pbp_df: pd.DataFrame,
    *,
    season: Optional[int],
    through_week: Optional[int],
) -> pd.DataFrame:
    pbp = pbp_df.copy()
    if season is not None and "season" in pbp.columns:
        pbp = pbp[pbp["season"] == season]
    if through_week is not None and "week" in pbp.columns:
        pbp = pbp[pbp["week"] <= through_week]
    if "posteam" not in pbp.columns:
        return pd.DataFrame()
    pbp = pbp[pbp["posteam"].notna()].copy()
    if {"qb_dropback", "rush_attempt"}.issubset(pbp.columns):
        pbp = pbp[
            (pbp["qb_dropback"].fillna(0.0) == 1.0)
            | (pbp["rush_attempt"].fillna(0.0) == 1.0)
        ].copy()
    return pbp


def _ensure_optional_columns(pbp: pd.DataFrame, notes: list[str]) -> None:
    defaults = {
        "game_id": "unknown",
        "season": 0,
        "week": 0,
        "epa": 0.0,
        "success": 0.0,
        "down": 0.0,
        "yards_gained": 0.0,
        "yardline_100": 100.0,
        "touchdown": 0.0,
        "interception": 0.0,
        "fumble_lost": 0.0,
        "sack": 0.0,
        "qb_dropback": 0.0,
        "rush_attempt": 0.0,
        "game_seconds_remaining": 3600.0,
        "score_differential": 0.0,
    }
    for column, default in defaults.items():
        if column not in pbp.columns:
            pbp[column] = default
            notes.append(f"`{column}` missing; defaulted for DOBA.")


def _build_team_components(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    pbp["is_early_down"] = pbp["down"].isin([1.0, 2.0])
    pbp["is_late_down"] = pbp["down"].isin([3.0, 4.0])
    pbp["is_explosive"] = pbp["yards_gained"].fillna(0.0) >= 20.0
    pbp["is_red_zone"] = pbp["yardline_100"].fillna(100.0) <= 20.0
    pbp["is_negative_play"] = (pbp["epa"].fillna(0.0) < 0.0) | (
        pbp["yards_gained"].fillna(0.0) < 0.0
    )
    pbp["turnover"] = pbp[["interception", "fumble_lost"]].fillna(0.0).max(axis=1)
    pbp["is_garbage_time"] = (pbp["game_seconds_remaining"].fillna(3600.0) <= 900.0) & (
        pbp["score_differential"].fillna(0.0).abs() >= 17.0
    )
    pbp["positive_epa"] = pbp["epa"].clip(lower=0.0)
    pbp["explosive_positive_epa"] = pbp["positive_epa"] * pbp["is_explosive"].astype(
        float
    )
    pbp["garbage_positive_epa"] = pbp["positive_epa"] * pbp["is_garbage_time"].astype(
        float
    )

    grouped = pbp.groupby("posteam", as_index=False).agg(
        team=("posteam", "first"),
        season=("season", "max"),
        through_week=("week", "max"),
        games_played=("game_id", "nunique"),
        plays=("epa", "count"),
        offensive_epa_per_play=("epa", "mean"),
        offensive_success_rate=("success", "mean"),
        early_down_epa=("epa", lambda s: s[pbp.loc[s.index, "is_early_down"]].mean()),
        early_down_success=(
            "success",
            lambda s: s[pbp.loc[s.index, "is_early_down"]].mean(),
        ),
        third_fourth_down_epa=(
            "epa",
            lambda s: s[pbp.loc[s.index, "is_late_down"]].mean(),
        ),
        third_fourth_down_success=(
            "success",
            lambda s: s[pbp.loc[s.index, "is_late_down"]].mean(),
        ),
        explosive_rate=("is_explosive", "mean"),
        red_zone_epa=("epa", lambda s: s[pbp.loc[s.index, "is_red_zone"]].mean()),
        red_zone_td_rate=(
            "touchdown",
            lambda s: s[pbp.loc[s.index, "is_red_zone"]].mean(),
        ),
        negative_play_rate=("is_negative_play", "mean"),
        turnover_rate=("turnover", "mean"),
        sack_rate=("sack", "mean"),
        pass_epa=("epa", lambda s: s[pbp.loc[s.index, "qb_dropback"] == 1.0].mean()),
        rush_epa=("epa", lambda s: s[pbp.loc[s.index, "rush_attempt"] == 1.0].mean()),
        positive_epa=("positive_epa", "sum"),
        explosive_positive_epa=("explosive_positive_epa", "sum"),
        garbage_positive_epa=("garbage_positive_epa", "sum"),
    )
    grouped["early_down_efficiency"] = grouped["early_down_epa"].fillna(
        grouped["early_down_success"]
    )
    grouped["third_fourth_down_efficiency"] = grouped["third_fourth_down_epa"].fillna(
        grouped["third_fourth_down_success"]
    )
    grouped["red_zone_efficiency"] = grouped["red_zone_epa"].fillna(
        grouped["red_zone_td_rate"]
    )
    grouped["pass_run_efficiency_gap"] = (
        grouped["pass_epa"].fillna(0.0) - grouped["rush_epa"].fillna(0.0)
    ).abs()
    grouped["explosive_epa_share"] = grouped["explosive_positive_epa"] / grouped[
        "positive_epa"
    ].replace(0, pd.NA)
    grouped["garbage_time_epa_share"] = grouped["garbage_positive_epa"] / grouped[
        "positive_epa"
    ].replace(0, pd.NA)
    grouped["dependency_penalty"] = (
        pd.to_numeric(grouped["pass_run_efficiency_gap"], errors="coerce").fillna(0.0)
        + pd.to_numeric(grouped["explosive_epa_share"], errors="coerce").fillna(0.0)
        + pd.to_numeric(grouped["garbage_time_epa_share"], errors="coerce").fillna(0.0)
    ) / 3.0

    fill_zero = [
        "early_down_efficiency",
        "third_fourth_down_efficiency",
        "red_zone_efficiency",
        "negative_play_rate",
        "turnover_rate",
        "sack_rate",
        "dependency_penalty",
    ]
    grouped[fill_zero] = (
        grouped[fill_zero].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    return grouped


def _add_doba_scores(team_frame: pd.DataFrame, weights: DOBAWeights) -> pd.DataFrame:
    result = team_frame.copy()
    z_columns = {
        "offensive_epa_per_play_z": "offensive_epa_per_play",
        "offensive_success_rate_z": "offensive_success_rate",
        "early_down_efficiency_z": "early_down_efficiency",
        "explosive_rate_z": "explosive_rate",
        "red_zone_efficiency_z": "red_zone_efficiency",
        "third_fourth_down_efficiency_z": "third_fourth_down_efficiency",
        "negative_play_rate_z": "negative_play_rate",
        "turnover_rate_z": "turnover_rate",
        "dependency_penalty_z": "dependency_penalty",
    }
    for z_col, source_col in z_columns.items():
        result[z_col] = _zscore(result[source_col])

    result["doba_score"] = (
        weights.offensive_epa_per_play * result["offensive_epa_per_play_z"]
        + weights.offensive_success_rate * result["offensive_success_rate_z"]
        + weights.early_down_efficiency * result["early_down_efficiency_z"]
        + weights.explosive_rate * result["explosive_rate_z"]
        + weights.red_zone_efficiency * result["red_zone_efficiency_z"]
        + weights.third_fourth_down_efficiency
        * result["third_fourth_down_efficiency_z"]
        + weights.negative_play_rate * result["negative_play_rate_z"]
        + weights.turnover_rate * result["turnover_rate_z"]
        + weights.dependency_penalty * result["dependency_penalty_z"]
    )
    result["label"] = result.apply(_doba_label, axis=1)
    result["public_tier"] = result.apply(_public_tier, axis=1)
    result["why_flagged"] = result.apply(_why_flagged, axis=1)
    return result


def _doba_label(row: pd.Series) -> str:
    doba = float(row["doba_score"])
    success = float(row["offensive_success_rate_z"])
    explosive = float(row["explosive_rate_z"])
    negative = float(row["negative_play_rate_z"])
    dependency = float(row["dependency_penalty_z"])
    efficiency = float(row["offensive_epa_per_play_z"])

    if doba >= 1.25 and success >= 0.50 and efficiency >= 0.50 and dependency < 1.25:
        return "Elite Machine"
    if doba <= -1.00 and efficiency <= -0.50 and success <= -0.50:
        return "Broken Offense"
    if (
        -0.10 <= doba < 1.00
        and dependency >= 0.75
        and (efficiency >= -0.25 or doba > 0.0)
    ):
        return "Inflated Production"
    if doba >= -0.50 and explosive >= 0.50 and (success < 0.25 or negative >= 0.50):
        return "Volatile Threat"
    if doba >= 0.40 and success >= -0.25:
        return "Sustainable Threat"
    if -0.40 <= doba <= 0.40 and dependency < 0.75:
        return "Functional Offense"
    if -1.00 < doba < -0.40 and efficiency < 0.0:
        return "Stuck in Mud"
    if doba <= -1.00:
        return "Broken Offense"
    if dependency >= 0.75:
        return "Inflated Production"
    return "Functional Offense"


def _public_tier(row: pd.Series) -> str:
    label = str(row["label"])
    if label == "Elite Machine":
        return "Elite"
    if label == "Sustainable Threat":
        return "Positive"
    if label in {"Volatile Threat", "Inflated Production"}:
        return "Fragile"
    if label in {"Stuck in Mud", "Broken Offense"}:
        return "Negative"
    return "Neutral"


def _why_flagged(row: pd.Series) -> str:
    positive = _top_driver(
        row,
        {
            "offensive_epa_per_play_z": "top driver is EPA per play",
            "offensive_success_rate_z": "top driver is success rate",
            "early_down_efficiency_z": "top driver is early-down efficiency",
            "explosive_rate_z": "top driver is explosive rate",
            "red_zone_efficiency_z": "top driver is red-zone efficiency",
            "third_fourth_down_efficiency_z": "top driver is late-down efficiency",
        },
        highest=True,
    )
    negative = _top_driver(
        row,
        {
            "negative_play_rate_z": "main concern is negative-play rate",
            "turnover_rate_z": "main concern is turnover rate",
            "dependency_penalty_z": "main concern is dependency/fragility",
            "offensive_success_rate_z": "main concern is weak success rate",
            "early_down_efficiency_z": "main concern is early-down efficiency",
        },
        highest=False,
    )
    return f"{positive}; {negative}; { _verdict_phrase(str(row['label'])) }"


def _top_driver(
    row: pd.Series,
    labels: dict[str, str],
    *,
    highest: bool,
) -> str:
    values = {column: float(row[column]) for column in labels}
    if highest:
        column = max(values, key=values.get)
        if values[column] < 0.25:
            return "no standout positive driver"
        return labels[column]

    bad_when_high = {"negative_play_rate_z", "turnover_rate_z", "dependency_penalty_z"}
    adjusted = {
        column: (value if column in bad_when_high else -value)
        for column, value in values.items()
    }
    column = max(adjusted, key=adjusted.get)
    if adjusted[column] < 0.25:
        return "no major negative or dependency driver"
    return labels[column]


def _verdict_phrase(label: str) -> str:
    verdicts = {
        "Elite Machine": "verdict: offense is efficient and repeatable",
        "Sustainable Threat": "verdict: offense has stable paths to value",
        "Functional Offense": "verdict: offense is usable but not especially dangerous",
        "Volatile Threat": "verdict: offense is dangerous but fragile",
        "Inflated Production": "verdict: production may be context or dependency driven",
        "Stuck in Mud": "verdict: offense lacks enough efficient, repeatable value",
        "Broken Offense": "verdict: offense is not creating enough reliable value",
    }
    return verdicts.get(label, "verdict: mixed offensive profile")


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(numeric.std(ddof=0))
    if std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (numeric - float(numeric.mean())) / std


def _metadata(
    *,
    season: Optional[int],
    through_week: Optional[int],
    min_games: int,
    qualified_team_count: int,
    weights: DOBAWeights,
    notes: list[str],
) -> DOBAMetadata:
    return DOBAMetadata(
        metric="doba",
        formula_version=DOBA_FORMULA_VERSION,
        season=season,
        through_week=through_week,
        min_games=min_games,
        qualified_team_count=qualified_team_count,
        weights=asdict(weights),
        built_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        notes=notes,
    )


def _doba_columns() -> list[str]:
    return [
        "rank",
        "team",
        "season",
        "through_week",
        "games_played",
        "offensive_epa_per_play",
        "offensive_success_rate",
        "early_down_efficiency",
        "third_fourth_down_efficiency",
        "explosive_rate",
        "red_zone_efficiency",
        "negative_play_rate",
        "turnover_rate",
        "dependency_penalty",
        "doba_score",
        "public_tier",
        "label",
        "why_flagged",
        "formula_version",
    ]


def _empty_doba_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_doba_columns())
