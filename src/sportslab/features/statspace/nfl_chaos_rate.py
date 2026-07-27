"""Experimental StatSpace NFL Chaos Rate calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

CHAOS_RATE_FORMULA_VERSION = 1


@dataclass(frozen=True)
class ChaosRateWeights:
    """Weights for StatSpace defensive Chaos Rate."""

    negative_epa_forced_rate: float = 0.25
    defensive_epa_per_play_allowed_inverted: float = 0.20
    success_rate_allowed_inverted: float = 0.15
    sack_rate: float = 0.15
    turnover_forced_rate: float = 0.10
    third_fourth_down_stop_rate: float = 0.10
    explosive_rate_allowed_inverted: float = 0.10
    penalty_first_down_rate_allowed: float = -0.10


@dataclass(frozen=True)
class ChaosRateMetadata:
    """Machine-readable metadata for one Chaos Rate build."""

    metric: str
    formula_version: int
    season: Optional[int]
    through_week: Optional[int]
    min_games: int
    qualified_team_count: int
    weights: dict[str, float]
    built_at: str
    notes: list[str]


def build_chaos_rate(
    pbp_df: pd.DataFrame,
    *,
    season: Optional[int] = None,
    through_week: Optional[int] = None,
    min_games: int = 4,
    weights: ChaosRateWeights | None = None,
) -> tuple[pd.DataFrame, ChaosRateMetadata]:
    """Build season-to-date defensive Chaos Rate rankings."""
    resolved_weights = weights or ChaosRateWeights()
    notes: list[str] = []
    if pbp_df.empty:
        return _empty_chaos_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_games=min_games,
            qualified_team_count=0,
            weights=resolved_weights,
            notes=["PBP data was empty."],
        )

    pbp = _filter_pbp(pbp_df, season=season, through_week=through_week)
    if pbp.empty:
        return _empty_chaos_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_games=min_games,
            qualified_team_count=0,
            weights=resolved_weights,
            notes=["No defensive PBP rows matched the requested filter."],
        )

    _ensure_optional_columns(pbp, notes)
    components = _build_team_components(pbp)
    qualified = components[components["games_played"] >= int(min_games)].copy()
    if qualified.empty:
        note = f"No teams reached the minimum games threshold ({min_games})."
        return _empty_chaos_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_games=min_games,
            qualified_team_count=0,
            weights=resolved_weights,
            notes=[note],
        )

    scored = _add_chaos_scores(qualified, resolved_weights)
    scored = scored.sort_values(
        ["chaos_rate", "negative_epa_forced_rate", "team"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    scored["formula_version"] = CHAOS_RATE_FORMULA_VERSION
    scored = scored[_chaos_columns()]
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
    if "defteam" not in pbp.columns:
        return pd.DataFrame()
    pbp = pbp[pbp["defteam"].notna()].copy()
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
        "sack": 0.0,
        "interception": 0.0,
        "fumble_lost": 0.0,
        "first_down_penalty": 0.0,
        "penalty_team": "",
        "qb_dropback": 0.0,
        "rush_attempt": 0.0,
    }
    for column, default in defaults.items():
        if column not in pbp.columns:
            pbp[column] = default
            notes.append(f"`{column}` missing; defaulted for Chaos Rate.")


def _build_team_components(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    pbp["is_late_down"] = pbp["down"].isin([3.0, 4.0])
    pbp["negative_epa_forced"] = pbp["epa"].fillna(0.0) < 0.0
    pbp["explosive_allowed"] = pbp["yards_gained"].fillna(0.0) >= 20.0
    pbp["turnover_forced"] = (
        pbp[["interception", "fumble_lost"]].fillna(0.0).max(axis=1)
    )
    pbp["late_down_stop"] = pbp["is_late_down"] & (
        pbp["success"].fillna(0.0) < 1.0
    )  # noqa: E501
    pbp["penalty_first_down_allowed"] = (
        pbp["first_down_penalty"].fillna(0.0) == 1.0
    ) & (pbp["penalty_team"].fillna("") == pbp["defteam"].fillna(""))

    grouped = pbp.groupby("defteam", as_index=False).agg(
        team=("defteam", "first"),
        season=("season", "max"),
        through_week=("week", "max"),
        games_played=("game_id", "nunique"),
        plays=("epa", "count"),
        defensive_epa_per_play_allowed=("epa", "mean"),
        defensive_success_rate_allowed=("success", "mean"),
        negative_epa_forced_rate=("negative_epa_forced", "mean"),
        sack_rate=("sack", "mean"),
        turnover_forced_rate=("turnover_forced", "mean"),
        explosive_rate_allowed=("explosive_allowed", "mean"),
        third_fourth_down_stop_rate=(
            "late_down_stop",
            lambda s: s[pbp.loc[s.index, "is_late_down"]].mean(),
        ),
        penalty_first_down_rate_allowed=("penalty_first_down_allowed", "mean"),
    )
    fill_zero = [
        "defensive_epa_per_play_allowed",
        "defensive_success_rate_allowed",
        "negative_epa_forced_rate",
        "sack_rate",
        "turnover_forced_rate",
        "explosive_rate_allowed",
        "third_fourth_down_stop_rate",
        "penalty_first_down_rate_allowed",
    ]
    grouped[fill_zero] = (
        grouped[fill_zero].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    return grouped


def _add_chaos_scores(
    team_frame: pd.DataFrame, weights: ChaosRateWeights
) -> pd.DataFrame:
    result = team_frame.copy()
    result["defensive_epa_per_play_allowed_inverted"] = -result[
        "defensive_epa_per_play_allowed"
    ]
    result["success_rate_allowed_inverted"] = -result[
        "defensive_success_rate_allowed"
    ]  # noqa: E501
    result["explosive_rate_allowed_inverted"] = -result[
        "explosive_rate_allowed"
    ]  # noqa: E501

    z_columns = {
        "negative_epa_forced_rate_z": "negative_epa_forced_rate",
        "defensive_epa_per_play_allowed_inverted_z": (
            "defensive_epa_per_play_allowed_inverted"
        ),
        "success_rate_allowed_inverted_z": "success_rate_allowed_inverted",
        "sack_rate_z": "sack_rate",
        "turnover_forced_rate_z": "turnover_forced_rate",
        "third_fourth_down_stop_rate_z": "third_fourth_down_stop_rate",
        "explosive_rate_allowed_inverted_z": "explosive_rate_allowed_inverted",
        "penalty_first_down_rate_allowed_z": "penalty_first_down_rate_allowed",
    }
    for z_col, source_col in z_columns.items():
        result[z_col] = _zscore(result[source_col])

    result["chaos_rate"] = (
        weights.negative_epa_forced_rate * result["negative_epa_forced_rate_z"]
        + weights.defensive_epa_per_play_allowed_inverted
        * result["defensive_epa_per_play_allowed_inverted_z"]
        + weights.success_rate_allowed_inverted
        * result["success_rate_allowed_inverted_z"]
        + weights.sack_rate * result["sack_rate_z"]
        + weights.turnover_forced_rate * result["turnover_forced_rate_z"]
        + weights.third_fourth_down_stop_rate
        * result["third_fourth_down_stop_rate_z"]  # noqa: E501
        + weights.explosive_rate_allowed_inverted
        * result["explosive_rate_allowed_inverted_z"]
        + weights.penalty_first_down_rate_allowed
        * result["penalty_first_down_rate_allowed_z"]
    )
    result["label"] = result.apply(_chaos_label, axis=1)
    result["public_tier"] = result.apply(_public_tier, axis=1)
    result["why_flagged"] = result.apply(_why_flagged, axis=1)
    return result


def _chaos_label(row: pd.Series) -> str:
    chaos = float(row["chaos_rate"])
    negative = float(row["negative_epa_forced_rate_z"])
    epa = float(row["defensive_epa_per_play_allowed_inverted_z"])
    success = float(row["success_rate_allowed_inverted_z"])
    sack = float(row["sack_rate_z"])
    turnover = float(row["turnover_forced_rate_z"])
    explosive = float(row["explosive_rate_allowed_inverted_z"])
    penalty = float(row["penalty_first_down_rate_allowed_z"])
    disruption = max(negative, sack, turnover)
    pressure_disruption = max(sack, turnover)
    efficiency = max(epa, success)
    poor_efficiency = epa <= -0.50 and success <= -0.50
    poor_disruption = disruption <= 0.25

    if chaos >= 1.20 and negative >= 0.50 and efficiency >= 0.50:
        return "Game Wrecker"
    if chaos <= -1.00 and poor_efficiency and poor_disruption:
        return "Broken Defense"
    if (
        (sack >= 0.75 or turnover >= 0.75)
        and (epa < -0.25 or success < -0.25 or explosive < -0.75)
        and chaos >= -0.50
    ):
        return "Fake Pressure"
    if efficiency >= 0.25 and pressure_disruption < 0.50 and chaos >= -0.30:
        return "Bend-Don't-Break"
    if chaos >= 0.35 and (pressure_disruption >= 0.50 or negative >= 0.75):
        return "Pressure Cooker"
    if disruption < 0.0 and chaos > -1.00:
        return "Passive Defense"
    if chaos <= -1.00:
        return "Broken Defense"
    if penalty >= 1.25 and chaos < 0.35:
        return "Passive Defense"
    if chaos >= 0.20 and pressure_disruption >= 0.50:
        return "Pressure Cooker"
    if efficiency >= 0.25:
        return "Bend-Don't-Break"
    return "Passive Defense"


def _public_tier(row: pd.Series) -> str:
    label = str(row["label"])
    if label == "Game Wrecker":
        return "Elite Disruption"
    if label == "Pressure Cooker":
        return "Disruptive"
    if label == "Bend-Don't-Break":
        return "Stable"
    if label == "Fake Pressure":
        return "Volatile"
    if label == "Broken Defense":
        return "Broken"
    return "Passive"


def _why_flagged(row: pd.Series) -> str:
    positive = _top_driver(
        row,
        {
            "negative_epa_forced_rate_z": "top driver is negative EPA forced",
            "defensive_epa_per_play_allowed_inverted_z": (
                "top driver is EPA prevention"
            ),
            "success_rate_allowed_inverted_z": (
                "top driver is success-rate prevention"
            ),
            "sack_rate_z": "top driver is sack rate",
            "turnover_forced_rate_z": "top driver is turnovers forced",
            "third_fourth_down_stop_rate_z": "top driver is late-down stops",
            "explosive_rate_allowed_inverted_z": (
                "top driver is explosive prevention"
            ),  # noqa: E501
        },
        highest=True,
    )
    concern = _top_driver(
        row,
        {
            "negative_epa_forced_rate_z": (
                "main concern is limited negative plays"
            ),  # noqa: E501
            "defensive_epa_per_play_allowed_inverted_z": (
                "main concern is EPA allowed"
            ),
            "success_rate_allowed_inverted_z": (
                "main concern is success rate allowed"
            ),  # noqa: E501
            "sack_rate_z": "main concern is limited sack production",
            "turnover_forced_rate_z": "main concern is limited turnovers",
            "explosive_rate_allowed_inverted_z": (
                "main concern is explosives allowed"
            ),  # noqa: E501
            "penalty_first_down_rate_allowed_z": (
                "main concern is penalty first downs allowed"
            ),
        },
        highest=False,
    )
    return f"{positive}; {concern}; {_verdict_phrase(str(row['label']))}"


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

    bad_when_high = {"penalty_first_down_rate_allowed_z"}
    adjusted = {
        column: (value if column in bad_when_high else -value)
        for column, value in values.items()
    }
    column = max(adjusted, key=adjusted.get)
    if adjusted[column] < 0.25:
        return "no major negative driver"
    return labels[column]


def _verdict_phrase(label: str) -> str:
    verdicts = {
        "Game Wrecker": (
            "verdict: defense creates frequent drive-altering chaos"
        ),  # noqa: E501
        "Pressure Cooker": (
            "verdict: defense generates disruption but has tradeoffs"
        ),  # noqa: E501
        "Bend-Don't-Break": (
            "verdict: defense prevents efficiency without constant heat"
        ),
        "Fake Pressure": (
            "verdict: splash plays are masking weaker down-to-down quality"
        ),
        "Passive Defense": "verdict: defense lacks enough disruptive pressure",
        "Broken Defense": "verdict: defense is neither efficient nor disruptive",  # noqa: E501
    }
    return verdicts.get(label, "verdict: mixed defensive disruption profile")


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
    weights: ChaosRateWeights,
    notes: list[str],
) -> ChaosRateMetadata:
    return ChaosRateMetadata(
        metric="chaos_rate",
        formula_version=CHAOS_RATE_FORMULA_VERSION,
        season=season,
        through_week=through_week,
        min_games=min_games,
        qualified_team_count=qualified_team_count,
        weights=asdict(weights),
        built_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        notes=notes,
    )


def _chaos_columns() -> list[str]:
    return [
        "rank",
        "team",
        "season",
        "through_week",
        "games_played",
        "defensive_epa_per_play_allowed",
        "defensive_success_rate_allowed",
        "negative_epa_forced_rate",
        "sack_rate",
        "turnover_forced_rate",
        "explosive_rate_allowed",
        "third_fourth_down_stop_rate",
        "penalty_first_down_rate_allowed",
        "chaos_rate",
        "public_tier",
        "label",
        "why_flagged",
        "formula_version",
    ]


def _empty_chaos_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_chaos_columns())
