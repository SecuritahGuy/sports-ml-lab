"""Experimental StatSpace NFL QB Lift Index calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

QB_LIFT_FORMULA_VERSION = 1
QB_LIFT_CALIBRATION_BASELINE_VERSION = "qb_lift_static_v1"
QB_LIFT_OPPONENT_ADJUSTMENT_VERSION = "def_pass_epa_allowed_v1"
QB_LIFT_COMPONENT_COLUMNS = [
    "epa_per_dropback",
    "cpoe",
    "third_fourth_down_epa",
    "pressure_proxy",
    "scramble_epa",
    "yac_dependency",
    "sack_rate",
    "garbage_time_inflation",
]


@dataclass(frozen=True)
class QBLiftWeights:
    """Weights for the public-data QB Lift Index formula."""

    epa_per_dropback: float = 0.35
    cpoe: float = 0.25
    third_fourth_down_epa: float = 0.20
    pressure_proxy: float = 0.15
    scramble_value: float = 0.15
    yac_dependency: float = -0.20
    sack_rate: float = -0.15
    garbage_time_inflation: float = -0.10


@dataclass(frozen=True)
class QBLiftMetadata:
    """Machine-readable metadata for one QB Lift build."""

    metric: str
    formula_version: int
    season: Optional[int]
    through_week: Optional[int]
    min_dropbacks: int
    qualified_qb_count: int
    weights: dict[str, float]
    built_at: str
    notes: list[str]


def build_qb_lift_index(
    pbp_df: pd.DataFrame,
    *,
    season: Optional[int] = None,
    through_week: Optional[int] = None,
    min_dropbacks: int = 100,
    weights: QBLiftWeights | None = None,
    calibration_artifact: Optional[pd.DataFrame] = None,
    opponent_adjusted: bool = False,
) -> tuple[pd.DataFrame, QBLiftMetadata]:
    """Build a season-to-date QB Lift Index table from nflverse PBP."""
    resolved_weights = weights or QBLiftWeights()
    notes: list[str] = []
    if pbp_df.empty:
        return _empty_qb_lift_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_dropbacks=min_dropbacks,
            qualified_qb_count=0,
            weights=resolved_weights,
            notes=["PBP data was empty."],
        )

    pbp = _filter_pbp(pbp_df, season=season, through_week=through_week)
    if pbp.empty:
        return _empty_qb_lift_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_dropbacks=min_dropbacks,
            qualified_qb_count=0,
            weights=resolved_weights,
            notes=["No PBP rows matched the requested filter."],
        )

    _ensure_optional_columns(pbp, notes)
    _validate_schema(pbp, notes)
    qb_frame = _build_qb_components(pbp)
    qualified = qb_frame[qb_frame["dropbacks"] >= int(min_dropbacks)].copy()
    if qualified.empty:
        return _empty_qb_lift_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_dropbacks=min_dropbacks,
            qualified_qb_count=0,
            weights=resolved_weights,
            notes=[f"No QBs reached the minimum dropback threshold ({min_dropbacks})."],
        )

    calibration_lookup = _calibration_lookup(calibration_artifact)
    scored = _add_qb_lift_scores(
        qualified,
        resolved_weights,
        calibration_lookup=calibration_lookup,
        opponent_adjusted=opponent_adjusted,
    )
    scored = scored.sort_values(
        ["qb_lift_index", "dropbacks", "player"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    scored = scored[_qb_lift_columns()]
    return scored, _metadata(
        season=season,
        through_week=through_week,
        min_dropbacks=min_dropbacks,
        qualified_qb_count=len(scored),
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
    if "qb_dropback" not in pbp.columns:
        return pd.DataFrame()
    pbp = pbp[pbp["qb_dropback"].fillna(0.0) == 1.0].copy()
    return pbp[pbp["passer_player_id"].notna()].copy()


def _ensure_optional_columns(pbp: pd.DataFrame, notes: list[str]) -> None:
    defaults = {
        "passer_player_name": "Unknown",
        "posteam": "UNK",
        "epa": 0.0,
        "success": 0.0,
        "cpoe": 0.0,
        "down": 0.0,
        "qb_scramble": 0.0,
        "sack": 0.0,
        "yards_after_catch": 0.0,
        "passing_yards": 0.0,
        "complete_pass": 0.0,
        "game_seconds_remaining": 3600.0,
        "score_differential": 0.0,
    }
    for column, default in defaults.items():
        if column not in pbp.columns:
            pbp[column] = default
            notes.append(f"`{column}` missing; defaulted for QB Lift.")


def _build_qb_components(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    if "defteam" in pbp.columns:
        defense_strength = pbp.groupby("defteam", as_index=True)["epa"].mean().to_dict()
        pbp["opponent_def_pass_epa_allowed"] = pbp["defteam"].map(defense_strength)
    else:
        pbp["opponent_def_pass_epa_allowed"] = 0.0
    pbp["is_late_down"] = pbp["down"].isin([3.0, 4.0])
    pbp["is_scramble"] = pbp["qb_scramble"].fillna(0.0) == 1.0
    pbp["is_garbage_time"] = (pbp["game_seconds_remaining"].fillna(3600.0) <= 900.0) & (
        pbp["score_differential"].fillna(0.0).abs() >= 17.0
    )
    pbp["yac_completion_yards"] = pbp["yards_after_catch"].fillna(0.0) * (
        pbp["complete_pass"].fillna(0.0) == 1.0
    ).astype(float)

    grouped = pbp.groupby("passer_player_id", as_index=False).agg(
        player=("passer_player_name", _mode_or_first),
        team=("posteam", _mode_or_first),
        season=("season", "max"),
        through_week=("week", "max"),
        dropbacks=("qb_dropback", "count"),
        epa_total=("epa", "sum"),
        epa_per_dropback=("epa", "mean"),
        cpoe=("cpoe", "mean"),
        success_rate=("success", "mean"),
        late_down_epa_total=(
            "epa",
            lambda s: s[pbp.loc[s.index, "is_late_down"]].sum(),
        ),
        late_down_dropbacks=("is_late_down", "sum"),
        scramble_epa_total=("epa", lambda s: s[pbp.loc[s.index, "is_scramble"]].sum()),
        scramble_dropbacks=("is_scramble", "sum"),
        sacks=("sack", "sum"),
        yac_yards=("yac_completion_yards", "sum"),
        passing_yards=("passing_yards", "sum"),
        opponent_def_pass_epa_allowed=("opponent_def_pass_epa_allowed", "mean"),
        garbage_time_dropbacks=("is_garbage_time", "sum"),
        garbage_time_epa=(
            "epa",
            lambda s: s[pbp.loc[s.index, "is_garbage_time"]].sum(),
        ),
    )
    grouped["third_fourth_down_epa"] = grouped["late_down_epa_total"] / grouped[
        "late_down_dropbacks"
    ].replace(0, pd.NA)
    grouped["scramble_epa"] = grouped["scramble_epa_total"] / grouped[
        "dropbacks"
    ].replace(0, pd.NA)
    grouped["sack_rate"] = grouped["sacks"] / grouped["dropbacks"].replace(0, pd.NA)
    grouped["pressure_proxy"] = 1.0 - grouped["sack_rate"]
    grouped["yac_dependency"] = grouped["yac_yards"] / grouped["passing_yards"].replace(
        0, pd.NA
    )
    grouped["garbage_time_share"] = grouped["garbage_time_dropbacks"] / grouped[
        "dropbacks"
    ].replace(0, pd.NA)
    grouped["garbage_time_inflation"] = (
        grouped["garbage_time_share"]
        * grouped["garbage_time_epa"].clip(lower=0.0)
        / grouped["dropbacks"].replace(0, pd.NA)
    )

    fill_zero = [
        "third_fourth_down_epa",
        "scramble_epa",
        "sack_rate",
        "pressure_proxy",
        "yac_dependency",
        "garbage_time_share",
        "garbage_time_inflation",
    ]
    grouped[fill_zero] = (
        grouped[fill_zero].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    grouped["opponent_def_pass_epa_allowed"] = pd.to_numeric(
        grouped["opponent_def_pass_epa_allowed"], errors="coerce"
    ).fillna(0.0)
    return grouped


def _add_qb_lift_scores(
    qb_frame: pd.DataFrame,
    weights: QBLiftWeights,
    *,
    calibration_lookup: Optional[dict[str, tuple[float, float]]] = None,
    opponent_adjusted: bool = False,
) -> pd.DataFrame:
    result = qb_frame.copy()
    z_columns = {
        "epa_per_dropback_z": "epa_per_dropback",
        "cpoe_z": "cpoe",
        "third_fourth_down_epa_z": "third_fourth_down_epa",
        "pressure_proxy_z": "pressure_proxy",
        "scramble_value_z": "scramble_epa",
        "yac_dependency_z": "yac_dependency",
        "sack_rate_z": "sack_rate",
        "garbage_time_inflation_z": "garbage_time_inflation",
    }
    for z_col, source_col in z_columns.items():
        if calibration_lookup and source_col in calibration_lookup:
            mean, std = calibration_lookup[source_col]
            result[z_col] = _zscore_against_baseline(result[source_col], mean=mean, std=std)
        else:
            result[z_col] = _zscore(result[source_col])

    result["qb_lift_index_raw"] = (
        weights.epa_per_dropback * result["epa_per_dropback_z"]
        + weights.cpoe * result["cpoe_z"]
        + weights.third_fourth_down_epa * result["third_fourth_down_epa_z"]
        + weights.pressure_proxy * result["pressure_proxy_z"]
        + weights.scramble_value * result["scramble_value_z"]
        + weights.yac_dependency * result["yac_dependency_z"]
        + weights.sack_rate * result["sack_rate_z"]
        + weights.garbage_time_inflation * result["garbage_time_inflation_z"]
    )
    result["qb_lift_index"] = result["qb_lift_index_raw"]

    result["opponent_def_pass_epa_allowed_z"] = _zscore(
        result["opponent_def_pass_epa_allowed"]
    )
    if opponent_adjusted:
        result["qb_lift_index_opp_adjusted"] = result["qb_lift_index_raw"] - (
            0.20 * result["opponent_def_pass_epa_allowed_z"]
        )
        result["opponent_adjustment_version"] = QB_LIFT_OPPONENT_ADJUSTMENT_VERSION
    else:
        result["qb_lift_index_opp_adjusted"] = result["qb_lift_index_raw"]
        result["opponent_adjustment_version"] = "none"

    result["support_dependency_score"] = (
        0.60 * result["yac_dependency_z"]
        - 0.25 * result["pressure_proxy_z"]
        - 0.15 * result["scramble_value_z"]
    )
    result["support_dependency_label"] = result["support_dependency_score"].apply(
        _support_dependency_label
    )
    result["label"] = result.apply(_qb_lift_label, axis=1)
    result["public_tier"] = result.apply(_public_tier, axis=1)
    result["why_flagged"] = result.apply(_qb_lift_why, axis=1)
    return result


def _qb_lift_label(row: pd.Series) -> str:
    lift = float(row["qb_lift_index"])
    support = float(row["support_dependency_score"])
    sack_risk = float(row["sack_rate_z"])
    scramble = float(row["scramble_value_z"])
    efficiency_bad = (
        float(row["epa_per_dropback_z"]) <= -0.50
        or float(row["third_fourth_down_epa_z"]) <= -0.75
        or float(row["cpoe_z"]) <= -0.75
    )
    raw_efficiency_not_terrible = float(row["epa_per_dropback_z"]) > -0.75

    if lift <= -0.75 and efficiency_bad:
        return "Anchor"
    if lift >= 1.00 and support < 0.75:
        return "Elevator"
    if lift >= 0.25 and (sack_risk >= 0.75 or scramble >= 0.75):
        return "Volatile Creator"
    if support >= 0.75 and lift < 0.75 and lift > -0.75 and raw_efficiency_not_terrible:
        return "Protected Passenger"
    if lift >= 0.50 and support >= 0.25:
        return "System Plus"
    if support >= 0.75:
        return "Supported Starter"
    if lift >= 0.50:
        return "System Plus"
    return "Supported Starter"


def _public_tier(row: pd.Series) -> str:
    lift = float(row["qb_lift_index"])
    support = float(row["support_dependency_score"])
    if lift >= 1.00:
        return "High Lift"
    if lift >= 0.50:
        return "Positive Lift"
    if lift <= -0.75:
        return "Negative Lift"
    if support >= 0.75:
        return "Support Dependent"
    return "Neutral"


def _support_dependency_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


def _qb_lift_why(row: pd.Series) -> str:
    positive = _top_driver(
        row,
        {
            "epa_per_dropback_z": "best positive driver is EPA per dropback",
            "cpoe_z": "best positive driver is accuracy over expectation",
            "third_fourth_down_epa_z": "best positive driver is late-down value",
            "pressure_proxy_z": "best positive driver is sack avoidance",
            "scramble_value_z": "best positive driver is scramble value",
        },
        highest=True,
    )
    negative = _top_driver(
        row,
        {
            "yac_dependency_z": "main support concern is YAC dependency",
            "sack_rate_z": "main negative driver is sack rate",
            "garbage_time_inflation_z": "main support concern is garbage-time production",
            "epa_per_dropback_z": "main negative driver is EPA per dropback",
            "cpoe_z": "main negative driver is accuracy over expectation",
            "third_fourth_down_epa_z": "main negative driver is late-down value",
        },
        highest=False,
    )
    verdict = _verdict_phrase(str(row["label"]))
    return f"{positive}; {negative}; {verdict}"


def _top_driver(
    row: pd.Series,
    labels: dict[str, str],
    *,
    highest: bool,
) -> str:
    values = {column: float(row[column]) for column in labels}
    if highest:
        column = max(values, key=values.get)
        value = values[column]
        if value < 0.25:
            return "no standout positive driver"
        return labels[column]

    support_columns = {
        "yac_dependency_z",
        "sack_rate_z",
        "garbage_time_inflation_z",
    }
    adjusted = {
        column: (value if column in support_columns else -value)
        for column, value in values.items()
    }
    column = max(adjusted, key=adjusted.get)
    value = adjusted[column]
    if value < 0.25:
        return "no major negative or support-dependency driver"
    return labels[column]


def _verdict_phrase(label: str) -> str:
    verdicts = {
        "Elevator": "verdict: QB is clearly lifting the offense",
        "System Plus": "verdict: efficient QB in a helpful environment",
        "Volatile Creator": "verdict: creates value but adds volatility",
        "Protected Passenger": "verdict: production is highly support dependent",
        "Supported Starter": "verdict: starter-level profile without a strong lift signal",
        "Anchor": "verdict: QB value is dragging the offense down",
    }
    return verdicts.get(label, "verdict: mixed QB profile")


def _mode_or_first(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return "Unknown"
    modes = clean.mode()
    if not modes.empty:
        return str(modes.iloc[0])
    return str(clean.iloc[0])


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(numeric.std(ddof=0))
    if std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (numeric - float(numeric.mean())) / std


def _zscore_against_baseline(
    series: pd.Series,
    *,
    mean: float,
    std: float,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    safe_std = float(std) if abs(float(std)) >= 1e-12 else 1.0
    return (numeric - float(mean)) / safe_std


def _calibration_lookup(
    artifact: Optional[pd.DataFrame],
) -> dict[str, tuple[float, float]]:
    if artifact is None or artifact.empty:
        return {}
    required = {"component_name", "mean", "std"}
    if not required.issubset(set(artifact.columns)):
        return {}
    lookup: dict[str, tuple[float, float]] = {}
    for row in artifact.itertuples(index=False):
        component = str(getattr(row, "component_name"))
        lookup[component] = (
            float(getattr(row, "mean")),
            float(getattr(row, "std")),
        )
    return lookup


def build_qb_lift_calibration_artifact(
    pbp_df: pd.DataFrame,
    *,
    test_season: int,
    min_dropbacks: int = 100,
    baseline_version: str = QB_LIFT_CALIBRATION_BASELINE_VERSION,
) -> pd.DataFrame:
    """Build walk-forward-safe static calibration baselines for QB Lift components.

    Calibration rows are computed using seasons strictly before ``test_season``.
    """
    base_columns = [
        "baseline_version",
        "calibration_start_season",
        "calibration_end_season",
        "component_name",
        "mean",
        "std",
        "sample_size",
        "min_dropbacks",
        "created_at",
    ]
    if pbp_df.empty or "season" not in pbp_df.columns:
        return pd.DataFrame(columns=base_columns)

    source = pbp_df[pbp_df["season"] < int(test_season)].copy()
    if source.empty:
        return pd.DataFrame(columns=base_columns)

    notes: list[str] = []
    _ensure_optional_columns(source, notes)
    _validate_schema(source, notes)
    source = _filter_pbp(source, season=None, through_week=None)
    if source.empty:
        return pd.DataFrame(columns=base_columns)

    components = _build_qb_components(source)
    components = components[components["dropbacks"] >= int(min_dropbacks)].copy()
    if components.empty:
        return pd.DataFrame(columns=base_columns)

    start_season = int(source["season"].min())
    end_season = int(source["season"].max())
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rows: list[dict[str, object]] = []
    for component in QB_LIFT_COMPONENT_COLUMNS:
        series = pd.to_numeric(components[component], errors="coerce").dropna()
        rows.append(
            {
                "baseline_version": baseline_version,
                "calibration_start_season": start_season,
                "calibration_end_season": end_season,
                "component_name": component,
                "mean": float(series.mean()) if not series.empty else 0.0,
                "std": float(series.std(ddof=0)) if not series.empty else 1.0,
                "sample_size": int(series.shape[0]),
                "min_dropbacks": int(min_dropbacks),
                "created_at": created_at,
            }
        )
    return pd.DataFrame(rows, columns=base_columns)


def _metadata(
    *,
    season: Optional[int],
    through_week: Optional[int],
    min_dropbacks: int,
    qualified_qb_count: int,
    weights: QBLiftWeights,
    notes: list[str],
) -> QBLiftMetadata:
    return QBLiftMetadata(
        metric="qb_lift_index",
        formula_version=QB_LIFT_FORMULA_VERSION,
        season=season,
        through_week=through_week,
        min_dropbacks=min_dropbacks,
        qualified_qb_count=qualified_qb_count,
        weights=asdict(weights),
        built_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        notes=notes,
    )


def _qb_lift_columns() -> list[str]:
    return [
        "rank",
        "player",
        "team",
        "season",
        "through_week",
        "dropbacks",
        "epa_per_dropback",
        "cpoe",
        "success_rate",
        "third_fourth_down_epa",
        "scramble_epa",
        "sack_rate",
        "yac_dependency",
        "garbage_time_share",
        "opponent_def_pass_epa_allowed",
        "qb_lift_index_raw",
        "qb_lift_index",
        "qb_lift_index_opp_adjusted",
        "opponent_adjustment_version",
        "support_dependency_score",
        "support_dependency_label",
        "label",
        "public_tier",
        "why_flagged",
    ]


def _empty_qb_lift_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_qb_lift_columns())


def _validate_schema(pbp: pd.DataFrame, notes: list[str]) -> None:
    """Validate schema for required and optional columns."""
    hard_required = {
        "season", "week", "game_id", "posteam", "defteam", "play_type", "epa"
    }
    strong_required_for_qb_index = {
        "cpoe", "air_yards", "yards_after_catch", "wp", "down", "ydstogo",
        "passer_player_id", "passer_player_name"
    }
    optional = {
        "weather", "roof", "surface", "spread_line", "total_line"
    }

    missing_hard = hard_required - set(pbp.columns)
    if missing_hard:
        raise ValueError(f"Missing required columns: {missing_hard}")

    missing_strong = strong_required_for_qb_index - set(pbp.columns)
    if missing_strong:
        notes.append(f"Missing strong-required columns: {missing_strong}")

    for column in optional:
        if column not in pbp.columns:
            notes.append(f"Optional column `{column}` is missing.")


def _filter_garbage_time(pbp: pd.DataFrame, mode: str = "score_diff") -> pd.DataFrame:
    """Filter garbage-time plays based on the selected mode."""
    if mode == "none":
        return pbp

    if mode == "score_diff":
        return pbp[
            ~((pbp["game_seconds_remaining"] <= 900) & (pbp["score_differential"].abs() >= 17))
        ]

    if mode == "wp":
        return pbp[
            ~((pbp["wp"] <= 0.05) | (pbp["wp"] >= 0.95))
        ]

    raise ValueError(f"Unknown garbage-time mode: {mode}")


def _calibrate_z_scores(
    pbp: pd.DataFrame, calibration_means: dict, calibration_stds: dict,
) -> pd.DataFrame:
    """Calibrate QB metrics to static z-scores using precomputed means and stds."""
    pbp = pbp.copy()

    for metric, mean in calibration_means.items():
        std = calibration_stds.get(metric, 1)
        if metric in pbp.columns:
            pbp[f"{metric}_z"] = (pbp[metric] - mean) / std
        else:
            raise ValueError(f"Metric {metric} not found in DataFrame columns.")

    return pbp


def _create_weekly_qb_snapshots(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly QB feature snapshots for model-ready data."""
    weekly_snapshots = (
        pbp.groupby(["season", "week", "passer_player_id"])
        .agg(
            air_epa_mean=("air_epa", "mean"),
            yac_epa_mean=("yac_epa", "mean"),
            success_rate=("success", "mean"),
            cpoe_adjusted_epa_mean=("cpoe_adjusted_epa", "mean"),
            plays=("play_id", "count"),
        )
        .reset_index()
    )

    return weekly_snapshots


# Expose selected internals for focused unit tests.
validate_schema = _validate_schema
filter_garbage_time = _filter_garbage_time
compute_qb_components = _build_qb_components
calibrate_z_scores = _calibrate_z_scores
create_weekly_qb_snapshots = _create_weekly_qb_snapshots
build_qb_lift_calibration_artifact_public = build_qb_lift_calibration_artifact
