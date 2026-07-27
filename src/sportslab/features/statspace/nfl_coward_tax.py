"""Experimental StatSpace NFL Coward Tax calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

COWARD_TAX_FORMULA_VERSION = 2


@dataclass(frozen=True)
class CowardTaxMetadata:
    """Machine-readable metadata for one Coward Tax build."""

    metric: str
    formula_version: int
    season: Optional[int]
    through_week: Optional[int]
    min_games: int
    qualified_team_count: int
    built_at: str
    notes: list[str]


def build_coward_tax(
    pbp_df: pd.DataFrame,
    *,
    season: Optional[int] = None,
    through_week: Optional[int] = None,
    min_games: int = 4,
) -> tuple[pd.DataFrame, CowardTaxMetadata]:
    """Build season-to-date Coward Tax rankings."""
    notes: list[str] = []
    if pbp_df.empty:
        return _empty_coward_tax_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_games=min_games,
            qualified_team_count=0,
            notes=["PBP data was empty."],
        )

    pbp = _filter_pbp(pbp_df, season=season, through_week=through_week)
    if pbp.empty:
        return _empty_coward_tax_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_games=min_games,
            qualified_team_count=0,
            notes=["No offensive PBP rows matched the requested filter."],
        )

    _ensure_optional_columns(pbp, notes)
    components = _build_team_components(pbp)
    qualified = components[components["games_played"] >= int(min_games)].copy()
    if qualified.empty:
        note = f"No teams reached the minimum games threshold ({min_games})."
        return _empty_coward_tax_frame(), _metadata(
            season=season,
            through_week=through_week,
            min_games=min_games,
            qualified_team_count=0,
            notes=[note],
        )

    scored = _add_scores(qualified)
    scored = scored.sort_values(
        ["coward_tax_wp", "conservative_decisions", "team"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    scored["formula_version"] = COWARD_TAX_FORMULA_VERSION
    scored = scored[_coward_tax_columns()]
    return scored, _metadata(
        season=season,
        through_week=through_week,
        min_games=min_games,
        qualified_team_count=len(scored),
        notes=notes,
    )


def classify_fourth_down_go_zone(row: pd.Series) -> tuple[bool, float]:
    """Return whether a fourth down is a likely go zone and its strength."""
    if float(row.get("down", 0.0) or 0.0) != 4.0:
        return False, 0.0

    ydstogo = float(row.get("ydstogo", 99.0) or 99.0)
    yardline = float(row.get("yardline_100", 100.0) or 100.0)
    qtr = float(row.get("qtr", 1.0) or 1.0)
    game_seconds = float(row.get("game_seconds_remaining", 3600.0) or 3600.0)
    wp = float(row.get("wp", 0.5) if pd.notna(row.get("wp", 0.5)) else 0.5)
    score_diff = float(row.get("score_differential", 0.0) or 0.0)

    strength = 0.0
    if ydstogo <= 1:
        strength += 0.55
    elif ydstogo <= 2:
        strength += 0.42
    elif ydstogo <= 4:
        strength += 0.22
    else:
        strength -= 0.25

    if yardline <= 5:
        strength += 0.45
    elif yardline <= 20:
        strength += 0.28
    elif yardline <= 40:
        strength += 0.25
    elif yardline <= 50:
        strength += 0.18
    elif yardline >= 75:
        strength -= 0.35
    elif yardline >= 65:
        strength -= 0.15

    if qtr >= 4 and score_diff < 0:
        strength += 0.22
    if game_seconds <= 900 and score_diff < 0:
        strength += 0.18
    if game_seconds <= 300 and score_diff < 0:
        strength += 0.18
    if qtr >= 4 and score_diff > 7 and wp >= 0.70:
        strength -= 0.35

    if 0.20 <= wp <= 0.80:
        strength += 0.10
    if wp < 0.25 and qtr >= 4 and score_diff < 0:
        strength += 0.20
    if wp > 0.85 and qtr >= 4 and score_diff > 0:
        strength -= 0.25

    strength = max(0.0, min(1.0, strength))
    return strength >= 0.65, strength


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
    return pbp[pbp["posteam"].notna()].copy()


def _ensure_optional_columns(pbp: pd.DataFrame, notes: list[str]) -> None:
    defaults = {
        "game_id": "unknown",
        "season": 0,
        "week": 0,
        "down": 0.0,
        "ydstogo": 99.0,
        "yardline_100": 100.0,
        "qtr": 1.0,
        "game_seconds_remaining": 3600.0,
        "half_seconds_remaining": 1800.0,
        "wp": 0.5,
        "score_differential": 0.0,
        "play_type": "",
        "two_point_attempt": 0.0,
        "extra_point_attempt": 0.0,
        "no_play": 0.0,
    }
    for column, default in defaults.items():
        if column not in pbp.columns:
            pbp[column] = default
            notes.append(f"`{column}` missing; defaulted for Coward Tax.")


def _build_team_components(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    pbp["play_type"] = pbp["play_type"].fillna("").astype(str)
    pbp["is_fourth_down"] = pbp["down"].fillna(0.0) == 4.0
    pbp["is_no_play"] = pbp["no_play"].fillna(0.0) == 1.0
    conservative_play_types = ["punt", "field_goal"]
    conservative_fourth = pbp["play_type"].isin(conservative_play_types)
    pbp["is_conservative_fourth"] = conservative_fourth
    pbp["is_aggressive_fourth"] = pbp["play_type"].isin(["run", "pass"])
    pbp["is_fourth_opportunity"] = (
        pbp["is_fourth_down"]
        & ~pbp["is_no_play"]
        & (pbp["is_conservative_fourth"] | pbp["is_aggressive_fourth"])
    )

    go_zone = pbp.apply(classify_fourth_down_go_zone, axis=1)
    pbp["likely_go_zone"] = [item[0] for item in go_zone]
    pbp["go_zone_strength"] = [item[1] for item in go_zone]
    pbp["leverage_weight"] = pbp.apply(_leverage_weight, axis=1)
    pbp["conservative_decision"] = (
        pbp["is_fourth_opportunity"]
        & pbp["likely_go_zone"]
        & pbp["is_conservative_fourth"]
    )
    pbp["aggressive_decision"] = (
        pbp["is_fourth_opportunity"]
        & pbp["likely_go_zone"]
        & pbp["is_aggressive_fourth"]
    )
    pbp["questionable_aggressive_decision"] = (
        pbp["is_fourth_opportunity"]
        & ~pbp["likely_go_zone"]
        & pbp["is_aggressive_fourth"]
    )
    pbp["estimated_wp_left"] = (
        pbp["conservative_decision"].astype(float)
        * pbp["go_zone_strength"]
        * pbp["leverage_weight"]
        * 0.012
    )
    pbp["aggression_credit"] = (
        pbp["aggressive_decision"].astype(float)
        * pbp["go_zone_strength"]
        * pbp["leverage_weight"]
        * 0.010
    )

    pbp["two_point_attempt_flag"] = pbp["two_point_attempt"].fillna(0.0) == 1.0
    score_diff = pbp["score_differential"].fillna(0.0)
    pbp["two_point_need"] = (
        (pbp["extra_point_attempt"].fillna(0.0) == 1.0)
        & (pbp["qtr"].fillna(1.0) >= 4.0)
        & score_diff.isin([-8.0, -5.0, -2.0, 1.0])
    )
    two_point_passive = pbp["two_point_need"] & ~pbp["two_point_attempt_flag"]
    pbp["two_point_passive"] = two_point_passive
    pbp["two_point_wp_left"] = pbp["two_point_passive"].astype(float) * 0.010
    pbp["two_point_credit"] = (
        pbp["two_point_need"] & pbp["two_point_attempt_flag"]
    ).astype(float) * 0.008
    pbp["aggressive_decision"] = (
        pbp["aggressive_decision"] | pbp["two_point_attempt_flag"]
    )

    pbp["end_half_passive"] = (
        (pbp["qtr"].fillna(0.0) == 2.0)
        & (pbp["half_seconds_remaining"].fillna(1800.0) <= 120.0)
        & (pbp["score_differential"].fillna(0.0) <= 7.0)
        & (pbp["score_differential"].fillna(0.0) >= -14.0)
        & (pbp["play_type"].isin(["run", "qb_kneel"]))
        & (pbp["yardline_100"].fillna(100.0) >= 35.0)
    )
    pbp["end_half_wp_left"] = pbp["end_half_passive"].astype(float) * 0.006
    pbp["total_wp_left"] = pbp[
        ["estimated_wp_left", "two_point_wp_left", "end_half_wp_left"]
    ].sum(axis=1)
    credit_columns = ["aggression_credit", "two_point_credit"]
    pbp["total_aggression_credit"] = pbp[credit_columns].sum(axis=1)

    grouped = pbp.groupby("posteam", as_index=False).agg(
        team=("posteam", "first"),
        season=("season", "max"),
        through_week=("week", "max"),
        games_played=("game_id", "nunique"),
        fourth_down_opportunities=("is_fourth_opportunity", "sum"),
        conservative_decisions=("conservative_decision", "sum"),
        aggressive_decisions=("aggressive_decision", "sum"),
        correct_aggressive_decisions=("aggressive_decision", "sum"),
        questionable_aggressive_decisions=(
            "questionable_aggressive_decision",
            "sum",
        ),
        coward_tax_wp=("total_wp_left", "sum"),
        aggression_credit_wp=("total_aggression_credit", "sum"),
    )
    numeric = [
        "fourth_down_opportunities",
        "conservative_decisions",
        "aggressive_decisions",
        "correct_aggressive_decisions",
        "questionable_aggressive_decisions",
        "coward_tax_wp",
        "aggression_credit_wp",
    ]
    grouped[numeric] = grouped[numeric].apply(pd.to_numeric, errors="coerce")
    grouped[numeric] = grouped[numeric].fillna(0.0)
    return grouped


def _leverage_weight(row: pd.Series) -> float:
    wp = float(row.get("wp", 0.5) if pd.notna(row.get("wp", 0.5)) else 0.5)
    qtr = float(row.get("qtr", 1.0) or 1.0)
    game_seconds = float(row.get("game_seconds_remaining", 3600.0) or 3600.0)
    score_diff = float(row.get("score_differential", 0.0) or 0.0)

    leverage = 1.0 + max(0.0, 0.5 - abs(wp - 0.5)) * 1.6
    if qtr >= 4:
        leverage += 0.25
    if game_seconds <= 900:
        leverage += 0.25
    if abs(score_diff) <= 8:
        leverage += 0.20
    return max(0.75, min(2.25, leverage))


def _add_scores(team_frame: pd.DataFrame) -> pd.DataFrame:
    result = team_frame.copy()
    games_played = result["games_played"].replace(0, pd.NA)
    tax_per_game = result["coward_tax_wp"] / games_played
    result["coward_tax_per_game"] = tax_per_game
    result["coward_tax_per_game"] = result["coward_tax_per_game"].fillna(0.0)
    credit_per_game = result["aggression_credit_wp"] / games_played
    result["aggression_credit_per_game"] = credit_per_game.fillna(0.0)
    result["decision_edge_wp"] = (
        result["aggression_credit_wp"] - result["coward_tax_wp"]
    )
    result["conservative_rate"] = result["conservative_decisions"] / result[
        "fourth_down_opportunities"
    ].replace(0, pd.NA)
    result["aggressive_rate"] = result["aggressive_decisions"] / result[
        "fourth_down_opportunities"
    ].replace(0, pd.NA)
    result[["conservative_rate", "aggressive_rate"]] = result[
        ["conservative_rate", "aggressive_rate"]
    ].apply(pd.to_numeric, errors="coerce")
    result[["conservative_rate", "aggressive_rate"]] = result[
        ["conservative_rate", "aggressive_rate"]
    ].fillna(0.0)
    tax_z = _zscore(result["coward_tax_per_game"])
    credit_z = _zscore(result["aggression_credit_per_game"])
    edge_z = _zscore(result["decision_edge_wp"])
    conservative_z = _zscore(result["conservative_rate"])
    aggressive_z = _zscore(result["aggressive_rate"])
    result["aggression_score"] = (
        0.50 * edge_z
        + 0.35 * credit_z
        + 0.15 * aggressive_z
        - 0.25 * tax_z
        - 0.15 * conservative_z
    )
    result["label"] = result.apply(_label, axis=1)
    result["public_tier"] = result.apply(_public_tier, axis=1)
    result["why_flagged"] = result.apply(_why_flagged, axis=1)
    return result


def _label(row: pd.Series) -> str:
    tax = float(row["coward_tax_wp"])
    credit = float(row["aggression_credit_wp"])
    edge = float(row["decision_edge_wp"])
    tax_pg = float(row["coward_tax_per_game"])
    conservative = float(row["conservative_decisions"])
    correct_aggressive = float(row["correct_aggressive_decisions"])
    opportunities = float(row["fourth_down_opportunities"])
    aggression = float(row["aggression_score"])
    if opportunities:
        aggressive_rate = correct_aggressive / opportunities
    else:
        aggressive_rate = 0.0

    if tax_pg >= 0.030 and conservative >= 5:
        return "Full Turtle"
    if tax_pg >= 0.014 and conservative >= 3 and edge < 0.0:
        return "Coward Tax"
    if (
        edge >= 0.20
        and credit >= 0.20
        and correct_aggressive >= 10
        and aggressive_rate >= 0.45
        and tax_pg < 0.006
    ):
        return "Sharp Aggressor"
    if edge >= 0.15 and credit >= 0.15 and correct_aggressive >= 7 and tax_pg < 0.010:
        return "Calculated Aggressor"
    if abs(edge) <= 0.06 and tax_pg < 0.012 and aggression < 1.25:
        return "Properly Priced"
    if tax <= 0.040 and conservative <= 1 and edge >= -0.03:
        return "Properly Priced"
    if tax_pg >= 0.010 or (tax_pg >= 0.006 and edge < 0.10):
        return "Cautious"
    if conservative > correct_aggressive:
        return "Cautious"
    return "Properly Priced"


def _public_tier(row: pd.Series) -> str:
    label = str(row["label"])
    if label == "Sharp Aggressor":
        return "Aggressive Edge"
    if label == "Calculated Aggressor":
        return "Positive"
    if label in {"Cautious", "Coward Tax"}:
        return "Taxed"
    if label == "Full Turtle":
        return "High Tax"
    return "Neutral"


def _why_flagged(row: pd.Series) -> str:
    conservative = int(row["conservative_decisions"])
    correct_aggressive = int(row["correct_aggressive_decisions"])
    questionable = int(row["questionable_aggressive_decisions"])
    tax = float(row["coward_tax_wp"])
    credit = float(row["aggression_credit_wp"])
    edge = float(row["decision_edge_wp"])
    tax_pg = float(row["coward_tax_per_game"])
    label = str(row["label"])
    return (
        f"{conservative} conservative go-zone decisions; "
        f"{correct_aggressive} correct aggressive decisions; "
        f"{questionable} questionable aggressive decisions; "
        f"{tax:.3f} tax WP ({tax_pg:.3f}/game); "
        f"{credit:.3f} aggression credit WP; "
        f"{edge:+.3f} decision edge WP; "
        f"{_verdict_phrase(label)}"
    )


def _verdict_phrase(label: str) -> str:
    verdicts = {
        "Sharp Aggressor": "verdict: aggressive in spots the model supports",
        "Calculated Aggressor": ("verdict: earns aggression credit without much tax"),
        "Properly Priced": "verdict: decision profile is close to neutral",
        "Cautious": "verdict: some conservative drag, but not a major tax",
        "Coward Tax": "verdict: meaningful win probability left on the table",
        "Full Turtle": "verdict: repeated high-leverage passivity",
    }
    return verdicts.get(label, "verdict: mixed aggression profile")


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
    notes: list[str],
) -> CowardTaxMetadata:
    return CowardTaxMetadata(
        metric="coward_tax",
        formula_version=COWARD_TAX_FORMULA_VERSION,
        season=season,
        through_week=through_week,
        min_games=min_games,
        qualified_team_count=qualified_team_count,
        built_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        notes=notes,
    )


def _coward_tax_columns() -> list[str]:
    return [
        "rank",
        "team",
        "season",
        "through_week",
        "games_played",
        "fourth_down_opportunities",
        "conservative_decisions",
        "aggressive_decisions",
        "correct_aggressive_decisions",
        "questionable_aggressive_decisions",
        "coward_tax_wp",
        "coward_tax_per_game",
        "aggression_credit_wp",
        "aggression_credit_per_game",
        "decision_edge_wp",
        "aggression_score",
        "public_tier",
        "label",
        "why_flagged",
        "formula_version",
    ]


def _empty_coward_tax_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_coward_tax_columns())
