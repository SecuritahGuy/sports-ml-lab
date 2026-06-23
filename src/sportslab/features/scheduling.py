"""Pregame scheduling and rest features — no leakage, chronological."""

import pandas as pd

# International stadium ID prefixes
_INTERNATIONAL_PREFIXES = ("LON", "FRA", "GER", "MEX", "SAO")

# Standard rest days between games
STANDARD_REST = 7
# Maximum rest days for a "short week"
SHORT_WEEK_THRESHOLD = 6
# Minimum rest days for a team coming off a bye
BYE_REST_THRESHOLD = 13


def compute_scheduling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scheduling and rest features known before kickoff.

    Processes games chronologically to ensure no future data leakage.
    Features that depend on prior games (consecutive road games) are
    computed during a single chronological pass.

    Args:
        df: Must contain columns: season, week, gameday, home_team,
            away_team, home_rest, away_rest, weekday, stadium_id,
            location, is_neutral.

    Returns:
        DataFrame with additional scheduling feature columns.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    # Boolean / flag features — no chronological pass needed
    out["home_short_week"] = (
        out["home_rest"].notna() & (out["home_rest"] <= SHORT_WEEK_THRESHOLD)
    ).astype(int)
    out["away_short_week"] = (
        out["away_rest"].notna() & (out["away_rest"] <= SHORT_WEEK_THRESHOLD)
    ).astype(int)
    out["home_off_bye"] = (
        out["home_rest"].notna() & (out["home_rest"] >= BYE_REST_THRESHOLD)
    ).astype(int)
    out["away_off_bye"] = (
        out["away_rest"].notna() & (out["away_rest"] >= BYE_REST_THRESHOLD)
    ).astype(int)
    out["thursday_flag"] = (out["weekday"] == "Thursday").astype(int)
    out["monday_flag"] = (out["weekday"] == "Monday").astype(int)
    out["is_international"] = (out["stadium_id"].str.startswith(_INTERNATIONAL_PREFIXES)).astype(
        int
    )

    # Consecutive road games — requires chronological pass
    out["home_consecutive_road"] = 0
    out["away_consecutive_road"] = 0

    consecutive_road: dict[str, int] = {}

    for idx, row in out.iterrows():
        home_team: str = row["home_team"]
        away_team: str = row["away_team"]
        is_neutral: bool = row["is_neutral"] or False

        # Record pregame consecutive road count (before this game)
        home_cr = consecutive_road.get(home_team, 0)
        away_cr = consecutive_road.get(away_team, 0)
        out.at[idx, "home_consecutive_road"] = home_cr
        out.at[idx, "away_consecutive_road"] = away_cr

        # Update counters for next game based on this game's location
        if is_neutral:
            consecutive_road[home_team] = home_cr + 1
        else:
            consecutive_road[home_team] = 0
        consecutive_road[away_team] = away_cr + 1

    return out


SCHEDULING_FEATURE_COLUMNS = [
    "home_short_week",
    "away_short_week",
    "home_off_bye",
    "away_off_bye",
    "thursday_flag",
    "monday_flag",
    "is_neutral",
    "is_international",
    "home_consecutive_road",
    "away_consecutive_road",
]
