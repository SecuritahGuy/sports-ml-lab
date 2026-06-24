"""Pregame injury report features from nflreadpy.

Aggregates player-level injury reports to team-level features
per game: total players ruled OUT, by position group, and by
designation severity (questionable, doubtful, out).
"""

from pathlib import Path

import nflreadpy
import pandas as pd
import polars as pl

SPORTSLAB_MIN_SEASON = 2021

SKILL_POSITIONS = {"WR", "RB", "TE"}
OL_POSITIONS = {"T", "G", "C"}
DEF_POSITIONS = {
    "DE",
    "DT",
    "LB",
    "CB",
    "S",
    "NT",
    "OLB",
    "ILB",
    "MLB",
    "DB",
    "DL",
    "EDGE",
    "SS",
    "FS",
}

INJURY_FEATURE_COLUMNS = [
    "home_injuries_out",
    "away_injuries_out",
    "injuries_out_diff",
    "home_injuries_qb_out",
    "away_injuries_qb_out",
    "injuries_qb_out_diff",
    "home_injuries_skill_out",
    "away_injuries_skill_out",
    "injuries_skill_out_diff",
    "home_injuries_ol_out",
    "away_injuries_ol_out",
    "injuries_ol_out_diff",
    "home_injuries_def_out",
    "away_injuries_def_out",
    "injuries_def_out_diff",
    "home_injuries_questionable",
    "away_injuries_questionable",
    "home_injuries_doubtful",
    "away_injuries_doubtful",
]


def load_injury_data(
    seasons: list[int] | None = None,
    cache_dir: str = "data/interim/nfl",
) -> pl.DataFrame:
    """Load nflreadpy injury reports with local caching.

    Args:
        seasons: Season years to load. Must be >= 2021.
        cache_dir: Directory for cached parquet files.

    Returns:
        Polars DataFrame with injury report data.
    """
    if seasons is None:
        seasons = list(range(SPORTSLAB_MIN_SEASON, 2026))

    bad = [s for s in seasons if s < SPORTSLAB_MIN_SEASON]
    if bad:
        raise ValueError(f"Seasons before {SPORTSLAB_MIN_SEASON} not allowed: {bad}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    fragments: list[pl.DataFrame] = []
    uncached: list[int] = []

    for s in seasons:
        cp = cache_path / f"injuries_{s}.parquet"
        if cp.exists():
            fragments.append(pl.read_parquet(str(cp)))
        else:
            uncached.append(s)

    if uncached:
        # Load each season individually to handle schema differences
        for s in uncached:
            season_df = nflreadpy.load_injuries(seasons=[s])
            if "season_type" in season_df.columns:
                season_df = season_df.drop("season_type")
            cp = cache_path / f"injuries_{s}.parquet"
            season_df.write_parquet(str(cp))
            fragments.append(season_df)

    # Ensure all fragments have matching columns
    if len(fragments) > 1:
        common_cols = set(fragments[0].columns)
        for f in fragments[1:]:
            common_cols &= set(f.columns)
        common_cols = sorted(common_cols)
        fragments = [f.select(common_cols) for f in fragments]

    if len(fragments) == 1:
        return fragments[0]
    return pl.concat(fragments)


def _normalize_position(pos: str | None) -> str | None:
    """Clean position string, handling whitespace issues."""
    if pos is None:
        return None
    pos = pos.strip()
    if pos == "" or pos == "\n    ":
        return None
    return pos


def compute_injury_features(
    df: pd.DataFrame,
    seasons: list[int] | None = None,
    cache_dir: str = "data/interim/nfl",
) -> pd.DataFrame:
    """Add team-level injury report features based on nflreadpy data.

    For each game in df, looks up the home and away teams' injury reports
    for that season and week.

    Args:
        df: Must contain columns: season, week, home_team, away_team.
        seasons: Season years to load injury data for.
        cache_dir: Directory for cached injury parquet files.

    Returns:
        DataFrame with added injury feature columns (zeros if no injury
        data available for a given game).
    """
    raw = load_injury_data(seasons=seasons, cache_dir=cache_dir)

    # Build a lookup: (season, week, team) -> dict of injury counts
    # We'll do this per game by grouping injury data by (season, week, team)
    injury_lookup: dict[tuple[int, int, str], dict[str, int]] = {}

    # Process in polars then convert to dict
    grouped = raw.group_by(["season", "week", "team"]).agg(
        [
            pl.col("report_status").alias("statuses"),
            pl.col("position").alias("positions"),
            pl.col("gsis_id").alias("player_ids"),
        ]
    )

    for row in grouped.iter_rows(named=True):
        season = row["season"]
        week = row["week"]
        team = row["team"]
        statuses = row["statuses"]
        positions = row["positions"]

        counts: dict[str, int] = {
            "out": 0,
            "qb_out": 0,
            "skill_out": 0,
            "ol_out": 0,
            "def_out": 0,
            "questionable": 0,
            "doubtful": 0,
        }

        for status, pos_raw in zip(statuses, positions):
            pos = _normalize_position(pos_raw)
            if pos is None:
                continue

            status_str = str(status) if status is not None else ""

            if status_str == "Out":
                counts["out"] += 1
                if pos == "QB":
                    counts["qb_out"] += 1
                if pos in SKILL_POSITIONS:
                    counts["skill_out"] += 1
                if pos in OL_POSITIONS:
                    counts["ol_out"] += 1
                if pos in DEF_POSITIONS:
                    counts["def_out"] += 1
            elif status_str == "Questionable":
                counts["questionable"] += 1
            elif status_str == "Doubtful":
                counts["doubtful"] += 1

        injury_lookup[(season, week, team)] = counts

    out = df.copy()
    home_counts_list = []
    away_counts_list = []

    for _, row in df.iterrows():
        season = int(row["season"])
        week = int(row["week"])
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        home_key = (season, week, home_team)
        away_key = (season, week, away_team)

        home_c = injury_lookup.get(
            home_key,
            {
                "out": 0,
                "qb_out": 0,
                "skill_out": 0,
                "ol_out": 0,
                "def_out": 0,
                "questionable": 0,
                "doubtful": 0,
            },
        )
        away_c = injury_lookup.get(
            away_key,
            {
                "out": 0,
                "qb_out": 0,
                "skill_out": 0,
                "ol_out": 0,
                "def_out": 0,
                "questionable": 0,
                "doubtful": 0,
            },
        )

        home_counts_list.append(home_c)
        away_counts_list.append(away_c)

    home_df = pd.DataFrame(home_counts_list)
    away_df = pd.DataFrame(away_counts_list)

    out["home_injuries_out"] = home_df["out"]
    out["away_injuries_out"] = away_df["out"]
    out["injuries_out_diff"] = home_df["out"] - away_df["out"]

    out["home_injuries_qb_out"] = home_df["qb_out"]
    out["away_injuries_qb_out"] = away_df["qb_out"]
    out["injuries_qb_out_diff"] = home_df["qb_out"] - away_df["qb_out"]

    out["home_injuries_skill_out"] = home_df["skill_out"]
    out["away_injuries_skill_out"] = away_df["skill_out"]
    out["injuries_skill_out_diff"] = home_df["skill_out"] - away_df["skill_out"]

    out["home_injuries_ol_out"] = home_df["ol_out"]
    out["away_injuries_ol_out"] = away_df["ol_out"]
    out["injuries_ol_out_diff"] = home_df["ol_out"] - away_df["ol_out"]

    out["home_injuries_def_out"] = home_df["def_out"]
    out["away_injuries_def_out"] = away_df["def_out"]
    out["injuries_def_out_diff"] = home_df["def_out"] - away_df["def_out"]

    out["home_injuries_questionable"] = home_df["questionable"]
    out["away_injuries_questionable"] = away_df["questionable"]

    out["home_injuries_doubtful"] = home_df["doubtful"]
    out["away_injuries_doubtful"] = away_df["doubtful"]

    return out
