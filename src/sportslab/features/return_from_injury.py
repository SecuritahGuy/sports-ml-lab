"""Return-from-injury rust features.

Identifies players returning from multi-game absences and computes
team-level rust scores. The hypothesis is that players need 1-2 games
to ramp up after a 2+ game absence, causing teams to underperform
their Elo expectation in that first game back.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sportslab.features.build_features import TARGET_COLUMN

POSITION_WEIGHTS = {
    "QB": 5.0,
    "RB": 3.0,
    "FB": 2.0,
    "WR": 2.0,
    "TE": 2.0,
    "OL": 1.5,
    "C": 1.5,
    "G": 1.5,
    "T": 1.5,
    "DT": 1.0,
    "DE": 1.0,
    "DL": 1.0,
    "NT": 1.0,
    "LB": 1.0,
    "ILB": 1.0,
    "OLB": 1.0,
    "CB": 1.0,
    "S": 1.0,
    "DB": 1.0,
    "K": 0.5,
    "P": 0.5,
    "LS": 0.5,
}

SKILL_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}
MIN_OUT_STREAK = 2
CACHE_DIR = Path("data/interim/nfl")


def _get_position_weight(position: str, default: float = 1.0) -> float:
    pos = str(position).strip().upper() if pd.notna(position) else ""
    return POSITION_WEIGHTS.get(pos, default)


def _load_injury_data(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    all_inj = []
    for season in seasons:
        s_df = nfl.import_injuries([season])
        if not s_df.empty:
            all_inj.append(s_df)
    if not all_inj:
        return pd.DataFrame()
    return pd.concat(all_inj, ignore_index=True)


def _find_return_events(injury_df: pd.DataFrame) -> pd.DataFrame:
    """Identify when a player returns from a multi-game absence.

    Returns a DataFrame with one row per return event:
      season, week, team, gsis_id, position, games_missed, return_type
    """
    if injury_df.empty:
        return pd.DataFrame()

    inj = injury_df.copy()
    inj = inj[inj["report_status"] == "Out"].copy()
    inj = inj.sort_values(["gsis_id", "season", "week"])

    events = []
    for (pid, season), grp in inj.groupby(["gsis_id", "season"]):
        grp = grp.sort_values("week")
        weeks_out = sorted(grp["week"].unique())

        # Find consecutive streaks
        streaks = []
        current = [weeks_out[0]]
        for w in weeks_out[1:]:
            if w == current[-1] + 1:
                current.append(w)
            else:
                streaks.append(current)
                current = [w]
        streaks.append(current)

        for streak in streaks:
            if len(streak) < MIN_OUT_STREAK:
                continue

            last_out_week = streak[-1]
            games_missed = len(streak)
            pos = grp["position"].iloc[0]
            team = grp["team"].iloc[0]
            primary_injury = grp["report_primary_injury"].iloc[0]

            events.append({
                "season": season,
                "gsis_id": pid,
                "position": pos,
                "team": team,
                "last_out_week": last_out_week,
                "games_missed": games_missed,
                "primary_injury": primary_injury,
                "return_week": last_out_week + 1,
            })

    if not events:
        return pd.DataFrame()

    result = pd.DataFrame(events)
    return result


def _compute_rust_score(row: pd.Series) -> dict:
    """Compute rust metrics for a single game from return events on both teams."""
    return {}


def compute_rust_features(
    df: pd.DataFrame,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute team-level rust scores from players returning from injury.

    For each game, identifies which players are playing their first game(s)
    back from a 2+ game absence and computes weighted rust scores.

    Returns df with added columns:
      home_rust_score, away_rust_score: weighted sum of returning players
      home_rust_qb, away_rust_qb: QB-specific rust
      home_rust_skill, away_rust_skill: skill position rust (RB/WR/TE)
      home_rust_games_missed, away_rust_games_missed: total games missed by returners
    """
    if seasons is None:
        eligible = df[df[TARGET_COLUMN].notna()]
        seasons = sorted(int(s) for s in eligible["season"].unique() if s != 2026)

    # Load injury data
    print(f"Loading injury data for seasons {seasons}...")
    injury_df = _load_injury_data(seasons)
    print(f"  Injury rows: {len(injury_df)}")

    # Find return events
    returns = _find_return_events(injury_df)
    if returns.empty:
        print("  No return events found")
        out_cols = [
            "home_rust_score", "away_rust_score",
            "home_rust_qb", "away_rust_qb",
            "home_rust_skill", "away_rust_skill",
            "home_rust_games_missed", "away_rust_games_missed",
        ]
        for c in out_cols:
            df[c] = 0.0
        return df

    print(f"  Return events found: {len(returns)}")
    print(f"  Total unique players returning: {returns['gsis_id'].nunique()}")

    # Show position breakdown
    for pos in ["QB", "RB", "WR", "TE"]:
        n = len(returns[returns["position"] == pos])
        if n:
            print(f"    {pos}: {n} return events")

    # Build per-game rust scores
    rust_rows = []
    for (season, week), grp in returns.groupby(["season", "return_week"]):
        for team_name in ["home", "away"]:
            rust_rows.append({
                "season": season,
                "week": week,
                "team_col": team_name,
                "returners": [],
            })

    rust_map = {}
    for _, event in returns.iterrows():
        key = (event["season"], event["return_week"], event["team"])
        if key not in rust_map:
            rust_map[key] = []
        rust_map[key].append(event)

    home_scores = []
    away_scores = []
    home_qb = []
    away_qb = []
    home_skill = []
    away_skill = []
    home_gm = []
    away_gm = []

    for _, row in df.iterrows():
        season = int(row["season"])
        week = int(row["week"])

        h_team = row["home_team"]
        a_team = row["away_team"]

        h_events = rust_map.get((season, week, h_team), [])
        a_events = rust_map.get((season, week, a_team), [])

        def _score(events):
            total = 0.0
            qb = 0.0
            skill = 0.0
            gm = 0
            for e in events:
                pos = str(e["position"]).strip().upper() if pd.notna(e["position"]) else ""
                weight = _get_position_weight(pos)
                gm += e["games_missed"]
                rust = weight * np.sqrt(e["games_missed"])
                total += rust
                if pos == "QB":
                    qb += rust
                if pos in SKILL_POSITIONS:
                    skill += rust
            return total, qb, skill, gm

        h_total, h_qb, h_skill, h_gm = _score(h_events)
        a_total, a_qb, a_skill, a_gm = _score(a_events)

        home_scores.append(h_total)
        away_scores.append(a_total)
        home_qb.append(h_qb)
        away_qb.append(a_qb)
        home_skill.append(h_skill)
        away_skill.append(a_skill)
        home_gm.append(h_gm)
        away_gm.append(a_gm)

    df["home_rust_score"] = home_scores
    df["away_rust_score"] = away_scores
    df["home_rust_qb"] = home_qb
    df["away_rust_qb"] = away_qb
    df["home_rust_skill"] = home_skill
    df["away_rust_skill"] = away_skill
    df["home_rust_games_missed"] = home_gm
    df["away_rust_games_missed"] = away_gm

    # Summary stats
    non_zero = (pd.Series(home_scores) > 0).sum() + (pd.Series(away_scores) > 0).sum()
    print(f"  Games with rust > 0: {non_zero}")
    print(f"  Max home rust: {max(home_scores) if home_scores else 0:.2f}")
    print(f"  Max away rust: {max(away_scores) if away_scores else 0:.2f}")

    return df
