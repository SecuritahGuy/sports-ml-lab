"""Compute player-level value scores from PBP data.

Uses 2025 PBP to assign each player a value score (0-100 per position).
Maps players to their 2026 teams and aggregates by position group.

Output: data/features/nfl/player_values_2026.parquet
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
ROSTER_CACHE = BASE / "data" / "features" / "nfl" / "rosters_2026.parquet"
OUTPUT_PATH = BASE / "data" / "features" / "nfl" / "player_values_2026.parquet"

POSITION_GROUP_MAP = {
    "QB": "qb",
    "RB": "skill", "FB": "skill", "TE": "skill", "WR": "skill",
    "OL": "ol", "C": "ol", "G": "ol", "T": "ol",
    "DL": "front", "DE": "front", "DT": "front", "NT": "front", "EDGE": "front",
    "LB": "lb", "ILB": "lb", "OLB": "lb", "MLB": "lb",
    "DB": "coverage", "CB": "coverage", "S": "coverage", "SS": "coverage", "FS": "coverage",
    "K": "st", "P": "st", "LS": "st", "KR": "st",
}

POS_ORDER = ["qb", "skill", "ol", "front", "lb", "coverage", "st"]


def _compute_offensive_epa(pbp):
    """Compute per-player EPA for offensive skill positions."""
    records = []

    passer = pbp[pbp["pass_attempt"] == 1].copy()
    if len(passer):
        grp = passer.groupby("passer_player_id")
        for pid, g in grp:
            records.append({
                "player_id": pid,
                "player_name": g["passer_player_name"].iloc[0],
                "team_2025": g["posteam"].iloc[0],
                "position": "QB",
                "off_epa": g["epa"].sum(),
                "count": len(g),
                "td": g["touchdown"].sum(),
                "turnover": (
                    g["interception"].sum() + g["fumble_lost"].sum()
                    if "fumble_lost" in g.columns
                    else g["interception"].sum()
                ),
            })

    rusher = pbp[pbp["rush_attempt"] == 1].copy()
    if len(rusher):
        grp = rusher.groupby("rusher_player_id")
        for pid, g in grp:
            records.append({
                "player_id": pid,
                "player_name": g["rusher_player_name"].iloc[0],
                "team_2025": g["posteam"].iloc[0],
                "position": "RB",
                "off_epa": g["epa"].sum(),
                "count": len(g),
                "td": g["touchdown"].sum(),
                "turnover": g["fumble_lost"].sum() if "fumble_lost" in g.columns else 0,
            })

    receiver = pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()].copy()
    if len(receiver):
        grp = receiver.groupby("receiver_player_id")
        for pid, g in grp:
            records.append({
                "player_id": pid,
                "player_name": g["receiver_player_name"].iloc[0],
                "team_2025": g["posteam"].iloc[0],
                "position": "REC",
                "off_epa": g["epa"].sum(),
                "count": len(g),
                "td": g["touchdown"].sum(),
                "turnover": g["fumble_lost"].sum() if "fumble_lost" in g.columns else 0,
            })

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df

    # Collapse players with multiple position roles (e.g., a RB who also receives)
    agg = df.groupby("player_id").agg(
        player_name=("player_name", "first"),
        team_2025=("team_2025", "first"),
        position=("position", lambda x: "REC" if "REC" in x.values else x.iloc[0]),
        off_epa=("off_epa", "sum"),
        count=("count", "sum"),
        td=("td", "sum"),
        turnover=("turnover", "sum"),
    ).reset_index()

    # Map REC/QB/RB to actual position from roster
    return agg


def _compute_defensive_stats(pbp):
    """Compute per-player defensive stats from PBP."""
    records = []

    def _extract_player(col_name, label):
        if col_name not in pbp.columns:
            return
        valid = pbp[pbp[col_name].notna() & (pbp[col_name] != "")]
        if len(valid) == 0:
            return
        grp = valid.groupby(col_name)
        for pid, g in grp:
            name_col = col_name.replace("_player_id", "_player_name")
            player_name = g[name_col].iloc[0] if name_col in g.columns else pid
            records.append({
                "player_id": pid,
                "player_name": player_name,
                "team_2025": g["defteam"].iloc[0] if "defteam" in g.columns else "",
                "stat_type": label,
                "stat_value": len(g),
            })

    _extract_player("sack_player_id", "sack")
    _extract_player("sack_player_1_player_id", "sack")
    _extract_player("interception_player_id", "int")
    _extract_player("solo_tackle_1_player_id", "tackle")
    _extract_player("solo_tackle_2_player_id", "tackle")
    _extract_player("assist_tackle_1_player_id", "assist_tackle")
    _extract_player("pass_defense_1_player_id", "pass_def")
    _extract_player("tackle_for_loss_1_player_id", "tfl")
    _extract_player("forced_fumble_player_1_player_id", "ff")
    _extract_player("fumble_recovery_1_player_id", "fr")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivot = df.pivot_table(
        index="player_id",
        columns="stat_type",
        values="stat_value",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    pivot.columns.name = None

    # Merge player info back
    info = df.drop_duplicates("player_id")[["player_id", "player_name", "team_2025"]]
    result = info.merge(pivot, on="player_id", how="left")
    return result


def compute_player_values(pbp=None, rosters=None):
    """Compute player value scores from PBP data.

    Args:
        pbp: DataFrame of PBP data (will load if None)
        rosters: DataFrame of 2026 roster (will load if None)

    Returns:
        DataFrame with player values per team.
    """
    if pbp is None:
        print("Loading 2025 PBP data...")
        import nfl_data_py as nfl
        pbp = nfl.import_pbp_data([2025])

    if rosters is None:
        if ROSTER_CACHE.exists():
            rosters = pd.read_parquet(ROSTER_CACHE)
        else:
            print("Loading 2026 rosters...")
            import nfl_data_py as nfl
            rosters = nfl.import_seasonal_rosters([2026])
            rosters.to_parquet(ROSTER_CACHE)

    print("Computing offensive EPA...")
    offense = _compute_offensive_epa(pbp)
    print(f"  {len(offense)} offensive players")

    print("Computing defensive stats...")
    defense = _compute_defensive_stats(pbp)
    print(f"  {len(defense)} defensive players")

    # Build player value from stats
    value_records = _build_value_scores(offense, defense, rosters, pbp)

    value_df = pd.DataFrame(value_records)
    if len(value_df):
        value_df.to_parquet(OUTPUT_PATH)
        print(f"Saved player values to {OUTPUT_PATH}")
        print(f"  {len(value_df)} players across {value_df['team_2026'].nunique()} teams")
    return value_df


def _build_value_scores(offense, defense, rosters, pbp):
    """Build composite value scores per player."""
    records = []

    # Process offensive players
    for _, row in offense.iterrows():
        pid = row["player_id"]
        # Map to 2026 team
        team_2026 = _map_to_2026_team(pid, rosters)

        value = row["off_epa"]
        pos_group = row["position"]
        # Determine actual position group
        if pos_group == "QB":
            pos_group = "qb"
        elif pos_group == "RB":
            pos_group = "skill"
        elif pos_group == "REC":
            pos_group = "skill"

        records.append({
            "player_id": pid,
            "player_name": row["player_name"],
            "team_2025": row["team_2025"],
            "team_2026": team_2026,
            "position": pos_group,
            "value": value,
            "detail": f"{row['count']} plays, {row['td']} TD, {row['turnover']} TO",
        })

    # Process defensive players
    if len(defense):
        for _, row in defense.iterrows():
            pid = row["player_id"]
            team_2026 = _map_to_2026_team(pid, rosters)

            # Compute a defensive value from combined stats
            sacks = float(row.get("sack", 0))
            ints = float(row.get("int", 0))
            tackles = float(row.get("tackle", 0))
            pass_def = float(row.get("pass_def", 0))
            tfl = float(row.get("tfl", 0))
            ff = float(row.get("ff", 0))
            fr = float(row.get("fr", 0))

            # Weighted composite
            value = (
                sacks * 5.0
                + ints * 4.0
                + tfl * 2.0
                + ff * 3.0
                + fr * 2.0
                + pass_def * 1.5
                + tackles * 0.5
            )

            pos_group = _position_from_roster(pid, rosters)
            detail_parts = []
            if sacks:
                detail_parts.append(f"{sacks:.0f} sack")
            if ints:
                detail_parts.append(f"{ints:.0f} INT")
            if tackles:
                detail_parts.append(f"{tackles:.0f} tackle")

            records.append({
                "player_id": pid,
                "player_name": row["player_name"],
                "team_2025": row["team_2025"],
                "team_2026": team_2026,
                "position": pos_group,
                "value": value,
                "detail": ", ".join(detail_parts) if detail_parts else "",
            })

    # Also add players from rosters with no PBP stats (rookies, etc.) at 0 value
    roster_pids = set(rosters["player_id"].unique())
    stat_pids = {r["player_id"] for r in records}
    missing = roster_pids - stat_pids
    for pid in missing:
        roster_row = rosters[rosters["player_id"] == pid]
        if len(roster_row) == 0:
            continue
        r = roster_row.iloc[0]
        pos_group = POSITION_GROUP_MAP.get(r["position"], "other")
        records.append({
            "player_id": pid,
            "player_name": r["player_name"],
            "team_2025": "",
            "team_2026": r["team"],
            "position": pos_group,
            "value": 0.0,
            "detail": "No 2025 stats (rookie/new team)",
        })

    return records


def _map_to_2026_team(player_id, rosters):
    """Map a player from 2025 stats to their 2026 team."""
    match = rosters[rosters["player_id"] == player_id]
    if len(match):
        return match.iloc[0]["team"]
    return ""


def _position_from_roster(player_id, rosters):
    """Get position group for a player from roster data."""
    match = rosters[rosters["player_id"] == player_id]
    if len(match):
        pos = match.iloc[0]["position"]
        return POSITION_GROUP_MAP.get(pos, "other")
    return "other"


def aggregate_by_team(value_df):
    """Aggregate player values by team and position group.

    Returns a DataFrame with team-level roster value stats.
    """
    if len(value_df) == 0:
        return pd.DataFrame()

    # Filter to players on 2026 teams
    current = value_df[value_df["team_2026"] != ""].copy()
    current["team_2026"] = current["team_2026"].str.upper()

    # Compute percentile ranks per position group
    for pos in POS_ORDER:
        mask = current["position"] == pos
        if mask.sum() > 1:
            vals = current.loc[mask, "value"]
            current.loc[mask, "pctl"] = vals.rank(pct=True) * 100

    current["pctl"] = current["pctl"].fillna(50.0)

    # Aggregate by team
    team_groups = current.groupby("team_2026")

    rows = []
    for team, grp in team_groups:
        row = {"team": team, "total_players": len(grp), "total_value": grp["value"].sum()}
        for pos in POS_ORDER:
            pos_grp = grp[grp["position"] == pos]
            row[f"{pos}_count"] = len(pos_grp)
            row[f"{pos}_value"] = pos_grp["value"].sum() if len(pos_grp) else 0
            row[f"{pos}_avg_pctl"] = round(pos_grp["pctl"].mean(), 1) if len(pos_grp) else 0
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_value", ascending=False)


def compute_team_value_added(value_df, team_aggs):
    """Compute net value added/lost per team (2026 roster vs expected baseline).

    The baseline is: what would an average team's roster be worth?
    """
    if len(team_aggs) == 0:
        return team_aggs

    result = team_aggs.copy()
    for pos in POS_ORDER:
        col = f"{pos}_avg_pctl"
        if col in result.columns:
            result[f"{pos}_delta"] = result[col] - 50.0

    delta_cols = [f"{pos}_delta" for pos in POS_ORDER if f"{pos}_delta" in result.columns]
    result["overall_delta"] = result[delta_cols].mean(axis=1)
    return result


if __name__ == "__main__":
    values = compute_player_values()
    print(f"\nComputed {len(values)} player values")
    team_vals = aggregate_by_team(values)
    print("\nTeam roster strength (top 10):")
    cols = ["team", "total_players", "total_value", "qb_avg_pctl",
            "skill_avg_pctl", "ol_avg_pctl", "front_avg_pctl"]
    print(team_vals[cols].head(10).to_string())
    print(f"\nSaved to {OUTPUT_PATH}")
