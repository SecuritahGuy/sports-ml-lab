"""Player-level injury recovery curve analysis.

Tracks individual player performance before/after multi-game injuries
to model recovery trajectories — how many weeks until a player returns
to pre-injury baseline, injury-type effects, compounding effects.

Pipeline:
  1. Build per-game player performance table (fantasy pts for all positions)
  2. Match injury return events (Out 2+ weeks → plays)
  3. Compute pre-injury baseline and post-return deficit trajectory
  4. Fit recovery curve model (deficit ~ weeks + position + type + count)
  5. Aggregate to team-level logit adjustment for prediction
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RECOVERY_MAX_WEEKS = 8
BASELINE_GAMES = 4
MIN_GAMES_OUT = 2

INJURY_TYPE_GROUPS = {
    "knee": ["knee", "acl", "meniscus", "mcl"],
    "ankle": ["ankle"],
    "hamstring": ["hamstring"],
    "concussion": ["concussion", "head"],
    "shoulder": ["shoulder", "collarbone", "clavicle"],
    "core": ["groin", "abdomen", "oblique", "hip", "quad"],
    "foot": ["foot", "toe", "heel"],
    "arm": ["elbow", "wrist", "hand", "thumb", "arm", "forearm", "bicep", "tricep", "pectoral"],
    "back": ["back", "neck", "rib", "chest"],
    "calf": ["calf"],
    "illness": ["illness"],
    "other": [],
}

POSITION_WEIGHTS = {
    "QB": 5.0,
    "RB": 3.0,
    "WR": 2.0,
    "TE": 1.5,
    "OL": 1.0,
    "DL": 1.0,
    "LB": 1.0,
    "DB": 1.0,
    "K": 0.5,
    "P": 0.3,
}


def _group_injury_type(raw: str) -> str:
    if pd.isna(raw) or not raw:
        return "other"
    r = str(raw).lower().strip()
    for group, keywords in INJURY_TYPE_GROUPS.items():
        for kw in keywords:
            if kw in r:
                return group
    return "other"


def build_player_game_table(
    seasons: list[int],
    snap_path: str = "data/raw/nfl/snap_counts.parquet",
    pbp_path: str = "data/raw/nfl/pbp.parquet",
) -> pd.DataFrame:
    """Build per-game player performance table using fantasy points.

    Uses PBP fantasy column as the primary performance metric (captures
    volume + scoring for all offensive positions). For non-skill players
    (OL/DL/LB/DB), uses snap% as availability proxy.
    """
    import nfl_data_py as nfl

    pbp = nfl.import_pbp_data(seasons, downcast=True)

    # ── QB per-game stats (passing + rushing) ──
    qb_stats = (
        pbp[pbp["passer_player_id"].notna()]
        .groupby(["game_id", "season", "week", "posteam",
                   "passer_player_id", "passer_player_name"])
        .agg(
            qb_epa=("qb_epa", "sum"),
            pass_epa=("epa", "sum"),
            pass_attempts=("pass_attempt", "sum"),
            completions=("complete_pass", "sum"),
            passing_yards=("passing_yards", "sum"),
            pass_td=("touchdown", "sum"),
            interceptions=("interception", "sum"),
            sacks=("sack", "sum"),
        )
        .reset_index()
        .rename(columns={
            "passer_player_id": "gsis_id",
            "passer_player_name": "player_name",
            "posteam": "team",
        })
    )
    qb_stats["position"] = "QB"

    # ── RB per-game stats (rushing + receiving) ──
    rb_rush = (
        pbp[(pbp["rush_attempt"] == 1)]
        .dropna(subset=["rusher_player_id"])
        .groupby(["game_id", "season", "week", "posteam",
                   "rusher_player_id", "rusher_player_name"])
        .agg(
            rush_attempts=("rush_attempt", "sum"),
            rushing_yards=("rushing_yards", "sum"),
            rush_td=("touchdown", "sum"),
        )
        .reset_index()
        .rename(columns={
            "rusher_player_id": "gsis_id",
            "rusher_player_name": "player_name",
            "posteam": "team",
        })
    )
    rb_recv = (
        pbp[(pbp["pass_attempt"] == 1) & (pbp["complete_pass"] == 1)]
        .dropna(subset=["receiver_player_id"])
        .groupby(["game_id", "season", "week", "posteam",
                   "receiver_player_id", "receiver_player_name"])
        .agg(
            targets=("pass_attempt", "sum"),
            receptions=("complete_pass", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            rec_td=("touchdown", "sum"),
        )
        .reset_index()
        .rename(columns={
            "receiver_player_id": "gsis_id",
            "receiver_player_name": "player_name",
        })
    )
    rb = rb_rush.merge(rb_recv, on=["game_id", "season", "week", "gsis_id"],
                        how="outer", suffixes=("", "_recv"))
    for c in ["player_name", "team"]:
        col = f"{c}_recv"
        if col in rb.columns:
            rb[c] = rb[c].fillna(rb[col])
            rb.drop(columns=[col], inplace=True)
    if "posteam" in rb.columns:
        rb["team"] = rb["team"].fillna(rb["posteam"])
        rb.drop(columns=["posteam"], inplace=True, errors="ignore")
    rb.fillna(0, inplace=True)
    rb["position"] = "RB"

    # ── WR/TE per-game stats (receiving) ──
    skill = (
        pbp[(pbp["pass_attempt"] == 1) & (pbp["complete_pass"] == 1)]
        .dropna(subset=["receiver_player_id"])
        .groupby(["game_id", "season", "week", "posteam",
                   "receiver_player_id", "receiver_player_name"])
        .agg(
            targets=("pass_attempt", "sum"),
            receptions=("complete_pass", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            rec_td=("touchdown", "sum"),
        )
        .reset_index()
        .rename(columns={
            "receiver_player_id": "gsis_id",
            "receiver_player_name": "player_name",
            "posteam": "team",
        })
    )
    skill["position"] = "SKILL"

    # Combine all offense
    for df in [qb_stats, rb, skill]:
        for c in ["qb_epa", "pass_epa", "pass_attempts", "completions",
                   "passing_yards", "pass_td", "interceptions", "sacks",
                   "rush_attempts", "rushing_yards", "rush_td",
                   "targets", "receptions", "receiving_yards", "rec_td"]:
            if c not in df.columns:
                df[c] = 0

    combined = pd.concat([qb_stats, rb, skill], ignore_index=True, sort=False)

    # Compute total yards and fantasy-style score per game
    combined["total_yards"] = (combined["passing_yards"] +
                                combined["rushing_yards"] +
                                combined["receiving_yards"])
    combined["total_td"] = (combined["pass_td"] +
                             combined["rush_td"] +
                             combined["rec_td"])
    # Simple fantasy pts: 1pt/25 pass yds, 1pt/10 rush yds, 1pt/10 recv yds,
    # 4pt pass TD, 6pt rush/recv TD
    combined["fantasy_pts"] = (
        combined["passing_yards"] / 25.0 +
        combined["rushing_yards"] / 10.0 +
        combined["receiving_yards"] / 10.0 +
        combined["pass_td"] * 4.0 +
        (combined["rush_td"] + combined["rec_td"]) * 6.0
    )

    # Load snap counts for position info
    snaps = nfl.import_snap_counts(seasons)
    players = nfl.import_players()

    # Map PFR IDs (from snaps) → gsis_id (from PBP)
    pfr_to_gsis = dict(zip(players["pfr_id"].dropna(), players["gsis_id"].dropna()))
    position_map = dict(zip(players["gsis_id"].dropna(), players["position"].dropna()))

    snaps["gsis_id"] = snaps["pfr_player_id"].map(pfr_to_gsis)

    def _assign_position(gsis_id):
        pos = position_map.get(gsis_id, "")
        p = str(pos).upper().strip()
        if p in ("QB",):
            return "QB"
        if p in ("RB", "FB"):
            return "RB"
        if p in ("WR",):
            return "WR"
        if p in ("TE",):
            return "TE"
        if p in ("C", "G", "T", "OT", "OG", "OC"):
            return "OL"
        if p in ("DE", "DT", "NT", "DL"):
            return "DL"
        if p in ("ILB", "OLB", "LB"):
            return "LB"
        if p in ("CB", "S", "DB", "FS", "SS"):
            return "DB"
        if p in ("K", "P", "LS"):
            return "ST"
        return "OTHER"

    combined["position"] = combined["gsis_id"].apply(_assign_position)
    # Fix SKILL → WR or TE using position map
    mask = combined["position"] == "SKILL"
    if mask.any():
        combined.loc[mask, "position"] = combined.loc[mask, "gsis_id"].apply(
            lambda g: position_map.get(g, "WR")
        )
    combined = combined.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)

    # Also include snap% for tracking playing time
    snap_pct = snaps[["gsis_id", "season", "week", "team", "offense_pct", "defense_pct"]].copy()
    snap_pct["season"] = snap_pct["season"].astype(int)
    snap_pct["week"] = snap_pct["week"].astype(int)
    snap_pct["snap_pct"] = snap_pct[["offense_pct", "defense_pct"]].max(axis=1).fillna(0)

    combined = combined.merge(
        snap_pct[["gsis_id", "season", "week", "snap_pct"]],
        on=["gsis_id", "season", "week"],
        how="left",
    )
    combined["snap_pct"] = combined["snap_pct"].fillna(0)

    return combined


def identify_return_events(
    player_games: pd.DataFrame,
    injury_data: pd.DataFrame,
    min_games_out: int = MIN_GAMES_OUT,
) -> pd.DataFrame:
    """Match injury data to player-game table, identify return events.

    A return event: player was Out 2+ consecutive weeks, then plays again.
    """
    inj = injury_data.copy()
    inj = inj.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
    inj["is_out"] = inj["report_status"].str.strip().str.lower().isin(["out", "doubtful"])
    inj["injury_group"] = inj["report_primary_injury"].apply(_group_injury_type)

    events = []
    for gsid, grp in inj.groupby("gsis_id"):
        grp = grp.sort_values(["season", "week"]).reset_index(drop=True)
        in_streak = False
        streak_start_s = None
        streak_start_w = None
        streak_len = 0
        injury_types = []

        for _, row in grp.iterrows():
            if row["is_out"]:
                if not in_streak:
                    in_streak = True
                    streak_start_s = row["season"]
                    streak_start_w = row["week"]
                    streak_len = 1
                    injury_types = [row["injury_group"]]
                else:
                    streak_len += 1
                    if row["injury_group"] not in injury_types:
                        injury_types.append(row["injury_group"])
            else:
                if in_streak and streak_len >= min_games_out:
                    events.append({
                        "gsis_id": gsid,
                        "position": row.get("position", ""),
                        "team": row.get("team", ""),
                        "return_season": row["season"],
                        "return_week": row["week"],
                        "games_missed": streak_len,
                        "injury_type": injury_types[0] if injury_types else "other",
                        "injury_types_raw": ",".join(injury_types),
                        "streak_start_season": streak_start_s,
                        "streak_start_week": streak_start_w,
                    })
                in_streak = False
                streak_len = 0
                injury_types = []

    result = pd.DataFrame(events)
    if len(result) == 0:
        return result

    # For each return event, get pre-injury baseline and post-return trajectory
    enriched = []
    for _, ev in result.iterrows():
        gsid = ev["gsis_id"]
        pg = player_games[player_games["gsis_id"] == gsid].sort_values(["season", "week"])

        if len(pg) == 0:
            continue

        # Pre-injury games
        pre = pg[
            (pg["season"] < ev["streak_start_season"]) |
            ((pg["season"] == ev["streak_start_season"]) & (pg["week"] < ev["streak_start_week"]))
        ].tail(BASELINE_GAMES)

        if len(pre) == 0:
            continue

        # Baseline fantasy
        base_fantasy = pre["fantasy_pts"].mean()
        base_epa = pre["qb_epa"].mean()
        base_snap = pre["snap_pct"].mean()
        base_yards = pre["total_yards"].mean()

        ev["baseline_fantasy"] = base_fantasy
        ev["baseline_qb_epa"] = base_epa
        ev["baseline_snap_pct"] = base_snap
        ev["baseline_yards"] = base_yards
        ev["baseline_games"] = len(pre)

        # Post-return games
        post = pg[
            (pg["season"] > ev["return_season"]) |
            ((pg["season"] == ev["return_season"]) & (pg["week"] >= ev["return_week"]))
        ].head(RECOVERY_MAX_WEEKS)

        ev["post_games_tracked"] = len(post)

        # Week-by-week tracking
        for i in range(RECOVERY_MAX_WEEKS):
            if i < len(post):
                w = post.iloc[i]
                ev[f"w{i+1}_fantasy_pts"] = w["fantasy_pts"]
                ev[f"w{i+1}_qb_epa"] = w["qb_epa"]
                ev[f"w{i+1}_snap_pct"] = w["snap_pct"]
                ev[f"w{i+1}_yards"] = w["total_yards"]
                ev[f"w{i+1}_fantasy_deficit"] = base_fantasy - w["fantasy_pts"]
                ev[f"w{i+1}_epa_deficit"] = base_epa - w["qb_epa"]
                ev[f"w{i+1}_yards_deficit"] = base_yards - w["total_yards"]
            else:
                for metric in ["fantasy_pts", "qb_epa", "snap_pct", "yards",
                               "fantasy_deficit", "epa_deficit", "yards_deficit"]:
                    ev[f"w{i+1}_{metric}"] = np.nan

        enriched.append(ev)

    return pd.DataFrame(enriched)


def fit_recovery_curve(returns: pd.DataFrame) -> dict:
    """Fit a simple recovery curve model.

    Returns position-specific average deficits by week post-return,
    plus linear regression coefficients.
    """
    records = []
    for _, ev in returns.iterrows():
        for w in range(1, RECOVERY_MAX_WEEKS + 1):
            def_col = f"w{w}_fantasy_deficit"
            val = ev.get(def_col)
            if val is not None and not np.isnan(val):
                records.append({
                    "gsis_id": ev["gsis_id"],
                    "position": ev.get("position", ""),
                    "injury_type": ev.get("injury_type", "other"),
                    "games_missed": ev.get("games_missed", 0),
                    "weeks_since_return": w,
                    "fantasy_deficit": val,
                    "season": ev.get("return_season", 0),
                })

    df = pd.DataFrame(records)
    if len(df) < 10:
        return {"records": df}

    # Per-position average curves
    curves = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = df[df["position"] == pos]
        if len(sub) < 5:
            continue
        curve = sub.groupby("weeks_since_return").agg(
            avg_deficit=("fantasy_deficit", "mean"),
            std_deficit=("fantasy_deficit", "std"),
            n=("fantasy_deficit", "count"),
        ).reset_index()
        curves[pos] = curve

    # Linear regression: deficit ~ weeks + games_missed + injury_type
    from sklearn.linear_model import LinearRegression

    for pos in ["QB", "RB", "WR", "TE"]:
        sub = df[df["position"] == pos]
        if len(sub) < 10:
            continue
        x = pd.get_dummies(
            sub[["weeks_since_return", "games_missed", "injury_type"]],
            columns=["injury_type"], drop_first=True,
        )
        y = sub["fantasy_deficit"].values
        lr = LinearRegression()
        lr.fit(x, y)
        curves[f"{pos}_model"] = {
            "coef": dict(zip(x.columns, lr.coef_)),
            "intercept": lr.intercept_,
            "r2": lr.score(x, y),
            "n": len(sub),
        }

    return {"records": df, "curves": curves}


def compute_game_recovery_adjustments(
    returns: pd.DataFrame,
    seasons: list[int],
) -> pd.DataFrame:
    """Compute per-game recovery adjustment for every eligible game.

    For each game, identifies returning players on each team and
    computes the total expected fantasy deficit (weighted by position).

    Returns a DataFrame with columns: game_id, home_recovery_adj, away_recovery_adj
    """
    # Build per-game aggregates
    games = returns.groupby(["return_season", "return_week", "team"]).agg(
        num_returning=("gsis_id", "count"),
        qb_returning=("position", lambda x: sum(1 for p in x if p == "QB")),
        rb_returning=("position", lambda x: sum(1 for p in x if p == "RB")),
        wr_returning=("position", lambda x: sum(1 for p in x if p == "WR")),
        te_returning=("position", lambda x: sum(1 for p in x if p == "TE")),
        total_fantasy_deficit_w1=(
            "w1_fantasy_deficit",
            lambda x: sum(v for v in x if not pd.isna(v)),
        ),
    ).reset_index()

    # Load feature table for game_id mapping
    ft = pd.read_parquet("data/features/nfl/feature_table.parquet")
    ft = ft[ft["season"].isin(seasons)].copy()

    home = ft[["game_id", "season", "week", "home_team"]].copy()
    home.rename(columns={"home_team": "team"}, inplace=True)
    home["side"] = "home"

    away = ft[["game_id", "season", "week", "away_team"]].copy()
    away.rename(columns={"away_team": "team"}, inplace=True)
    away["side"] = "away"

    all_games = pd.concat([home, away], ignore_index=True)
    all_games = all_games.merge(
        games,
        left_on=["season", "week", "team"],
        right_on=["return_season", "return_week", "team"],
        how="left",
    )

    for c in ["num_returning", "qb_returning", "rb_returning", "wr_returning",
              "te_returning", "total_fantasy_deficit_w1"]:
        all_games[c] = all_games[c].fillna(0)

    # Weighted recovery adjustment (fantasy pts → logit space)
    # QB: boost (negative deficit = they play better)
    # RB/WR/TE: penalty (positive deficit = they play worse)
    # Scale: 1 fantasy pt ≈ 0.05 logit (calibrated from Elo prob scale)
    fantasy_to_logit = 0.05

    all_games["recovery_adj"] = (
        -all_games["total_fantasy_deficit_w1"] * fantasy_to_logit
    )

    # Pivot to home/away
    home_adj = all_games[all_games["side"] == "home"][["game_id", "recovery_adj"]]
    home_adj.rename(columns={"recovery_adj": "home_recovery_adj"}, inplace=True)
    away_adj = all_games[all_games["side"] == "away"][["game_id", "recovery_adj"]]
    away_adj.rename(columns={"recovery_adj": "away_recovery_adj"}, inplace=True)

    result = home_adj.merge(away_adj, on="game_id", how="outer").fillna(0)
    return result


def run_recovery_analysis(
    seasons: Optional[list[int]] = None,
    min_games_out: int = MIN_GAMES_OUT,
    report_path: str = "reports/experiments/player_recovery_analysis.md",
) -> str:
    """Run full recovery analysis and generate report."""
    import nfl_data_py as nfl

    if seasons is None:
        seasons = [2021, 2022, 2023, 2024, 2025]

    print("=== Player Recovery Analysis ===")
    print(f"Seasons: {seasons}")

    # Load injury data
    print("\nLoading injury data...")
    inj = nfl.import_injuries(seasons)
    inj["gsis_id"] = inj["gsis_id"].str.strip()
    print(f"  Injury rows: {len(inj)}")

    # Build player-game table
    print("\nBuilding player-game performance table (5 seasons of PBP)...")
    pg = build_player_game_table(seasons)
    print(f"  Player-game rows: {len(pg)}")

    # Identify return events
    print("\nIdentifying return events...")
    returns = identify_return_events(pg, inj, min_games_out=min_games_out)
    print(f"  Return events: {len(returns)}")

    # Fit recovery curves
    print("\nFitting recovery curves...")
    model = fit_recovery_curve(returns)
    records = model["records"]
    print(f"  Recovery records: {len(records)}")

    # Print summary
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = returns[returns["position"] == pos]
        if len(sub) == 0:
            continue
        print(f"\n  {pos} ({len(sub)} events):")
        w1 = sub["w1_fantasy_deficit"].dropna()
        w2 = sub.get("w2_fantasy_deficit", pd.Series(dtype=float)).dropna()
        b1 = sub["baseline_fantasy"].dropna()
        a1 = sub.get("w1_fantasy_pts", pd.Series(dtype=float)).dropna()
        if len(w1) > 0:
            print(f"    Baseline fantasy: {b1.mean():.1f}")
            print(f"    Week 1 fantasy: {a1.mean():.1f}")
            print(f"    Week 1 deficit: {w1.mean():+.2f} (n={len(w1)})")
        if len(w2) > 0:
            a2 = sub.get("w2_fantasy_pts", pd.Series(dtype=float)).dropna()
            print(f"    Week 2 fantasy: {a2.mean():.1f}")
            print(f"    Week 2 deficit: {w2.mean():+.2f} (n={len(w2)})")

        # Injury type breakdown
        for it in ["knee", "ankle", "hamstring", "concussion"]:
            it_sub = sub[sub["injury_type"] == it]
            it_w1 = it_sub["w1_fantasy_deficit"].dropna()
            if len(it_w1) >= 3:
                print(f"    {it}: w1 deficit = {it_w1.mean():+.2f} (n={len(it_w1)})")

    # Count multiple-injury players
    returns["player_season"] = (
        returns["return_season"].astype(str) + "_" + returns["gsis_id"]
    )
    inj_counts = returns.groupby("player_season").size().reset_index(name="num_injuries")
    returns = returns.merge(inj_counts, on="player_season")
    multi = returns[returns["num_injuries"] > 1]
    print(f"\n  Players with 2+ injuries in a season: {multi['gsis_id'].nunique()}")

    for pos in ["QB", "RB", "WR", "TE"]:
        single = returns[(returns["position"] == pos) & (returns["num_injuries"] == 1)]
        repeat = returns[(returns["position"] == pos) & (returns["num_injuries"] > 1)]
        s_w1 = single["w1_fantasy_deficit"].dropna()
        r_w1 = repeat["w1_fantasy_deficit"].dropna()
        if len(s_w1) >= 3 and len(r_w1) >= 3:
            print(f"  {pos} single injury: w1 = {s_w1.mean():+.2f} (n={len(s_w1)})")
            print(f"  {pos} repeat injury: w1 = {r_w1.mean():+.2f} (n={len(r_w1)})")

    # Write report
    report = _build_report(returns, model, seasons, min_games_out)
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report)
    print(f"\nReport: {rp}")
    return str(rp)


def _build_report(
    returns: pd.DataFrame,
    model: dict,
    seasons: list[int],
    min_games_out: int,
) -> str:
    lines = [
        "# Player Recovery Analysis",
        "",
        f"Seasons: {', '.join(str(s) for s in seasons)}",
        f"Min games out: {min_games_out}",
        f"Baseline games: {BASELINE_GAMES}",
        f"Return events: {len(returns)}",
        "",
        "## Return Events by Position",
        "",
        "| Position | Events | Avg Games Missed | Week 1 Deficit | "
        "Week 2 Deficit | Week 4 Deficit |",
        "|----------|--------|-----------------|----------------|----------------|----------------|",
    ]

    for pos in ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]:
        sub = returns[returns["position"] == pos]
        if len(sub) == 0:
            continue
        w1 = sub["w1_fantasy_deficit"].dropna()
        w2 = sub.get("w2_fantasy_deficit", pd.Series(dtype=float)).dropna()
        w4 = sub.get("w4_fantasy_deficit", pd.Series(dtype=float)).dropna()
        lines.append(
            f"| {pos} | {len(sub)} | {sub['games_missed'].mean():.1f}"
            f" | {w1.mean():+.2f} (n={len(w1)})"
            f" | {w2.mean():+.2f} (n={len(w2)})"
            f" | {w4.mean():+.2f} (n={len(w4)}) |"
        )

    lines.extend([
        "",
        "## Injury Type Breakdown (Week 1 Deficit)",
        "",
        "| Position | Injury Type | N | Avg Deficit |",
        "|----------|------------|----|-------------|",
    ])

    for pos in ["QB", "RB", "WR", "TE"]:
        sub = returns[returns["position"] == pos]
        if len(sub) == 0:
            continue
        for it in ["knee", "ankle", "hamstring", "concussion", "core",
                   "calf", "foot", "shoulder", "arm", "back", "other"]:
            it_sub = sub[sub["injury_type"] == it]
            w1 = it_sub["w1_fantasy_deficit"].dropna()
            if len(w1) >= 2:
                lines.append(f"| {pos} | {it} | {len(w1)} | {w1.mean():+.2f} |")

    lines.extend([
        "",
        "## Multiple Injury Compounding",
        "",
    ])

    for pos in ["QB", "RB", "WR", "TE"]:
        single = returns[(returns["position"] == pos) & (returns["num_injuries"] == 1)]
        repeat = returns[(returns["position"] == pos) & (returns["num_injuries"] > 1)]
        s_w1 = single["w1_fantasy_deficit"].dropna()
        r_w1 = repeat["w1_fantasy_deficit"].dropna()
        if len(s_w1) >= 2 and len(r_w1) >= 2:
            lines.append(f"- **{pos}**: Single injury w1={s_w1.mean():+.2f} (n={len(s_w1)}), "
                         f"Repeat injury w1={r_w1.mean():+.2f} (n={len(r_w1)})")

    lines.extend([
        "",
        "## Recovery Curve Model",
        "",
    ])

    curves = model.get("curves", {})
    for pos in ["QB", "RB", "WR", "TE"]:
        curve = curves.get(pos)
        if curve is not None:
            lines.append(f"### {pos}")
            lines.append("")
            lines.append("| Week | Avg Deficit | Std | N |")
            lines.append("|------|------------|-----|---|")
            for _, r in curve.iterrows():
                lines.append(
                    f"| {r['weeks_since_return']} | {r['avg_deficit']:+.2f}"
                    f" | {r['std_deficit']:.2f} | {r['n']} |"
                )
            lines.append("")

        model_coef = curves.get(f"{pos}_model")
        if model_coef:
            lines.append(
                f"- Linear model R² = {model_coef['r2']:.3f} (n={model_coef['n']})"
            )
            lines.append(f"- Intercept: {model_coef['intercept']:.3f}")
            top_coefs = sorted(
                model_coef["coef"].items(),
                key=lambda x: abs(x[1]), reverse=True,
            )[:5]
            for k, v in top_coefs:
                lines.append(f"- {k}: {v:+.4f}")
            lines.append("")

    lines.extend([
        "---",
        "Auto-generated by player_recovery.py",
    ])

    return "\n".join(lines)
