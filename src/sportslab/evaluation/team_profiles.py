"""Per-team-season profiles for all StatSpace metrics."""


import pandas as pd

from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN
from sportslab.features.epa import load_pbp_data
from sportslab.features.statspace import (
    compute_statspace_chaos_rate,
    compute_statspace_coward_tax,
    compute_statspace_doba,
    compute_statspace_fdr,
    compute_statspace_qb_lift,
    schedule_to_nfl_historical_games,
)

_FDR_VALUE_COL = ["fraud_detector_rating", "record_strength_z", "underlying_quality_z",
                  "elo_edge_z", "suspicion_z", "luck_gap_z", "close_game_luck_z",
                  "turnover_luck_z"]


def build_team_profiles(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    output_path: str = "reports/team_profiles.csv",
) -> pd.DataFrame:
    """Build per-team-season profiles with all StatSpace metrics."""

    ft = pd.read_parquet(ft_path)
    all_seasons = sorted(ft[ft[MODEL_ELIGIBLE_COLUMN]]["season"].unique())
    all_seasons = [s for s in all_seasons if s != 2026]

    print(f"Loading PBP for {all_seasons}...")
    pbp = load_pbp_data(seasons=all_seasons)
    print(f"PBP loaded: {pbp.shape}")

    # Build profiles keyed by (team, season)
    profiles = {}

    for s in all_seasons:
        s_pbp = pbp[pbp["season"] == s].copy()

        # Coward Tax
        try:
            ct = compute_statspace_coward_tax(s_pbp, season=s)
            if ct is not None and not ct.empty:
                for _, r in ct.iterrows():
                    k = (r["team"], s)
                    profiles[k] = profiles.get(k, {"team": r["team"], "season": s})
                    profiles[k]["coward_tax_per_game"] = r.get("coward_tax_per_game")
                    profiles[k]["aggression_score"] = r.get("aggression_score")
                    profiles[k]["aggressive_decisions"] = r.get("aggressive_decisions")
                    profiles[k]["conservative_decisions"] = r.get("conservative_decisions")
        except Exception as e:
            print(f"  Coward Tax {s} failed: {e}")

        # Chaos Rate
        try:
            cr = compute_statspace_chaos_rate(s_pbp, season=s)
            if cr is not None and not cr.empty:
                for _, r in cr.iterrows():
                    k = (r["team"], s)
                    profiles[k] = profiles.get(k, {"team": r["team"], "season": s})
                    profiles[k]["chaos_rate"] = r.get("chaos_rate")
        except Exception as e:
            print(f"  Chaos Rate {s} failed: {e}")

        # DOBA
        try:
            db = compute_statspace_doba(s_pbp, season=s)
            if db is not None and not db.empty:
                for _, r in db.iterrows():
                    k = (r["team"], s)
                    profiles[k] = profiles.get(k, {"team": r["team"], "season": s})
                    profiles[k]["doba_score"] = r.get("doba_score")
        except Exception as e:
            print(f"  DOBA {s} failed: {e}")

        # QB Lift (primary QB per team-season)
        try:
            qb = compute_statspace_qb_lift(s_pbp, season=s, min_dropbacks=50)
            if qb is not None and not qb.empty:
                # For each team, take the QB with highest qb_lift_index (the starter)
                qb_sorted = qb.sort_values("qb_lift_index", ascending=False)
                for _, r in qb_sorted.iterrows():
                    k = (r["team"], s)
                    if k in profiles or True:
                        profiles.setdefault(k, {"team": r["team"], "season": s})
                    # Only set if not already set (first = highest lift = starter)
                    if "qb_lift_index" not in profiles[k]:
                        profiles[k]["qb_lift_index"] = r.get("qb_lift_index")
                        profiles[k]["qb_player"] = r.get("player")
                        profiles[k]["support_dependency_score"] = r.get("support_dependency_score")
                        profiles[k]["support_dependency_label"] = r.get("support_dependency_label")
                        profiles[k]["dropbacks"] = r.get("dropbacks")
                        profiles[k]["epa_per_dropback"] = r.get("epa_per_dropback")
                        profiles[k]["cpoe"] = r.get("cpoe")
        except Exception as e:
            print(f"  QB Lift {s} failed: {e}")

    # FDR - requires schedule, compute separately
    try:
        for s in all_seasons:
            s_sched = ft[(ft["season"] == s) & ft[MODEL_ELIGIBLE_COLUMN]].copy()
            if s_sched.empty:
                continue
            s_pbp = pbp[pbp["season"] == s].copy()
            games = schedule_to_nfl_historical_games(s_sched)
            if not games:
                continue
            fdr = compute_statspace_fdr(games, pbp_df=s_pbp, season=s)
            if fdr is not None and not fdr.empty:
                for _, r in fdr.iterrows():
                    k = (r["team"], s)
                    profiles.setdefault(k, {"team": r["team"], "season": s})
                    profiles[k]["fraud_detector_rating"] = r.get("fraud_detector_rating")
    except Exception as e:
        print(f"  FDR failed: {e}")

    result = pd.DataFrame(list(profiles.values()))
    if not result.empty:
        result = result.sort_values(["season", "team"]).reset_index(drop=True)
        result.to_csv(output_path, index=False)
        print(f"\nTeam profiles saved to {output_path}")
        print(f"  {len(result)} team-seasons, {len(result.columns)} columns")
        print(f"  Columns: {', '.join(result.columns)}")
    else:
        print("No profiles generated")

    return result
