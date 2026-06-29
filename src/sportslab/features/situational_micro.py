"""Situational micro-features — divisional, first-year coach, surface mismatch.

Chronologically computed, pregame-safe, no leakage. No market features.
"""

from typing import Dict, List, Tuple

import pandas as pd

SITUATIONAL_MICRO_COLUMNS = [
    # Divisional interaction
    "div_home_qb_changed",
    "div_away_qb_changed",
    # First-year coach
    "home_first_year_coach",
    "away_first_year_coach",
    "coach_change_diff",
    # Surface mismatch
    "away_surface_mismatch",
    "away_grass_to_turf",
    "away_turf_to_grass",
]

SURFACE_GRASS = "grass"
SURFACE_TURF = "turf"
SURFACE_UNKNOWN = "unknown"


def _map_surface(surface: str) -> str:
    """Normalize a surface string to grass/turf/unknown."""
    s = str(surface).strip().lower()
    if s == "grass":
        return SURFACE_GRASS
    if s in ("fieldturf", "matrixturf", "sportturf", "a_turf", "astroturf"):
        return SURFACE_TURF
    return SURFACE_UNKNOWN


def _get_team_usual_surface(df: pd.DataFrame) -> Dict[str, str]:
    """Compute each team's most common home surface from non-neutral games."""
    home = df[df["is_neutral"] != 1][["home_team", "surface"]].copy()
    home["surface_cat"] = home["surface"].apply(_map_surface)
    mode_map = {}
    for team, grp in home.groupby("home_team"):
        cats = grp["surface_cat"].value_counts()
        # Filter out unknown for mode
        non_unknown = cats[cats.index != SURFACE_UNKNOWN]
        if not non_unknown.empty:
            mode_map[team] = non_unknown.index[0]
        else:
            mode_map[team] = SURFACE_UNKNOWN
    return mode_map


def compute_situational_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add situational micro-features.

    Requires columns: season, week, gameday, home_team, away_team,
    home_coach, away_coach, home_qb_changed, away_qb_changed, div_game,
    surface, is_neutral.

    Must be called AFTER compute_qb_features and compute_situational_features.

    Args:
        df: DataFrame sorted chronologically with required columns.

    Returns:
        DataFrame with added SITUATIONAL_MICRO_COLUMNS.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    # ── Divisional interaction features ──
    div = out["div_game"].fillna(0).astype(int)
    home_qbc = out.get("home_qb_changed", pd.Series(0)).fillna(0).astype(int)
    away_qbc = out.get("away_qb_changed", pd.Series(0)).fillna(0).astype(int)
    out["div_home_qb_changed"] = (div * home_qbc).values
    out["div_away_qb_changed"] = (div * away_qbc).values

    # ── First-year coach features ──
    # Track which coaches are in their first season with a team
    team_coach_seasons: Dict[Tuple[str, str], int] = {}  # (team, coach) -> first season seen

    home_fyc: List[int] = []
    away_fyc: List[int] = []

    for _, row in out.iterrows():
        season = int(row["season"])

        for side, team_col, coach_col in [
            ("home", "home_team", "home_coach"),
            ("away", "away_team", "away_coach"),
        ]:
            team = str(row[team_col])
            coach = str(row[coach_col])
            key = (team, coach)

            if key in team_coach_seasons:
                first_season = team_coach_seasons[key]
                is_first = 1 if season == first_season else 0
            else:
                team_coach_seasons[key] = season
                is_first = 1

            if side == "home":
                home_fyc.append(is_first)
            else:
                away_fyc.append(is_first)

    out["home_first_year_coach"] = home_fyc
    out["away_first_year_coach"] = away_fyc
    out["coach_change_diff"] = (
        pd.Series(home_fyc, dtype=int) - pd.Series(away_fyc, dtype=int)
    ).values

    # ── Surface mismatch features ──
    team_usual = _get_team_usual_surface(out)
    game_surface = out["surface"].apply(_map_surface)

    away_mismatch = []
    away_g2t = []
    away_t2g = []

    for _, row in out.iterrows():
        away_team = str(row["away_team"])
        game_surf = (
            game_surface.iloc[row.name] if hasattr(game_surface, "iloc") else game_surface[row.name]
        )
        away_usual = team_usual.get(away_team, SURFACE_UNKNOWN)

        is_neutral_flag = bool(row.get("is_neutral", False))

        if is_neutral_flag:
            away_mismatch.append(0)
            away_g2t.append(0)
            away_t2g.append(0)
        elif away_usual == SURFACE_UNKNOWN or game_surf == SURFACE_UNKNOWN:
            away_mismatch.append(0)
            away_g2t.append(0)
            away_t2g.append(0)
        else:
            mismatch = 1 if away_usual != game_surf else 0
            away_mismatch.append(mismatch)
            away_g2t.append(1 if (away_usual == SURFACE_GRASS and game_surf == SURFACE_TURF) else 0)
            away_t2g.append(1 if (away_usual == SURFACE_TURF and game_surf == SURFACE_GRASS) else 0)

    out["away_surface_mismatch"] = away_mismatch
    out["away_grass_to_turf"] = away_g2t
    out["away_turf_to_grass"] = away_t2g

    return out
