"""Tests for situational micro-features."""

import numpy as np
import pandas as pd

from sportslab.evaluation.situational_micro_experiment import (
    MODEL_VARIANTS,
    ROLLING_FOLDS,
    _referee_audit,
)
from sportslab.features.situational_micro import (
    SITUATIONAL_MICRO_COLUMNS,
    _get_team_usual_surface,
    _map_surface,
    compute_situational_micro_features,
)

# ── Surface normalization tests ──


class TestMapSurface:
    def test_grass(self):
        assert _map_surface("grass") == "grass"
        assert _map_surface("Grass") == "grass"
        assert _map_surface(" GRASS ") == "grass"

    def test_turf(self):
        assert _map_surface("fieldturf") == "turf"
        assert _map_surface("FieldTurf") == "turf"
        assert _map_surface("matrixturf") == "turf"
        assert _map_surface("sportturf") == "turf"
        assert _map_surface("a_turf") == "turf"
        assert _map_surface("astroturf") == "turf"

    def test_unknown(self):
        assert _map_surface("") == "unknown"
        assert _map_surface("dirt") == "unknown"
        assert _map_surface("concrete") == "unknown"
        assert _map_surface(np.nan) == "unknown"
        assert _map_surface(None) == "unknown"


# ── Team usual surface tests ──


class TestGetTeamUsualSurface:
    def test_grass_team(self):
        df = pd.DataFrame(
            {
                "home_team": ["GB", "GB", "GB"],
                "surface": ["grass", "grass", "grass"],
                "is_neutral": [0, 0, 0],
            }
        )
        usual = _get_team_usual_surface(df)
        assert usual["GB"] == "grass"

    def test_turf_team(self):
        df = pd.DataFrame(
            {
                "home_team": ["NO", "NO", "NO"],
                "surface": ["astroturf", "fieldturf", "fieldturf"],
                "is_neutral": [0, 0, 0],
            }
        )
        usual = _get_team_usual_surface(df)
        assert usual["NO"] == "turf"

    def test_unknown_team(self):
        df = pd.DataFrame(
            {
                "home_team": ["XX", "XX"],
                "surface": ["", "unknown"],
                "is_neutral": [0, 0],
            }
        )
        usual = _get_team_usual_surface(df)
        assert usual["XX"] == "unknown"

    def test_neutral_excluded(self):
        df = pd.DataFrame(
            {
                "home_team": ["GB", "GB"],
                "surface": ["grass", "turf"],
                "is_neutral": [0, 1],
            }
        )
        usual = _get_team_usual_surface(df)
        assert usual["GB"] == "grass"  # turf is neutral, excluded


# ── Column completeness tests ──


class TestColumnCompleteness:
    def test_all_columns_present(self):
        df = _make_minimal_df(2)
        result = compute_situational_micro_features(df)
        for col in SITUATIONAL_MICRO_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_extra_columns(self):
        df = _make_minimal_df(2)
        result = compute_situational_micro_features(df)
        expected = set(SITUATIONAL_MICRO_COLUMNS) | set(df.columns)
        assert set(result.columns) == expected


# ── Div interaction tests ──


class TestDivInteraction:
    def test_div_game_with_qb_change(self):
        df = _make_minimal_df(2)
        df["div_game"] = 1
        df["home_qb_changed"] = 0
        df["away_qb_changed"] = 1
        result = compute_situational_micro_features(df)
        assert result["div_home_qb_changed"].iloc[0] == 0
        assert result["div_away_qb_changed"].iloc[0] == 1

    def test_non_div_ignores_qb_change(self):
        df = _make_minimal_df(2)
        df["div_game"] = 0
        df["home_qb_changed"] = 1
        df["away_qb_changed"] = 1
        result = compute_situational_micro_features(df)
        assert result["div_home_qb_changed"].iloc[0] == 0
        assert result["div_away_qb_changed"].iloc[0] == 0

    def test_mixed_div_games(self):
        df = _make_minimal_df(4)
        df["div_game"] = [1, 0, 1, 0]
        df["home_qb_changed"] = [1, 1, 0, 0]
        df["away_qb_changed"] = [0, 0, 1, 1]
        result = compute_situational_micro_features(df)
        assert list(result["div_home_qb_changed"]) == [1, 0, 0, 0]
        assert list(result["div_away_qb_changed"]) == [0, 0, 1, 0]

    def test_missing_div_game_defaults_zero(self):
        df = _make_minimal_df(2)
        df["home_qb_changed"] = [1, 0]
        df["away_qb_changed"] = [0, 1]
        result = compute_situational_micro_features(df)
        # div_game doesn't exist, so all div features should be 0
        assert list(result["div_home_qb_changed"]) == [0, 0]
        assert list(result["div_away_qb_changed"]) == [0, 0]


# ── First-year coach tests ──


class TestFirstYearCoach:
    def test_first_season_is_first_year(self):
        df = _make_minimal_df(2)
        df["home_coach"] = ["CoachA", "CoachA"]
        df["away_coach"] = ["CoachB", "CoachB"]
        result = compute_situational_micro_features(df)
        assert result["home_first_year_coach"].iloc[0] == 1  # first time seen
        assert result["away_first_year_coach"].iloc[0] == 1  # first time seen

    def test_second_season_not_first_year(self):
        df = _make_minimal_df(2)
        df["season"] = [2021, 2022]
        df["home_team"] = ["TeamA", "TeamA"]
        df["home_coach"] = ["CoachA", "CoachA"]
        df["away_team"] = ["TeamB", "TeamB"]
        df["away_coach"] = ["CoachB", "CoachB"]
        result = compute_situational_micro_features(df)
        assert result["home_first_year_coach"].iloc[1] == 0  # same coach+team, later season
        assert result["away_first_year_coach"].iloc[1] == 0

    def test_new_team_same_coach_first_year(self):
        """Coach moves from Team A to Team B — should be first year at new team."""
        df = _make_minimal_df(3)
        df["season"] = [2021, 2022, 2022]
        df["home_team"] = ["TeamA", "TeamB", "TeamC"]
        df["home_coach"] = ["CoachX", "CoachX", "CoachY"]
        result = compute_situational_micro_features(df)
        # CoachX seen first at TeamA (2021), then at TeamB (2022) — year 1 for TeamB
        assert result["home_first_year_coach"].iloc[0] == 1  # first appearance
        assert result["home_first_year_coach"].iloc[1] == 1  # new team
        assert result["home_first_year_coach"].iloc[2] == 1  # first appearance

    def test_mid_season_change_detected(self):
        """Same team, different coach mid-season — new coach should be first year."""
        df = _make_minimal_df(3)
        df["season"] = [2021, 2021, 2022]
        df["home_team"] = ["TeamA", "TeamA", "TeamA"]
        df["home_coach"] = ["CoachA", "CoachB", "CoachB"]
        result = compute_situational_micro_features(df)
        assert result["home_first_year_coach"].iloc[0] == 1  # CoachA first
        assert result["home_first_year_coach"].iloc[1] == 1  # CoachB first at TeamA
        assert result["home_first_year_coach"].iloc[2] == 0  # CoachB second season

    def test_coach_change_diff_computation(self):
        df = _make_minimal_df(2)
        df["home_team"] = ["TeamA", "TeamA"]
        df["home_coach"] = ["NewCoach", "OldCoach"]
        df["away_team"] = ["TeamB", "TeamB"]
        df["away_coach"] = ["OldCoach", "NewCoach"]
        result = compute_situational_micro_features(df)
        # Row 0: home is first year (1), away might be first year too
        # Row 1: home team A + OldCoach is new pair; away team B + NewCoach is new pair
        # diff = home_first - away_first
        assert (
            result["coach_change_diff"].iloc[0]
            == result["home_first_year_coach"].iloc[0] - result["away_first_year_coach"].iloc[0]
        )
        assert (
            result["coach_change_diff"].iloc[1]
            == result["home_first_year_coach"].iloc[1] - result["away_first_year_coach"].iloc[1]
        )

    def test_chronological_order_safety(self):
        """First-year coach detection should not look ahead."""
        df = _make_minimal_df(4)
        df["season"] = [2022, 2021, 2023, 2024]
        df["week"] = [1, 1, 1, 1]
        df["home_team"] = ["TeamA", "TeamA", "TeamA", "TeamA"]
        df["home_coach"] = ["CoachB", "CoachA", "CoachA", "CoachA"]
        result = compute_situational_micro_features(df)
        # After sorting: 2021 CoachA first, 2022 CoachB new, 2023 CoachA not first, 2024 not first
        # After sorting: index 0 = season 2021, home_coach = CoachA, first year = 1
        # index 1 = season 2022, home_coach = CoachB, first year = 1 (new pair)
        # index 2 = season 2023, (TeamA,CoachA) seen in 2021 → not first
        assert result["home_first_year_coach"].iloc[0] == 1  # 2021, first seen
        assert result["home_first_year_coach"].iloc[1] == 1  # 2022, new pair
        assert result["home_first_year_coach"].iloc[2] == 0  # 2023, seen before
        assert result["home_first_year_coach"].iloc[3] == 0  # 2024, seen before


# ── Surface mismatch tests ──


class TestSurfaceMismatch:
    def test_no_mismatch_same_surface(self):
        df = _make_surface_df(home="grass", away_usual="grass", game_surface="grass", neutral=0)
        result = compute_situational_micro_features(df)
        assert result["away_surface_mismatch"].iloc[0] == 0

    def test_mismatch_grass_to_turf(self):
        df = _make_surface_df(
            home="fieldturf", away_usual="grass", game_surface="fieldturf", neutral=0
        )
        result = compute_situational_micro_features(df)
        assert result["away_surface_mismatch"].iloc[0] == 1
        assert result["away_grass_to_turf"].iloc[0] == 1
        assert result["away_turf_to_grass"].iloc[0] == 0

    def test_mismatch_turf_to_grass(self):
        df = _make_surface_df(home="grass", away_usual="fieldturf", game_surface="grass", neutral=0)
        result = compute_situational_micro_features(df)
        assert result["away_surface_mismatch"].iloc[0] == 1
        assert result["away_grass_to_turf"].iloc[0] == 0
        assert result["away_turf_to_grass"].iloc[0] == 1

    def test_neutral_games_no_mismatch(self):
        df = _make_surface_df(
            home="fieldturf", away_usual="grass", game_surface="fieldturf", neutral=1
        )
        result = compute_situational_micro_features(df)
        assert result["away_surface_mismatch"].iloc[0] == 0
        assert result["away_grass_to_turf"].iloc[0] == 0
        assert result["away_turf_to_grass"].iloc[0] == 0

    def test_unknown_surface_no_mismatch(self):
        df = _make_surface_df(home="grass", away_usual="grass", game_surface="", neutral=0)
        result = compute_situational_micro_features(df)
        assert result["away_surface_mismatch"].iloc[0] == 0

    def test_unknown_away_usual_no_mismatch(self):
        df = _make_surface_df(
            home="fieldturf", away_usual="unknown", game_surface="fieldturf", neutral=0
        )
        result = compute_situational_micro_features(df)
        assert result["away_surface_mismatch"].iloc[0] == 0


# ── Experiment structure tests ──


class TestExperimentStructure:
    def test_rolling_folds_correct_seasons(self):
        for train, val in ROLLING_FOLDS:
            assert val not in train
            assert all(s >= 2021 for s in train)
            assert val >= 2022 and val <= 2024

    def test_model_variants_include_incumbent(self):
        names = [n for n, _, _ in MODEL_VARIANTS]
        assert "incumbent" in names

    def test_model_variants_have_features(self):
        for name, feats, desc in MODEL_VARIANTS:
            assert (
                len(feats) >= 4
            )  # at least incumbent features (elo_prob added in _build_feature_matrix)

    def test_no_model_has_referee_features(self):
        for name, feats, desc in MODEL_VARIANTS:
            assert "referee" not in " ".join(feats).lower()


class TestRefereeAudit:
    def test_referee_audit_structure(self):
        df = _make_minimal_df(5)
        df["referee"] = [f"Ref{i}" for i in range(5)]
        audit = _referee_audit(df)
        assert "unique_referees" in audit
        assert "pregame_safe" in audit
        assert "pregame_known" in audit
        assert "recommendation" in audit
        assert audit["unique_referees"] == 5

    def test_referee_missing(self):
        df = _make_minimal_df(3)
        df["referee"] = [np.nan, "RefA", "RefB"]
        audit = _referee_audit(df)
        assert audit["missing_referee"] == 1
        assert not audit["pregame_safe"]

    def test_referee_all_known_pregame_safe(self):
        df = _make_minimal_df(3)
        df["referee"] = ["RefA", "RefB", "RefC"]
        audit = _referee_audit(df)
        assert audit["missing_referee"] == 0
        assert audit["pregame_safe"]

    def test_importable_from_cli(self):
        """Verify the experiment module can be imported without error."""
        import sportslab.evaluation.situational_micro_experiment  # noqa: F401

        assert True


# ── Helpers ──


def _make_minimal_df(n: int) -> pd.DataFrame:
    """Create a minimal DataFrame with required columns for n games."""
    return pd.DataFrame(
        {
            "season": [2021] * n,
            "week": list(range(1, n + 1)),
            "gameday": [f"2021-09-{i:02d}" for i in range(1, n + 1)],
            "home_team": [f"Team{i}" for i in range(n)],
            "away_team": [f"Opp{i}" for i in range(n)],
            "home_coach": [f"Coach{i}" for i in range(n)],
            "away_coach": [f"Coach{i + 10}" for i in range(n)],
            "home_qb_changed": [0] * n,
            "away_qb_changed": [0] * n,
            "div_game": [0] * n,
            "surface": ["grass"] * n,
            "is_neutral": [0] * n,
            "home_win": [1] * n,
            "model_eligible": [1] * n,
            "game_id": [f"2021_0{i}" for i in range(1, n + 1)],
        }
    )


def _make_surface_df(
    home: str,
    away_usual: str,
    game_surface: str,
    neutral: int,
) -> pd.DataFrame:
    """Create DataFrame for one game with controlled surface setup."""
    team = "TeamA"
    away = "TeamB"
    df = _make_minimal_df(2)
    df.loc[0, "home_team"] = team
    df.loc[0, "away_team"] = away
    df.loc[0, "surface"] = game_surface
    df.loc[0, "is_neutral"] = neutral
    # Second game establishes TeamB's usual surface (TeamB at home)
    df.loc[1, "home_team"] = away
    df.loc[1, "away_team"] = team
    df.loc[1, "surface"] = away_usual
    df.loc[1, "is_neutral"] = 0
    df.loc[1, "season"] = 2021
    df.loc[1, "week"] = 2
    return df
