"""Tests for scheduling/rest feature computation and experiment."""

import pandas as pd

from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, NEUTRAL_COLUMN, TARGET_COLUMN
from sportslab.features.scheduling import SCHEDULING_FEATURE_COLUMNS, compute_scheduling_features


def _make_minimal_df() -> pd.DataFrame:
    """Create a minimal 2-team DataFrame over a few weeks."""
    rows = []
    for season in [2021, 2022]:
        for week in range(1, 5):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "gameday": f"{season}-09-0{week}",
                    "home_team": "CHI",
                    "away_team": "GB",
                    "home_rest": 7,
                    "away_rest": 7,
                    "weekday": "Sunday",
                    "stadium_id": "CHI00",
                    "location": "Home",
                    "is_neutral": False,
                    "home_win": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    df[TARGET_COLUMN] = df["home_win"]
    df[MODEL_ELIGIBLE_COLUMN] = True
    df[NEUTRAL_COLUMN] = False
    return df


class TestSchedulingFeatureFlags:
    def test_home_short_week_flag(self):
        df = _make_minimal_df()
        df.loc[0, "home_rest"] = 4
        result = compute_scheduling_features(df)
        assert result.loc[0, "home_short_week"] == 1

    def test_away_short_week_flag(self):
        df = _make_minimal_df()
        df.loc[0, "away_rest"] = 6
        result = compute_scheduling_features(df)
        assert result.loc[0, "away_short_week"] == 1

    def test_normal_rest_not_short(self):
        df = _make_minimal_df()
        result = compute_scheduling_features(df)
        assert result.loc[0, "home_short_week"] == 0
        assert result.loc[0, "away_short_week"] == 0

    def test_home_off_bye_flag(self):
        df = _make_minimal_df()
        df.loc[0, "home_rest"] = 14
        result = compute_scheduling_features(df)
        assert result.loc[0, "home_off_bye"] == 1

    def test_away_off_bye_flag(self):
        df = _make_minimal_df()
        df.loc[0, "away_rest"] = 13
        result = compute_scheduling_features(df)
        assert result.loc[0, "away_off_bye"] == 1

    def test_thursday_flag(self):
        df = _make_minimal_df()
        df.loc[0, "weekday"] = "Thursday"
        result = compute_scheduling_features(df)
        assert result.loc[0, "thursday_flag"] == 1

    def test_monday_flag(self):
        df = _make_minimal_df()
        df.loc[0, "weekday"] = "Monday"
        result = compute_scheduling_features(df)
        assert result.loc[0, "monday_flag"] == 1

    def test_sunday_not_flagged(self):
        df = _make_minimal_df()
        result = compute_scheduling_features(df)
        assert result.loc[0, "thursday_flag"] == 0
        assert result.loc[0, "monday_flag"] == 0

    def test_international_flag(self):
        df = _make_minimal_df()
        df.loc[0, "stadium_id"] = "LON00"
        df.loc[0, "is_neutral"] = True
        result = compute_scheduling_features(df)
        assert result.loc[0, "is_international"] == 1

    def test_domestic_not_international(self):
        df = _make_minimal_df()
        result = compute_scheduling_features(df)
        assert result.loc[0, "is_international"] == 0


class TestConsecutiveRoad:
    def test_first_game_defaults(self):
        df = _make_minimal_df()
        result = compute_scheduling_features(df)
        # First game: both teams start at 0
        assert result.loc[0, "home_consecutive_road"] == 0
        assert result.loc[0, "away_consecutive_road"] == 0

    def test_away_team_accumulates(self):
        """Away team playing consecutive road games should increment."""
        rows = [
            {
                "season": 2021,
                "week": 1,
                "gameday": "2021-09-12",
                "home_team": "GB",
                "away_team": "CHI",
                "home_rest": 7,
                "away_rest": 7,
                "weekday": "Sunday",
                "stadium_id": "GBP00",
                "location": "Home",
                "is_neutral": False,
                "home_win": 1.0,
            },
            {
                "season": 2021,
                "week": 2,
                "gameday": "2021-09-19",
                "home_team": "MIN",
                "away_team": "CHI",
                "home_rest": 7,
                "away_rest": 7,
                "weekday": "Sunday",
                "stadium_id": "MIN00",
                "location": "Home",
                "is_neutral": False,
                "home_win": 1.0,
            },
        ]
        df = pd.DataFrame(rows)
        df[TARGET_COLUMN] = df["home_win"]
        df[MODEL_ELIGIBLE_COLUMN] = True

        result = compute_scheduling_features(df)
        # Game 1: away_team=CHI, 0 consecutive before this game
        assert result.loc[0, "away_consecutive_road"] == 0
        # Game 2: away_team=CHI again, should have 1 (was away week 1)
        assert result.loc[1, "away_consecutive_road"] == 1

    def test_home_team_resets_after_home_game(self):
        """Home team at home should have 0 consecutive road."""
        rows = [
            {
                "season": 2021,
                "week": 1,
                "gameday": "2021-09-12",
                "home_team": "CHI",
                "away_team": "GB",
                "home_rest": 7,
                "away_rest": 7,
                "weekday": "Sunday",
                "stadium_id": "CHI00",
                "location": "Home",
                "is_neutral": False,
                "home_win": 1.0,
            },
            {
                "season": 2021,
                "week": 2,
                "gameday": "2021-09-19",
                "home_team": "CHI",
                "away_team": "GB",
                "home_rest": 7,
                "away_rest": 7,
                "weekday": "Sunday",
                "stadium_id": "CHI00",
                "location": "Home",
                "is_neutral": False,
                "home_win": 1.0,
            },
        ]
        df = pd.DataFrame(rows)
        df[TARGET_COLUMN] = df["home_win"]
        df[MODEL_ELIGIBLE_COLUMN] = True
        df[NEUTRAL_COLUMN] = False

        result = compute_scheduling_features(df)
        # Home team CHI: both games at home, consecutive_road = 0
        assert result.loc[0, "home_consecutive_road"] == 0
        assert result.loc[1, "home_consecutive_road"] == 0

    def test_neutral_counts_as_road_for_home_team(self):
        """Home team at neutral site increments consecutive road."""
        rows = [
            {
                "season": 2021,
                "week": 1,
                "gameday": "2021-09-12",
                "home_team": "CHI",
                "away_team": "GB",
                "home_rest": 7,
                "away_rest": 7,
                "weekday": "Sunday",
                "stadium_id": "LON00",
                "location": "Neutral",
                "is_neutral": True,
                "home_win": 1.0,
            },
            {
                "season": 2021,
                "week": 2,
                "gameday": "2021-09-19",
                "home_team": "CHI",
                "away_team": "GB",
                "home_rest": 7,
                "away_rest": 7,
                "weekday": "Sunday",
                "stadium_id": "LON00",
                "location": "Neutral",
                "is_neutral": True,
                "home_win": 1.0,
            },
        ]
        df = pd.DataFrame(rows)
        df[TARGET_COLUMN] = df["home_win"]
        df[MODEL_ELIGIBLE_COLUMN] = True

        result = compute_scheduling_features(df)
        # Home team at neutral: counts as road
        assert result.loc[0, "home_consecutive_road"] == 0  # before first game
        assert result.loc[1, "home_consecutive_road"] == 1  # after one neutral game


class TestFlagsFromComputedData:
    def test_all_scheduling_columns_present(self):
        """Verify all SCHEDULING_FEATURE_COLUMNS are computed."""
        df = _make_minimal_df()
        result = compute_scheduling_features(df)
        for col in SCHEDULING_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_missing_values_in_scheduling_cols(self):
        """No NaN values in computed scheduling features."""
        df = _make_minimal_df()
        result = compute_scheduling_features(df)
        for col in SCHEDULING_FEATURE_COLUMNS:
            assert result[col].isna().sum() == 0, f"NaN in {col}"


class TestScheduleRestExperiment:
    def test_holdout_not_used_in_validation(self):
        """Verify experiment's rolling folds exclude 2025."""
        from sportslab.evaluation.schedule_rest_experiment import HOLDOUT_SEASON, ROLLING_FOLDS

        fold_seasons = set()
        for train_seasons, val_season in ROLLING_FOLDS:
            fold_seasons.update(train_seasons)
            fold_seasons.add(val_season)
        assert HOLDOUT_SEASON not in fold_seasons

    def test_folds_only_2021_plus(self):
        from sportslab.evaluation.schedule_rest_experiment import ROLLING_FOLDS

        for train_seasons, val_season in ROLLING_FOLDS:
            for s in train_seasons + [val_season]:
                assert s >= 2021
