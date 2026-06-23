"""Tests for season-specific regression experiment."""

import pandas as pd

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
    qb_change_across_seasons,
    run_season_regression_experiment,
)


def _qb_df():
    """Mini DataFrame tracking QB changes across seasons."""
    rows = []
    # 2021: KC has Mahomes (QB1), BUF has Allen (QB2)
    for week in [1, 2]:
        rows.append(
            {
                "season": 2021,
                "week": week,
                "gameday": f"2021-09-{week:02d}",
                "home_team": "KC",
                "away_team": "BUF",
                "home_qb_id": "QB1",
                "away_qb_id": "QB2",
                "home_win": 1,
                "away_score": 14,
                "away_rest": 7,
                "home_rest": 7,
            }
        )
    # 2022: BUF keeps Allen, KC has Maye (QB3) — QB CHANGE for KC
    for week in [1, 2]:
        rows.append(
            {
                "season": 2022,
                "week": week,
                "gameday": f"2022-09-{week:02d}",
                "home_team": "BUF",
                "away_team": "KC",
                "home_qb_id": "QB2",
                "away_qb_id": "QB3",
                "home_win": 1,
                "away_score": 14,
                "away_rest": 7,
                "home_rest": 7,
            }
        )
    # 2023: KC keeps Maye, BUF keeps Allen — both stable
    for week in [1, 2]:
        rows.append(
            {
                "season": 2023,
                "week": week,
                "gameday": f"2023-09-{week:02d}",
                "home_team": "KC",
                "away_team": "BUF",
                "home_qb_id": "QB3",
                "away_qb_id": "QB2",
                "home_win": 1,
                "away_score": 14,
                "away_rest": 7,
                "home_rest": 7,
            }
        )
    return pd.DataFrame(rows)


class TestQBChangeDetection:
    def test_detects_qb_change(self):
        df = _qb_df()
        change_map = qb_change_across_seasons(df)
        # 2021 has no prior season
        assert 2021 not in change_map or change_map.get(2021, []) == []
        # 2022: KC changed QB1->QB3
        assert "KC" in change_map.get(2022, [])
        # 2023: no changes (both teams kept QBs)
        changes_2023 = change_map.get(2023, [])
        assert "KC" not in changes_2023
        assert "BUF" not in changes_2023

    def test_no_change_same_qb(self):
        df = _qb_df()
        # In 2023, BUF has QB2 both years
        change_map = qb_change_across_seasons(df)
        assert "BUF" not in change_map.get(2023, [])

    def test_unknown_qb_handled(self):
        df = _qb_df()
        df.loc[0, "home_qb_id"] = None
        change_map = qb_change_across_seasons(df)
        # Should not crash
        assert isinstance(change_map, dict)

    def test_change_map_contains_season_keys(self):
        df = _qb_df()
        change_map = qb_change_across_seasons(df)
        for s in [2021, 2022, 2023]:
            assert s in change_map


class TestBuildOverrides:
    def test_qb_change_team_in_overrides(self):
        df = _qb_df()
        overrides = build_team_regression_overrides(
            df, preseason_regression=0.20, qb_change_bonus=0.30
        )
        # KC had QB change entering 2022 (QB1->QB3)
        assert overrides is not None
        assert "KC" in overrides
        assert overrides["KC"] == 0.50  # 0.20 + 0.30

    def test_stable_team_not_in_overrides(self):
        df = _qb_df()
        overrides = build_team_regression_overrides(
            df, preseason_regression=0.20, qb_change_bonus=0.30
        )
        # BUF had no QB change
        assert overrides is None or "BUF" not in overrides

    def test_bonus_capped_at_one(self):
        df = _qb_df()
        overrides = build_team_regression_overrides(
            df, preseason_regression=0.50, qb_change_bonus=0.60
        )
        assert overrides["KC"] <= 1.0

    def test_no_bonus_returns_none(self):
        df = _qb_df()
        overrides = build_team_regression_overrides(
            df, preseason_regression=0.0, qb_change_bonus=0.0
        )
        assert overrides is None

    def test_season_outside_train_ignored(self):
        df = _qb_df()
        # KC changed QB entering 2022; 2022 > min season so should be included
        overrides = build_team_regression_overrides(
            df, preseason_regression=0.20, qb_change_bonus=0.30
        )
        assert overrides is not None
        assert "KC" in overrides


class TestExperiment:
    def test_importable(self):
        assert callable(run_season_regression_experiment)

    def test_folds_exclude_holdout(self):
        for train_s, val_s in ROLLING_FOLDS:
            all_seasons = list(train_s) + [val_s]
            assert HOLDOUT_SEASON not in all_seasons

    def test_folds_sequential(self):
        expected_folds = [
            ([2021], 2022),
            ([2021, 2022], 2023),
            ([2021, 2022, 2023], 2024),
        ]
        assert len(ROLLING_FOLDS) == 3
        for i, (train_s, val_s) in enumerate(ROLLING_FOLDS):
            assert train_s == expected_folds[i][0], f"Fold {i} train mismatch"
            assert val_s == expected_folds[i][1], f"Fold {i} val mismatch"
