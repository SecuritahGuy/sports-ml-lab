"""Tests for coach+QB season regression experiment."""

import pandas as pd

from sportslab.evaluation.coach_season_regression_experiment import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    build_team_regression_overrides,
    identity_across_seasons,
    run_coach_season_regression_experiment,
)


def _change_df():
    """Mini DataFrame with QB and coach changes across seasons."""
    rows = []
    # 2021: KC has Mahomes(QB1)/Reid(C1), BUF has Allen(QB2)/McDermott(C2)
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
                "home_coach": "C1",
                "away_coach": "C2",
                "home_win": 1,
                "away_score": 14,
                "away_rest": 7,
                "home_rest": 7,
            }
        )
    # 2022: KC keeps both, BUF keeps both — stable
    for week in [1, 2]:
        rows.append(
            {
                "season": 2022,
                "week": week,
                "gameday": f"2022-09-{week:02d}",
                "home_team": "BUF",
                "away_team": "KC",
                "home_qb_id": "QB2",
                "away_qb_id": "QB1",
                "home_coach": "C2",
                "away_coach": "C1",
                "home_win": 1,
                "away_score": 14,
                "away_rest": 7,
                "home_rest": 7,
            }
        )
    # 2023: KC keeps QB1 but gets new coach C3; BUF unchanged
    for week in [1, 2]:
        rows.append(
            {
                "season": 2023,
                "week": week,
                "gameday": f"2023-09-{week:02d}",
                "home_team": "KC",
                "away_team": "BUF",
                "home_qb_id": "QB1",
                "away_qb_id": "QB2",
                "home_coach": "C3",
                "away_coach": "C2",
                "home_win": 1,
                "away_score": 14,
                "away_rest": 7,
                "home_rest": 7,
            }
        )
    return pd.DataFrame(rows)


class TestIdentityDetection:
    def test_qb_change_detected(self):
        df = _change_df()
        cm = identity_across_seasons(df, "home_qb_id", "away_qb_id")
        # 2021 has no prior
        assert cm.get(2021, []) == []
        # 2022: no QB changes (KC: QB1->QB1, BUF: QB2->QB2)
        assert cm.get(2022, []) == []
        # 2023: no QB changes (KC: QB1->QB1, BUF: QB2->QB2)
        assert cm.get(2023, []) == []

    def test_coach_change_detected(self):
        df = _change_df()
        cm = identity_across_seasons(df, "home_coach", "away_coach")
        # 2021: no prior
        # 2022: no coach changes (KC: C1->C1, BUF: C2->C2)
        assert "KC" not in cm.get(2022, [])
        # 2023: KC changed coach C1->C3
        assert "KC" in cm.get(2023, [])

    def test_no_change_same(self):
        df = _change_df()
        cm = identity_across_seasons(df, "home_coach", "away_coach")
        # BUF kept C2 throughout
        assert "BUF" not in cm.get(2023, [])

    def test_missing_id_handled(self):
        df = _change_df()
        df.loc[0, "home_coach"] = None
        cm = identity_across_seasons(df, "home_coach", "away_coach")
        assert isinstance(cm, dict)


class TestBuildOverrides:
    def test_qb_only_bonus(self):
        df = _change_df()
        overrides = build_team_regression_overrides(
            df, preseason_regression=0.1, qb_change_bonus=0.2, coach_change_bonus=0.0
        )
        # No QB changes in this data across seasons, so maybe empty
        # Actually KC changed coach, but coach_bonus=0 so no coach override
        assert overrides is None or "KC" not in overrides

    def test_coach_only_bonus(self):
        df = _change_df()
        overrides = build_team_regression_overrides(
            df, preseason_regression=0.1, qb_change_bonus=0.0, coach_change_bonus=0.3
        )
        # KC changed coach entering 2023
        assert overrides is not None
        assert "KC" in overrides
        assert overrides["KC"] == 0.40  # 0.1 + 0.3

    def test_combined_bonus_capped(self):
        df = _change_df()
        # KC has coach change but not QB change → just coach bonus
        overrides = build_team_regression_overrides(
            df, preseason_regression=0.5, qb_change_bonus=0.4, coach_change_bonus=0.3
        )
        assert "KC" in overrides
        assert overrides["KC"] <= 1.0

    def test_no_bonus_returns_none(self):
        df = _change_df()
        overrides = build_team_regression_overrides(
            df, preseason_regression=0.0, qb_change_bonus=0.0, coach_change_bonus=0.0
        )
        assert overrides is None


class TestExperiment:
    def test_importable(self):
        assert callable(run_coach_season_regression_experiment)

    def test_folds_exclude_holdout(self):
        for train_s, val_s in ROLLING_FOLDS:
            all_seasons = list(train_s) + [val_s]
            assert HOLDOUT_SEASON not in all_seasons

    def test_folds_sequential(self):
        expected = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
        assert ROLLING_FOLDS == expected
