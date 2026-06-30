"""Tests for roster_availability module."""

import pandas as pd
import pytest

from sportslab.features.roster_availability import (
    ROSTER_AVAILABILITY_COLUMNS,
    compute_roster_availability,
)


@pytest.fixture
def sample_df():
    """Minimal DataFrame with injury columns."""
    return pd.DataFrame({
        "season": [2024, 2024],
        "week": [1, 2],
        "gameday": ["2024-09-08", "2024-09-15"],
        "home_team": ["KC", "BUF"],
        "away_team": ["BAL", "MIA"],
        "home_qb_out": [0, 1],
        "away_qb_out": [0, 0],
        "home_ol_out": [0, 2],
        "away_ol_out": [1, 0],
        "home_skill_out": [0, 1],
        "away_skill_out": [0, 0],
        "home_front_out": [0, 0],
        "away_front_out": [0, 1],
        "home_lb_out": [0, 0],
        "away_lb_out": [0, 0],
        "home_coverage_out": [0, 0],
        "away_coverage_out": [1, 0],
        "home_total_out": [0, 3],
        "away_total_out": [2, 1],
    })


class TestRosterAvailability:
    def test_column_completeness(self, sample_df):
        """All expected columns should be present after compute."""
        result = compute_roster_availability(sample_df)
        for col in ROSTER_AVAILABILITY_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_availability_range(self, sample_df):
        """Availability should be in [0, 1]."""
        result = compute_roster_availability(sample_df)
        for col in ROSTER_AVAILABILITY_COLUMNS:
            assert result[col].between(0, 1).all(), f"{col} out of [0, 1] range"

    def test_fully_healthy(self, sample_df):
        """Games with no outs should have availability = 1."""
        result = compute_roster_availability(sample_df)
        # Game 1 home: no OL outs -> availability = 1
        assert result.loc[0, "home_ol_availability"] == 1.0

    def test_depleted_ol(self, sample_df):
        """Game with 2/5 OL out should have availability = 0.6."""
        result = compute_roster_availability(sample_df)
        # Game 2 home: 2 OL out, depth=5 -> depletion = 0.4, avail = 0.6
        assert result.loc[1, "home_ol_availability"] == 0.6

    def test_single_ol_out(self, sample_df):
        """Game with 1/5 OL out should have availability = 0.8."""
        result = compute_roster_availability(sample_df)
        # Game 1 away: 1 OL out, depth=5 -> depletion = 0.2, avail = 0.8
        assert result.loc[0, "away_ol_availability"] == 0.8

    def test_qb_availability(self, sample_df):
        """QB with 1 out should have availability = 0.0 (depth=1)."""
        result = compute_roster_availability(sample_df)
        # Game 2 home: 1 QB out, depth=1 -> avail = 0
        assert result.loc[1, "home_qb_availability"] == 0.0

    def test_overall_availability(self, sample_df):
        """Overall availability should be mean of position groups."""
        result = compute_roster_availability(sample_df)
        # Game 1 home: all groups healthy -> overall = 1.0
        assert result.loc[0, "home_overall_availability"] == 1.0

    def test_auto_injury_compute(self):
        """Should handle missing injury columns gracefully."""
        df = pd.DataFrame({
            "season": [2024],
            "week": [1],
            "gameday": ["2024-09-08"],
            "home_team": ["KC"],
            "away_team": ["BAL"],
        })
        result = compute_roster_availability(df)
        assert "home_qb_availability" in result.columns

    def test_st_availability_default(self, sample_df):
        """ST should default to 1.0 when st_out column is missing."""
        result = compute_roster_availability(sample_df)
        assert result.loc[0, "away_st_availability"] == 1.0

    def test_importability(self):
        """Module should import correctly."""
        from sportslab.features import roster_availability
        assert hasattr(roster_availability, "compute_roster_availability")
