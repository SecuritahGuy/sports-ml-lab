"""Tests for roster_strength module (V1)."""

import numpy as np
import pandas as pd
import pytest

from sportslab.ratings.roster_strength import (
    ALL_ROSTER_COLUMNS,
    POSITION_WEIGHTS,
    compute_roster_adjusted_elo_prob,
    compute_roster_strength,
)


@pytest.fixture
def sample_df():
    """Minimal DataFrame with required columns."""
    return pd.DataFrame({
        "season": [2024, 2024],
        "week": [1, 2],
        "gameday": ["2024-09-08", "2024-09-15"],
        "home_team": ["KC", "BUF"],
        "away_team": ["BAL", "MIA"],
        "home_qb_id": ["Patrick Mahomes", "Josh Allen"],
        "away_qb_id": ["Lamar Jackson", "Tua Tagovailoa"],
        "home_elo_pre": [1600.0, 1550.0],
        "away_elo_pre": [1500.0, 1520.0],
    })


class TestRosterStrengthV1:
    def test_column_completeness(self, sample_df):
        """All expected roster columns should be present."""
        result = compute_roster_strength(sample_df)
        for col in ALL_ROSTER_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_qb_points_populated(self, sample_df):
        """QB points should be non-zero when QB data is available."""
        result = compute_roster_strength(sample_df)
        assert result["home_roster_qb_points"].values[0] != 0.0

    def test_ol_points_range(self, sample_df):
        """Non-QB points should be in [-weight, weight]."""
        result = compute_roster_strength(sample_df)
        w = POSITION_WEIGHTS["ol"]
        ol_pts = result["home_roster_ol_points"]
        assert ol_pts.between(-w, w).all()

    def test_skill_points_range(self, sample_df):
        """Skill points should be in [-weight, weight]."""
        result = compute_roster_strength(sample_df)
        w = POSITION_WEIGHTS["skill"]
        assert result["home_roster_skill_points"].between(-w, w).all()

    def test_front_points_range(self, sample_df):
        """Front points should be in [-weight, weight]."""
        result = compute_roster_strength(sample_df)
        w = POSITION_WEIGHTS["front"]
        assert result["home_roster_front_points"].between(-w, w).all()

    def test_injury_adj_negative(self, sample_df):
        """Injury adjustment should be <= 0 when there are injuries."""
        result = compute_roster_strength(sample_df)
        # No injury columns in minimal df -> total_out = 0 -> injury_adj = 0
        assert (result["home_roster_injury_adj"] <= 0).all()

    def test_total_adjustment_computed(self, sample_df):
        """Total adjustment should be a finite float."""
        result = compute_roster_strength(sample_df)
        assert np.isfinite(result["home_roster_total_adjustment"].values).all()

    def test_adjusted_elo_prob(self):
        """compute_roster_adjusted_elo_prob should return valid probabilities."""
        home_elo = np.array([1600.0, 1550.0])
        away_elo = np.array([1500.0, 1520.0])
        home_adt = np.array([0.5, -0.3])
        away_adt = np.array([-0.2, 0.1])
        prob = compute_roster_adjusted_elo_prob(home_elo, away_elo, home_adt, away_adt)
        assert bool(np.all(prob >= 0)) and bool(np.all(prob <= 1))

    def test_adjusted_elo_preferred_favored(self):
        """Positive home adjustment should increase home win prob."""
        home_elo = np.array([1500.0])
        away_elo = np.array([1500.0])
        prob_base = compute_roster_adjusted_elo_prob(
            home_elo, away_elo, np.array([0.0]), np.array([0.0])
        )
        prob_boosted = compute_roster_adjusted_elo_prob(
            home_elo, away_elo, np.array([50.0]), np.array([0.0])
        )
        assert prob_boosted[0] > prob_base[0]

    def test_fully_healthy_gives_zero_points(self, sample_df):
        """When all availability scores = 1, non-QB points should be weight."""
        # compute_roster_availability without injury columns defaults to 1.0
        result = compute_roster_strength(sample_df)
        # When avail = 1: weight * (2*1 - 1) = weight
        assert result["home_roster_ol_points"].values[0] == POSITION_WEIGHTS["ol"]

    def test_away_columns_populated(self, sample_df):
        """Away roster columns should also be populated."""
        result = compute_roster_strength(sample_df)
        assert result["away_roster_total_adjustment"].values[0] != 0.0
