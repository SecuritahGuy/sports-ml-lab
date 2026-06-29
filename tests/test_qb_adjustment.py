"""Tests for QB adjustment features (shrunken Elo-point QB ratings)."""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.qb_adjustment import (
    PRIOR_IMPACT,
    PRIOR_STARTS,
    compute_qb_adjusted_elo_prob,
    compute_qb_adjustments,
)


def _make_minimal_df(**overrides) -> pd.DataFrame:
    """Create a minimal DataFrame with required columns."""
    defaults = {
        "season": [2021, 2021],
        "week": [1, 2],
        "gameday": ["2021-09-09", "2021-09-19"],
        "home_team": ["KC", "KC"],
        "away_team": ["CLE", "LAC"],
        "home_elo_pre": [1500.0, 1520.0],
        "away_elo_pre": [1500.0, 1480.0],
        "elo_prob": [0.5, 0.5],
        "home_qb_id": ["00-MAHOMES", "00-MAHOMES"],
        "away_qb_id": ["00-BAKER", "00-HERB"],
        "home_qb_name": ["P.Mahomes", "P.Mahomes"],
        "away_qb_name": ["B.Mayfield", "J.Herbert"],
        "home_win": [1, 0],
        "home_score": [33, 20],
        "away_score": [29, 31],
    }
    df = pd.DataFrame(defaults)
    for k, v in overrides.items():
        df[k] = v
    return df


class TestComputeQBAdjustments:
    def test_adds_expected_columns(self):
        df = _make_minimal_df()
        result = compute_qb_adjustments(df)
        for col in ["home_qb_adj", "away_qb_adj", "home_qb_starts", "away_qb_starts"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_first_game_zero_starts(self):
        df = _make_minimal_df()
        result = compute_qb_adjustments(df)
        assert result["home_qb_starts"].iloc[0] == 0
        assert result["away_qb_starts"].iloc[0] == 0

    def test_second_game_has_one_start(self):
        df = _make_minimal_df()
        result = compute_qb_adjustments(df)
        assert result["home_qb_starts"].iloc[1] == 1
        assert result["away_qb_starts"].iloc[1] == 0  # Herbert first start

    def test_missing_qb_id_safe(self):
        df = _make_minimal_df(home_qb_id=[None, "00-ALLEN"])
        result = compute_qb_adjustments(df)
        assert result["home_qb_adj"].iloc[0] == 0.0
        assert result["home_qb_starts"].iloc[0] == 0

    def test_adjustments_finite_and_bounded(self):
        df = _make_minimal_df()
        # Add more games to build up QB history
        extra = pd.DataFrame({
            "season": [2021, 2021, 2021, 2021, 2021],
            "week": [3, 4, 5, 6, 7],
            "gameday": ["2021-09-26", "2021-10-03", "2021-10-10", "2021-10-17", "2021-10-24"],
            "home_team": ["KC", "KC", "KC", "KC", "KC"],
            "away_team": ["LAC", "LV", "DEN", "WAS", "NYG"],
            "home_elo_pre": [1520.0, 1540.0, 1560.0, 1550.0, 1570.0],
            "away_elo_pre": [1490.0, 1480.0, 1470.0, 1500.0, 1460.0],
            "elo_prob": [0.55, 0.60, 0.65, 0.58, 0.68],
            "home_qb_id": ["00-MAHOMES"] * 5,
            "away_qb_id": ["00-HERB", "00-CARR", "00-LOCK", "00-HEIN", "00-JONES"],
            "home_qb_name": ["P.Mahomes"] * 5,
            "away_qb_name": ["J.Herbert", "D.Carr", "D.Lock", "T.Heinicke", "D.Jones"],
            "home_win": [1] * 5,
            "home_score": [30, 35, 28, 24, 31],
            "away_score": [20, 17, 13, 21, 10],
        })
        df_full = pd.concat([df, extra], ignore_index=True)
        result = compute_qb_adjustments(df_full)

        for col in ["home_qb_adj", "away_qb_adj"]:
            assert np.all(np.isfinite(result[col])), f"Non-finite values in {col}"
            assert np.all(np.abs(result[col]) <= 120), f"Unbounded values in {col}"

    def test_high_impact_qb_positive_adj(self):
        """A QB who always wins when expected to lose should get positive adj."""
        np.random.seed(42)
        rows = []
        for i in range(34):  # 2 seasons of starts
            rows.append({
                "season": 2021 if i < 17 else 2022,
                "week": (i % 17) + 1,
                "gameday": f"2021-09-{i+10:02d}" if i < 17 else f"2022-09-{i-7:02d}",
                "home_team": "KC",
                "away_team": "LV",
                "home_elo_pre": 1400.0,  # Bad team
                "away_elo_pre": 1550.0,  # Good opponent
                "elo_prob": 0.3,  # Expected to lose
                "home_qb_id": "00-MAHOMES",
                "away_qb_id": "00-CARR",
                "home_qb_name": "P.Mahomes",
                "away_qb_name": "D.Carr",
                "home_win": 1,  # Wins anyway
                "home_score": 28,
                "away_score": 17,
            })
        df = pd.DataFrame(rows)
        result = compute_qb_adjustments(df)
        # After many starts, Mahomes should have positive adjustment
        assert result["home_qb_adj"].iloc[-1] > 5, (
            f"Expected positive adjustment for dominant QB, got {result['home_qb_adj'].iloc[-1]}"
        )

    def test_backup_qb_shrunk_toward_baseline(self):
        """A QB with very few starts should be shrunk near replacement level."""
        rows = []
        for i in range(5):
            rows.append({
                "season": 2021,
                "week": i + 1,
                "gameday": f"2021-09-{i+10:02d}",
                "home_team": "NYJ",
                "away_team": "NE",
                "home_elo_pre": 1450.0,
                "away_elo_pre": 1520.0,
                "elo_prob": 0.35,
                "home_qb_id": "00-BACKUP",
                "away_qb_id": "00-BRADY",
                "home_qb_name": "J.Backup",
                "away_qb_name": "T.Brady",
                "home_win": 0,
                "home_score": 10,
                "away_score": 24,
            })
        df = pd.DataFrame(rows)
        result = compute_qb_adjustments(df)
        # 5 starts + 17 prior = shrunk strongly toward prior_impact
        # expected shrunken = (obs*n + prior*n) / (n+prior)
        # 5 starts, all losses when expected to lose ~35% => obs ~ -0.35
        # After 5 starts, should still be near prior
        adj = result["home_qb_adj"].iloc[-1]
        # Expected shrunken impact = (-0.35*5 + -0.03*17) / 22 = -0.103
        # Elo = 400*log10((0.5-0.103)/(0.5+0.103)) ~ -72
        # With PRIOR_STARTS=17, shrunk but still reflects poor performance
        assert adj < -20, f"Backup QB adj {adj:.1f} should be negative"

    def test_tie_game_is_safe(self):
        df = _make_minimal_df(home_win=[pd.NA, 0])
        result = compute_qb_adjustments(df)
        # Tie shouldn't crash; QB adj should be finite (replacement-level prior)
        assert np.isfinite(result["home_qb_adj"].iloc[0])
        assert np.isfinite(result["away_qb_adj"].iloc[0])

    def test_season_validation_rejects_pre2021(self):
        df = _make_minimal_df(season=[2020, 2020])
        with pytest.raises(ValueError, match="2021"):
            compute_qb_adjustments(df)

    def test_empty_df(self):
        df = pd.DataFrame(columns=["season", "week", "gameday", "home_team", "away_team",
                                    "home_elo_pre", "away_elo_pre", "elo_prob",
                                    "home_qb_id", "away_qb_id", "home_win"])
        result = compute_qb_adjustments(df)
        assert len(result) == 0

    def test_multiple_qbs_same_team(self):
        """Team changes QBs mid-season; both should have separate ratings."""
        rows = []
        for i in range(10):
            qb = "00-STARTER" if i < 5 else "00-BACKUP"
            rows.append({
                "season": 2021,
                "week": i + 1,
                "gameday": f"2021-09-{i+10:02d}",
                "home_team": "DAL",
                "away_team": "NYG",
                "home_elo_pre": 1500.0,
                "away_elo_pre": 1480.0,
                "elo_prob": 0.53,
                "home_qb_id": qb,
                "away_qb_id": "00-JONES",
                "home_qb_name": f"QB{'S' if i < 5 else 'B'}",
                "away_qb_name": "D.Jones",
                "home_win": [1, 1, 0, 1, 0, 0, 1, 0, 0, 1][i],
                "home_score": 24,
                "away_score": 17,
            })
        df = pd.DataFrame(rows)
        result = compute_qb_adjustments(df)
        # Game 6 (index 5) is the first with backup: backup should have 0 prior starts
        msg = "Backup first start should have 0 prior starts"
        assert result["home_qb_starts"].iloc[5] == 0, msg
        # Game 7 (index 6): backup has 1 start
        assert result["home_qb_starts"].iloc[6] == 1


class TestQBAdjustedEloProb:
    def test_no_adjustment_matches_standard_elo(self):
        home_elo = np.array([1500.0, 1520.0])
        away_elo = np.array([1500.0, 1480.0])
        home_adj = np.array([0.0, 0.0])
        away_adj = np.array([0.0, 0.0])
        hfa = 40.0

        probs = compute_qb_adjusted_elo_prob(home_elo, away_elo, home_adj, away_adj, hfa)

        expected_first = 1.0 / (1.0 + 10.0 ** (-40.0 / 400.0))
        assert abs(probs[0] - expected_first) < 1e-10

    def test_positive_home_adj_increases_home_prob(self):
        probs_base = compute_qb_adjusted_elo_prob(
            np.array([1500.0]), np.array([1500.0]),
            np.array([0.0]), np.array([0.0]), hfa=40.0,
        )
        probs_boosted = compute_qb_adjusted_elo_prob(
            np.array([1500.0]), np.array([1500.0]),
            np.array([50.0]), np.array([0.0]), hfa=40.0,
        )
        assert probs_boosted[0] > probs_base[0]

    def test_positive_away_adj_decreases_home_prob(self):
        probs_base = compute_qb_adjusted_elo_prob(
            np.array([1500.0]), np.array([1500.0]),
            np.array([0.0]), np.array([0.0]), hfa=40.0,
        )
        probs_boosted = compute_qb_adjusted_elo_prob(
            np.array([1500.0]), np.array([1500.0]),
            np.array([0.0]), np.array([50.0]), hfa=40.0,
        )
        assert probs_boosted[0] < probs_base[0]

    def test_probabilities_in_0_1_range(self):
        home_elo = np.array([1300.0, 1700.0])
        away_elo = np.array([1700.0, 1300.0])
        home_adj = np.array([100.0, -100.0])
        away_adj = np.array([-100.0, 100.0])
        probs = compute_qb_adjusted_elo_prob(home_elo, away_elo, home_adj, away_adj, hfa=40.0)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_extreme_adjustments_clamped(self):
        probs = compute_qb_adjusted_elo_prob(
            np.array([1500.0]), np.array([1500.0]),
            np.array([500.0]), np.array([-500.0]), hfa=0.0,
        )
        assert 0.0 < probs[0] < 1.0


class TestImportability:
    def test_module_imports(self):
        from sportslab.features import qb_adjustment
        assert hasattr(qb_adjustment, "compute_qb_adjustments")

    def test_constants_accessible(self):
        assert PRIOR_STARTS >= 1
        assert PRIOR_IMPACT < 0  # replacement-level should be negative
