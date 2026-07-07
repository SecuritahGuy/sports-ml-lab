"""RALPH Loop 2: Leakage and production-safety tests.

Verifies that:
1. Market data is never in the incumbent feature set
2. Score/result columns are never used as features
3. Rolling windows exclude the current game
4. Holdout season is never used in training
5. Promotion logic is validation-based, not holdout-informed
6. Oracle QB data is blocked in live mode
7. Rehearsal outputs are properly isolated from live
"""

from pathlib import Path

import pandas as pd
import pytest

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.predict_incumbent import FEATURE_COLS
from sportslab.features.build_features import (
    LEAKAGE_COLUMNS,
)

# ── Incumbent feature safety ──


def test_no_market_in_incumbent_features():
    """Incumbent FEATURE_COLS must not contain market columns."""
    market_cols = {"home_moneyline", "away_moneyline", "spread_line",
                   "home_spread_odds", "away_spread_odds", "market_prob",
                   "home_market_prob", "away_market_prob", "market_home_win_prob",
                   "implied_home_prob", "no_vig_home_prob", "spread_prob"}
    overlap = set(FEATURE_COLS) & market_cols
    assert not overlap, f"Market columns found in FEATURE_COLS: {overlap}"


def test_no_scores_in_incumbent_features():
    """Incumbent FEATURE_COLS must not contain score/result columns."""
    score_cols = {"home_score", "away_score", "result", "total", "overtime",
                  "home_win", "is_tie"}
    overlap = set(FEATURE_COLS) & score_cols
    assert not overlap, f"Score/result columns found in FEATURE_COLS: {overlap}"


def test_incumbent_feature_count():
    """Incumbent must use exactly 4 features (v3.0.0 without elo_prob in FEATURE_COLS)."""
    assert len(FEATURE_COLS) == 4, (
        f"Expected 4 features, got {len(FEATURE_COLS)}: {FEATURE_COLS}"
    )


def test_leakage_columns_not_in_feature_table():
    """LEAKAGE_COLUMNS should include score + target + tie."""
    required = {"home_score", "away_score", "result", "home_win", "is_tie"}
    assert required.issubset(set(LEAKAGE_COLUMNS)), (
        f"Missing from LEAKAGE_COLUMNS: {required - set(LEAKAGE_COLUMNS)}"
    )


# ── Rolling window safety ──


def test_rolling_excludes_current():
    """Rolling mean must exclude current value via shift(1)."""
    from sportslab.features.turnovers import _compute_rolling
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = _compute_rolling(s, window=3)
    # At index 3, should be mean of [10, 20, 30] = 20.0, NOT including 40.0
    assert result.iloc[3] == 20.0, (
        f"Expected 20.0 (mean of prior 3 values), got {result.iloc[3]}"
    )
    assert result.iloc[0] != result.iloc[0]  # NaN check
    assert result.iloc[1] == 10.0  # mean of [10]
    assert result.iloc[2] == 15.0  # mean of [10, 20]


def test_rolling_first_game_zero():
    """Rolling mean of first game should be NaN (no prior values)."""
    from sportslab.features.turnovers import _compute_rolling
    s = pd.Series([5.0, 10.0, 15.0])
    result = _compute_rolling(s, window=3)
    # First value should be NaN (no prior values after shift)
    assert pd.isna(result.iloc[0]), "First rolling value should be NaN"


# ── Holdout isolation ──


def test_holdout_not_in_all_seasons():
    """HOLDOUT_SEASON must not be in experiment_config ALL_SEASONS."""
    from sportslab.evaluation.experiment_config import ALL_SEASONS
    assert HOLDOUT_SEASON not in ALL_SEASONS, (
        f"Holdout {HOLDOUT_SEASON} is in ALL_SEASONS {ALL_SEASONS}"
    )


def test_holdout_not_in_historical_seasons():
    """Holdout must not appear in HISTORICAL_SEASONS in predict_future/rehearsal."""
    from sportslab.evaluation.predict_future import HISTORICAL_SEASONS
    assert HOLDOUT_SEASON not in HISTORICAL_SEASONS, (
        f"Holdout {HOLDOUT_SEASON} found in predict_future HISTORICAL_SEASONS"
    )


def test_rolling_folds_val_not_holdout():
    """No rolling fold should use HOLDOUT_SEASON as validation."""
    for train, val in ROLLING_FOLDS:
        assert val != HOLDOUT_SEASON, (
            f"Fold uses holdout {HOLDOUT_SEASON} as validation season"
        )


def test_holdout_2025():
    """HOLDOUT_SEASON must be 2025 (project-standard)."""
    assert HOLDOUT_SEASON == 2025, (
        f"Expected HOLDOUT_SEASON=2025, got {HOLDOUT_SEASON}"
    )


# ── Live mode guards ──


def test_live_mode_blocks_oracle():
    """predict_future should raise ValueError in live mode without qb_input."""
    from sportslab.evaluation.predict_future import predict_future
    with pytest.raises(ValueError, match="oracle"):
        predict_future(season=2021, week=1, mode="live")


def test_predict_week_live_blocks_oracle():
    """predict_week should reject live mode without qb_input."""
    from sportslab.evaluation.weekly_pipeline import predict_week
    with pytest.raises(ValueError, match="oracle"):
        predict_week(season=2021, week=1, mode="live")


# ── Rehearsal isolation ──


def test_rehearsal_uses_separate_paths():
    """Rehearsal outputs should be under reports/predictions/rehearsal/."""
    from sportslab.evaluation.rehearsal_season import REHEARSAL_BASE
    base = str(REHEARSAL_BASE)
    assert "rehearsal" in base, f"Rehearsal base should contain 'rehearsal': {base}"
    assert "live" not in base.lower(), (
        f"Rehearsal base should not contain 'live': {base}"
    )


# ── Experiment promotion integrity ──


def test_leaderboard_no_diagnostic_promoted():
    """No diagnostic-labeled entry should be marked as 'promoted' in leaderboard."""
    lb_path = Path("reports/benchmarks/leaderboard.csv")
    if not lb_path.exists():
        pytest.skip("leaderboard.csv not found")
    lb = pd.read_csv(lb_path)
    # If the CSV has a 'status' column, check for contamination
    if "status" in lb.columns:
        diag_promoted = lb[(lb["status"] == "diagnostic") & (lb["decision"] == "promoted")]
        assert diag_promoted.empty, (
            f"Found diagnostic entries marked as promoted:\n{diag_promoted}"
        )
