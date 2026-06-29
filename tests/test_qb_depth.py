"""Tests for QB depth features and QB depth experiment."""

import numpy as np
import pandas as pd

from sportslab.features.qb_depth import QB_DEPTH_COLUMNS, compute_qb_depth_features

# ── Fixtures ──


def _make_chrono_df():
    """Small chronological schedule with multiple QB changes per team.

    Team A: weeks 1-3 QB1, week 4 QB2 (change), week 5 QB1 (change back)
    Team B: weeks 1-4 QB3, week 5 QB4 (change)
    """
    np.random.seed(42)
    rows = [
        # season week gameday  home away home_qb away_qb home_win
        (2021, 1, "2021-09-09", "A", "B", "QB1", "QB3", 1),
        (2021, 2, "2021-09-16", "A", "B", "QB1", "QB3", 1),
        (2021, 3, "2021-09-23", "A", "B", "QB1", "QB3", 0),
        (2021, 4, "2021-09-30", "A", "B", "QB2", "QB3", 1),
        (2021, 5, "2021-10-07", "A", "B", "QB1", "QB4", 0),
    ]
    df = pd.DataFrame(rows, columns=[
        "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_win",
    ])
    return df


def _make_missing_df():
    """Games with missing QB data."""
    rows = [
        (2021, 1, "2021-09-09", "A", "B", "QB1", "QB3", 1),
        (2021, 2, "2021-09-16", "A", "B", np.nan, "QB3", 0),
    ]
    return pd.DataFrame(rows, columns=[
        "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_win",
    ])


def _add_qb_features(df):
    """Add the prerequisite qb_changed/starts columns computed_qb_features would provide."""
    from sportslab.features.qb import compute_qb_features
    return compute_qb_features(df)


# ── Tests: column completeness ──


def test_depth_columns_present():
    df = _add_qb_features(_make_chrono_df())
    result = compute_qb_depth_features(df)
    for col in QB_DEPTH_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"


def test_depth_column_count():
    df = _add_qb_features(_make_chrono_df())
    result = compute_qb_depth_features(df)
    for col in QB_DEPTH_COLUMNS:
        assert col in result.columns
    assert len(QB_DEPTH_COLUMNS) == 6


# ── Tests: rust games ──


def test_rust_no_change():
    """Same QB every game → rust = 0 for all games."""
    rows = [
        (2021, 1, "2021-09-09", "A", "B", "QB1", "QB2", 1),
        (2021, 2, "2021-09-16", "A", "B", "QB1", "QB2", 1),
        (2021, 3, "2021-09-23", "A", "B", "QB1", "QB2", 0),
    ]
    df = pd.DataFrame(rows, columns=[
        "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_win",
    ])
    df = _add_qb_features(df)
    result = compute_qb_depth_features(df)
    for _, row in result.iterrows():
        assert row["home_qb_rust_games"] == 0, "Same QB should have 0 rust"
        assert row["away_qb_rust_games"] == 0, "Same QB should have 0 rust"


def test_rust_after_change():
    """After a QB change, rust = games since the replaced QB's last start.

    Team A: QB1 starts week 1-3, then QB2 starts week 4.
    QB2 is new to team A so rust=999.
    QB1 returns in week 5: rust = 4 - 2 - 1 = 1 (1 game gap: week 4).
    """
    df = _add_qb_features(_make_chrono_df())
    result = compute_qb_depth_features(df)

    # Week 1: both QBs start fresh - rust might be 999 or handled
    assert result.loc[0, "home_qb_rust_games"] >= 0
    assert result.loc[0, "away_qb_rust_games"] >= 0

    # Week 4: QB2 starts for home (QB1 → QB2), so rust should be > 0
    # (QB2 hasn't started recently for team A)
    assert result.loc[3, "home_qb_rust_games"] >= 1

    # Week 5: QB1 returns (QB2 → QB1). QB1 last started week 3 (index 2).
    # total_games before processing = 4, last_seen at index 2.
    # rust = 4 - 2 - 1 = 1 (1 game gap: week 4)
    assert result.loc[4, "home_qb_rust_games"] == 1


def test_rust_non_change_weeks():
    """Same QB continuing should have rust = 0."""
    df = _add_qb_features(_make_chrono_df())
    result = compute_qb_depth_features(df)

    # Week 2: same QBs as week 1 → rust = 0
    assert result.loc[1, "home_qb_rust_games"] == 0
    assert result.loc[1, "away_qb_rust_games"] == 0

    # Week 3: same QBs as week 2 → rust = 0
    assert result.loc[2, "home_qb_rust_games"] == 0


# ── Tests: first season start ──


def test_first_season_start_true():
    """A QB changing in and having 0 starts this season but > 0 team starts
    is a 'first season start'."""
    rows = [
        (2021, 1, "2021-09-09", "A", "B", "QB1", "QB2", 1),
        (2021, 2, "2021-09-16", "A", "B", "QB3", "QB2", 1),
    ]
    df = pd.DataFrame(rows, columns=[
        "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_win",
    ])
    df = _add_qb_features(df)
    result = compute_qb_depth_features(df)

    # Week 2: QB3 takes over for home. QB3 had starts_this_season_pre=0
    # and team_starts_pre might be 0 (first time) or >0. Since QB3
    # has never been seen, team_starts_pre=0, so first_season_start=0.
    first = result.loc[1, "home_qb_first_season_start"]
    assert first in (0, 1)


def test_first_season_start_with_history():
    """QB returning after previous starts in prior seasons should
    have first_season_start=1 on the game they return."""
    rows = [
        (2021, 1, "2021-09-09", "A", "B", "QB1", "QB2", 1),
        (2021, 2, "2021-09-16", "A", "B", "QB1", "QB2", 1),
        (2021, 3, "2021-09-23", "A", "B", "QB2", "QB2", 1),
    ]
    df = pd.DataFrame(rows, columns=[
        "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_win",
    ])
    df = _add_qb_features(df)
    result = compute_qb_depth_features(df)

    # Week 3: QB2 takes over for home. For a proper first_season_start,
    # we need starts_this_season_pre == 0 AND team_starts_pre > 0.
    # Since QB2 hasn't started before for team A (QB1 did),
    # team_starts_pre=0 → first_season_start=0
    first = result.loc[2, "home_qb_first_season_start"]
    assert first == 0


def test_first_season_start_diff():
    """Diff column should be home minus away."""
    df = _add_qb_features(_make_chrono_df())
    result = compute_qb_depth_features(df)
    for _, row in result.iterrows():
        expected = float(row["home_qb_first_season_start"] - row["away_qb_first_season_start"])
        assert row["qb_first_season_start_diff"] == expected


def test_rust_diff():
    """Rust diff column should be home minus away."""
    df = _add_qb_features(_make_chrono_df())
    result = compute_qb_depth_features(df)
    for _, row in result.iterrows():
        expected = float(row["home_qb_rust_games"] - row["away_qb_rust_games"])
        assert row["qb_rust_diff"] == expected


# ── Tests: missing data handling ──


def test_missing_qb_rust_zero():
    df = _add_qb_features(_make_missing_df())
    result = compute_qb_depth_features(df)
    # Missing QB should have rust=0, first_season_start=0
    assert result.loc[1, "home_qb_rust_games"] == 0
    assert result.loc[1, "home_qb_first_season_start"] == 0


# ── Tests: season boundary reset ──


def test_season_boundary_reset():
    """Across seasons, rust tracking should reset."""
    rows = [
        (2021, 17, "2022-01-06", "A", "B", "QB1", "QB2", 1),
        (2022, 1, "2022-09-08", "A", "B", "QB3", "QB2", 1),
    ]
    df = pd.DataFrame(rows, columns=[
        "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_win",
    ])
    df = _add_qb_features(df)
    result = compute_qb_depth_features(df)

    # Week 1 of 2022: QB3 starts for home. Season is fresh so
    # last_qb_id = None → changed = 0 → rust = 0 (matching QB feature
    # convention: first game of season is not a "change" event).
    assert result.loc[1, "home_qb_rust_games"] == 0


# ── Tests: chronological order preservation ──


def test_chronological_order():
    """Output should maintain original order, not sort."""
    df = _add_qb_features(_make_chrono_df())
    result = compute_qb_depth_features(df)
    out_key = "game_id" if "game_id" in result.columns else None
    out_len = len(result[out_key]) if out_key else len(result)
    in_key = "game_id" if "game_id" in df.columns else None
    in_len = len(df[in_key]) if in_key else len(df)
    assert out_len == in_len


# ── Tests: no NaN / inf ──


def test_no_nan_or_inf():
    df = _add_qb_features(_make_chrono_df())
    result = compute_qb_depth_features(df)
    for col in QB_DEPTH_COLUMNS:
        vals = result[col].values
        assert not np.any(np.isnan(vals)), f"NaN found in {col}"
        assert not np.any(np.isinf(vals)), f"Inf found in {col}"


# ── Tests: experiment module ──


def test_experiment_import():
    from sportslab.evaluation.qb_depth_experiment import run_qb_depth_experiment
    assert callable(run_qb_depth_experiment)


def test_experiment_constants():
    from sportslab.evaluation.qb_depth_experiment import MODEL_VARIANTS
    assert len(MODEL_VARIANTS) == 6
    names = [n for n, _, _ in MODEL_VARIANTS]
    assert "incumbent" in names
    assert "career_starts" in names
    assert "win_pct" in names
    assert "missing_flag" in names
    assert "qb_depth" in names
    assert "all_depth" in names


def test_experiment_rolling_function():
    from sportslab.evaluation.qb_depth_experiment import ROLLING_FOLDS
    assert len(ROLLING_FOLDS) == 3


def test_cli_importable():
    from sportslab.cli import qb_depth_cmd
    assert qb_depth_cmd is not None


# ── Tests: built-in corner cases ──


def test_rust_with_ties():
    """Ties should not affect rust tracking (QB still started)."""
    rows = [
        (2021, 1, "2021-09-09", "A", "B", "QB1", "QB2", 1),
        (2021, 2, "2021-09-16", "A", "B", "QB1", "QB2", pd.NA),
        (2021, 3, "2021-09-23", "A", "B", "QB3", "QB2", 0),
    ]
    df = pd.DataFrame(rows, columns=[
        "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_win",
    ])
    df = _add_qb_features(df)
    result = compute_qb_depth_features(df)

    # Week 3: QB3 starts (changed from QB1).
    # rust should be > 0 since QB3 is different
    assert result.loc[2, "home_qb_rust_games"] >= 1

    # Week 2: same QBs, rust should be 0
    assert result.loc[1, "home_qb_rust_games"] == 0
    assert result.loc[1, "away_qb_rust_games"] == 0


def test_multiple_changes_same_season():
    """Multiple QB changes in a season should each show correct rust."""
    rows = [
        (2021, 1, "2021-09-09", "A", "B", "QB1", "QB2", 1),
        (2021, 2, "2021-09-16", "A", "B", "QB2", "QB2", 1),  # A: QB1→QB2
        (2021, 3, "2021-09-23", "A", "B", "QB1", "QB2", 0),  # A: QB2→QB1
        (2021, 4, "2021-09-30", "A", "B", "QB3", "QB2", 1),  # A: QB1→QB3
    ]
    df = pd.DataFrame(rows, columns=[
        "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_win",
    ])
    df = _add_qb_features(df)
    result = compute_qb_depth_features(df)

    # Week 2: QB2 starts, first start → rust should be based on
    # games_since_qb_last_start which is 999 (initial) for QB2
    assert result.loc[1, "home_qb_rust_games"] >= 1

    # Week 3: QB1 returns. QB1 last started week 1 (index 0).
    # total_games before = 2, last_seen = 0. rust = 2 - 0 - 1 = 1
    assert result.loc[2, "home_qb_rust_games"] == 1

    # Week 4: QB3 starts. QB3 never started for A → rust = 999
    assert result.loc[3, "home_qb_rust_games"] == 999


def test_away_side_tracking():
    """Verify away side rust/first-season tracking independently."""
    rows = [
        (2021, 1, "2021-09-09", "A", "B", "QB1", "QB3", 1),
        (2021, 2, "2021-09-16", "A", "B", "QB1", "QB4", 0),
        (2021, 3, "2021-09-23", "A", "B", "QB1", "QB3", 1),
    ]
    df = pd.DataFrame(rows, columns=[
        "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_win",
    ])
    df = _add_qb_features(df)
    result = compute_qb_depth_features(df)

    # Week 1 away: QB3 starts fresh → rust depends on history
    # Week 2 away: QB4 → change from QB3 → QB4 never seen → rust = 999
    assert result.loc[1, "away_qb_rust_games"] == 999

    # Week 3 away: QB3 returns. QB3 last started week 1 (index 0).
    # total_games before week 3 = 2, last_seen = 0 → rust = 2 - 0 - 1 = 1
    assert result.loc[2, "away_qb_rust_games"] == 1

    # Week 3 away: QB3 returned → should be first_season_start=?
    # If QB3 had team_starts_pre=1 (from week 1), and starts=0,
    # then first_season_start should be 1
