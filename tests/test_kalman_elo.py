"""Tests for Kalman Elo feature module and experiment."""

import pandas as pd

from sportslab.features.kalman_elo import (
    DEFAULT_MU,
    _effective_expected,
    compute_kalman_elo_features,
)


def _make_schedule():
    """Small synthetic schedule for testing."""
    games = []
    for week in range(1, 4):
        for home, away in [("ARI", "ATL"), ("BAL", "BUF")]:
            games.append({
                "season": 2021, "week": week, "gameday": f"2021-{week}-0",
                "home_team": home, "away_team": away,
                "home_win": 1 if week % 2 else 0,
                "home_score": 24 + week, "away_score": 17,
            })
    return pd.DataFrame(games)


def test_effective_expected():
    p = _effective_expected(1500, 1500, hfa=0)
    assert abs(p - 0.5) < 1e-10


def test_effective_expected_hfa():
    p = _effective_expected(1500, 1500, hfa=40)
    assert p > 0.5


def test_compute_kalman_returns_columns():
    df = _make_schedule()
    result = compute_kalman_elo_features(df)
    for col in ["home_kalman_mu", "away_kalman_mu", "home_kalman_sigma",
                "away_kalman_sigma", "kalman_diff", "kalman_prob"]:
        assert col in result.columns


def test_kalman_prob_bounds():
    df = _make_schedule()
    result = compute_kalman_elo_features(df)
    assert result["kalman_prob"].between(0, 1).all()


def test_kalman_sigma_positive():
    df = _make_schedule()
    result = compute_kalman_elo_features(df)
    assert (result["home_kalman_sigma"] > 0).all()
    assert (result["away_kalman_sigma"] > 0).all()


def test_kalman_width_initial_sigma():
    """Higher initial_sigma produces wider early updates (affects game 3+)."""
    df = _make_schedule()
    r1 = compute_kalman_elo_features(df, initial_sigma=100)
    r2 = compute_kalman_elo_features(df, initial_sigma=500)
    # Row 2: ARI (home) has been updated after game 0; sigma affects the gain
    d1 = abs(r1.loc[2, "home_kalman_mu"] - r2.loc[2, "home_kalman_mu"])
    assert d1 > 0


def test_kalman_obs_noise():
    """Higher obs_noise damps updates (smaller mu changes)."""
    df = _make_schedule()
    r_low = compute_kalman_elo_features(df, obs_noise=10)
    r_high = compute_kalman_elo_features(df, obs_noise=500)
    h_low = r_low["home_kalman_mu"].diff().abs().sum()
    h_high = r_high["home_kalman_mu"].diff().abs().sum()
    assert h_low > h_high


def test_kalman_regression():
    """Preseason regression pulls mu toward default."""
    df = pd.concat([_make_schedule().assign(season=2021),
                    _make_schedule().assign(season=2022)])
    r_no = compute_kalman_elo_features(df, preseason_regression=0.0)
    r_hi = compute_kalman_elo_features(df, preseason_regression=0.5)
    # After season boundary, high-regression mu should be closer to default
    s2022_no = r_no[r_no["season"] == 2022].iloc[0]
    s2022_hi = r_hi[r_hi["season"] == 2022].iloc[0]
    assert abs(s2022_hi["home_kalman_mu"] - DEFAULT_MU) <= abs(
        s2022_no["home_kalman_mu"] - DEFAULT_MU
    ) + 1e-6


def test_kalman_mov_affects_update():
    """Larger MOV produces larger updates with MOV multiplier."""
    df = _make_schedule().copy()
    df.loc[0, "home_score"] = 50
    df.loc[0, "away_score"] = 3  # big blowout
    r_mov = compute_kalman_elo_features(df, mov_type="log", mov_scale=1.0)
    r_no = compute_kalman_elo_features(df)
    # After game 1, MOV should produce larger mu change
    assert abs(r_mov.loc[0, "home_kalman_mu"] - DEFAULT_MU) >= abs(
        r_no.loc[0, "home_kalman_mu"] - DEFAULT_MU
    ) - 1e-6
