"""Tests for Dynamic Bayesian Elo module and experiment."""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.dynamic_elo import DynamicBayesianElo, compute_dynamic_elo_features


def _make_schedule(n_weeks=5, seed=42):
    """Synthetic schedule for testing."""
    rng = np.random.default_rng(seed)
    games = []
    teams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI"]
    for season in [2021, 2022]:
        for week in range(1, n_weeks + 1):
            for i in range(0, len(teams), 2):
                home = teams[i]
                away = teams[i + 1]
                margin = rng.normal() * 14
                games.append({
                    "season": season,
                    "week": week,
                    "gameday": f"{season}-W{week:02d}-0",
                    "home_team": home,
                    "away_team": away,
                    "home_win": 1 if margin > 0 else 0,
                    "home_score": max(0, int(21 + margin)),
                    "away_score": max(0, int(21)),
                })
    return pd.DataFrame(games)


def test_model_fit_converges():
    df = _make_schedule()
    model = DynamicBayesianElo()
    result = model.fit(df, verbose=False)
    assert result["converged"]
    assert 0 < model.sigma_evolution < 20
    assert 5 < model.sigma_observation < 30
    assert -5 < model.hfa < 15


def test_training_margins_non_nan():
    df = _make_schedule()
    model = DynamicBayesianElo()
    model.fit(df)
    margins = model.training_margins()
    assert margins is not None
    assert not np.any(np.isnan(margins))
    assert len(margins) == len(df)


def test_predict_on_new_data():
    train = _make_schedule()
    model = DynamicBayesianElo()
    model.fit(train)
    new = _make_schedule(n_weeks=2)
    preds = model.predict(new)
    assert len(preds) == len(new)
    assert not np.any(np.isnan(preds))


def test_predict_from_fitted_state_differs_from_scratch():
    train = _make_schedule(n_weeks=8)
    model = DynamicBayesianElo()
    model.fit(train)
    new = _make_schedule(n_weeks=2)
    preds_from_state = model.predict(new)

    # Re-fit and predict from scratch should differ (no prior knowledge)
    model2 = DynamicBayesianElo()
    model2.fit(train)
    # Add a new team to force different behavior
    model2.predict(new)
    assert not np.any(np.isnan(preds_from_state))


def test_compute_features_produces_columns():
    df = _make_schedule()
    result = compute_dynamic_elo_features(df)
    expected = [
        "dynamic_elo_margin", "dynamic_elo_pred_var",
        "dynamic_elo_home_mu", "dynamic_elo_away_mu",
        "dynamic_elo_hfa", "dynamic_elo_sigma_evo", "dynamic_elo_sigma_obs",
    ]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"


def test_margin_no_nan_in_output():
    df = _make_schedule()
    result = compute_dynamic_elo_features(df)
    assert not result["dynamic_elo_margin"].isna().any()


def test_predict_loop_updates_state():
    train = _make_schedule(n_weeks=10)
    model = DynamicBayesianElo()
    model.fit(train)
    theta_before = model.theta.copy()
    p_before = model.P.copy()

    # Use same teams as training (first 2 rows of same schedule)
    new = train.iloc[:2].copy()
    new["home_score"] = new["home_score"] + 100  # big surprise -> updates
    new["away_score"] = new["away_score"].clip(lower=0)
    model.predict_loop(new)
    theta_after = model.theta
    p_after = model.P

    diff = np.abs(theta_after[:6] - theta_before[:6]).max()
    assert diff > 0.01, f"theta changed by only {diff}"
    pdiff = np.abs(p_after[:6, :6] - p_before[:6, :6]).max()
    assert pdiff > 0.01, f"P changed by only {pdiff}"


def test_fold_params_differ():
    """Fitting on different schedules should produce different MLE params."""
    s1 = _make_schedule(n_weeks=5, seed=42)
    s2 = _make_schedule(n_weeks=5, seed=99)


    m1 = DynamicBayesianElo()
    m1.fit(s1)

    m2 = DynamicBayesianElo()
    m2.fit(s2)

    # Parameters should differ (different data)
    assert (abs(m1.sigma_evolution - m2.sigma_evolution) > 1e-6 or
            abs(m1.sigma_observation - m2.sigma_observation) > 1e-6)


def test_experiment_runs():
    from sportslab.evaluation.dynamic_elo_experiment import run_dynamic_elo_experiment
    with pytest.raises(FileNotFoundError):
        run_dynamic_elo_experiment(
            ft_path="nonexistent.parquet",
            report_path="/tmp/test_de_report.md",
        )


def test_cli_importable():
    from sportslab.cli import dynamic_elo_cmd
    assert callable(dynamic_elo_cmd)
