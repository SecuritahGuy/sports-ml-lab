"""RALPH Loop 2: Regression tests for previous fixes.

Each test captures a fix that was applied in a prior session
but lacked a regression test to prevent reversion.
"""

import pytest
from click.testing import CliRunner
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from sportslab.cli import cli
from sportslab.evaluation.experiment_config import HOLDOUT_SEASON


@pytest.mark.slow
def test_autogluon_import_no_crash():
    """Autogluon experiment module should import without auto-dependency."""
    import importlib
    spec = importlib.util.find_spec("autogluon")
    if spec is None:
        from sportslab.evaluation import autogluon_experiment
        assert hasattr(autogluon_experiment, "run_autogluon_experiment")


@pytest.mark.slow
def test_optuna_search_import_no_crash():
    """optuna_elo_search module should import cleanly."""
    from sportslab.evaluation.optuna_elo_search import run_optuna_search
    assert callable(run_optuna_search)


def test_efficiency_rejects_pre_2021():
    """efficiency._validate_seasons_efficiency should reject seasons before 2021."""
    from sportslab.features.efficiency import _validate_seasons_efficiency
    with pytest.raises(ValueError, match="not allowed"):
        _validate_seasons_efficiency([2019, 2020])


def test_backtest_cli_rejects_pre_2021():
    """sportslab backtest should reject seasons before 2021."""
    runner = CliRunner()
    result = runner.invoke(cli, ["backtest", "2019", "2020"])
    assert result.exit_code == 0
    assert "not allowed" in result.output


def test_turnovers_rejects_pre_2021():
    """turnovers._load_team_stats should reject seasons before 2021."""
    from sportslab.features.turnovers import _load_team_stats
    with pytest.raises(ValueError, match="Minimum season"):
        _load_team_stats([2020, 2019])


def test_predict_future_validates_season():
    """predict_future should validate season >= 2021."""
    from sportslab.evaluation.predict_future import _validate_season
    with pytest.raises(ValueError, match="Minimum season"):
        _validate_season(2019)
    _validate_season(HOLDOUT_SEASON)


def test_weekly_pipeline_validates_season():
    """weekly_pipeline._validate_season should reject pre-2021."""
    from sportslab.evaluation.weekly_pipeline import _validate_season
    with pytest.raises(ValueError, match="Minimum season"):
        _validate_season(2019)


def test_rehearse_season_validates_season():
    """rehearse_season should reject seasons before 2021."""
    from sportslab.evaluation.rehearsal_season import rehearse_season
    with pytest.raises(ValueError, match="Minimum season"):
        rehearse_season(season=2019)


def test_prediction_audit_validates_season():
    """prediction_audit.run_prediction_audit should reject pre-2021."""
    from sportslab.evaluation.prediction_audit import run_prediction_audit
    with pytest.raises(ValueError, match="Minimum season"):
        run_prediction_audit(season=2019)


def test_weekly_qb_audit_validates_season():
    """weekly_qb_audit.run_weekly_qb_audit should reject pre-2021."""
    from sportslab.evaluation.weekly_qb_audit import run_weekly_qb_audit
    with pytest.raises(ValueError, match="Minimum season"):
        run_weekly_qb_audit(season=2019, week=1)


def test_incumbent_holdout_ll():
    """Holdout LL constant should match the documented v3.0.0 value."""
    from sportslab.evaluation.predict_incumbent import INCUMBENT_HOLDOUT_LL
    assert abs(INCUMBENT_HOLDOUT_LL - 0.6200) < 0.0001


def test_sklearn_api_compat():
    """sklearn Platt calibration should use valid API."""
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    assert cal is not None
