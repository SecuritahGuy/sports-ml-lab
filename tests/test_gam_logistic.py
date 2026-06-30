"""Tests for GAM/spline logistic experiment."""

import numpy as np
from sklearn.preprocessing import SplineTransformer

from sportslab.evaluation.gam_logistic_experiment import (
    N_KNOTS_VALUES,
    DEGREE_VALUES,
    C_VALUES,
    LINEAR_FEATURE_COLS,
    _sigmoid,
    _logit,
    _build_gate_mask,
)


def test_module_importable():
    from sportslab.evaluation import gam_logistic_experiment
    assert hasattr(gam_logistic_experiment, "run_gam_logistic_experiment")


def test_cli_importable():
    from sportslab.cli import gam_logistic_cmd
    assert callable(gam_logistic_cmd)


def test_linear_feature_count():
    assert len(LINEAR_FEATURE_COLS) == 7
    assert "home_qb_changed" in LINEAR_FEATURE_COLS
    assert "rest_diff" in LINEAR_FEATURE_COLS
    assert "is_dome" in LINEAR_FEATURE_COLS
    assert "div_game" in LINEAR_FEATURE_COLS


def test_hyperparameter_ranges():
    assert N_KNOTS_VALUES == [3, 4, 5]
    assert DEGREE_VALUES == [2, 3]
    assert C_VALUES == [0.1, 1.0, 10.0]
    assert len(N_KNOTS_VALUES) * len(DEGREE_VALUES) * len(C_VALUES) == 18


def test_sigmoid_logit_inverse():
    p = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    assert np.allclose(_sigmoid(_logit(p)), p, atol=1e-10)


def test_spline_transformer_works():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, size=(100, 1))
    spline = SplineTransformer(n_knots=4, degree=3, knots="uniform")
    out = spline.fit_transform(x)
    assert out.shape[0] == 100
    # For 1 input feature: n_features * (n_knots - 1 + degree) = 4 - 1 + 3 = 6
    assert out.shape[1] == 6
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


def test_spline_varied_knots():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, size=(100, 1))
    for k in N_KNOTS_VALUES:
        for d in DEGREE_VALUES:
            spline = SplineTransformer(n_knots=k, degree=d, knots="uniform")
            out = spline.fit_transform(x)
            assert out.shape[0] == 100
            assert not np.any(np.isnan(out))


def test_gate_mask():
    import pandas as pd
    df = pd.DataFrame({
        "home_qb_changed": [1, 0],
        "away_qb_changed": [0, 0],
        "home_qb_team_starts_pre": [10.0, 30.0],
        "away_qb_team_starts_pre": [20.0, 20.0],
    })
    mask = _build_gate_mask(df)
    assert bool(mask[0])   # home changed
    assert not bool(mask[1])  # neither


def test_experiment_uses_experiment_config():
    from sportslab.evaluation.gam_logistic_experiment import (
        HOLDOUT_SEASON, ROLLING_FOLDS,
    )
    from sportslab.evaluation.experiment_config import (
        HOLDOUT_SEASON as CONFIG_H,
        ROLLING_FOLDS as CONFIG_F,
    )
    assert HOLDOUT_SEASON == CONFIG_H
    assert ROLLING_FOLDS == CONFIG_F


def test_holdout_not_in_folds():
    from sportslab.evaluation.gam_logistic_experiment import (
        HOLDOUT_SEASON, ROLLING_FOLDS,
    )
    for _, val_season in ROLLING_FOLDS:
        assert val_season != HOLDOUT_SEASON


def test_experiment_function_signature():
    from sportslab.evaluation.gam_logistic_experiment import (
        run_gam_logistic_experiment,
    )
    import inspect
    sig = inspect.signature(run_gam_logistic_experiment)
    assert "ft_path" in sig.parameters
    assert "report_path" in sig.parameters
    assert "output_csv" in sig.parameters


def test_no_pre_2021_seasons():
    from sportslab.evaluation.experiment_config import ALL_SEASONS
    assert min(ALL_SEASONS) >= 2021


def test_v3_0_0_constants_preserved():
    from sportslab.evaluation.gam_logistic_experiment import (
        V3_VAL_LL, V3_HOLDOUT_LL,
    )
    assert V3_VAL_LL == 0.6305
    assert V3_HOLDOUT_LL == 0.6200
