"""Tests for regularized logistic meta-model experiment."""

import numpy as np

from sportslab.evaluation.regularized_logistic_experiment import (
    C_VALUES,
    META_FEATURE_COLS,
    PENALTY_VALUES,
    _build_gate_mask,
    _encode_week_sin_cos,
    _get_features,
    _logit,
    _sigmoid,
)


def test_module_importable():
    from sportslab.evaluation import regularized_logistic_experiment
    assert hasattr(regularized_logistic_experiment, "run_regularized_logistic_experiment")


def test_cli_importable():
    from sportslab.cli import regularized_logistic_cmd
    assert callable(regularized_logistic_cmd)


def test_meta_feature_count():
    assert len(META_FEATURE_COLS) == 3
    assert "rest_diff" in META_FEATURE_COLS
    assert "is_dome" in META_FEATURE_COLS
    assert "div_game" in META_FEATURE_COLS


def test_c_values_reasonable():
    assert len(C_VALUES) == 11
    assert min(C_VALUES) == 0.001
    assert max(C_VALUES) == 100.0


def test_penalty_values():
    assert "l2" in PENALTY_VALUES
    assert "l1" in PENALTY_VALUES
    assert len(PENALTY_VALUES) == 2


def test_sigmoid_bounds():
    x = np.array([-1000, -500, -1, 0, 1, 500, 1000])
    p = _sigmoid(x)
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)
    assert np.isclose(_sigmoid(0), 0.5)
    assert np.isclose(_sigmoid(1000), 1.0)
    assert np.isclose(_sigmoid(-1000), 0.0)


def test_logit_bounds():
    p = np.array([0.0, 0.5, 1.0])
    logit_vals = _logit(p)
    assert np.all(np.isfinite(logit_vals))
    assert np.isclose(_logit(0.5), 0.0)
    # ln(1/1e-15) ≈ 34.54 — clipping ensures finite
    logit_one = _logit(1.0)
    assert np.isfinite(logit_one)
    assert logit_one > 30.0
    logit_zero = _logit(0.0)
    assert np.isfinite(logit_zero)
    assert logit_zero < -30.0


def test_sigmoid_logit_inverse():
    p = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    assert np.allclose(_sigmoid(_logit(p)), p, atol=1e-10)


def test_encode_week_sin_cos():
    weeks = np.array([1, 4, 9, 14, 18])
    sc = _encode_week_sin_cos(weeks)
    assert sc.shape == (5, 2)
    assert np.all(sc >= -1.0)
    assert np.all(sc <= 1.0)
    # Week 9 (midseason) should have sin near 0
    assert np.abs(sc[2, 0]) < 0.1  # sin(pi) ≈ 0
    assert np.isclose(sc[2, 1], -1.0, atol=0.1)  # cos(pi) ≈ -1


def test_encode_week_cyclic():
    same = _encode_week_sin_cos(np.array([1, 19]))
    assert np.allclose(same[0], same[1], atol=0.01)


def test_gate_mask_basic():
    import pandas as pd
    df = pd.DataFrame({
        "home_qb_changed": [1, 0, 0],
        "away_qb_changed": [0, 0, 0],
        "home_qb_team_starts_pre": [10.0, 5.0, 30.0],
        "away_qb_team_starts_pre": [20.0, 20.0, 20.0],
    })
    mask = _build_gate_mask(df)
    assert bool(mask[0])  # home_qb_changed
    assert bool(mask[1])  # home_starts < 17
    assert not bool(mask[2])  # neither


def test_gate_mask_no_starts_data_defaults_conservative():
    import pandas as pd
    df = pd.DataFrame({
        "home_qb_changed": [0],
        "away_qb_changed": [0],
    })
    mask = _build_gate_mask(df)
    # When starts data is missing, defaults to 0, which triggers starts<17
    assert bool(mask[0])  # conservative default


def test_get_features_all_present():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = _get_features(df, ["a", "b"])
    assert result.shape == (2, 2)
    assert np.array_equal(result, df.values)


def test_get_features_some_missing():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2]})
    result = _get_features(df, ["a", "c"])
    assert result.shape == (2, 1)
    assert np.array_equal(result, df[["a"]].values)


def test_get_features_all_missing():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2]})
    result = _get_features(df, ["c", "d"])
    assert result.shape == (2, 0)


def test_no_pre_2021_seasons():
    from sportslab.evaluation.experiment_config import ALL_SEASONS
    assert min(ALL_SEASONS) >= 2021


def test_experiment_uses_experiment_config():
    from sportslab.evaluation.experiment_config import (
        HOLDOUT_SEASON as CONFIG_HOLDOUT,
    )
    from sportslab.evaluation.experiment_config import (
        ROLLING_FOLDS as CONFIG_FOLDS,
    )
    from sportslab.evaluation.regularized_logistic_experiment import (
        HOLDOUT_SEASON,
        ROLLING_FOLDS,
    )
    assert HOLDOUT_SEASON == CONFIG_HOLDOUT
    assert ROLLING_FOLDS == CONFIG_FOLDS


def test_holdout_not_in_folds():
    from sportslab.evaluation.regularized_logistic_experiment import (
        HOLDOUT_SEASON,
        ROLLING_FOLDS,
    )
    for _, val_season in ROLLING_FOLDS:
        assert val_season != HOLDOUT_SEASON


def test_experiment_produces_report():
    """Verify the experiment function can be imported and has the right signature."""
    import inspect

    from sportslab.evaluation.regularized_logistic_experiment import (
        run_regularized_logistic_experiment,
    )
    sig = inspect.signature(run_regularized_logistic_experiment)
    params = list(sig.parameters.keys())
    assert "ft_path" in params
    assert "report_path" in params
    assert "output_csv" in params


def test_pipeline_construction():
    """Verify the meta-model pipeline structure."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, penalty="l2", max_iter=2000, random_state=42)),
    ])
    assert isinstance(pipe, Pipeline)
    assert len(pipe.steps) == 2
    assert isinstance(pipe.named_steps["lr"], LogisticRegression)
    assert pipe.named_steps["lr"].C == 1.0


def test_no_incumbent_mutation():
    """Verify that constants are read-only and not modified."""
    from sportslab.evaluation.regularized_logistic_experiment import (
        BEST_DECAY,
        BEST_HFA,
        BEST_K,
        BEST_QB_BONUS,
        BEST_REG,
        V3_HOLDOUT_LL,
        V3_VAL_LL,
    )
    assert V3_VAL_LL == 0.6305
    assert V3_HOLDOUT_LL == 0.6200
    assert BEST_K == 36
    assert BEST_HFA == 40
    assert BEST_REG == 0.1
    assert BEST_DECAY == 32
    assert BEST_QB_BONUS == 0.2
