"""Tests for gradient boosting diagnostic experiment."""

import inspect

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from sportslab.evaluation.gradient_boosting_diagnostic import (
    FEATURE_COLS,
    LEARNING_RATE_VALUES,
    MAX_BINS_VALUES,
    MAX_DEPTH_VALUES,
    MIN_SAMPLES_LEAF_VALUES,
)
from sportslab.evaluation.experiment_config import (
    HOLDOUT_SEASON as CH,
    ROLLING_FOLDS as CF,
)


def test_module_importable():
    from sportslab.evaluation import gradient_boosting_diagnostic
    assert hasattr(gradient_boosting_diagnostic, "run_gradient_boosting_diagnostic")


def test_cli_importable():
    from sportslab.cli import gradient_boosting_cmd
    assert callable(gradient_boosting_cmd)


def test_feature_cols():
    assert len(FEATURE_COLS) == 8
    assert "elo_prob" in FEATURE_COLS
    assert "home_qb_changed" in FEATURE_COLS
    assert "rest_diff" in FEATURE_COLS


def test_hyperparameter_ranges():
    assert LEARNING_RATE_VALUES == [0.01, 0.05, 0.1]
    assert MAX_DEPTH_VALUES == [2, 3]
    assert MIN_SAMPLES_LEAF_VALUES == [20, 50, 100]
    assert MAX_BINS_VALUES == [64, 128]
    total = (len(LEARNING_RATE_VALUES) * len(MAX_DEPTH_VALUES)
             * len(MIN_SAMPLES_LEAF_VALUES) * len(MAX_BINS_VALUES))
    assert total == 36


def test_hgb_early_stopping():
    """Verify HGBClassifier supports early stopping."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, size=(200, 4))
    y = (x[:, 0] + x[:, 1] > 1.0).astype(int)
    hgb = HistGradientBoostingClassifier(
        max_iter=500, early_stopping=True, validation_fraction=0.2,
        n_iter_no_change=10, random_state=42,
    )
    hgb.fit(x, y)
    assert hasattr(hgb, "predict_proba")
    prob = hgb.predict_proba(x[:10])[:, 1]
    assert np.all(prob >= 0.0)
    assert np.all(prob <= 1.0)
    assert hgb.n_iter_ <= 500


def test_hgb_conservative_params():
    """Verify conservative params produce valid outputs."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, size=(100, 4))
    y = (x[:, 0] > 0.5).astype(int)
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=2, min_samples_leaf=50,
        max_iter=200, early_stopping=True, random_state=42,
    )
    hgb.fit(x, y)
    prob = hgb.predict_proba(x)[:, 1]
    assert np.all(np.isfinite(prob))
    assert np.min(prob) >= 0.0
    assert np.max(prob) <= 1.0


def test_experiment_uses_experiment_config():
    from sportslab.evaluation.gradient_boosting_diagnostic import (
        HOLDOUT_SEASON, ROLLING_FOLDS,
    )
    assert HOLDOUT_SEASON == CH
    assert ROLLING_FOLDS == CF


def test_holdout_not_in_folds():
    from sportslab.evaluation.gradient_boosting_diagnostic import (
        HOLDOUT_SEASON, ROLLING_FOLDS,
    )
    for _, val_season in ROLLING_FOLDS:
        assert val_season != HOLDOUT_SEASON


def test_no_pre_2021():
    from sportslab.evaluation.experiment_config import ALL_SEASONS
    assert min(ALL_SEASONS) >= 2021


def test_experiment_signature():
    from sportslab.evaluation.gradient_boosting_diagnostic import (
        run_gradient_boosting_diagnostic,
    )
    sig = inspect.signature(run_gradient_boosting_diagnostic)
    assert "ft_path" in sig.parameters
    assert "report_path" in sig.parameters
