"""Tests for learned overlay experiment."""

import inspect

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.learned_overlay_experiment import (
    C_VALUES,
    FEATURE_SETS,
    MIN_PROMOTION_DELTA,
    PENALTIES,
    _logit,
    _sigmoid,
)
from sportslab.evaluation.experiment_config import (
    ALL_SEASONS,
    HOLDOUT_SEASON as CH,
    ROLLING_FOLDS as CF,
)


def test_module_importable():
    from sportslab.evaluation import learned_overlay_experiment
    assert hasattr(learned_overlay_experiment, "run_learned_overlay_experiment")


def test_cli_importable():
    from sportslab.cli import learned_overlay_cmd
    assert callable(learned_overlay_cmd)


def test_feature_sets():
    assert len(FEATURE_SETS) == 5
    assert "base" in FEATURE_SETS
    assert "base+adj" in FEATURE_SETS
    assert "base+depth" in FEATURE_SETS
    assert "all" in FEATURE_SETS
    assert "adj_only" in FEATURE_SETS
    assert len(FEATURE_SETS["base"]) == 5
    assert len(FEATURE_SETS["base+adj"]) == 7
    assert len(FEATURE_SETS["adj_only"]) == 3


def test_feature_set_columns_exist():
    for set_name, cols in FEATURE_SETS.items():
        assert all(isinstance(c, str) for c in cols), f"{set_name} has non-string cols"
        assert len(cols) == len(set(cols)), f"{set_name} has duplicates"


def test_hyperparameter_ranges():
    assert 0.001 in C_VALUES
    assert 1.0 in C_VALUES
    assert 1000.0 in C_VALUES
    assert "l2" in PENALTIES
    assert "l1" in PENALTIES
    total = len(FEATURE_SETS) * len(C_VALUES) * len(PENALTIES)
    assert total == 100


def test_promotion_delta():
    assert MIN_PROMOTION_DELTA == 0.001


def test_logit_sigmoid_inverses():
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    reconstructed = _sigmoid(_logit(p))
    assert np.allclose(p, reconstructed, atol=1e-10)


def test_logit_clipping():
    p = np.array([0.0, 1.0])
    logit_vals = _logit(p)
    assert np.all(np.isfinite(logit_vals))


def test_sigmoid_clipping():
    x = np.array([-1000, 1000])
    s = _sigmoid(x)
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0.0)
    assert np.all(s <= 1.0)


def test_logistic_with_saga():
    """Verify saga solver works with both L1 and L2."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, size=(100, 5))
    y = (x[:, 0] + x[:, 1] > 1.0).astype(int)

    for penalty in ["l1", "l2"]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=0.1, penalty=penalty, solver="saga",
                max_iter=5000, random_state=42,
            )),
        ])
        pipe.fit(x, y)
        prob = pipe.predict_proba(x[:10])[:, 1]
        assert np.all(prob >= 0.0)
        assert np.all(prob <= 1.0)


def test_experiment_uses_experiment_config():
    from sportslab.evaluation.learned_overlay_experiment import (
        HOLDOUT_SEASON, ROLLING_FOLDS,
    )
    assert HOLDOUT_SEASON == CH
    assert ROLLING_FOLDS == CF


def test_holdout_not_in_folds():
    from sportslab.evaluation.learned_overlay_experiment import (
        HOLDOUT_SEASON, ROLLING_FOLDS,
    )
    for _, val_season in ROLLING_FOLDS:
        assert val_season != HOLDOUT_SEASON


def test_no_pre_2021():
    assert min(ALL_SEASONS) >= 2021


def test_experiment_signature():
    from sportslab.evaluation.learned_overlay_experiment import (
        run_learned_overlay_experiment,
    )
    sig = inspect.signature(run_learned_overlay_experiment)
    assert "ft_path" in sig.parameters
    assert "report_path" in sig.parameters
