"""Tests for Elo parameter ensemble experiment."""

import numpy as np
from click.testing import CliRunner

from sportslab.cli import cli
from sportslab.evaluation.elo_ensemble_experiment import (
    ENSEMBLE_CONFIGS,
    _apply_qb_overlay,
    _build_qb_gate_mask,
    _get_features,
    _logit,
    _sigmoid,
)


def test_ensemble_configs_count():
    """Should have at least 5 ensemble configs."""
    assert len(ENSEMBLE_CONFIGS) >= 5


def test_ensemble_configs_format():
    """Each config should be a tuple of (K, HFA, reg, decay, label)."""
    for cfg in ENSEMBLE_CONFIGS:
        assert len(cfg) == 5
        k, hfa, reg, decay, label = cfg
        assert isinstance(k, (int, float))
        assert isinstance(hfa, (int, float))
        assert isinstance(reg, float)
        assert decay is None or isinstance(decay, (int, float))
        assert isinstance(label, str)


def test_ensemble_configs_diversity():
    """Ensemble configs should span different regions of parameter space."""
    k_values = [c[0] for c in ENSEMBLE_CONFIGS]
    hfa_values = [c[1] for c in ENSEMBLE_CONFIGS]
    reg_values = [c[2] for c in ENSEMBLE_CONFIGS]
    assert min(k_values) <= 28
    assert max(k_values) >= 44
    assert min(hfa_values) <= 25
    assert max(hfa_values) >= 40
    assert min(reg_values) <= 0.05
    assert max(reg_values) >= 0.2


def test_ensemble_configs_includes_champion():
    """V3.0.0 champion config should be in the ensemble."""
    labels = [c[4] for c in ENSEMBLE_CONFIGS]
    assert any("champion" in label or "champ" in label for label in labels)


def test_sigmoid_bounds():
    """Sigmoid should map to (0, 1) for moderate inputs."""
    x = np.array([-10, -1, 0, 1, 10])
    p = _sigmoid(x)
    assert np.all(p > 0) and np.all(p < 1)
    assert np.isclose(p[2], 0.5)  # sigmoid(0) = 0.5


def test_sigmoid_clip():
    """Sigmoid should not overflow on extreme inputs."""
    x = np.array([1e6, -1e6])
    p = _sigmoid(x)
    assert np.isclose(p[0], 1.0)
    assert np.isclose(p[1], 0.0)


def test_logit_inverse():
    """logit(sigmoid(x)) should recover x for moderate values."""
    x = np.array([-5, -2, 0, 2, 5])
    recovered = _logit(_sigmoid(x))
    assert np.allclose(x, recovered, atol=1e-10)


def test_logit_clip():
    """logit should return finite values for extreme probabilities."""
    p = np.array([1e-15, 0.5, 1 - 1e-15])
    logits = _logit(p)
    assert np.all(np.isfinite(logits))


def test_get_features_available():
    """Should return features as array when columns exist."""
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    feats = _get_features(df, ["a", "b"])
    assert feats.shape == (2, 2)
    assert np.allclose(feats[0], [1, 3])


def test_get_features_missing():
    """Should return empty array when no columns match."""
    import pandas as pd
    df = pd.DataFrame({"a": [1]})
    feats = _get_features(df, ["x", "y"])
    assert feats.size == 0


def test_get_features_partial():
    """Should return only available columns."""
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    feats = _get_features(df, ["a", "c"])
    assert feats.shape == (2, 1)
    assert np.allclose(feats[:, 0], [1, 2])


def test_apply_qb_overlay_shape():
    """QB overlay should preserve input shape."""
    n = 10
    base_logit = np.zeros(n)
    gate_mask = np.array([True, False] * 5)
    home_adj = np.ones(n) * 20
    away_adj = np.zeros(n)
    result = _apply_qb_overlay(base_logit, gate_mask, home_adj, away_adj)
    assert result.shape == (n,)


def test_apply_qb_overlay_gate_off():
    """When gate is off, output should match input."""
    n = 5
    base_logit = np.random.randn(n)
    gate_mask = np.zeros(n, dtype=bool)
    result = _apply_qb_overlay(base_logit, gate_mask, np.zeros(n), np.zeros(n))
    assert np.allclose(result, base_logit)


def test_build_qb_gate_mask_no_starts():
    """Should fall back to qb_changed when starts column missing."""
    import pandas as pd
    df = pd.DataFrame({
        "home_qb_changed": [1, 0, 0],
        "away_qb_changed": [0, 0, 1],
    })
    mask = _build_qb_gate_mask(df)
    assert mask.dtype == bool
    assert mask[0]  # home changed
    assert not mask[1]  # neither changed
    assert mask[2]  # away changed


def test_cli_importable():
    """elo-ensemble CLI command should exist."""
    runner = CliRunner()
    result = runner.invoke(cli, ["elo-ensemble", "--help"])
    assert result.exit_code == 0
    assert "ensemble" in result.output.lower() or "ensemble" in result.output.lower()


def test_run_ensemble_importable():
    """run_elo_ensemble should be importable."""
    from sportslab.evaluation.elo_ensemble_experiment import run_elo_ensemble
    assert callable(run_elo_ensemble)
