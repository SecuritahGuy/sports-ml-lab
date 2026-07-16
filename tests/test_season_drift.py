"""RALPH Loop 2: Drift detection for duplicated season constants.

If these tests fail, a copy-pasted constant has drifted from the canonical
source. Fix by importing from experiment_config.py or build_features.py
instead of maintaining a local copy.
"""

import importlib.util

import pytest

from sportslab.evaluation.experiment_config import (
    ALL_SEASONS,
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
)
from sportslab.features.build_features import SPORTSLAB_MIN_SEASON

# ── Files that define their own ROLLING_FOLDS (should match experiment_config) ──

ROLLING_FOLDS_FILE_LIST = []

# ── Files that define their own HOLDOUT_SEASON (should match experiment_config) ──

HOLDOUT_SEASON_FILE_LIST = [
    "sportslab.evaluation.adaptive_k_experiment",
    "sportslab.evaluation.calibration_audit",
    "sportslab.evaluation.calibration_improvements_experiment",
    "sportslab.evaluation.calibration_remediation",
    "sportslab.evaluation.coach_qb_tenure_experiment",
    "sportslab.evaluation.coach_season_regression_experiment",
    "sportslab.evaluation.combined_features_experiment",
    "sportslab.evaluation.confidence_calibration_experiment",
    "sportslab.evaluation.decayed_elo_experiment",
    "sportslab.evaluation.elo_feature_selection_redo",
    "sportslab.evaluation.elo_tuning",
    "sportslab.evaluation.expressive_models_experiment",
    "sportslab.evaluation.feature_selection_experiment",
    "sportslab.evaluation.home_away_elo_experiment",
    "sportslab.evaluation.injury_features_experiment",
    "sportslab.evaluation.margin_aware_elo",
    "sportslab.evaluation.market_baseline",
    "sportslab.evaluation.market_benchmark",
    "sportslab.evaluation.optuna_feature_selection_experiment",
    "sportslab.evaluation.predict_incumbent",
    "sportslab.evaluation.qb_ablation",
    "sportslab.evaluation.qb_continuity",
    "sportslab.evaluation.qb_depth_experiment",
    "sportslab.evaluation.qb_features_experiment",
    "sportslab.evaluation.qb_gated_experience",
    "sportslab.evaluation.qb_lift_experiment",
    "sportslab.evaluation.qb_magnitude_experiment",
    "sportslab.evaluation.residual_blending_experiment",
    "sportslab.evaluation.residual_diagnostics",
    "sportslab.evaluation.rolling_mov_sensitivity",
    "sportslab.evaluation.rolling_origin_elo_validation",
    "sportslab.evaluation.schedule_rest_experiment",
    "sportslab.evaluation.situational_micro_experiment",
    "sportslab.evaluation.team_hfa_experiment",
    "sportslab.evaluation.train_baseline",
    "sportslab.evaluation.turnover_experiment",
    "sportslab.evaluation.weather_features_experiment",
]

# ── Files that define their own SPORTSLAB_MIN_SEASON (should match build_features) ──

MIN_SEASON_FILE_LIST = [
    "sportslab.data.ingest_nfl",
    "sportslab.features.epa",
    "sportslab.features.team_stats",
    "sportslab.features.efficiency",
]


@pytest.mark.parametrize("module_name", ROLLING_FOLDS_FILE_LIST)
def test_rolling_folds_drift(module_name):
    """Local ROLLING_FOLDS should match canonical experiment_config."""
    mod = importlib.import_module(module_name)
    local = getattr(mod, "ROLLING_FOLDS", None)
    if local is not None:
        assert local == ROLLING_FOLDS, (
            f"{module_name}.ROLLING_FOLDS has drifted from experiment_config.\n"
            f"  Local: {local}\n"
            f"  Canonical: {ROLLING_FOLDS}\n"
            f"  Fix: import from experiment_config instead of defining locally."
        )


@pytest.mark.parametrize("module_name", HOLDOUT_SEASON_FILE_LIST)
def test_holdout_season_drift(module_name):
    """Local HOLDOUT_SEASON should match canonical experiment_config."""
    mod = importlib.import_module(module_name)
    local = getattr(mod, "HOLDOUT_SEASON", None)
    if local is not None:
        assert local == HOLDOUT_SEASON, (
            f"{module_name}.HOLDOUT_SEASON has drifted from experiment_config.\n"
            f"  Local: {local}\n"
            f"  Canonical: {HOLDOUT_SEASON}\n"
            f"  Fix: import from experiment_config instead of defining locally."
        )


@pytest.mark.parametrize("module_name", MIN_SEASON_FILE_LIST)
def test_min_season_drift(module_name):
    """Local SPORTSLAB_MIN_SEASON should match canonical build_features."""
    mod = importlib.import_module(module_name)
    local = getattr(mod, "SPORTSLAB_MIN_SEASON", None)
    if local is not None:
        assert local == SPORTSLAB_MIN_SEASON, (
            f"{module_name}.SPORTSLAB_MIN_SEASON has drifted from build_features.\n"
            f"  Local: {local}\n"
            f"  Canonical: {SPORTSLAB_MIN_SEASON}\n"
            f"  Fix: import from build_features instead of defining locally."
        )


def test_canonical_rolling_folds_is_list_of_tuples():
    """Canonical ROLLING_FOLDS should be a list of (train, val) tuples."""
    assert isinstance(ROLLING_FOLDS, list)
    for fold in ROLLING_FOLDS:
        assert isinstance(fold, tuple) and len(fold) == 2


def test_canonical_all_seasons_does_not_include_holdout():
    """ALL_SEASONS in experiment_config should NOT include the holdout year."""
    assert HOLDOUT_SEASON not in ALL_SEASONS, (
        f"ALL_SEASONS includes holdout season {HOLDOUT_SEASON}. "
        f"Training data must not include holdout."
    )


def test_canonical_min_season_is_2000():
    """SPORTSLAB_MIN_SEASON must be 2000 per project rules."""
    assert SPORTSLAB_MIN_SEASON == 2000
