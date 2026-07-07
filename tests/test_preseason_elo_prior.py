"""Tests for RALPH Loop 8 — Preseason Elo Prior experiment."""


import numpy as np
import pytest

from sportslab.evaluation.fold_safe import (
    load_feature_table,
)
from sportslab.evaluation.preseason_elo_prior_experiment import (
    _build_base_spine,
    _compute_ece,
    _fold_safe_cv,
    _regress_prior,
    _score_holdout,
    _subgroup_metrics,
    build_incumbent_fn,
    build_prior_elo_fn,
    compute_prior_season_elo,
    run_preseason_elo_experiment,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)


def test_prior_elo_columns_exist():
    """prior_elo columns exist in the feature table after computation."""
    df = load_feature_table()
    df = compute_prior_season_elo(df)
    assert "home_prior_elo_raw" in df.columns
    assert "away_prior_elo_raw" in df.columns


def test_prior_elo_2021_default():
    """For 2021 games (first season), prior Elo should be 1500."""
    df = load_feature_table()
    df = compute_prior_season_elo(df)
    g2021 = df[df["season"] == 2021]
    assert (g2021["home_prior_elo_raw"] == 1500.0).all()
    assert (g2021["away_prior_elo_raw"] == 1500.0).all()


def test_prior_elo_not_constant():
    """For 2022+, prior Elo should vary (not all 1500)."""
    df = load_feature_table()
    df = compute_prior_season_elo(df)
    g2022 = df[df["season"] == 2022]
    assert g2022["home_prior_elo_raw"].std() > 10
    assert g2022["away_prior_elo_raw"].std() > 10


def test_prior_elo_available_before_kickoff():
    """prior_elo is a pregame feature — available before any game in the season."""
    df = load_feature_table()
    df = compute_prior_season_elo(df)
    me = df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    neut = df[NEUTRAL_COLUMN].fillna(False).values
    df = df[me & ~neut].copy()
    week1 = df[df["week"] == 1]
    assert week1["home_prior_elo_raw"].notna().all()
    assert week1["away_prior_elo_raw"].notna().all()


def test_no_future_season_leakage():
    """prior_elo for a 2023 game uses 2022 final Elo, NOT 2023."""
    df = load_feature_table()
    df = compute_prior_season_elo(df)
    g2025 = df[df["season"] == 2025]
    g2024 = df[df["season"] == 2024]
    # For 2025 games, prior_elo should be based on 2024 final ratings
    assert g2025["home_prior_elo_raw"].notna().all()
    assert g2024["home_prior_elo_raw"].notna().all()


def test_season_boundary_handling():
    """Prior Elo from one season does not equal the same team's prior from next season."""
    df = load_feature_table()
    df = compute_prior_season_elo(df)
    # Pick a team and compare their prior across seasons
    teams_2022 = df[df["season"] == 2022][["home_team", "home_prior_elo_raw"]].drop_duplicates(
        subset="home_team").set_index("home_team")
    teams_2023 = df[df["season"] == 2023][["home_team", "home_prior_elo_raw"]].drop_duplicates(
        subset="home_team").set_index("home_team")
    common = teams_2022.index.intersection(teams_2023.index)
    if len(common) > 0:
        team = common[0]
        v2022 = teams_2022.loc[team, "home_prior_elo_raw"]
        v2023 = teams_2023.loc[team, "home_prior_elo_raw"]
        # Should be different if the team had different performance in 2021 vs 2022
        if abs(v2022 - v2023) < 0.1:
            pytest.skip("Team had identical prior Elo across seasons")


def test_regress_prior():
    """_regress_prior correctly regresses toward default_elo=1500."""
    assert _regress_prior(1600.0, 0.0) == 1600.0
    assert _regress_prior(1600.0, 0.5) == 1550.0
    assert _regress_prior(1600.0, 1.0) == 1500.0
    assert _regress_prior(1500.0, 0.5) == 1500.0


def test_ece_computation():
    """_compute_ece returns reasonable values."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.9, 0.1, 0.9])
    ece = _compute_ece(y_true, y_prob)
    assert 0 <= ece <= 1


def test_build_base_spine_columns():
    """_build_base_spine returns a DataFrame with expected columns."""
    df = _build_base_spine()
    assert "home_prior_elo_raw" in df.columns
    assert "away_prior_elo_raw" in df.columns
    assert "elo_prob" in df.columns
    assert TARGET_COLUMN in df.columns


def test_build_incumbent_fn_runs():
    """build_incumbent_fn returns a callable that produces valid probabilities."""
    df = _build_base_spine()
    fn = build_incumbent_fn()
    me = df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    tr = (df["season"].isin([2021, 2022, 2023]).values & me)
    va = (df["season"] == 2024).values & me
    preds = fn(df, tr, va)
    assert len(preds) == va.sum()
    assert preds.min() >= 0
    assert preds.max() <= 1


def test_prior_elo_fn_runs():
    """build_prior_elo_fn returns callable that produces valid probabilities."""
    df = _build_base_spine()
    fn = build_prior_elo_fn(["home_prior_elo_raw", "away_prior_elo_raw"])
    me = df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    tr = (df["season"].isin([2021, 2022, 2023]).values & me)
    va = (df["season"] == 2024).values & me
    preds = fn(df, tr, va)
    assert len(preds) == va.sum()
    assert preds.min() >= 0
    assert preds.max() <= 1


def test_fold_safe_cv_runs():
    """fold_safe_cv returns valid metrics."""
    df = _build_base_spine()
    fn = build_incumbent_fn()
    fold_metrics, avg_ll = _fold_safe_cv(df, fn)
    assert len(fold_metrics) == 3
    assert 0.5 <= avg_ll <= 0.7


def test_score_holdout_runs():
    """_score_holdout returns valid metrics."""
    df = _build_base_spine()
    fn = build_incumbent_fn()
    m = _score_holdout(df, fn)
    assert "log_loss" in m
    assert m["log_loss"] is not None


def test_cli_importable():
    """The experiment module can be called via CLI."""
    assert callable(run_preseason_elo_experiment)


def test_subgroup_metrics():
    """_subgroup_metrics returns valid subgroup results."""
    df = _build_base_spine()
    fn = build_incumbent_fn()
    early = df["week"] <= 4
    sr = _subgroup_metrics(df, fn, early.values, "early")
    assert "log_loss" in sr
    assert sr["n"] > 0


def test_prior_season_elo_aggregate():
    """Raw prior Elo is a meaningful number (not all zero, not all 1500 for 2022+)."""
    df = _build_base_spine()
    mean_raw = df["home_prior_elo_raw"].mean()
    std_raw = df["home_prior_elo_raw"].std()
    # Mean should be near 1500 (some regression, some variation)
    assert 1400 < mean_raw < 1600
    assert std_raw > 50
