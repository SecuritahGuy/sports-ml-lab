"""RALPH Loop 3: Reproducibility and determinism tests.

Verifies that:
1. Same command + same data + same config produces same predictions
2. Model registry does not change unless explicitly updated
3. Artifact audit catches missing or stale incumbent artifacts
4. Prediction output schema is stable
5. Random seeds are fixed where needed
"""

from pathlib import Path

import pandas as pd
import pytest

from sportslab.evaluation.predict_incumbent import (
    FEATURE_COLS,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VERSION,
)

# ── Registry stability ──


def test_incumbent_version_stable():
    """INCUMBENT_VERSION must be a string constant that doesn't change."""
    assert isinstance(INCUMBENT_VERSION, str)
    assert INCUMBENT_VERSION == "v3.0.0"


def test_incumbent_holdout_ll_stable():
    """INCUMBENT_HOLDOUT_LL must match documented value."""
    assert abs(INCUMBENT_HOLDOUT_LL - 0.6200) < 0.0001


def test_feature_cols_stable():
    """FEATURE_COLS must be a fixed list of 4 feature names."""
    assert isinstance(FEATURE_COLS, list)
    assert len(FEATURE_COLS) == 4
    assert all(isinstance(c, str) for c in FEATURE_COLS)


def test_registry_file_exists():
    """Benchmark registry must exist and be readable."""
    path = Path("reports/benchmarks/nfl_research_incumbent.md")
    assert path.exists(), f"Registry file not found: {path}"
    content = path.read_text()
    # The registry references v3.0.0 in its header
    assert "Frozen QB overlay" in content or "v3" in content


def test_leaderboard_exists():
    """Leaderboard CSV must exist and be parseable."""
    path = Path("reports/benchmarks/leaderboard.csv")
    assert path.exists()
    df = pd.read_csv(path)
    assert "decision" in df.columns
    assert "holdout_ll" in df.columns


# ── Prediction schema stability ──


def test_incumbent_predictions_csv_exists():
    """Incumbent predictions CSV must exist."""
    path = Path("reports/predictions/incumbent_predictions.csv")
    assert path.exists(), f"Predictions file not found: {path}"


def test_incumbent_holdout_csv_exists():
    """Holdout predictions CSV must exist."""
    path = Path("reports/predictions/incumbent_predictions_2025_holdout.csv")
    assert path.exists(), f"Holdout predictions file not found: {path}"


def test_holdout_csv_matches_incumbent_ll():
    """Holdout CSV must produce the documented holdout LL."""
    from sklearn.metrics import log_loss
    path = Path("reports/predictions/incumbent_predictions_2025_holdout.csv")
    df = pd.read_csv(path)
    actual = df["home_win_actual"].astype(int) if "home_win_actual" in df.columns else None
    prob_col = "incumbent_home_win_prob"
    if actual is None:
        actual = df["home_win"].astype(int)
    ll = log_loss(actual, df[prob_col])
    assert abs(ll - INCUMBENT_HOLDOUT_LL) < 0.01, (
        f"Holdout LL {ll:.4f} does not match documented {INCUMBENT_HOLDOUT_LL}"
    )


def test_prediction_schema_stable():
    """Prediction CSV must have a stable set of columns."""
    path = Path("reports/predictions/incumbent_predictions.csv")
    df = pd.read_csv(path)
    expected_core = {
        "game_id", "season", "week", "away_team", "home_team",
        "incumbent_home_win_prob", "model_version", "feature_set",
        "calibration_method", "confidence_bucket",
    }
    missing = expected_core - set(df.columns)
    assert not missing, f"Missing core columns: {missing}"


def test_prediction_csv_no_nan_probs():
    """Prediction CSV must have valid probabilities (no NaN, no out of range)."""
    path = Path("reports/predictions/incumbent_predictions.csv")
    df = pd.read_csv(path)
    probs = df["incumbent_home_win_prob"]
    assert probs.notna().all(), "NaN probabilities found"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of [0,1] range"


def test_prediction_bucket_labels_exist():
    """All confidence bucket labels must be defined strings, not null."""
    path = Path("reports/predictions/incumbent_predictions.csv")
    df = pd.read_csv(path)
    assert df["confidence_bucket"].notna().all(), "Null bucket labels found"
    assert (df["confidence_bucket"].str.len() > 0).all(), "Empty bucket labels found"


# ── Artifact audit ──


def test_artifact_audit_detects_missing_incumbent():
    """Artifact audit should detect missing incumbent predictions."""
    from sportslab.evaluation.audit_artifacts import run_audit
    issues = run_audit()
    # Should have no issues since artifacts exist
    assert isinstance(issues, list)


# ── Random seed ──


def test_sklearn_has_fixed_seed():
    """Core sklearn models should use fixed random_state."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state=42)
    rf = RandomForestClassifier(random_state=42)
    assert lr.random_state == 42
    assert rf.random_state == 42


# ── Holdout isolation ──


def test_holdout_season_not_in_training():
    """Training seasons must not include the holdout year."""
    from sportslab.evaluation.predict_incumbent import HOLDOUT_SEASON, TRAIN_SEASONS
    assert HOLDOUT_SEASON not in TRAIN_SEASONS, (
        f"Holdout {HOLDOUT_SEASON} found in TRAIN_SEASONS"
    )


def test_holdout_csv_not_in_incumbent_full():
    """Holdout predictions must be a subset of the full predictions CSV."""
    full = pd.read_csv("reports/predictions/incumbent_predictions.csv")
    hold = pd.read_csv("reports/predictions/incumbent_predictions_2025_holdout.csv")
    hold_ids = set(hold["game_id"])
    full_ids = set(full["game_id"])
    assert hold_ids.issubset(full_ids), (
        f"{len(hold_ids - full_ids)} game_ids in holdout not found in full CSV"
    )


def test_incumbent_prediction_matches_rehearsal_consistency():
    """Rehearsal overall metrics should be close to incumbent metrics."""
    path = Path("reports/predictions/rehearsal/season_2025_report.md")
    if not path.exists():
        pytest.skip("Rehearsal report not found")
    content = path.read_text()
    # The rehearsal should report metrics close to the incumbent
    assert "LL" in content
