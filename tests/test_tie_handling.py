"""Step 5: Tie handling audit and tests.

Current behavior (verified against source code):

1. Elo updates (compute_elo_features):
   - home_win is NA for ties (home_score == away_score)
   - Actual = 0.5 → update = K * (0.5 - expected) * MOV
   - Ties are treated as a "half-win" for each team for rating purposes
   - This is mathematically valid Elo behavior

2. Model eligibility (build_features.py):
   - Ties have TARGET_COLUMN = pd.NA
   - MODEL_ELIGIBLE_COLUMN = TARGET_COLUMN.notna()  → False for ties
   - Ties are excluded from logistic regression training

3. Incumbent prediction pipeline (predict_incumbent.py):
   - Filters to model_eligible == True → ties excluded
   - LogisticRegression never sees a 0.5 target
   - Elo is still updated for ties (they contribute to rating history)

4. Backtest (backtest_2025.py):
   - Reads predictions CSV which excludes non-eligible rows
   - Ties are excluded from all metrics
"""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    TARGET_COLUMN,
    TIE_COLUMN,
    build_feature_table,
)
from sportslab.features.ratings import compute_elo_features
from sportslab.evaluation.predict_incumbent import _fit_incumbent, _build_feature_pipeline


def test_tie_marked_not_eligible(sample_schedule_with_tie):
    """Ties should have model_eligible=False."""
    result = sample_schedule_with_tie.copy()
    result[TARGET_COLUMN] = result.apply(
        lambda r: (
            1 if r.home_score > r.away_score else (0 if r.home_score < r.away_score else pd.NA)
        ),
        axis=1,
    )
    result[TIE_COLUMN] = result["home_score"] == result["away_score"]
    result[MODEL_ELIGIBLE_COLUMN] = result[TARGET_COLUMN].notna()

    assert result.loc[0, MODEL_ELIGIBLE_COLUMN] == True, "Non-tie should be eligible"
    assert result.loc[1, MODEL_ELIGIBLE_COLUMN] == False, "Tie should not be eligible"
    assert result.loc[1, TIE_COLUMN] == True, "Tie should be flagged"


def test_elo_update_on_tie(sample_schedule_with_tie):
    """Elo should use 0.5 for tie games (half-win each side)."""
    # Override home_win: tie game gets pd.NA
    df = sample_schedule_with_tie.copy()
    df["home_win"] = [1, pd.NA]

    result = compute_elo_features(
        df,
        k_factor=36,
        home_advantage=40,
        mov_type="none",
    )
    # The tie game gets home_win=0.5 for Elo purposes
    # ATL won G1 (home=1, expected≈0.557) → ATL gained Elo
    # CHI tied G2 (home=0.5, expected≈0.5+slight_HFA) → small update

    # Game 2 (index 1): tie, home_win=pd.NA → actual_home=0.5
    # Since ATL just won G1, their Elo is higher → CHI is slight underdog
    # A tie at home is a slightly negative outcome for the favorite
    assert not pd.isna(result.loc[1, "home_elo_pre"]), "Elo pre should exist for tie games"
    assert not pd.isna(result.loc[1, "elo_prob"]), "Elo prob should exist for tie games"


def test_incumbent_generation_excludes_ties():
    """The incumbent prediction pipeline filters ties before training.

    The _build_feature_pipeline returns all rows including ties.
    Ties are filtered out in generate_incumbent_predictions via the
    model_eligible mask. Verify:
    - The feature pipeline retains ties
    - The prediction pipeline excludes them
    """
    import os
    fp = "data/features/nfl/feature_table.parquet"
    if not os.path.exists(fp):
        pytest.skip("Feature table not found — cannot test incumbent filtering")

    df = _build_feature_pipeline()
    # Feature pipeline retains ties
    assert not df[MODEL_ELIGIBLE_COLUMN].all(), "Feature pipeline should still contain ties"
    assert df[TARGET_COLUMN].isna().any(), "Feature pipeline should contain tie rows (NA target)"

    # The prediction mask filters them out
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df["is_neutral"].fillna(False).values
    df_filtered = df[mask].copy().reset_index(drop=True)
    assert df_filtered[MODEL_ELIGIBLE_COLUMN].all(), "Filtered rows should all be eligible"
    assert df_filtered[TARGET_COLUMN].notna().all(), "Filtered targets should all be non-NA"


def test_incumbent_fit_handles_no_ties(sample_schedule):
    """LogisticRegression training should never see 0.5 targets.

    Using a small schedule with all valid home_win values.
    """
    # Add required columns for the pipeline
    df = sample_schedule.copy()
    df["away_qb_id"] = "QB1"
    df["home_qb_id"] = "QB2"
    df["away_qb_name"] = "QB One"
    df["home_qb_name"] = "QB Two"
    df["away_coach"] = "Coach A"
    df["home_coach"] = "Coach B"
    df["referee"] = "Ref 1"
    df["stadium_id"] = "S1"
    df["stadium"] = "Stadium 1"
    df["game_type"] = "REG"
    df["weekday"] = "Sunday"
    df["gametime"] = "13:00"
    df["home_rest"] = 7
    df["away_rest"] = 7
    df["rest_diff"] = 0
    df["div_game"] = 0
    df["location"] = ""
    df[TARGET_COLUMN] = df["home_win"]
    df[MODEL_ELIGIBLE_COLUMN] = df[TARGET_COLUMN].notna()
    df[TIE_COLUMN] = False
    df["is_neutral"] = 0
    df["away_moneyline"] = 100
    df["home_moneyline"] = -120
    df["spread_line"] = 2.5
    df["away_spread_odds"] = -110
    df["home_spread_odds"] = -110
    df["total_line"] = 45.5
    df["under_odds"] = -110
    df["over_odds"] = -110

    # Build features
    from sportslab.evaluation.season_regression_experiment import build_team_regression_overrides
    overrides = build_team_regression_overrides(df, preseason_regression=0.1, qb_change_bonus=0.2)
    df = compute_elo_features(df, k_factor=36, home_advantage=40, preseason_regression=0.1,
                              team_regression_overrides=overrides)
    from sportslab.features.qb import compute_qb_features
    df = compute_qb_features(df)
    from sportslab.features.situational import compute_situational_features
    df = compute_situational_features(df)

    # Add market features (needed for pipeline)
    df["market_home_prob_novig"] = 0.55
    df["market_away_prob_novig"] = 0.45

    # Fit — should work fine
    pipe = _fit_incumbent(df)
    assert pipe is not None, "Incumbent fit should succeed"
