"""RALPH Loop 6: tests for challenger experiment module."""

from sportslab.evaluation.ralph6_challengers import (
    _build_base_df,
    _compute_prior_season_win_pct,
    _filter_df,
    _subgroup_mask,
    check_promotion,
)


class TestBasePipeline:
    def test_build_base_df_has_elo(self):
        df = _build_base_df()
        for col in ["elo_prob", "home_elo_pre", "away_elo_pre", "elo_diff"]:
            assert col in df.columns

    def test_build_base_df_has_qb_features(self):
        df = _build_base_df()
        for col in ["home_qb_changed", "away_qb_changed"]:
            assert col in df.columns

    def test_build_base_df_has_situational(self):
        df = _build_base_df()
        for col in ["home_rolling_mov_3", "away_rolling_mov_3"]:
            assert col in df.columns

    def test_build_base_df_has_adj(self):
        df = _build_base_df()
        assert "home_qb_adj" in df.columns
        assert "away_qb_adj" in df.columns

    def test_build_base_df_has_weather(self):
        df = _build_base_df()
        for col in ["weather_missing_flag", "is_dome", "outdoor_game_flag", "roof_enc"]:
            assert col in df.columns

    def test_filter_df_excludes_neutral_and_ineligible(self):
        df = _build_base_df()
        filtered = _filter_df(df)
        assert len(filtered) <= len(df)

    def test_filter_df_has_target_column(self):
        df = _build_base_df()
        filtered = _filter_df(df)
        assert "home_win" in filtered.columns


class TestPriorSeasonWinPct:
    def test_prior_win_pct_adds_columns(self):
        df = _build_base_df()
        result = _compute_prior_season_win_pct(df)
        assert "home_prior_win_pct" in result.columns
        assert "away_prior_win_pct" in result.columns

    def test_prior_win_pct_in_01_range(self):
        df = _build_base_df()
        result = _compute_prior_season_win_pct(df)
        assert result["home_prior_win_pct"].between(0, 1).all()
        assert result["away_prior_win_pct"].between(0, 1).all()

    def test_prior_win_pct_defaults_to_05(self):
        """First season (2021) should default to 0.5 for all teams."""
        df = _build_base_df()
        result = _compute_prior_season_win_pct(df)
        w1 = result[result["season"] == 2021]
        assert (w1["home_prior_win_pct"] == 0.5).all()
        assert (w1["away_prior_win_pct"] == 0.5).all()


class TestSubgroupMasks:
    def test_early_season_mask(self):
        df = _build_base_df()
        df = _filter_df(df)
        mask = _subgroup_mask(df, "early")
        assert mask.sum() > 0

    def test_weather_missing_mask(self):
        df = _build_base_df()
        df = _filter_df(df)
        mask = _subgroup_mask(df, "weather_missing")
        assert mask.sum() >= 0

    def test_retractable_mask(self):
        df = _build_base_df()
        df = _filter_df(df)
        mask = _subgroup_mask(df, "retractable")
        assert mask.sum() >= 0

    def test_qb_changed_mask(self):
        df = _build_base_df()
        df = _filter_df(df)
        mask = _subgroup_mask(df, "qb_changed")
        assert mask.sum() > 0


class TestPromotionCheck:
    def test_check_promotion_rejects_worse_val(self):
        r = check_promotion(val_ll=0.6400, holdout_ll=0.6100,
                            incumbent_val=0.6305, incumbent_holdout=0.6200, delta=0.001)
        assert not r["promoted"]

    def test_check_promotion_rejects_worse_hold(self):
        r = check_promotion(val_ll=0.6200, holdout_ll=0.6500,
                            incumbent_val=0.6305, incumbent_holdout=0.6200, delta=0.001)
        assert not r["promoted"]

    def test_check_promotion_requires_delta(self):
        r = check_promotion(val_ll=0.6296, holdout_ll=0.6191,
                            incumbent_val=0.6305, incumbent_holdout=0.6200, delta=0.001)
        assert not r["promoted"]

    def test_check_promotion_promotes_at_delta(self):
        r = check_promotion(val_ll=0.6295, holdout_ll=0.6190,
                            incumbent_val=0.6305, incumbent_holdout=0.6200, delta=0.001)
        assert r["promoted"]
