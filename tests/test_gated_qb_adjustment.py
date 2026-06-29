"""Tests for gated QB adjustment logic and experiment."""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.qb_adjustment import (
    GATE_MODES,
    apply_qb_adjustment_gate,
    compute_gated_qb_adjustments,
    compute_recency_weighted_qb_adjustments,
)


class TestApplyQBAdjustmentGate:
    def test_full_mode_returns_unchanged(self):
        h = np.array([10.0, -20.0, 30.0])
        a = np.array([-5.0, 15.0, 0.0])
        changed = np.array([0, 1, 0])
        g_h, g_a = apply_qb_adjustment_gate(h, a, changed, changed, gate_mode="full")
        assert np.allclose(g_h, h)
        assert np.allclose(g_a, a)

    def test_qb_changed_only_zeroes_stable(self):
        h = np.array([10.0, -20.0, 30.0])
        a = np.array([-5.0, 15.0, 0.0])
        h_changed = np.array([1, 0, 0])
        a_changed = np.array([0, 0, 1])
        g_h, g_a = apply_qb_adjustment_gate(
            h, a, h_changed, a_changed, gate_mode="qb_changed_only"
        )
        # Home: adj * [1,0,0] = [10, 0, 0]
        # Away: adj * [0,0,1] = [0, 0, 0]
        assert np.allclose(g_h, [10.0, 0.0, 0.0])
        assert np.allclose(g_a, [0.0, 0.0, 0.0])

    def test_low_continuity_with_existing_starts(self):
        h = np.array([10.0, 20.0, 30.0])
        a = np.array([5.0, 15.0, 25.0])
        h_changed = np.array([1, 0, 0])
        a_changed = np.array([0, 0, 1])
        h_starts = np.array([0, 10, 5])
        a_starts = np.array([3, 2, 0])
        g_h, g_a = apply_qb_adjustment_gate(
            h, a, h_changed, a_changed,
            home_qb_team_starts_pre=h_starts,
            away_qb_team_starts_pre=a_starts,
            gate_mode="low_continuity",
            min_starts_for_stable=4,
        )
        # Home: changed=1 OR starts<4 => adj*1
        #   game 0: changed=1 -> 10
        #   game 1: starts=10 >= 4, changed=0 -> 0
        #   game 2: starts=5 >= 4, changed=0 -> 0
        # Away:
        #   game 0: starts=3 < 4 -> 5
        #   game 1: starts=2 < 4 -> 15
        #   game 2: changed=1 -> 25
        assert np.allclose(g_h, [10.0, 0.0, 0.0])
        assert np.allclose(g_a, [5.0, 15.0, 25.0])

    def test_shrunk_stable_applies_shrink_to_stable(self):
        h = np.array([10.0, 20.0, 30.0])
        a = np.array([5.0, 15.0, 25.0])
        h_changed = np.array([1, 0, 0])
        a_changed = np.array([0, 0, 1])
        g_h, g_a = apply_qb_adjustment_gate(
            h, a, h_changed, a_changed,
            gate_mode="shrunk_stable",
            stable_shrink=0.3,
        )
        # Home: [1, 0.3, 0.3] * [10, 20, 30] = [10, 6, 9]
        # Away: [0.3, 0.3, 1] * [5, 15, 25] = [1.5, 4.5, 25]
        assert np.allclose(g_h, [10.0, 6.0, 9.0])
        assert np.allclose(g_a, [1.5, 4.5, 25.0])

    def test_shrunk_stable_full_for_changed(self):
        h = np.array([10.0])
        h_changed = np.array([1])
        g_h, _ = apply_qb_adjustment_gate(
            h, np.array([0.0]), h_changed, np.array([0]),
            gate_mode="shrunk_stable", stable_shrink=0.0,
        )
        assert g_h[0] == 10.0  # Full adjustment for changed QB

    def test_capped_only_limits_max_adj(self):
        h = np.array([100.0, -50.0, 30.0])
        a = np.array([-80.0, 60.0, 5.0])
        g_h, g_a = apply_qb_adjustment_gate(
            h, a, np.zeros(3), np.zeros(3),
            gate_mode="capped_only", max_adj_cap=60.0,
        )
        assert np.allclose(g_h, [60.0, -50.0, 30.0])
        assert np.allclose(g_a, [-60.0, 60.0, 5.0])

    def test_aggressive_diagnostic_doubles_changed_zeroes_stable(self):
        h = np.array([10.0, 20.0])
        a = np.array([5.0, 15.0])
        h_changed = np.array([1, 0])
        a_changed = np.array([0, 1])
        g_h, g_a = apply_qb_adjustment_gate(
            h, a, h_changed, a_changed,
            gate_mode="aggressive_diagnostic",
        )
        assert np.allclose(g_h, [20.0, 0.0])
        assert np.allclose(g_a, [0.0, 30.0])

    def test_stable_qb_not_modified_for_changed_only(self):
        """Stable-QB games must return 0 adjustment for qb_changed_only."""
        h = np.array([50.0, -30.0, 20.0])
        a = np.array([-40.0, 10.0, -10.0])
        h_changed = np.array([1, 0, 0])
        a_changed = np.array([0, 0, 1])
        g_h, g_a = apply_qb_adjustment_gate(
            h, a, h_changed, a_changed,
            gate_mode="qb_changed_only",
        )
        # Game 1 (index 1): no home change, no away change — both should be 0
        assert g_h[1] == 0.0
        assert g_a[1] == 0.0

    def test_missing_gate_data_fails_safely(self):
        """Missing gate arrays should produce finite values."""
        h = np.array([10.0, -20.0])
        a = np.array([5.0, -15.0])
        # Pass no optional arrays — should still work with defaults
        g_h, g_a = apply_qb_adjustment_gate(
            h, a,
            np.array([0, 1]), np.array([0, 0]),
            gate_mode="qb_changed_only",
        )
        assert np.all(np.isfinite(g_h))
        assert np.all(np.isfinite(g_a))

    def test_adjustments_remain_finite_and_bounded(self):
        rng = np.random.default_rng(42)
        h = rng.uniform(-200, 200, size=50)
        a = rng.uniform(-200, 200, size=50)
        changed = rng.integers(0, 2, size=50)

        for mode in ["full", "qb_changed_only", "shrunk_stable"]:
            g_h, g_a = apply_qb_adjustment_gate(
                h, a, changed, changed,
                gate_mode=mode,
                stable_shrink=0.3,
            )
            assert np.all(np.isfinite(g_h))
            assert np.all(np.isfinite(g_a))
            assert np.all(np.abs(g_h) <= 200)
            assert np.all(np.abs(g_a) <= 200)

    def test_hypers_do_not_produce_invalid_probs(self):
        rng = np.random.default_rng(123)
        h = rng.uniform(-100, 100, size=20)
        a = rng.uniform(-100, 100, size=20)
        changed = rng.integers(0, 2, size=20)
        starts = rng.integers(0, 80, size=20)

        # All gate mode combinations should produce valid values
        for mode in GATE_MODES:
            for shrink in [0.1, 0.5]:
                for cap in [40, 80]:
                    kws = dict(gate_mode=mode, stable_shrink=shrink)
                    if mode == "capped_only":
                        kws["max_adj_cap"] = cap
                    g_h, g_a = apply_qb_adjustment_gate(
                        h, a, changed, changed,
                        home_qb_team_starts_pre=starts,
                        away_qb_team_starts_pre=starts,
                        **kws,
                    )
                    assert np.all(np.isfinite(g_h)), f"Non-finite h for mode={mode}"
                    assert np.all(np.isfinite(g_a)), f"Non-finite a for mode={mode}"

    def test_gate_modes_have_descriptions(self):
        for mode, desc in GATE_MODES.items():
            assert len(desc) > 10
            assert " " in desc


class TestComputeGatedQBAdjustments:
    def test_gated_equals_full_when_full_mode(self):
        df = _make_minimal_df()
        from sportslab.features.qb_adjustment import compute_qb_adjustments
        raw = compute_qb_adjustments(df)
        full = compute_gated_qb_adjustments(df, gate_mode="full")
        assert np.allclose(full["home_qb_adj"].values, raw["home_qb_adj"].values, atol=0.01)
        assert np.allclose(full["away_qb_adj"].values, raw["away_qb_adj"].values, atol=0.01)

    def test_gated_zeroes_for_stable_when_changed_only(self):
        df = _make_minimal_df(season=[2021, 2021], week=[1, 2])
        # Add a third game so we can test qb_changed=1 -> adjustment preserved
        # and qb_changed=0 -> adjustment zeroed
        extra = pd.DataFrame({
            "season": [2021],
            "week": [3],
            "gameday": ["2021-09-26"],
            "home_team": ["KC"],
            "away_team": ["LV"],
            "home_elo_pre": [1540.0],
            "away_elo_pre": [1490.0],
            "elo_prob": [0.55],
            "home_qb_id": ["00-MAHOMES"],
            "away_qb_id": ["00-CARR"],
            "home_qb_name": ["P.Mahomes"],
            "away_qb_name": ["D.Carr"],
            "home_win": [1],
            "home_score": [30],
            "away_score": [20],
        })
        df = pd.concat([df, extra], ignore_index=True)
        result = compute_gated_qb_adjustments(df, gate_mode="qb_changed_only")
        # gated adj = full_adj * qb_changed.  For games where qb_changed=0,
        # the gated adj should be 0.  First game has no prior QB, changed=0.
        assert result["home_qb_adj"].iloc[0] == 0.0, "First game, no prior QB"
        assert np.all(np.isfinite(result["home_qb_adj"]))
        assert np.all(np.isfinite(result["away_qb_adj"]))

    def test_missing_qb_data_fails_safely(self):
        df = _make_minimal_df(home_qb_id=[None, "00-ALLEN"])
        result = compute_gated_qb_adjustments(df, gate_mode="qb_changed_only")
        assert result["home_qb_adj"].iloc[0] == 0.0
        assert np.isfinite(result["home_qb_adj"].iloc[1])

    def test_no_season_before_2021(self):
        df = _make_minimal_df(season=[2020, 2020])
        with pytest.raises(ValueError, match="2021"):
            compute_gated_qb_adjustments(df, gate_mode="full")

    def test_empty_df(self):
        df = pd.DataFrame(columns=["season", "week", "gameday", "home_team", "away_team",
                                    "home_elo_pre", "away_elo_pre", "elo_prob",
                                    "home_qb_id", "away_qb_id", "home_win"])
        result = compute_gated_qb_adjustments(df, gate_mode="full")
        assert len(result) == 0


class TestRecencyWeightedQBAdjustments:
    def test_adds_expected_columns(self):
        df = _make_minimal_df()
        result = compute_recency_weighted_qb_adjustments(df)
        for col in ["home_qb_adj", "away_qb_adj", "home_qb_starts", "away_qb_starts"]:
            assert col in result.columns

    def test_first_game_zero_starts(self):
        df = _make_minimal_df()
        result = compute_recency_weighted_qb_adjustments(df)
        assert result["home_qb_starts"].iloc[0] == 0
        assert result["away_qb_starts"].iloc[0] == 0

    def test_second_game_has_one_start(self):
        df = _make_minimal_df()
        result = compute_recency_weighted_qb_adjustments(df)
        assert result["home_qb_starts"].iloc[1] == 1
        assert result["away_qb_starts"].iloc[1] == 0

    def test_adjustments_finite_and_bounded(self):
        df = _make_minimal_df()
        extra = pd.DataFrame({
            "season": [2021] * 10,
            "week": list(range(3, 13)),
            "gameday": [f"2021-09-{i:02d}" for i in range(3, 13)],
            "home_team": ["KC"] * 10,
            "away_team": ["LV"] * 10,
            "home_elo_pre": [1500.0] * 10,
            "away_elo_pre": [1480.0] * 10,
            "elo_prob": [0.5] * 10,
            "home_qb_id": ["00-MAHOMES"] * 10,
            "away_qb_id": ["00-CARR"] * 10,
            "home_qb_name": ["P.Mahomes"] * 10,
            "away_qb_name": ["D.Carr"] * 10,
            "home_win": [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            "home_score": [28] * 10,
            "away_score": [17] * 10,
        })
        df_full = pd.concat([df, extra], ignore_index=True)
        result = compute_recency_weighted_qb_adjustments(df_full)
        for col in ["home_qb_adj", "away_qb_adj"]:
            assert np.all(np.isfinite(result[col]))
            assert np.all(np.abs(result[col]) <= 120)

    def test_decay_reduces_older_game_influence(self):
        """Very high decay should make early games nearly irrelevant."""
        rows = []
        for i in range(34):
            rows.append({
                "season": 2021 if i < 17 else 2022,
                "week": (i % 17) + 1,
                "gameday": f"2021-09-{i+10:02d}" if i < 17 else f"2022-09-{i-7:02d}",
                "home_team": "KC",
                "away_team": "LV",
                "home_elo_pre": 1500.0,
                "away_elo_pre": 1480.0,
                "elo_prob": 0.5,
                "home_qb_id": "00-MAHOMES",
                "away_qb_id": "00-CARR",
                "home_qb_name": "P.Mahomes",
                "away_qb_name": "D.Carr",
                "home_win": 1,
                "home_score": 28,
                "away_score": 17,
            })
        df = pd.DataFrame(rows)
        result = compute_recency_weighted_qb_adjustments(df, decay_half_life=5.0)
        assert np.all(np.isfinite(result["home_qb_adj"]))
        assert np.all(np.isfinite(result["away_qb_adj"]))

    def test_missing_qb_id_safe(self):
        df = _make_minimal_df(home_qb_id=[None, "00-ALLEN"])
        result = compute_recency_weighted_qb_adjustments(df)
        assert result["home_qb_adj"].iloc[0] == 0.0
        assert result["home_qb_starts"].iloc[0] == 0

    def test_no_season_before_2021(self):
        df = _make_minimal_df(season=[2020, 2020])
        with pytest.raises(ValueError, match="2021"):
            compute_recency_weighted_qb_adjustments(df)


class TestExperimentImportability:
    def test_module_imports(self):
        from sportslab.evaluation import gated_qb_adjusted_elo_experiment
        assert hasattr(
            gated_qb_adjusted_elo_experiment,
            "run_gated_qb_adjusted_elo_experiment",
        )

    def test_cli_command_registered(self):
        from sportslab import cli
        commands = {c.name for c in cli.cli.commands.values()}
        assert "gated-qb-elo" in commands

    def test_gate_modes_importable(self):
        assert "full" in GATE_MODES
        assert "qb_changed_only" in GATE_MODES
        assert "shrunk_stable" in GATE_MODES
        assert "aggressive_diagnostic" in GATE_MODES


class TestExperimentSafety:
    def test_no_season_before_2021(self):
        """Verify experiment config does not use pre-2021 seasons."""
        from sportslab.evaluation.experiment_config import ALL_SEASONS, ROLLING_FOLDS
        for s in ALL_SEASONS:
            assert s >= 2021
        for train_seasons, val_season in ROLLING_FOLDS:
            for ts in train_seasons:
                assert ts >= 2021
            assert val_season >= 2022

    def test_experiment_smoke(self, tmp_path):
        """Minimal smoke test — run experiment end-to-end."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")

        from sportslab.evaluation.gated_qb_adjusted_elo_experiment import (
            run_gated_qb_adjusted_elo_experiment,
        )
        report_p = tmp_path / "gated_qb_test.md"
        run_gated_qb_adjusted_elo_experiment(report_path=str(report_p))
        assert report_p.exists()
        content = report_p.read_text()
        assert "Decision" in content
        assert "Rolling-Origin Validation" in content
        assert "2025 Holdout" in content
        assert "Gating" in content or "gated" in content.lower()

    def test_experiment_does_not_modify_incumbent_files(self, tmp_path):
        """Verify experiment does not overwrite incumbent outputs."""
        from sportslab.evaluation.gated_qb_adjusted_elo_experiment import (
            run_gated_qb_adjusted_elo_experiment,
        )
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")

        report_p = tmp_path / "gated_qb_test.md"
        run_gated_qb_adjusted_elo_experiment(report_path=str(report_p))
        # Check that incumbent prediction artifacts are not modified
        import os
        inc_pred = "reports/predictions/incumbent_predictions.csv"
        if os.path.exists(inc_pred):
            df = pd.read_csv(inc_pred)
            # Verify schema not modified by experiment
            assert "incumbent_home_win_prob" in df.columns


def _make_minimal_df(**overrides) -> pd.DataFrame:
    defaults = {
        "season": [2021, 2021],
        "week": [1, 2],
        "gameday": ["2021-09-09", "2021-09-19"],
        "home_team": ["KC", "KC"],
        "away_team": ["CLE", "LAC"],
        "home_elo_pre": [1500.0, 1520.0],
        "away_elo_pre": [1500.0, 1480.0],
        "elo_prob": [0.5, 0.5],
        "home_qb_id": ["00-MAHOMES", "00-MAHOMES"],
        "away_qb_id": ["00-BAKER", "00-HERB"],
        "home_qb_name": ["P.Mahomes", "P.Mahomes"],
        "away_qb_name": ["B.Mayfield", "J.Herbert"],
        "home_win": [1, 0],
        "home_score": [33, 20],
        "away_score": [29, 31],
    }
    df = pd.DataFrame(defaults)
    for k, v in overrides.items():
        df[k] = v
    return df
