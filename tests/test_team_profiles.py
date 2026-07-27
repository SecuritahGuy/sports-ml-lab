"""Tests for team profiles module."""


from sportslab.evaluation.team_profiles import build_team_profiles


class TestTeamProfiles:
    def test_importable(self):
        from sportslab.evaluation import team_profiles
        assert hasattr(team_profiles, "build_team_profiles")

    def test_team_profiles_produces_csv(self, tmp_path):
        output = tmp_path / "profiles.csv"
        df = build_team_profiles(output_path=str(output))
        assert len(df) > 0
        assert output.exists()
        content = output.read_text()
        assert "season" in content
        assert "team" in content
        assert "doba_score" in content
        assert "chaos_rate" in content
        assert "aggression_score" in content
        assert "qb_lift_index" in content

    def test_team_profiles_all_teams_each_season(self, tmp_path):
        output = tmp_path / "profiles2.csv"
        df = build_team_profiles(output_path=str(output))
        for s in [2021, 2022, 2023, 2024, 2025]:
            subset = df[df["season"] == s]
            assert len(subset) > 0, f"No teams for season {s}"
