"""Tests for the NFL ingestion module — no network calls."""

from datetime import datetime

import pandas as pd
import pytest

from sportslab.data.ingest_nfl import (
    NFLREADPY_AVAILABLE,
    _build_metadata,
    _normalize_table,
    _validate_seasons,
)


class TestNormalizeTable:
    """Test _normalize_table with various input types."""

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = _normalize_table(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_polars_like_to_pandas(self):
        class FakePolars:
            def to_pandas(self):
                return pd.DataFrame({"x": [10, 20]})

        result = _normalize_table(FakePolars())
        assert isinstance(result, pd.DataFrame)
        assert result["x"].tolist() == [10, 20]

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Cannot normalize"):
            _normalize_table("not a dataframe")


class TestBuildMetadata:
    """Test _build_metadata output structure."""

    def test_metadata_keys(self):
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        meta = _build_metadata(
            source_function="load_schedules",
            seasons_requested=[2021, 2022],
            seasons_found=[2021, 2022],
            df=df,
            output_path="data/raw/nfl/schedules.parquet",
        )
        expected_keys = {
            "package",
            "nflreadpy_version",
            "seasons_requested",
            "seasons_found",
            "row_count",
            "column_count",
            "columns",
            "created_at",
            "output_path",
            "source_function",
        }
        assert set(meta.keys()) == expected_keys

    def test_metadata_values(self):
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        meta = _build_metadata(
            source_function="load_schedules",
            seasons_requested=[2021],
            seasons_found=[2021],
            df=df,
            output_path="data/raw/nfl/schedules.parquet",
        )
        assert meta["package"] == "nflreadpy"
        assert meta["source_function"] == "load_schedules"
        assert meta["seasons_requested"] == [2021]
        assert meta["row_count"] == 2
        assert meta["column_count"] == 2
        assert meta["columns"] == ["col_a", "col_b"]
        assert meta["output_path"] == "data/raw/nfl/schedules.parquet"
        assert isinstance(meta["created_at"], str)
        # Validate ISO format timestamp
        datetime.fromisoformat(meta["created_at"])

    def test_nflreadpy_version_present(self):
        df = pd.DataFrame({"a": [1]})
        meta = _build_metadata(
            source_function="load_schedules",
            seasons_requested=[2022],
            seasons_found=[2022],
            df=df,
            output_path="output.parquet",
        )
        if NFLREADPY_AVAILABLE:
            assert meta["nflreadpy_version"] != "unknown"
            assert isinstance(meta["nflreadpy_version"], str)
        else:
            assert meta["nflreadpy_version"] == "unknown"


class TestNflreadpyAvailability:
    """Test that nflreadpy availability flag is set correctly."""

    def test_availability_flag(self):
        # If nflreadpy is installed this should be True
        try:
            import nflreadpy  # noqa: F401

            assert NFLREADPY_AVAILABLE is True
        except ImportError:
            assert NFLREADPY_AVAILABLE is False


class TestValidateSeasons:
    """Test _validate_seasons with various inputs."""

    def test_valid_seasons_2021_2022(self):
        """Seasons 2021 and later are accepted by ingestion."""
        _validate_seasons([2021, 2022])

    def test_valid_seasons_2025(self):
        """Single season 2025 is accepted."""
        _validate_seasons([2025])

    def test_season_2020_rejected(self):
        """Season 2020 (pre-2021) raises ValueError."""
        with pytest.raises(ValueError, match="2021"):
            _validate_seasons([2020])

    def test_mixed_seasons_rejected(self):
        """Mixed seasons with a pre-2021 entry raise ValueError."""
        with pytest.raises(ValueError, match="2021"):
            _validate_seasons([2020, 2025])

    def test_all_bad_seasons_rejected(self):
        """Multiple pre-2021 seasons all raise ValueError."""
        with pytest.raises(ValueError, match="2021"):
            _validate_seasons([2000, 2005, 2009])

    def test_season_2010_rejected(self):
        """Season 2010 (pre-2021) raises ValueError."""
        with pytest.raises(ValueError, match="2021"):
            _validate_seasons([2010])

    def test_error_message_contains_project_scope(self):
        """Error message explicitly mentions the min season."""
        with pytest.raises(ValueError) as exc:
            _validate_seasons([2000])
        msg = str(exc.value)
        assert "2021" in msg
        assert "current" in msg or "only" in msg


class TestOutputPaths:
    """Test that output paths are constructed as expected."""

    def test_default_output_structure(self):
        from pathlib import Path

        data_dir = Path("data/raw/nfl")
        parquet = data_dir / "schedules.parquet"
        metadata = data_dir / "schedules_metadata.json"
        assert str(parquet) == "data/raw/nfl/schedules.parquet"
        assert str(metadata) == "data/raw/nfl/schedules_metadata.json"
