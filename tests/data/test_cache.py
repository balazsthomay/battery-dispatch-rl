"""Tests for data cache module."""

from pathlib import Path

import pandas as pd
import pytest

from bess_dispatch.data.cache import cache_exists, cache_path, load_from_cache, save_to_cache


class TestCachePath:
    """Tests for cache_path function."""

    def test_returns_correct_structure(self):
        path = cache_path("data", "DE_LU", "day_ahead_prices", 2023)
        assert path == Path("data/DE_LU/day_ahead_prices_2023.parquet")

    def test_different_zone(self):
        path = cache_path("data", "ES", "day_ahead_prices", 2024)
        assert path == Path("data/ES/day_ahead_prices_2024.parquet")

    def test_different_data_type(self):
        path = cache_path("data", "NL", "wind_solar", 2023)
        assert path == Path("data/NL/wind_solar_2023.parquet")

    def test_custom_data_dir(self):
        path = cache_path("/tmp/mydata", "DE_LU", "day_ahead_prices", 2023)
        assert path == Path("/tmp/mydata/DE_LU/day_ahead_prices_2023.parquet")


class TestSaveAndLoadCache:
    """Tests for save_to_cache and load_from_cache round-trip."""

    def test_round_trip(self, tmp_path, sample_prices):
        path = tmp_path / "test.parquet"
        df = sample_prices.to_frame()
        save_to_cache(df, path)
        loaded = load_from_cache(path)
        # Parquet may not preserve freq metadata, so check values only
        pd.testing.assert_frame_equal(loaded, df, check_freq=False)

    def test_creates_parent_directories(self, tmp_path, sample_prices):
        path = tmp_path / "sub" / "dir" / "test.parquet"
        df = sample_prices.to_frame()
        save_to_cache(df, path)
        assert path.exists()

    def test_load_missing_raises(self, tmp_path):
        path = tmp_path / "nonexistent.parquet"
        with pytest.raises(FileNotFoundError, match="Cache file not found"):
            load_from_cache(path)


class TestCacheExists:
    """Tests for cache_exists function."""

    def test_returns_true_when_exists(self, tmp_path, sample_prices):
        path = tmp_path / "test.parquet"
        save_to_cache(sample_prices.to_frame(), path)
        assert cache_exists(path) is True

    def test_returns_false_when_missing(self, tmp_path):
        path = tmp_path / "nonexistent.parquet"
        assert cache_exists(path) is False
