"""Tests for data loader module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bess_dispatch.config import DataConfig, MarketConfig
from bess_dispatch.data.loader import (
    MarketData,
    MarketDataLoader,
    load_sample,
    validate_prices,
)
from bess_dispatch.data.cache import cache_path


class TestValidatePrices:
    """Tests for validate_prices function."""

    def test_raises_on_empty_series(self):
        empty = pd.Series(dtype=float)
        with pytest.raises(ValueError, match="Price series is empty"):
            validate_prices(empty)

    def test_raises_on_too_many_missing(self):
        """More than 10% missing should raise."""
        prices = pd.Series([np.nan] * 20 + [50.0] * 80, dtype=float)
        with pytest.raises(ValueError, match="Too many missing values"):
            validate_prices(prices, max_missing_fraction=0.1)

    def test_forward_fills_small_gaps(self, sample_prices):
        """Gaps up to max_gap_hours should be filled."""
        prices = sample_prices.copy()
        # Introduce a 2-hour gap
        prices.iloc[10:12] = np.nan
        result = validate_prices(prices, max_gap_hours=3)
        assert not result.isna().any()

    def test_backward_fills_small_gaps_at_start(self):
        """NaN at start should be backward-filled."""
        hours = pd.date_range("2023-01-02", periods=30, freq="h", tz="Europe/Berlin")
        values = [np.nan, np.nan] + [50.0 + i for i in range(28)]
        prices = pd.Series(values, index=hours, name="price_eur_mwh")
        result = validate_prices(prices, max_gap_hours=3)
        assert not result.isna().any()

    def test_raises_on_unfillable_gaps(self):
        """Gaps larger than max_gap_hours should raise after filling attempts."""
        # Use enough data points so the missing fraction is under 10%
        hours = pd.date_range("2023-01-02", periods=100, freq="h", tz="Europe/Berlin")
        prices = pd.Series([50.0] * 100, index=hours, name="price_eur_mwh")
        # Introduce an 8-hour gap in the middle (unfillable with max_gap_hours=3)
        prices.iloc[45:53] = np.nan
        with pytest.raises(ValueError, match="Unfillable gaps remain"):
            validate_prices(prices, max_gap_hours=3)

    def test_passes_clean_data(self, sample_prices):
        """Clean data should pass through unchanged."""
        result = validate_prices(sample_prices)
        pd.testing.assert_series_equal(result, sample_prices)


class TestMarketDataLoader:
    """Tests for MarketDataLoader class."""

    def test_loads_from_cache_when_available(self, tmp_path, sample_prices):
        """Should load from cache instead of calling API."""
        # Setup: save prices to expected cache location
        zone = "DE_LU"
        year = 2023
        cache_dir = tmp_path / zone
        cache_dir.mkdir()
        cache_file = cache_dir / f"day_ahead_prices_{year}.parquet"
        sample_prices.to_frame().to_parquet(cache_file)

        data_config = DataConfig(data_dir=str(tmp_path))
        loader = MarketDataLoader(data_config=data_config)

        # Should load from cache without needing a client
        prices = loader.load_prices(zone, year)
        assert len(prices) == 168
        assert prices.name == "price_eur_mwh"

    def test_load_market_data_returns_market_data(self, tmp_path, sample_prices):
        """load_market_data should return a MarketData instance."""
        zone = "DE_LU"
        year = 2023
        cache_dir = tmp_path / zone
        cache_dir.mkdir()
        cache_file = cache_dir / f"day_ahead_prices_{year}.parquet"
        sample_prices.to_frame().to_parquet(cache_file)

        data_config = DataConfig(data_dir=str(tmp_path))
        loader = MarketDataLoader(data_config=data_config)

        market_data = loader.load_market_data(zone, year)
        assert isinstance(market_data, MarketData)
        assert market_data.zone == zone
        assert market_data.year == year
        assert len(market_data.prices) == 168

    def test_uses_default_market_config_zone(self, tmp_path, sample_prices):
        """Should use MarketConfig default zone if none provided."""
        zone = "DE_LU"
        cache_dir = tmp_path / zone
        cache_dir.mkdir()
        cache_file = cache_dir / "day_ahead_prices_2023.parquet"
        sample_prices.to_frame().to_parquet(cache_file)

        data_config = DataConfig(data_dir=str(tmp_path))
        market_config = MarketConfig(zone="DE_LU")
        loader = MarketDataLoader(data_config=data_config, market_config=market_config)

        prices = loader.load_prices(year=2023)
        assert len(prices) == 168

    def test_client_property_lazy_init(self):
        """Client should not be created until accessed."""
        loader = MarketDataLoader()
        assert loader._client is None

    @patch("bess_dispatch.data.loader.EntsoeClient")
    def test_fetches_from_api_when_cache_missing(self, mock_client_cls, tmp_path, sample_prices):
        """When cache is missing, should fetch from API and cache the result."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.fetch_day_ahead_prices.return_value = sample_prices

        data_config = DataConfig(data_dir=str(tmp_path))
        loader = MarketDataLoader(data_config=data_config, client=mock_client)

        prices = loader.load_prices("DE_LU", 2023)
        mock_client.fetch_day_ahead_prices.assert_called_once()
        assert len(prices) == 168

        # Verify it was cached
        path = cache_path(str(tmp_path), "DE_LU", "day_ahead_prices", 2023)
        assert path.exists()


class TestLoadSample:
    """Tests for load_sample function."""

    def test_load_sample_returns_market_data(self):
        """load_sample should return a valid MarketData with 168 hourly prices."""
        market_data = load_sample()
        assert isinstance(market_data, MarketData)
        assert market_data.zone == "DE_LU"
        assert market_data.year == 2023
        assert len(market_data.prices) == 168
        assert market_data.prices.name == "price_eur_mwh"

    def test_load_sample_prices_realistic_range(self):
        """Sample prices should be in a realistic range."""
        market_data = load_sample()
        assert market_data.prices.min() > -50  # no extreme negatives
        assert market_data.prices.max() < 300  # no extreme positives
        assert 20 < market_data.prices.mean() < 100  # reasonable mean

    def test_load_sample_with_explicit_path(self, tmp_path, sample_prices):
        """load_sample should accept an explicit path."""
        path = tmp_path / "custom_sample.parquet"
        sample_prices.to_frame().to_parquet(path)
        market_data = load_sample(sample_path=path)
        assert isinstance(market_data, MarketData)
        assert len(market_data.prices) == 168

    def test_load_sample_raises_on_missing_path(self, tmp_path):
        """load_sample should raise FileNotFoundError for missing path."""
        path = tmp_path / "nonexistent.parquet"
        with pytest.raises(FileNotFoundError, match="Sample data not found"):
            load_sample(sample_path=path)
