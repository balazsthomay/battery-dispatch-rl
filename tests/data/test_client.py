"""Tests for data client module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bess_dispatch.data.client import BIDDING_ZONES, EntsoeClient


class TestBiddingZones:
    """Tests for BIDDING_ZONES mapping."""

    def test_has_four_zones(self):
        assert len(BIDDING_ZONES) == 4

    def test_contains_expected_zones(self):
        expected = {"DE_LU", "ES", "NL", "PL"}
        assert set(BIDDING_ZONES.keys()) == expected

    def test_values_are_entsoe_area_codes(self):
        for zone, code in BIDDING_ZONES.items():
            assert code.startswith("10Y"), f"Zone {zone} code {code} does not start with 10Y"


class TestEntsoeClient:
    """Tests for EntsoeClient wrapper."""

    def test_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            # Ensure ENTSOE_API_KEY is not in env
            with patch.dict("os.environ", {"ENTSOE_API_KEY": ""}, clear=False):
                with pytest.raises(ValueError, match="ENTSOE_API_KEY not set"):
                    EntsoeClient(api_key="")

    def test_raises_when_env_var_empty(self):
        with patch.dict("os.environ", {"ENTSOE_API_KEY": ""}):
            with pytest.raises(ValueError, match="ENTSOE_API_KEY not set"):
                EntsoeClient()

    @patch("bess_dispatch.data.client.EntsoePandasClient")
    def test_accepts_explicit_api_key(self, mock_client_cls):
        client = EntsoeClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"
        mock_client_cls.assert_called_once_with(api_key="test-key-123")

    @patch("bess_dispatch.data.client.EntsoePandasClient")
    def test_fetch_day_ahead_prices(self, mock_client_cls):
        """Test fetch_day_ahead_prices with mocked entsoe client."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        hours = pd.date_range("2023-01-01", periods=24, freq="h", tz="Europe/Berlin")
        mock_prices = pd.Series(range(24), index=hours, dtype=float)
        mock_client.query_day_ahead_prices.return_value = mock_prices

        # Execute
        client = EntsoeClient(api_key="test-key")
        start = pd.Timestamp("2023-01-01", tz="Europe/Berlin")
        end = pd.Timestamp("2023-01-02", tz="Europe/Berlin")
        result = client.fetch_day_ahead_prices("DE_LU", start, end)

        # Verify
        mock_client.query_day_ahead_prices.assert_called_once_with(
            BIDDING_ZONES["DE_LU"], start=start, end=end
        )
        assert result.name == "price_eur_mwh"
        assert len(result) == 24

    @patch("bess_dispatch.data.client.EntsoePandasClient")
    def test_fetch_day_ahead_prices_resamples_subhourly(self, mock_client_cls):
        """Test that sub-hourly data gets resampled to hourly."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # 15-min resolution data (96 points for 24h)
        quarter_hours = pd.date_range(
            "2023-01-01", periods=96, freq="15min", tz="Europe/Berlin"
        )
        mock_prices = pd.Series(range(96), index=quarter_hours, dtype=float)
        mock_client.query_day_ahead_prices.return_value = mock_prices

        client = EntsoeClient(api_key="test-key")
        start = pd.Timestamp("2023-01-01", tz="Europe/Berlin")
        end = pd.Timestamp("2023-01-02", tz="Europe/Berlin")
        result = client.fetch_day_ahead_prices("DE_LU", start, end)

        # Should be resampled to hourly
        assert len(result) == 24
        assert result.name == "price_eur_mwh"

    @patch("bess_dispatch.data.client.EntsoePandasClient")
    def test_fetch_wind_solar_forecast(self, mock_client_cls):
        """Test fetch_wind_solar_forecast with mocked client."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        hours = pd.date_range("2023-01-01", periods=24, freq="h", tz="Europe/Berlin")
        mock_gen = pd.DataFrame(
            {"wind": range(24), "solar": range(24)},
            index=hours,
        )
        mock_client.query_wind_and_solar_forecast.return_value = mock_gen

        client = EntsoeClient(api_key="test-key")
        start = pd.Timestamp("2023-01-01", tz="Europe/Berlin")
        end = pd.Timestamp("2023-01-02", tz="Europe/Berlin")
        result = client.fetch_wind_solar_forecast("DE_LU", start, end)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24
        mock_client.query_wind_and_solar_forecast.assert_called_once()

    @patch("bess_dispatch.data.client.EntsoePandasClient")
    def test_fetch_wind_solar_forecast_resamples(self, mock_client_cls):
        """Test that sub-hourly wind/solar data gets resampled."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        quarter_hours = pd.date_range(
            "2023-01-01", periods=96, freq="15min", tz="Europe/Berlin"
        )
        mock_gen = pd.DataFrame(
            {"wind": range(96), "solar": range(96)},
            index=quarter_hours,
        )
        mock_client.query_wind_and_solar_forecast.return_value = mock_gen

        client = EntsoeClient(api_key="test-key")
        start = pd.Timestamp("2023-01-01", tz="Europe/Berlin")
        end = pd.Timestamp("2023-01-02", tz="Europe/Berlin")
        result = client.fetch_wind_solar_forecast("DE_LU", start, end)

        assert len(result) == 24
