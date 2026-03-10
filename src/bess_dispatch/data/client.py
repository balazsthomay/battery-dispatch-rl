"""ENTSO-E API client wrapper."""

from __future__ import annotations

import os

import pandas as pd
from entsoe import EntsoePandasClient

# Bidding zone mapping
BIDDING_ZONES: dict[str, str] = {
    "DE_LU": "10Y1001A1001A82H",
    "ES": "10YES-REE------0",
    "NL": "10YNL----------L",
    "PL": "10YPL-AREA-----S",
}


class EntsoeClient:
    """Thin wrapper around entsoe-py client."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ENTSOE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ENTSOE_API_KEY not set. Pass api_key or set environment variable."
            )
        self.client = EntsoePandasClient(api_key=self.api_key)

    def fetch_day_ahead_prices(
        self, zone: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Fetch day-ahead prices, resample to hourly if needed."""
        country_code = BIDDING_ZONES[zone]
        prices = self.client.query_day_ahead_prices(
            country_code, start=start, end=end
        )
        # Resample to hourly (some zones have 15-min resolution)
        if (
            hasattr(prices.index, "freq")
            and prices.index.freq
            and prices.index.freq.n < 60
        ):
            prices = prices.resample("h").mean()
        elif len(prices) > 0:
            # Check actual interval
            intervals = prices.index.to_series().diff().dropna()
            if len(intervals) > 0 and intervals.median() < pd.Timedelta(hours=1):
                prices = prices.resample("h").mean()
        prices.name = "price_eur_mwh"
        return prices

    def fetch_wind_solar_forecast(
        self, zone: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Fetch wind+solar generation forecast."""
        country_code = BIDDING_ZONES[zone]
        gen = self.client.query_wind_and_solar_forecast(
            country_code, start=start, end=end
        )
        if isinstance(gen, pd.DataFrame):
            gen = gen.resample("h").mean()
        return gen
