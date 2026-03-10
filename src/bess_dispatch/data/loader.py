"""Market data loading with caching and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bess_dispatch.config import DataConfig, MarketConfig
from bess_dispatch.data.cache import cache_exists, cache_path, load_from_cache, save_to_cache
from bess_dispatch.data.client import EntsoeClient


@dataclass
class MarketData:
    """Container for market data used by the environment."""

    prices: pd.Series  # hourly day-ahead prices (EUR/MWh)
    wind_solar: pd.DataFrame | None = None  # optional wind+solar forecast
    zone: str = "DE_LU"
    year: int = 2023


def validate_prices(
    prices: pd.Series,
    max_gap_hours: int = 3,
    max_missing_fraction: float = 0.1,
) -> pd.Series:
    """Validate and clean price series. Raises ValueError on too many gaps."""
    if prices.empty:
        raise ValueError("Price series is empty")

    missing_frac = prices.isna().sum() / len(prices)
    if missing_frac > max_missing_fraction:
        raise ValueError(
            f"Too many missing values: {missing_frac:.1%} > {max_missing_fraction:.0%}"
        )

    # Forward-fill gaps up to max_gap_hours
    prices = prices.ffill(limit=max_gap_hours)

    # If still NaN, backward-fill remaining small gaps
    prices = prices.bfill(limit=max_gap_hours)

    if prices.isna().any():
        raise ValueError(
            f"Unfillable gaps remain after filling up to {max_gap_hours}h"
        )

    return prices


class MarketDataLoader:
    """Load market data with cache-first strategy."""

    def __init__(
        self,
        data_config: DataConfig | None = None,
        market_config: MarketConfig | None = None,
        client: EntsoeClient | None = None,
    ) -> None:
        self.data_config = data_config or DataConfig()
        self.market_config = market_config or MarketConfig()
        self._client = client  # lazy init

    @property
    def client(self) -> EntsoeClient:
        if self._client is None:
            self._client = EntsoeClient()
        return self._client

    def load_prices(self, zone: str | None = None, year: int = 2023) -> pd.Series:
        """Load prices: cache first, then API."""
        zone = zone or self.market_config.zone
        path = cache_path(self.data_config.data_dir, zone, "day_ahead_prices", year)

        if cache_exists(path):
            df = load_from_cache(path)
            prices = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
            prices.name = "price_eur_mwh"
            return validate_prices(
                prices,
                self.data_config.max_gap_hours,
                self.data_config.max_missing_fraction,
            )

        # Fetch from API
        start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin")
        prices = self.client.fetch_day_ahead_prices(zone, start, end)

        # Cache it
        save_to_cache(prices.to_frame(), path)

        return validate_prices(
            prices,
            self.data_config.max_gap_hours,
            self.data_config.max_missing_fraction,
        )

    def load_market_data(
        self, zone: str | None = None, year: int = 2023
    ) -> MarketData:
        """Load full market data package."""
        zone = zone or self.market_config.zone
        prices = self.load_prices(zone, year)
        return MarketData(prices=prices, zone=zone, year=year)


def load_sample(sample_path: Path | None = None) -> MarketData:
    """Load the committed sample dataset for testing/demo.

    Args:
        sample_path: Optional explicit path to sample parquet file.
            If None, searches relative to this file's location and CWD.
    """
    if sample_path is not None:
        path = sample_path
    else:
        # Try relative to source file (works for installed packages)
        path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "sample"
            / "de_lu_sample.parquet"
        )
        if not path.exists():
            # Try relative to CWD (works when running from project root)
            path = Path.cwd() / "data" / "sample" / "de_lu_sample.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Sample data not found at {path}")

    df = pd.read_parquet(path)
    prices = df.iloc[:, 0]
    prices.name = "price_eur_mwh"
    return MarketData(prices=prices, zone="DE_LU", year=2023)
