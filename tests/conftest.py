"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices():
    """1 week of synthetic hourly prices."""
    rng = np.random.default_rng(42)
    hours = pd.date_range("2023-01-02", periods=168, freq="h", tz="Europe/Berlin")
    base = 50 + 30 * np.sin(2 * np.pi * np.arange(168) / 24 - np.pi / 2)
    noise = rng.normal(0, 10, 168)
    prices = pd.Series(base + noise, index=hours, name="price_eur_mwh")
    return prices


@pytest.fixture
def battery_config():
    from bess_dispatch.config import BatteryConfig

    return BatteryConfig()


@pytest.fixture
def data_config(tmp_path):
    from bess_dispatch.config import DataConfig

    return DataConfig(data_dir=str(tmp_path))


@pytest.fixture
def long_prices():
    """4 weeks of synthetic hourly prices (672 hours) — enough for env episodes."""
    rng = np.random.default_rng(42)
    n = 672
    hours = pd.date_range("2023-01-02", periods=n, freq="h", tz="Europe/Berlin")
    base = 50 + 30 * np.sin(2 * np.pi * np.arange(n) / 24 - np.pi / 2)
    noise = rng.normal(0, 10, n)
    prices = pd.Series(base + noise, index=hours, name="price_eur_mwh")
    return prices


@pytest.fixture
def market_data(long_prices):
    from bess_dispatch.data.loader import MarketData

    return MarketData(prices=long_prices, zone="DE_LU", year=2023)


@pytest.fixture
def bess_env(market_data, battery_config):
    from bess_dispatch.env.bess_env import BESSEnv

    env = BESSEnv(
        market_data=market_data,
        battery_config=battery_config,
        random_start=False,
        seed=42,
    )
    return env
