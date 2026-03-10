"""Tests for config module."""

import dataclasses

import pytest

from bess_dispatch.config import (
    BatteryConfig,
    DataConfig,
    MarketConfig,
    TrainingConfig,
)


class TestBatteryConfig:
    """Tests for BatteryConfig dataclass."""

    def test_default_values(self):
        cfg = BatteryConfig()
        assert cfg.capacity_mwh == 1.0
        assert cfg.max_power_mw == 1.0
        assert cfg.efficiency == 0.92
        assert cfg.initial_soc == 0.5
        assert cfg.min_soc == 0.0
        assert cfg.max_soc == 1.0
        assert cfg.degradation_k == 2.0
        assert cfg.degradation_cost == 50.0
        assert cfg.calendar_aging_per_hour == 1e-6

    def test_frozen(self):
        cfg = BatteryConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.capacity_mwh = 2.0  # type: ignore[misc]

    def test_custom_values(self):
        cfg = BatteryConfig(capacity_mwh=2.0, efficiency=0.95)
        assert cfg.capacity_mwh == 2.0
        assert cfg.efficiency == 0.95
        # Other defaults remain
        assert cfg.max_power_mw == 1.0


class TestMarketConfig:
    """Tests for MarketConfig dataclass."""

    def test_default_values(self):
        cfg = MarketConfig()
        assert cfg.zone == "DE_LU"
        assert cfg.currency == "EUR"

    def test_frozen(self):
        cfg = MarketConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.zone = "ES"  # type: ignore[misc]

    def test_custom_values(self):
        cfg = MarketConfig(zone="NL", currency="USD")
        assert cfg.zone == "NL"
        assert cfg.currency == "USD"


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.algorithm == "DQN"
        assert cfg.total_timesteps == 100_000
        assert cfg.gamma == 0.99
        assert cfg.learning_rate == 3e-4
        assert cfg.exploration_fraction == 0.2
        assert cfg.episode_length == 168
        assert cfg.n_envs == 4
        assert cfg.seed == 42

    def test_frozen(self):
        cfg = TrainingConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.seed = 99  # type: ignore[misc]

    def test_custom_values(self):
        cfg = TrainingConfig(algorithm="PPO", total_timesteps=50_000)
        assert cfg.algorithm == "PPO"
        assert cfg.total_timesteps == 50_000


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        cfg = DataConfig()
        assert cfg.data_dir == "data"
        assert cfg.cache_format == "parquet"
        assert cfg.max_gap_hours == 3
        assert cfg.max_missing_fraction == 0.1

    def test_frozen(self):
        cfg = DataConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.data_dir = "/tmp"  # type: ignore[misc]

    def test_custom_values(self):
        cfg = DataConfig(data_dir="/custom", max_gap_hours=6)
        assert cfg.data_dir == "/custom"
        assert cfg.max_gap_hours == 6
