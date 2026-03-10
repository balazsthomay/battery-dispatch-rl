"""Configuration management for BESS Dispatch."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConfig:
    """Battery energy storage system parameters."""

    capacity_mwh: float = 1.0
    max_power_mw: float = 1.0
    efficiency: float = 0.92  # per direction, ~84.6% round-trip
    initial_soc: float = 0.5  # fraction
    min_soc: float = 0.0
    max_soc: float = 1.0
    degradation_k: float = 2.0  # DoD exponent
    degradation_cost: float = 50.0  # EUR/unit degradation
    calendar_aging_per_hour: float = 1e-6


@dataclass(frozen=True)
class MarketConfig:
    """Electricity market parameters."""

    zone: str = "DE_LU"
    currency: str = "EUR"


@dataclass(frozen=True)
class TrainingConfig:
    """RL training hyperparameters."""

    algorithm: str = "DQN"
    total_timesteps: int = 100_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    exploration_fraction: float = 0.2
    episode_length: int = 168  # 1 week of hourly steps
    n_envs: int = 4
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    """Data loading and caching parameters."""

    data_dir: str = "data"
    cache_format: str = "parquet"
    max_gap_hours: int = 3  # forward-fill up to this
    max_missing_fraction: float = 0.1  # raise if more than 10% missing
