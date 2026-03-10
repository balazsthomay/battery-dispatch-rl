"""Gymnasium BESS dispatch environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from bess_dispatch.config import BatteryConfig
from bess_dispatch.data.loader import MarketData
from bess_dispatch.env.battery import BatteryState, apply_action


class BESSEnv(gym.Env):
    """Battery Energy Storage System dispatch environment.

    Observation space (32 dims):
      [0]     SoC (0-1)
      [1]     Current price (normalized)
      [2:26]  24h price history (normalized) — last 24 prices
      [26]    sin(hour / 24 * 2pi)
      [27]    cos(hour / 24 * 2pi)
      [28]    sin(day_of_week / 7 * 2pi)
      [29]    cos(day_of_week / 7 * 2pi)
      [30]    Wind forecast (normalized, 0 if unavailable)
      [31]    Solar forecast (normalized, 0 if unavailable)

    Action space: Box(-1, 1, shape=(1,)) — continuous power level
      -1 = full charge, +1 = full discharge

    Episodes: configurable length (default 168 steps = 1 week)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: MarketData,
        battery_config: BatteryConfig | None = None,
        episode_length: int = 168,
        random_start: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.market_data = market_data
        self.battery_config = battery_config or BatteryConfig()
        self.episode_length = episode_length
        self.random_start = random_start

        # Price data as numpy array
        self.prices = market_data.prices.values.astype(np.float32)
        self.timestamps = market_data.prices.index

        # Price normalization stats
        self.price_mean = float(np.mean(self.prices))
        self.price_std = float(np.std(self.prices)) or 1.0

        # Wind/solar data (if available)
        self.has_renewables = market_data.wind_solar is not None
        if self.has_renewables:
            ws = market_data.wind_solar
            wind_sum = ws.filter(like="wind").sum(axis=1)
            solar_sum = ws.filter(like="solar").sum(axis=1)
            wind_max = max(float(wind_sum.max()), 1.0)
            solar_max = max(float(solar_sum.max()), 1.0)
            self.wind = (wind_sum.values / wind_max).astype(np.float32)
            self.solar = (solar_sum.values / solar_max).astype(np.float32)

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State
        self._step_idx = 0
        self._start_idx = 0
        self._state = BatteryState(soc=self.battery_config.initial_soc)
        self._rng = np.random.default_rng(seed)

    def _get_obs(self) -> np.ndarray:
        """Build 32-dim observation vector."""
        idx = self._start_idx + self._step_idx
        # Clamp idx to valid range for safety at episode boundary
        idx = min(idx, len(self.prices) - 1)

        obs = np.zeros(32, dtype=np.float32)

        # SoC
        obs[0] = self._state.soc

        # Current price (normalized)
        obs[1] = (self.prices[idx] - self.price_mean) / self.price_std

        # 24h price history (normalized)
        for i in range(24):
            hist_idx = idx - 24 + i
            if 0 <= hist_idx < len(self.prices):
                obs[2 + i] = (self.prices[hist_idx] - self.price_mean) / self.price_std

        # Cyclic time features
        ts = self.timestamps[idx]
        if hasattr(ts, "hour"):
            hour = ts.hour
        else:
            hour = pd.Timestamp(ts).hour

        if hasattr(ts, "dayofweek"):
            dow = ts.dayofweek
        else:
            dow = pd.Timestamp(ts).dayofweek

        obs[26] = np.sin(2 * np.pi * hour / 24)
        obs[27] = np.cos(2 * np.pi * hour / 24)
        obs[28] = np.sin(2 * np.pi * dow / 7)
        obs[29] = np.cos(2 * np.pi * dow / 7)

        # Wind/solar (if available)
        if self.has_renewables:
            if idx < len(self.wind):
                obs[30] = self.wind[idx]
            if idx < len(self.solar):
                obs[31] = self.solar[idx]

        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_idx = 0
        self._state = BatteryState(soc=self.battery_config.initial_soc)

        # Need at least 24 steps before start (for history) + episode_length after
        min_start = 24
        max_start = len(self.prices) - self.episode_length

        if self.random_start and max_start > min_start:
            self._start_idx = int(
                self._rng.integers(min_start, max_start)
            )
        else:
            self._start_idx = min_start

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step."""
        action_val = float(np.clip(action[0], -1.0, 1.0))

        idx = self._start_idx + self._step_idx
        price = float(self.prices[idx])

        new_state, reward, info = apply_action(
            self._state, action_val, price, self.battery_config
        )
        self._state = new_state
        self._step_idx += 1

        terminated = False
        truncated = self._step_idx >= self.episode_length

        return self._get_obs(), float(reward), terminated, truncated, info
