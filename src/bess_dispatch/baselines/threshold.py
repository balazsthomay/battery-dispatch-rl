"""Threshold-based baseline -- buy low, sell high using price percentiles."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ThresholdPolicy:
    """Percentile-based charge/discharge policy.

    Charges when price < low_percentile, discharges when price > high_percentile.
    Idles otherwise. Power levels are configurable to balance revenue against
    degradation costs.
    """

    def __init__(
        self,
        low_pct: float = 25.0,
        high_pct: float = 75.0,
        charge_power: float = 1.0,
        discharge_power: float = 1.0,
    ) -> None:
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.charge_power = charge_power
        self.discharge_power = discharge_power
        self.low_threshold: float | None = None
        self.high_threshold: float | None = None
        self.price_mean: float = 0.0
        self.price_std: float = 1.0

    def fit(self, prices: pd.Series | np.ndarray) -> ThresholdPolicy:
        """Compute thresholds from training prices.

        Also stores mean/std for denormalizing observations.

        Args:
            prices: Historical price series.

        Returns:
            self for method chaining.
        """
        prices_arr = np.asarray(prices, dtype=float)
        self.low_threshold = float(np.percentile(prices_arr, self.low_pct))
        self.high_threshold = float(np.percentile(prices_arr, self.high_pct))
        self.price_mean = float(np.mean(prices_arr))
        self.price_std = float(np.std(prices_arr)) or 1.0
        return self

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """Predict action from observation.

        obs[1] is the normalized current price: actual = obs[1] * price_std + price_mean

        Args:
            obs: Observation array, shape (32,) or (batch, 32).
            deterministic: Ignored (always deterministic).

        Returns:
            (action, None) matching SB3 predict() interface.

        Raises:
            RuntimeError: If fit() has not been called yet.
        """
        if self.low_threshold is None:
            raise RuntimeError("Must call fit() before predict()")

        single = obs.ndim == 1
        if single:
            obs = obs.reshape(1, -1)

        actions = np.zeros((obs.shape[0], 1), dtype=np.float32)
        for i in range(obs.shape[0]):
            # Denormalize price from observation
            actual_price = obs[i, 1] * self.price_std + self.price_mean
            if actual_price <= self.low_threshold:
                actions[i, 0] = -self.charge_power  # charge
            elif actual_price >= self.high_threshold:
                actions[i, 0] = self.discharge_power  # discharge
            # else idle (0.0)

        if single:
            return actions[0], None
        return actions, None

    @staticmethod
    def tune(
        prices: pd.Series | np.ndarray,
        market_data,  # MarketData for env creation
        battery_config=None,
        low_range: tuple[float, ...] = (10, 15, 20, 25, 30),
        high_range: tuple[float, ...] = (70, 75, 80, 85, 90),
        power_range: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0),
    ) -> ThresholdPolicy:
        """Grid search best percentile thresholds and power levels.

        Creates a BESSEnv and evaluates each (low_pct, high_pct, power)
        combination over a single deterministic episode.

        Args:
            prices: Price series for fitting thresholds.
            market_data: MarketData for creating the evaluation environment.
            battery_config: Optional battery config override.
            low_range: Candidate low percentiles.
            high_range: Candidate high percentiles.
            power_range: Candidate charge/discharge power levels in [0, 1].

        Returns:
            Best-performing ThresholdPolicy.
        """
        from bess_dispatch.config import BatteryConfig
        from bess_dispatch.env.bess_env import BESSEnv

        config = battery_config or BatteryConfig()
        best_reward = -np.inf
        best_policy = None

        for low_pct in low_range:
            for high_pct in high_range:
                if low_pct >= high_pct:
                    continue
                for power in power_range:
                    policy = ThresholdPolicy(
                        low_pct=low_pct,
                        high_pct=high_pct,
                        charge_power=power,
                        discharge_power=power,
                    )
                    policy.fit(prices)

                    # Run single evaluation episode
                    env = BESSEnv(market_data, config, random_start=False, seed=42)
                    obs, _ = env.reset()
                    total_reward = 0.0
                    done = False
                    while not done:
                        action, _ = policy.predict(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        total_reward += reward
                        done = terminated or truncated

                    if total_reward > best_reward:
                        best_reward = total_reward
                        best_policy = policy

        return best_policy
