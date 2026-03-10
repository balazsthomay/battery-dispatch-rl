"""Tests for threshold-based baseline policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess_dispatch.baselines.threshold import ThresholdPolicy


class TestThresholdPolicyFit:
    """Tests for ThresholdPolicy.fit()."""

    def test_fit_sets_thresholds(self, long_prices):
        """fit() should compute and store percentile thresholds."""
        policy = ThresholdPolicy(low_pct=25.0, high_pct=75.0)
        result = policy.fit(long_prices)

        assert result is policy  # returns self for chaining
        assert policy.low_threshold is not None
        assert policy.high_threshold is not None
        assert policy.low_threshold < policy.high_threshold

    def test_fit_stores_normalization_stats(self, long_prices):
        """fit() should store price mean and std for denormalization."""
        policy = ThresholdPolicy()
        policy.fit(long_prices)

        expected_mean = float(np.mean(long_prices.values))
        expected_std = float(np.std(long_prices.values))

        assert policy.price_mean == pytest.approx(expected_mean)
        assert policy.price_std == pytest.approx(expected_std)

    def test_fit_with_numpy_array(self):
        """fit() should accept numpy array in addition to pd.Series."""
        prices = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        policy = ThresholdPolicy(low_pct=25.0, high_pct=75.0)
        policy.fit(prices)

        assert policy.low_threshold == pytest.approx(np.percentile(prices, 25.0))
        assert policy.high_threshold == pytest.approx(np.percentile(prices, 75.0))


class TestThresholdPolicyPredict:
    """Tests for ThresholdPolicy.predict()."""

    def test_unfitted_predict_raises(self):
        """predict() before fit() should raise RuntimeError."""
        policy = ThresholdPolicy()
        obs = np.zeros(32, dtype=np.float32)
        with pytest.raises(RuntimeError, match="fit"):
            policy.predict(obs)

    def test_predict_charges_on_low_price(self):
        """Low price should produce charge action (negative)."""
        prices = np.linspace(10, 100, 100)
        policy = ThresholdPolicy(low_pct=25.0, high_pct=75.0)
        policy.fit(prices)

        # Create obs with very low normalized price
        obs = np.zeros(32, dtype=np.float32)
        # obs[1] = (actual_price - mean) / std
        # Set to a very low price (well below low_threshold)
        obs[1] = (10.0 - policy.price_mean) / policy.price_std

        action, state = policy.predict(obs)
        assert action[0] == pytest.approx(-1.0)  # default charge_power=1.0
        assert state is None

    def test_predict_charges_with_custom_power(self):
        """Custom charge_power should be respected."""
        prices = np.linspace(10, 100, 100)
        policy = ThresholdPolicy(low_pct=25.0, high_pct=75.0, charge_power=0.3)
        policy.fit(prices)

        obs = np.zeros(32, dtype=np.float32)
        obs[1] = (10.0 - policy.price_mean) / policy.price_std

        action, _ = policy.predict(obs)
        assert action[0] == pytest.approx(-0.3)

    def test_predict_discharges_on_high_price(self):
        """High price should produce discharge action (positive)."""
        prices = np.linspace(10, 100, 100)
        policy = ThresholdPolicy(low_pct=25.0, high_pct=75.0)
        policy.fit(prices)

        obs = np.zeros(32, dtype=np.float32)
        # Set to a very high price
        obs[1] = (100.0 - policy.price_mean) / policy.price_std

        action, _ = policy.predict(obs)
        assert action[0] == pytest.approx(1.0)

    def test_predict_idles_on_mid_price(self):
        """Mid-range price should produce idle action (0)."""
        prices = np.linspace(10, 100, 100)
        policy = ThresholdPolicy(low_pct=25.0, high_pct=75.0)
        policy.fit(prices)

        obs = np.zeros(32, dtype=np.float32)
        # Mid-range price (right between thresholds)
        mid_price = (policy.low_threshold + policy.high_threshold) / 2
        obs[1] = (mid_price - policy.price_mean) / policy.price_std

        action, _ = policy.predict(obs)
        assert action[0] == pytest.approx(0.0)

    def test_predict_batched(self):
        """Batched predict should return correct shape."""
        prices = np.linspace(10, 100, 100)
        policy = ThresholdPolicy(low_pct=25.0, high_pct=75.0)
        policy.fit(prices)

        batch = np.zeros((4, 32), dtype=np.float32)
        # Low price
        batch[0, 1] = (10.0 - policy.price_mean) / policy.price_std
        # High price
        batch[1, 1] = (100.0 - policy.price_mean) / policy.price_std
        # Mid price
        mid = (policy.low_threshold + policy.high_threshold) / 2
        batch[2, 1] = (mid - policy.price_mean) / policy.price_std
        batch[3, 1] = (mid - policy.price_mean) / policy.price_std

        actions, state = policy.predict(batch)
        assert actions.shape == (4, 1)
        assert actions[0, 0] == pytest.approx(-1.0)
        assert actions[1, 0] == pytest.approx(1.0)
        assert actions[2, 0] == pytest.approx(0.0)
        assert actions[3, 0] == pytest.approx(0.0)
        assert state is None


class TestThresholdPolicyTune:
    """Tests for ThresholdPolicy.tune()."""

    def test_tune_returns_threshold_policy(self, long_prices, market_data, battery_config):
        """tune() should return a fitted ThresholdPolicy."""
        policy = ThresholdPolicy.tune(
            long_prices,
            market_data,
            battery_config,
            low_range=(20, 30),
            high_range=(70, 80),
            power_range=(0.2, 0.5),
        )
        assert isinstance(policy, ThresholdPolicy)
        assert policy.low_threshold is not None
        assert policy.high_threshold is not None

    def test_tune_outperforms_do_nothing(self, long_prices, market_data, battery_config):
        """Tuned threshold policy should outperform do-nothing."""
        from bess_dispatch.baselines.do_nothing import DoNothingPolicy
        from bess_dispatch.env.bess_env import BESSEnv

        # Evaluate do-nothing
        env = BESSEnv(market_data, battery_config, random_start=False, seed=42)
        do_nothing = DoNothingPolicy()
        obs, _ = env.reset()
        dn_reward = 0.0
        done = False
        while not done:
            action, _ = do_nothing.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            dn_reward += reward
            done = terminated or truncated

        # Evaluate tuned threshold (includes power_range search)
        tuned = ThresholdPolicy.tune(
            long_prices,
            market_data,
            battery_config,
            low_range=(20, 30),
            high_range=(70, 80),
            power_range=(0.1, 0.2, 0.3, 0.5),
        )
        env = BESSEnv(market_data, battery_config, random_start=False, seed=42)
        obs, _ = env.reset()
        th_reward = 0.0
        done = False
        while not done:
            action, _ = tuned.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            th_reward += reward
            done = terminated or truncated

        assert th_reward >= dn_reward
