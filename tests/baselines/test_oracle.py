"""Tests for oracle (perfect foresight) baseline."""

from __future__ import annotations

import numpy as np
import pytest

from bess_dispatch.baselines.oracle import OraclePolicy, OracleResult, solve_oracle
from bess_dispatch.config import BatteryConfig


class TestSolveOracle:
    """Tests for the solve_oracle function."""

    def test_returns_optimal_status(self):
        """solve_oracle should return an optimal solution on a simple problem."""
        prices = np.array([20.0] * 12 + [80.0] * 12)  # low then high
        result = solve_oracle(prices)
        assert result.status in ("optimal", "optimal_inaccurate")

    def test_result_has_correct_fields(self):
        """OracleResult should contain all expected fields."""
        prices = np.array([30.0, 50.0, 70.0, 50.0, 30.0])
        result = solve_oracle(prices)

        assert isinstance(result, OracleResult)
        assert result.actions.shape == (5,)
        assert result.soc_trajectory.shape == (6,)  # T+1
        assert isinstance(result.revenue, float)
        assert isinstance(result.degradation, float)
        assert isinstance(result.net_reward, float)
        assert isinstance(result.status, str)

    def test_charges_low_discharges_high(self):
        """On step-function prices, oracle should charge low and discharge high."""
        # 12h at 20 EUR, then 12h at 80 EUR
        prices = np.array([20.0] * 12 + [80.0] * 12)
        config = BatteryConfig()
        result = solve_oracle(prices, config)

        assert result.status in ("optimal", "optimal_inaccurate")

        # During low prices (first half), net action should be negative (charging)
        low_actions = result.actions[:12]
        # During high prices (second half), net action should be positive (discharging)
        high_actions = result.actions[12:]

        # Overall, the policy should buy low and sell high
        assert np.sum(low_actions) < 0, "Should charge during low prices"
        assert np.sum(high_actions) > 0, "Should discharge during high prices"

    def test_soc_within_bounds(self):
        """SoC trajectory should stay within [min_soc, max_soc]."""
        rng = np.random.default_rng(42)
        prices = 50 + 30 * np.sin(2 * np.pi * np.arange(48) / 24) + rng.normal(0, 5, 48)
        config = BatteryConfig()
        result = solve_oracle(prices, config)

        assert result.status in ("optimal", "optimal_inaccurate")
        assert np.all(result.soc_trajectory >= config.min_soc - 1e-6)
        assert np.all(result.soc_trajectory <= config.max_soc + 1e-6)

    def test_positive_revenue_on_volatile_prices(self):
        """Oracle should earn positive net reward on volatile prices."""
        # Create strongly oscillating prices
        prices = np.array([10.0, 90.0] * 12)  # very high spread
        result = solve_oracle(prices)

        assert result.status in ("optimal", "optimal_inaccurate")
        assert result.net_reward > 0, "Oracle should profit on volatile prices"

    def test_custom_config(self):
        """Oracle should respect custom battery config."""
        prices = np.array([20.0] * 6 + [80.0] * 6)
        config = BatteryConfig(
            capacity_mwh=2.0,
            max_power_mw=0.5,
            efficiency=0.95,
            initial_soc=0.3,
        )
        result = solve_oracle(prices, config)

        assert result.status in ("optimal", "optimal_inaccurate")
        # Initial SoC should be respected
        assert result.soc_trajectory[0] == pytest.approx(0.3, abs=1e-6)


class TestOraclePolicy:
    """Tests for the OraclePolicy wrapper."""

    def test_predict_returns_actions(self):
        """OraclePolicy.predict() should return precomputed actions."""
        prices = np.array([20.0] * 6 + [80.0] * 6)
        policy = OraclePolicy(prices)

        obs = np.zeros(32, dtype=np.float32)
        for i in range(12):
            action, state = policy.predict(obs)
            assert action.shape == (1,)
            assert state is None

    def test_predict_past_end_returns_zero(self):
        """predict() past the action array should return 0."""
        prices = np.array([50.0, 60.0, 70.0])
        policy = OraclePolicy(prices)

        obs = np.zeros(32, dtype=np.float32)
        for _ in range(3):
            policy.predict(obs)

        # Past end
        action, _ = policy.predict(obs)
        assert action[0] == pytest.approx(0.0)

    def test_reset_restarts_index(self):
        """reset() should restart the action index."""
        prices = np.array([20.0] * 6 + [80.0] * 6)
        policy = OraclePolicy(prices)

        obs = np.zeros(32, dtype=np.float32)
        first_action, _ = policy.predict(obs)
        policy.reset()
        restarted_action, _ = policy.predict(obs)

        assert first_action[0] == pytest.approx(restarted_action[0])

    def test_through_env(self, market_data, battery_config):
        """OraclePolicy should run through env without errors."""
        from bess_dispatch.env.bess_env import BESSEnv

        env = BESSEnv(market_data, battery_config, random_start=False, seed=42)
        obs, _ = env.reset()

        # Get the actual prices the env will use
        start_idx = env._start_idx
        episode_prices = env.prices[start_idx : start_idx + env.episode_length]

        policy = OraclePolicy(episode_prices, battery_config)

        total_reward = 0.0
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Oracle should at worst break even or better on sinusoidal prices
        # (it has perfect foresight)
        assert total_reward > -10.0  # loose bound, just check it runs


class TestOracleRanking:
    """Test that oracle >= threshold >= do-nothing."""

    def test_oracle_ge_threshold_ge_do_nothing(self, market_data, battery_config):
        """Oracle reward >= threshold reward >= do-nothing reward."""
        from bess_dispatch.baselines.do_nothing import DoNothingPolicy
        from bess_dispatch.baselines.threshold import ThresholdPolicy
        from bess_dispatch.env.bess_env import BESSEnv

        def run_episode(policy, env, oracle=False):
            obs, _ = env.reset()
            if hasattr(policy, "reset"):
                policy.reset()
            total = 0.0
            done = False
            while not done:
                action, _ = policy.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            return total

        # Do-nothing
        env = BESSEnv(market_data, battery_config, random_start=False, seed=42)
        dn_reward = run_episode(DoNothingPolicy(), env)

        # Threshold (tuned with power search to handle degradation costs)
        threshold = ThresholdPolicy.tune(
            market_data.prices,
            market_data,
            battery_config,
            low_range=(20, 30),
            high_range=(70, 80),
            power_range=(0.1, 0.2, 0.3, 0.5),
        )
        env = BESSEnv(market_data, battery_config, random_start=False, seed=42)
        th_reward = run_episode(threshold, env)

        # Oracle (needs actual episode prices)
        env = BESSEnv(market_data, battery_config, random_start=False, seed=42)
        obs, _ = env.reset()
        start_idx = env._start_idx
        episode_prices = env.prices[start_idx : start_idx + env.episode_length]
        oracle = OraclePolicy(episode_prices, battery_config)
        env = BESSEnv(market_data, battery_config, random_start=False, seed=42)
        or_reward = run_episode(oracle, env)

        assert or_reward >= th_reward - 1e-6, (
            f"Oracle ({or_reward:.2f}) should >= threshold ({th_reward:.2f})"
        )
        assert th_reward >= dn_reward - 1e-6, (
            f"Threshold ({th_reward:.2f}) should >= do-nothing ({dn_reward:.2f})"
        )
