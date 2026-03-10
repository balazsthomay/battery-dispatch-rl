"""Tests for do-nothing baseline policy."""

from __future__ import annotations

import numpy as np
import pytest

from bess_dispatch.baselines.do_nothing import DoNothingPolicy


class TestDoNothingPolicy:
    """Tests for the DoNothingPolicy."""

    def test_predict_single_returns_zero(self):
        """Single observation returns action=0."""
        policy = DoNothingPolicy()
        obs = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        action, state = policy.predict(obs)
        assert action.shape == (1,)
        assert action[0] == pytest.approx(0.0)
        assert state is None

    def test_predict_batched_returns_zeros(self):
        """Batched observations return zeros of correct shape."""
        policy = DoNothingPolicy()
        batch_size = 8
        obs = np.random.default_rng(42).standard_normal((batch_size, 32)).astype(
            np.float32
        )
        actions, state = policy.predict(obs)
        assert actions.shape == (batch_size, 1)
        np.testing.assert_array_equal(actions, 0.0)
        assert state is None

    def test_predict_dtype_is_float32(self):
        """Action dtype should be float32."""
        policy = DoNothingPolicy()
        obs = np.zeros(32, dtype=np.float32)
        action, _ = policy.predict(obs)
        assert action.dtype == np.float32

    def test_through_env_zero_revenue(self, market_data, battery_config):
        """Running do-nothing through the env yields zero revenue, only calendar degradation."""
        from bess_dispatch.env.bess_env import BESSEnv

        env = BESSEnv(
            market_data=market_data,
            battery_config=battery_config,
            random_start=False,
            seed=42,
        )
        policy = DoNothingPolicy()
        obs, _ = env.reset()

        total_revenue = 0.0
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_revenue += info["revenue"]
            total_reward += reward
            done = terminated or truncated
            steps += 1

        # No revenue from idling
        assert total_revenue == pytest.approx(0.0)
        # SoC shouldn't change
        assert obs[0] == pytest.approx(battery_config.initial_soc, abs=1e-5)
        # Reward should be negative (only calendar degradation)
        assert total_reward < 0.0
        # Should have run for episode_length steps
        assert steps == 168
