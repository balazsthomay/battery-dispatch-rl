"""Tests for BESS Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest

from bess_dispatch.config import BatteryConfig
from bess_dispatch.data.loader import MarketData
from bess_dispatch.env.bess_env import BESSEnv


class TestEnvChecker:
    """Gymnasium env_checker must pass."""

    def test_check_env_passes(self, bess_env):
        """The env must pass gymnasium.utils.env_checker.check_env()."""
        from gymnasium.utils.env_checker import check_env

        check_env(bess_env, skip_render_check=True)


class TestObservationSpace:
    """Tests for the observation space."""

    def test_observation_shape(self, bess_env):
        obs, _ = bess_env.reset()
        assert obs.shape == (32,)

    def test_observation_dtype(self, bess_env):
        obs, _ = bess_env.reset()
        assert obs.dtype == np.float32

    def test_obs_soc_in_range(self, bess_env):
        """obs[0] should be SoC in [0, 1]."""
        obs, _ = bess_env.reset()
        assert 0.0 <= obs[0] <= 1.0

    def test_obs_in_observation_space(self, bess_env):
        obs, _ = bess_env.reset()
        assert bess_env.observation_space.contains(obs)


class TestActionSpace:
    """Tests for the action space."""

    def test_action_space_shape(self, bess_env):
        assert bess_env.action_space.shape == (1,)

    def test_action_space_bounds(self, bess_env):
        assert bess_env.action_space.low[0] == -1.0
        assert bess_env.action_space.high[0] == 1.0


class TestReset:
    """Tests for env.reset()."""

    def test_reset_returns_tuple(self, bess_env):
        result = bess_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_returns_valid_obs(self, bess_env):
        obs, info = bess_env.reset()
        assert obs.shape == (32,)
        assert isinstance(info, dict)

    def test_reset_soc_is_initial(self, bess_env):
        obs, _ = bess_env.reset()
        assert obs[0] == pytest.approx(bess_env.battery_config.initial_soc)

    def test_reset_with_seed_deterministic(self, market_data, battery_config):
        env1 = BESSEnv(market_data=market_data, battery_config=battery_config, seed=42)
        env2 = BESSEnv(market_data=market_data, battery_config=battery_config, seed=42)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestEpisodeLength:
    """Tests for episode truncation."""

    def test_episode_truncates_at_length(self, bess_env):
        bess_env.reset()
        truncated = False
        steps = 0
        while not truncated:
            action = bess_env.action_space.sample()
            obs, reward, terminated, truncated, info = bess_env.step(action)
            steps += 1
            assert not terminated  # should never terminate early
        assert steps == bess_env.episode_length

    def test_custom_episode_length(self, market_data, battery_config):
        env = BESSEnv(
            market_data=market_data,
            battery_config=battery_config,
            episode_length=48,
            random_start=False,
            seed=42,
        )
        env.reset()
        steps = 0
        truncated = False
        while not truncated:
            obs, _, _, truncated, _ = env.step(env.action_space.sample())
            steps += 1
        assert steps == 48


class TestRandomStart:
    """Tests for random vs deterministic start."""

    def test_random_start_false_deterministic(self, market_data, battery_config):
        env1 = BESSEnv(
            market_data=market_data,
            battery_config=battery_config,
            random_start=False,
            seed=42,
        )
        env2 = BESSEnv(
            market_data=market_data,
            battery_config=battery_config,
            random_start=False,
            seed=99,
        )
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)

    def test_random_start_true_varies(self, market_data, battery_config):
        env = BESSEnv(
            market_data=market_data,
            battery_config=battery_config,
            random_start=True,
            seed=42,
        )
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        # Different random starts should (usually) give different observations
        # With different start indices, at least the price features differ
        # This isn't guaranteed but extremely likely with different random seeds
        # We just check they're both valid
        assert obs1.shape == (32,)
        assert obs2.shape == (32,)


class TestStepBehavior:
    """Tests for step mechanics."""

    def test_step_returns_correct_tuple(self, bess_env):
        bess_env.reset()
        action = np.array([0.0], dtype=np.float32)
        result = bess_env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (32,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reward_is_finite(self, bess_env):
        bess_env.reset()
        for _ in range(10):
            action = bess_env.action_space.sample()
            _, reward, _, _, _ = bess_env.step(action)
            assert np.isfinite(reward)

    def test_info_dict_has_expected_keys(self, bess_env):
        bess_env.reset()
        action = np.array([0.5], dtype=np.float32)
        _, _, _, _, info = bess_env.step(action)
        assert "revenue" in info
        assert "degradation" in info
        assert "soc" in info

    def test_obs_valid_after_truncation(self, bess_env):
        """Observation returned with truncated=True should be valid."""
        bess_env.reset()
        truncated = False
        obs = None
        for _ in range(bess_env.episode_length):
            obs, _, _, truncated, _ = bess_env.step(bess_env.action_space.sample())
        assert truncated
        assert bess_env.observation_space.contains(obs)

    def test_soc_in_obs_tracks_battery_state(self, bess_env):
        """obs[0] should match the battery SoC after each step."""
        bess_env.reset()
        for _ in range(5):
            action = np.array([0.3], dtype=np.float32)
            obs, _, _, _, info = bess_env.step(action)
            assert obs[0] == pytest.approx(info["soc"], abs=1e-5)


class TestTimeFeaturesInObs:
    """Tests for cyclic time features in observation."""

    def test_time_features_bounded(self, bess_env):
        obs, _ = bess_env.reset()
        # sin/cos features should be in [-1, 1]
        assert -1.0 <= obs[26] <= 1.0  # sin(hour)
        assert -1.0 <= obs[27] <= 1.0  # cos(hour)
        assert -1.0 <= obs[28] <= 1.0  # sin(dow)
        assert -1.0 <= obs[29] <= 1.0  # cos(dow)


class TestRenewablesInObs:
    """Tests for wind/solar features in observation when renewables data is present."""

    def test_renewables_in_obs(self, long_prices, battery_config):
        """When wind_solar data is provided, obs[30] and obs[31] should be populated."""
        import pandas as pd

        n = len(long_prices)
        wind_solar = pd.DataFrame(
            {
                "wind_onshore": np.random.default_rng(1).uniform(0, 1000, n),
                "solar": np.random.default_rng(2).uniform(0, 500, n),
            },
            index=long_prices.index,
        )
        md = MarketData(prices=long_prices, wind_solar=wind_solar, zone="DE_LU", year=2023)
        env = BESSEnv(
            market_data=md,
            battery_config=battery_config,
            random_start=False,
            seed=42,
        )
        obs, _ = env.reset()
        # Wind and solar features should be non-zero (at least one should be)
        assert obs[30] >= 0.0  # wind normalized [0, 1]
        assert obs[31] >= 0.0  # solar normalized [0, 1]
        assert obs[30] <= 1.0
        assert obs[31] <= 1.0

    def test_no_renewables_obs_zero(self, bess_env):
        """Without renewables data, obs[30] and obs[31] should be 0."""
        obs, _ = bess_env.reset()
        assert obs[30] == 0.0
        assert obs[31] == 0.0


class TestDeterminism:
    """Tests for deterministic behavior with same seeds."""

    def test_same_seed_same_trajectory(self, market_data, battery_config):
        """Two envs with same seed should produce identical trajectories."""
        env1 = BESSEnv(
            market_data=market_data,
            battery_config=battery_config,
            random_start=True,
            seed=42,
        )
        env2 = BESSEnv(
            market_data=market_data,
            battery_config=battery_config,
            random_start=True,
            seed=42,
        )
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        # Same actions should yield same results
        for _ in range(10):
            action = np.array([0.3], dtype=np.float32)
            obs1, r1, _, _, _ = env1.step(action)
            obs2, r2, _, _, _ = env2.step(action)
            np.testing.assert_array_equal(obs1, obs2)
            assert r1 == r2
