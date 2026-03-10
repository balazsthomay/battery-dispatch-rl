"""Tests for environment wrappers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from bess_dispatch.config import BatteryConfig, TrainingConfig
from bess_dispatch.data.loader import MarketData
from bess_dispatch.env.wrappers import (
    DISCRETE_ACTIONS,
    DiscreteActionWrapper,
    make_env,
    make_vec_env,
)


class TestDiscreteActionWrapper:
    """Tests for DiscreteActionWrapper."""

    def test_action_space_is_discrete(self, bess_env):
        wrapped = DiscreteActionWrapper(bess_env)
        assert isinstance(wrapped.action_space, spaces.Discrete)
        assert wrapped.action_space.n == 5

    def test_action_mapping_full_charge(self, bess_env):
        wrapped = DiscreteActionWrapper(bess_env)
        result = wrapped.action(0)
        np.testing.assert_array_almost_equal(result, np.array([-1.0]))

    def test_action_mapping_idle(self, bess_env):
        wrapped = DiscreteActionWrapper(bess_env)
        result = wrapped.action(2)
        np.testing.assert_array_almost_equal(result, np.array([0.0]))

    def test_action_mapping_full_discharge(self, bess_env):
        wrapped = DiscreteActionWrapper(bess_env)
        result = wrapped.action(4)
        np.testing.assert_array_almost_equal(result, np.array([1.0]))

    def test_all_discrete_actions(self, bess_env):
        wrapped = DiscreteActionWrapper(bess_env)
        expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for i, exp in enumerate(expected):
            result = wrapped.action(i)
            assert result[0] == pytest.approx(exp)

    def test_step_through_wrapper(self, bess_env):
        wrapped = DiscreteActionWrapper(bess_env)
        obs, _ = wrapped.reset()
        assert obs.shape == (32,)
        obs, reward, terminated, truncated, info = wrapped.step(2)  # idle
        assert obs.shape == (32,)
        assert isinstance(reward, float)


class TestMakeEnv:
    """Tests for make_env factory function."""

    def test_returns_bess_env(self, market_data):
        from bess_dispatch.env.bess_env import BESSEnv

        env = make_env(market_data)
        assert isinstance(env, BESSEnv)

    def test_returns_wrapped_discrete(self, market_data):
        env = make_env(market_data, discrete=True)
        assert isinstance(env, DiscreteActionWrapper)

    def test_continuous_action_space(self, market_data):
        env = make_env(market_data, discrete=False)
        assert isinstance(env.action_space, spaces.Box)

    def test_discrete_action_space(self, market_data):
        env = make_env(market_data, discrete=True)
        assert isinstance(env.action_space, spaces.Discrete)

    def test_custom_config(self, market_data):
        config = BatteryConfig(capacity_mwh=2.0)
        env = make_env(market_data, battery_config=config)
        assert env.battery_config.capacity_mwh == 2.0

    def test_env_steps_work(self, market_data):
        env = make_env(market_data, seed=42)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.0], dtype=np.float32)
        )
        assert obs.shape == (32,)


class TestMakeVecEnv:
    """Tests for make_vec_env factory function."""

    def test_returns_vec_env(self, market_data):
        from stable_baselines3.common.vec_env import VecNormalize

        venv = make_vec_env(market_data)
        assert isinstance(venv, VecNormalize)

    def test_without_normalize(self, market_data):
        from stable_baselines3.common.vec_env import DummyVecEnv

        venv = make_vec_env(market_data, normalize=False)
        assert isinstance(venv, DummyVecEnv)

    def test_discrete_vec_env(self, market_data):
        venv = make_vec_env(market_data, discrete=True, normalize=False)
        # Should have Discrete action space
        assert isinstance(venv.action_space, spaces.Discrete)

    def test_n_envs(self, market_data):
        tc = TrainingConfig(n_envs=2)
        venv = make_vec_env(market_data, training_config=tc, normalize=False)
        assert venv.num_envs == 2

    def test_step_through_vec_env(self, market_data):
        tc = TrainingConfig(n_envs=2)
        venv = make_vec_env(market_data, training_config=tc, normalize=False)
        obs = venv.reset()
        assert obs.shape == (2, 32)

        # Step with batch of actions
        actions = np.array([[0.0], [0.5]], dtype=np.float32)
        obs, rewards, dones, infos = venv.step(actions)
        assert obs.shape == (2, 32)
        assert rewards.shape == (2,)

    def test_step_through_discrete_vec_env(self, market_data):
        tc = TrainingConfig(n_envs=2)
        venv = make_vec_env(market_data, training_config=tc, discrete=True, normalize=False)
        obs = venv.reset()
        assert obs.shape == (2, 32)

        actions = np.array([0, 4])  # full charge, full discharge
        obs, rewards, dones, infos = venv.step(actions)
        assert obs.shape == (2, 32)


class TestDiscreteActionsArray:
    """Tests for the DISCRETE_ACTIONS constant."""

    def test_discrete_actions_values(self):
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(DISCRETE_ACTIONS, expected)

    def test_discrete_actions_length(self):
        assert len(DISCRETE_ACTIONS) == 5
