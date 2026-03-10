"""Tests for the unified evaluation harness."""

from __future__ import annotations

import numpy as np
import pytest

from bess_dispatch.agents.evaluate import (
    EpisodeDetail,
    EvaluationResult,
    evaluate_policy,
)
from bess_dispatch.baselines.do_nothing import DoNothingPolicy


class TestEvaluatePolicy:
    """Tests for evaluate_policy function."""

    def test_returns_evaluation_result(self, market_data, battery_config):
        """evaluate_policy should return an EvaluationResult."""
        policy = DoNothingPolicy()
        result = evaluate_policy(
            policy,
            market_data,
            battery_config,
            n_episodes=1,
            random_start=False,
            seed=42,
        )
        assert isinstance(result, EvaluationResult)

    def test_n_episodes_produces_correct_count(self, market_data, battery_config):
        """n_episodes parameter should control number of EpisodeDetail objects."""
        policy = DoNothingPolicy()
        result = evaluate_policy(
            policy,
            market_data,
            battery_config,
            n_episodes=3,
            random_start=True,
            seed=42,
        )
        assert len(result.episodes) == 3

    def test_episode_detail_tracks_data(self, market_data, battery_config):
        """EpisodeDetail should track rewards, actions, SoC trajectory."""
        policy = DoNothingPolicy()
        result = evaluate_policy(
            policy,
            market_data,
            battery_config,
            n_episodes=1,
            episode_length=168,
            random_start=False,
            seed=42,
        )

        ep = result.episodes[0]
        assert isinstance(ep, EpisodeDetail)
        assert len(ep.actions) == 168
        assert len(ep.rewards) == 168
        # SoC trajectory includes initial state
        assert len(ep.soc_trajectory) == 169
        assert ep.total_reward == pytest.approx(sum(ep.rewards))

    def test_do_nothing_near_zero_revenue(self, market_data, battery_config):
        """Evaluate with do-nothing should produce zero revenue."""
        policy = DoNothingPolicy()
        result = evaluate_policy(
            policy,
            market_data,
            battery_config,
            n_episodes=1,
            random_start=False,
            seed=42,
        )

        assert result.mean_revenue == pytest.approx(0.0)
        assert result.mean_reward < 0.0  # only degradation
        assert result.mean_cycles == pytest.approx(0.0)

    def test_mean_std_computed_correctly(self, market_data, battery_config):
        """Mean and std should be correctly computed across episodes."""
        policy = DoNothingPolicy()
        result = evaluate_policy(
            policy,
            market_data,
            battery_config,
            n_episodes=5,
            random_start=True,
            seed=42,
        )

        rewards = [ep.total_reward for ep in result.episodes]
        assert result.mean_reward == pytest.approx(float(np.mean(rewards)))
        assert result.std_reward == pytest.approx(float(np.std(rewards)))

    def test_oracle_policy_with_evaluate(self, market_data, battery_config):
        """Oracle policy should work with evaluate_policy (has reset method)."""
        from bess_dispatch.baselines.oracle import OraclePolicy
        from bess_dispatch.env.bess_env import BESSEnv

        # Get episode prices
        env = BESSEnv(market_data, battery_config, random_start=False, seed=42)
        obs, _ = env.reset()
        start_idx = env._start_idx
        episode_prices = env.prices[start_idx : start_idx + env.episode_length]

        policy = OraclePolicy(episode_prices, battery_config)
        result = evaluate_policy(
            policy,
            market_data,
            battery_config,
            n_episodes=1,
            random_start=False,
            seed=42,
        )

        assert isinstance(result, EvaluationResult)
        assert len(result.episodes) == 1
        # Oracle should have positive or near-zero revenue on sinusoidal prices
        assert result.mean_reward > -10.0


class TestEpisodeDetail:
    """Tests for the EpisodeDetail dataclass."""

    def test_default_values(self):
        """EpisodeDetail should have sensible defaults."""
        ep = EpisodeDetail()
        assert ep.total_reward == 0.0
        assert ep.total_revenue == 0.0
        assert ep.total_degradation == 0.0
        assert ep.total_cycles == 0.0
        assert ep.actions == []
        assert ep.soc_trajectory == []
        assert ep.prices == []
        assert ep.rewards == []

    def test_mutable_lists_independent(self):
        """Each EpisodeDetail should have independent lists."""
        ep1 = EpisodeDetail()
        ep2 = EpisodeDetail()
        ep1.actions.append(1.0)
        assert ep2.actions == []
