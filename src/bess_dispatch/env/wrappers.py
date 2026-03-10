"""Environment wrappers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bess_dispatch.config import BatteryConfig, TrainingConfig
from bess_dispatch.data.loader import MarketData
from bess_dispatch.env.bess_env import BESSEnv

# Discrete action levels: {-1, -0.5, 0, 0.5, 1}
DISCRETE_ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)


class DiscreteActionWrapper(gym.ActionWrapper):
    """Maps Discrete(5) actions to continuous power levels for DQN."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Discrete(5)

    def action(self, action: int) -> np.ndarray:
        return np.array([DISCRETE_ACTIONS[action]], dtype=np.float32)


def make_env(
    market_data: MarketData,
    battery_config: BatteryConfig | None = None,
    episode_length: int = 168,
    random_start: bool = True,
    discrete: bool = False,
    seed: int | None = None,
) -> gym.Env:
    """Factory function for a single BESS environment."""
    env = BESSEnv(
        market_data=market_data,
        battery_config=battery_config,
        episode_length=episode_length,
        random_start=random_start,
        seed=seed,
    )
    if discrete:
        env = DiscreteActionWrapper(env)
    return env


def make_vec_env(
    market_data: MarketData,
    battery_config: BatteryConfig | None = None,
    training_config: TrainingConfig | None = None,
    discrete: bool = False,
    normalize: bool = True,
) -> DummyVecEnv | VecNormalize:
    """Create vectorized environment for training."""
    tc = training_config or TrainingConfig()

    def _make_env(rank: int):
        def _init():
            return make_env(
                market_data=market_data,
                battery_config=battery_config,
                episode_length=tc.episode_length,
                random_start=True,
                discrete=discrete,
                seed=tc.seed + rank,
            )

        return _init

    venv = DummyVecEnv([_make_env(i) for i in range(tc.n_envs)])

    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    return venv
