"""Unified evaluation harness for any policy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bess_dispatch.config import BatteryConfig
from bess_dispatch.data.loader import MarketData
from bess_dispatch.env.bess_env import BESSEnv


class NormalizedPolicy:
    """Wraps an SB3 model with VecNormalize obs normalization for raw-env eval.

    Handles both discrete (DQN) and continuous (SAC) action spaces.
    For DQN, maps discrete actions back to continuous power levels.
    """

    # Maps Discrete(5) indices to continuous power levels
    DISCRETE_ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

    def __init__(self, model, venv, discrete: bool = False) -> None:
        self.model = model
        self.venv = venv
        self.discrete = discrete

    def predict(self, obs):
        norm_obs = self.venv.normalize_obs(obs)
        action, state = self.model.predict(norm_obs, deterministic=True)
        if self.discrete:
            # Map discrete action index to continuous power level
            idx = int(action)
            action = np.array([self.DISCRETE_ACTIONS[idx]], dtype=np.float32)
        return action, state


@dataclass
class EpisodeDetail:
    """Detailed results from a single evaluation episode."""

    total_reward: float = 0.0
    total_revenue: float = 0.0
    total_degradation: float = 0.0
    total_cycles: float = 0.0
    actions: list[float] = field(default_factory=list)
    soc_trajectory: list[float] = field(default_factory=list)
    prices: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Aggregate results across evaluation episodes."""

    episodes: list[EpisodeDetail]
    mean_reward: float
    std_reward: float
    mean_revenue: float
    mean_degradation: float
    mean_cycles: float


def evaluate_policy(
    policy,  # anything with .predict(obs) -> (action, _)
    market_data: MarketData,
    battery_config: BatteryConfig | None = None,
    n_episodes: int = 1,
    episode_length: int = 168,
    random_start: bool = False,
    seed: int = 42,
) -> EvaluationResult:
    """Run a policy through the environment and collect detailed results.

    Args:
        policy: Any object with a .predict(obs) method returning (action, state).
            Optionally may have a .reset() method (e.g., OraclePolicy).
        market_data: Market data for environment creation.
        battery_config: Battery configuration. Uses defaults if None.
        n_episodes: Number of evaluation episodes to run.
        episode_length: Steps per episode.
        random_start: Whether to randomize episode start positions.
        seed: Base random seed (incremented per episode).

    Returns:
        EvaluationResult with per-episode details and aggregate statistics.
    """
    config = battery_config or BatteryConfig()
    episodes: list[EpisodeDetail] = []

    for ep in range(n_episodes):
        env = BESSEnv(
            market_data=market_data,
            battery_config=config,
            episode_length=episode_length,
            random_start=random_start,
            seed=seed + ep,
        )
        obs, _ = env.reset()

        # Reset oracle policies if they have a reset method
        if hasattr(policy, "reset"):
            policy.reset()

        detail = EpisodeDetail()
        detail.soc_trajectory.append(float(obs[0]))  # initial SoC

        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            detail.total_reward += reward
            detail.total_revenue += info.get("revenue", 0.0)
            detail.total_degradation += info.get("degradation", 0.0)
            detail.rewards.append(float(reward))
            detail.actions.append(
                float(action[0]) if hasattr(action, "__len__") else float(action)
            )
            detail.soc_trajectory.append(float(info.get("soc", obs[0])))
            detail.prices.append(float(info.get("actual_power", 0.0)))
            detail.total_cycles = env._state.total_cycles

            done = terminated or truncated

        episodes.append(detail)

    rewards = [ep.total_reward for ep in episodes]
    return EvaluationResult(
        episodes=episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_revenue=float(np.mean([ep.total_revenue for ep in episodes])),
        mean_degradation=float(np.mean([ep.total_degradation for ep in episodes])),
        mean_cycles=float(np.mean([ep.total_cycles for ep in episodes])),
    )
