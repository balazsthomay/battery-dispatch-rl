"""RL agent training orchestration."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN, SAC
from stable_baselines3.common.vec_env import VecNormalize

from bess_dispatch.config import BatteryConfig, TrainingConfig
from bess_dispatch.data.loader import MarketData
from bess_dispatch.env.wrappers import make_vec_env


def train_dqn(
    market_data: MarketData,
    battery_config: BatteryConfig | None = None,
    training_config: TrainingConfig | None = None,
    save_dir: str = "results/models",
) -> tuple[DQN, VecNormalize]:
    """Train a DQN agent with discrete actions.

    Uses DiscreteActionWrapper (5 actions: {-1, -0.5, 0, 0.5, 1}).

    Args:
        market_data: Market data for environment creation.
        battery_config: Battery configuration. Uses defaults if None.
        training_config: Training hyperparameters. Uses defaults if None.
        save_dir: Directory to save model and VecNormalize stats.

    Returns:
        Tuple of (trained DQN model, VecNormalize environment).
    """
    tc = training_config or TrainingConfig(algorithm="DQN")

    venv = make_vec_env(
        market_data=market_data,
        battery_config=battery_config,
        training_config=tc,
        discrete=True,
        normalize=True,
    )

    model = DQN(
        "MlpPolicy",
        venv,
        learning_rate=tc.learning_rate,
        gamma=tc.gamma,
        exploration_fraction=tc.exploration_fraction,
        buffer_size=min(tc.total_timesteps, 100_000),
        verbose=1,
        seed=tc.seed,
    )

    model.learn(total_timesteps=tc.total_timesteps)

    # Save model and VecNormalize stats
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save(save_path / "dqn_model")
    venv.save(str(save_path / "vecnormalize.pkl"))

    return model, venv


def train_sac(
    market_data: MarketData,
    battery_config: BatteryConfig | None = None,
    training_config: TrainingConfig | None = None,
    save_dir: str = "results/models",
) -> tuple[SAC, VecNormalize]:
    """Train a SAC agent with continuous actions.

    Args:
        market_data: Market data for environment creation.
        battery_config: Battery configuration. Uses defaults if None.
        training_config: Training hyperparameters. Uses defaults if None.
        save_dir: Directory to save model and VecNormalize stats.

    Returns:
        Tuple of (trained SAC model, VecNormalize environment).
    """
    tc = training_config or TrainingConfig(algorithm="SAC")

    venv = make_vec_env(
        market_data=market_data,
        battery_config=battery_config,
        training_config=tc,
        discrete=False,
        normalize=True,
    )

    model = SAC(
        "MlpPolicy",
        venv,
        learning_rate=tc.learning_rate,
        gamma=tc.gamma,
        ent_coef="auto",
        buffer_size=min(tc.total_timesteps, 100_000),
        verbose=1,
        seed=tc.seed,
    )

    model.learn(total_timesteps=tc.total_timesteps)

    # Save model and VecNormalize stats
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save(save_path / "sac_model")
    venv.save(str(save_path / "vecnormalize.pkl"))

    return model, venv


def load_model(
    model_path: str,
    vecnorm_path: str | None = None,
    algorithm: str = "DQN",
    market_data: MarketData | None = None,
    battery_config: BatteryConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> tuple:
    """Load a trained model and optionally VecNormalize stats.

    Args:
        model_path: Path to the saved model (without .zip extension).
        vecnorm_path: Path to VecNormalize stats pickle. If provided with
            market_data, recreates the normalized environment.
        algorithm: Algorithm name ("DQN" or "SAC").
        market_data: Market data for environment recreation (needed with vecnorm).
        battery_config: Battery configuration for environment recreation.
        training_config: Training config for environment recreation.

    Returns:
        Tuple of (model, venv_or_None).

    Raises:
        ValueError: If algorithm is not "DQN" or "SAC".
    """
    if algorithm.upper() == "DQN":
        model_cls = DQN
        discrete = True
    elif algorithm.upper() == "SAC":
        model_cls = SAC
        discrete = False
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if vecnorm_path and market_data:
        tc = training_config or TrainingConfig()
        venv = make_vec_env(
            market_data=market_data,
            battery_config=battery_config,
            training_config=tc,
            discrete=discrete,
            normalize=True,
        )
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        model = model_cls.load(model_path, env=venv)
        return model, venv
    else:
        model = model_cls.load(model_path)
        return model, None
