"""Tests for RL agent training."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.vec_env import VecNormalize

from bess_dispatch.config import BatteryConfig, TrainingConfig


class TestTrainDQN:
    """Tests for DQN training."""

    def test_smoke_train_dqn(self, market_data, tmp_path):
        """DQN training runs without error and returns model + venv."""
        from bess_dispatch.agents.train import train_dqn

        tc = TrainingConfig(
            algorithm="DQN",
            total_timesteps=500,
            n_envs=1,
            seed=42,
        )
        model, venv = train_dqn(
            market_data,
            training_config=tc,
            save_dir=str(tmp_path / "dqn"),
        )

        assert isinstance(model, DQN)
        assert isinstance(venv, VecNormalize)
        # Model file should exist
        assert (tmp_path / "dqn" / "dqn_model.zip").exists()
        assert (tmp_path / "dqn" / "vecnormalize.pkl").exists()

    def test_dqn_default_configs(self, market_data, tmp_path):
        """DQN training works with default configs (None)."""
        from bess_dispatch.agents.train import train_dqn

        tc = TrainingConfig(
            algorithm="DQN",
            total_timesteps=500,
            n_envs=1,
            seed=42,
        )
        model, venv = train_dqn(
            market_data,
            battery_config=None,
            training_config=tc,
            save_dir=str(tmp_path / "dqn_default"),
        )
        assert isinstance(model, DQN)

    def test_dqn_save_load_roundtrip(self, market_data, tmp_path):
        """Train, save, load — loaded model produces same predictions."""
        from bess_dispatch.agents.train import load_model, train_dqn

        tc = TrainingConfig(
            algorithm="DQN",
            total_timesteps=500,
            n_envs=1,
            seed=42,
        )
        model, venv = train_dqn(
            market_data,
            training_config=tc,
            save_dir=str(tmp_path / "dqn_rt"),
        )

        # Get prediction from trained model
        obs = venv.reset()
        action_orig, _ = model.predict(obs, deterministic=True)

        # Load model
        loaded_model, loaded_venv = load_model(
            model_path=str(tmp_path / "dqn_rt" / "dqn_model"),
            vecnorm_path=str(tmp_path / "dqn_rt" / "vecnormalize.pkl"),
            algorithm="DQN",
            market_data=market_data,
            training_config=tc,
        )

        # Get prediction from loaded model
        obs_loaded = loaded_venv.reset()
        action_loaded, _ = loaded_model.predict(obs_loaded, deterministic=True)

        assert isinstance(loaded_model, DQN)
        assert isinstance(loaded_venv, VecNormalize)
        # Loaded venv should not be training
        assert loaded_venv.training is False
        assert loaded_venv.norm_reward is False


class TestTrainSAC:
    """Tests for SAC training."""

    def test_smoke_train_sac(self, market_data, tmp_path):
        """SAC training runs without error and returns model + venv."""
        from bess_dispatch.agents.train import train_sac

        tc = TrainingConfig(
            algorithm="SAC",
            total_timesteps=500,
            n_envs=1,
            seed=42,
        )
        model, venv = train_sac(
            market_data,
            training_config=tc,
            save_dir=str(tmp_path / "sac"),
        )

        assert isinstance(model, SAC)
        assert isinstance(venv, VecNormalize)
        assert (tmp_path / "sac" / "sac_model.zip").exists()
        assert (tmp_path / "sac" / "vecnormalize.pkl").exists()

    def test_sac_save_load_roundtrip(self, market_data, tmp_path):
        """Train, save, load — loaded SAC model works."""
        from bess_dispatch.agents.train import load_model, train_sac

        tc = TrainingConfig(
            algorithm="SAC",
            total_timesteps=500,
            n_envs=1,
            seed=42,
        )
        model, venv = train_sac(
            market_data,
            training_config=tc,
            save_dir=str(tmp_path / "sac_rt"),
        )

        loaded_model, loaded_venv = load_model(
            model_path=str(tmp_path / "sac_rt" / "sac_model"),
            vecnorm_path=str(tmp_path / "sac_rt" / "vecnormalize.pkl"),
            algorithm="SAC",
            market_data=market_data,
            training_config=tc,
        )

        assert isinstance(loaded_model, SAC)
        assert isinstance(loaded_venv, VecNormalize)
        assert loaded_venv.training is False
        assert loaded_venv.norm_reward is False


class TestLoadModel:
    """Tests for load_model function."""

    def test_unknown_algorithm_raises(self):
        """load_model with unknown algorithm raises ValueError."""
        from bess_dispatch.agents.train import load_model

        with pytest.raises(ValueError, match="Unknown algorithm"):
            load_model(
                model_path="dummy_path",
                algorithm="PPO",
            )

    def test_load_without_vecnorm(self, market_data, tmp_path):
        """load_model without vecnorm_path returns (model, None)."""
        from bess_dispatch.agents.train import load_model, train_dqn

        tc = TrainingConfig(
            algorithm="DQN",
            total_timesteps=500,
            n_envs=1,
            seed=42,
        )
        train_dqn(
            market_data,
            training_config=tc,
            save_dir=str(tmp_path / "dqn_novn"),
        )

        model, venv = load_model(
            model_path=str(tmp_path / "dqn_novn" / "dqn_model"),
            vecnorm_path=None,
            algorithm="DQN",
        )

        assert isinstance(model, DQN)
        assert venv is None
