"""Tests for CLI module."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from bess_dispatch.cli import main


class TestCLI:
    """Tests for CLI entry point."""

    def test_main_group_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "BESS Dispatch RL Optimizer CLI" in result.output

    def test_download_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--help"])
        assert result.exit_code == 0
        assert "--zone" in result.output
        assert "--year" in result.output
        assert "--api-key" in result.output

    def test_train_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "--algorithm" in result.output
        assert "--timesteps" in result.output
        assert "--use-sample" in result.output

    def test_evaluate_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--algorithm" in result.output
        assert "--n-episodes" in result.output

    def test_baselines_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["baselines", "--help"])
        assert result.exit_code == 0
        assert "--use-sample" in result.output


class TestDownloadCommand:
    """Tests for the download subcommand."""

    @patch("bess_dispatch.data.loader.MarketDataLoader")
    @patch("bess_dispatch.data.client.EntsoePandasClient")
    def test_download_default_args(self, mock_entsoe_cls, mock_loader_cls):
        """Test download with default zone and year."""
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader

        runner = CliRunner()
        result = runner.invoke(main, ["download", "--api-key", "test-key"])

        assert result.exit_code == 0, result.output
        assert "Downloading DE_LU prices for 2023" in result.output

    @patch("bess_dispatch.data.loader.MarketDataLoader")
    @patch("bess_dispatch.data.client.EntsoePandasClient")
    def test_download_custom_zone_and_years(self, mock_entsoe_cls, mock_loader_cls):
        """Test download with custom zone and multiple years."""
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["download", "--api-key", "test-key", "--zone", "ES", "--year", "2022", "--year", "2023"],
        )

        assert result.exit_code == 0, result.output
        assert "Downloading ES prices for 2022" in result.output
        assert "Downloading ES prices for 2023" in result.output


class TestTrainCommand:
    """Tests for the train subcommand."""

    @patch("bess_dispatch.agents.train.train_dqn")
    @patch("bess_dispatch.data.loader.load_sample")
    def test_train_dqn_with_sample(self, mock_load_sample, mock_train_dqn):
        """Train DQN with sample data."""
        mock_market = MagicMock()
        mock_load_sample.return_value = mock_market
        mock_train_dqn.return_value = (MagicMock(), MagicMock())

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["train", "--algorithm", "DQN", "--use-sample", "--timesteps", "500"],
        )

        assert result.exit_code == 0, result.output
        assert "Training DQN" in result.output
        assert "Model saved" in result.output
        mock_load_sample.assert_called_once()
        mock_train_dqn.assert_called_once()

    @patch("bess_dispatch.agents.train.train_sac")
    @patch("bess_dispatch.data.loader.load_sample")
    def test_train_sac_with_sample(self, mock_load_sample, mock_train_sac):
        """Train SAC with sample data."""
        mock_market = MagicMock()
        mock_load_sample.return_value = mock_market
        mock_train_sac.return_value = (MagicMock(), MagicMock())

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["train", "--algorithm", "SAC", "--use-sample", "--timesteps", "500"],
        )

        assert result.exit_code == 0, result.output
        assert "Training SAC" in result.output
        mock_train_sac.assert_called_once()


class TestEvaluateCommand:
    """Tests for the evaluate subcommand."""

    @patch("bess_dispatch.agents.evaluate.evaluate_policy")
    @patch("bess_dispatch.agents.train.load_model")
    @patch("bess_dispatch.data.loader.load_sample")
    def test_evaluate_with_sample(self, mock_load_sample, mock_load_model, mock_eval):
        """Evaluate command runs with mocked model."""
        mock_market = MagicMock()
        mock_load_sample.return_value = mock_market
        mock_model = MagicMock()
        mock_venv = MagicMock()
        mock_load_model.return_value = (mock_model, mock_venv)

        mock_result = MagicMock()
        mock_result.mean_reward = 42.0
        mock_result.std_reward = 5.0
        mock_result.mean_revenue = 100.0
        mock_result.mean_degradation = 0.5
        mock_eval.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "evaluate",
                "--model", "some/path/dqn_model",
                "--algorithm", "DQN",
                "--use-sample",
                "--n-episodes", "3",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Mean reward: 42.00" in result.output
        assert "Mean revenue: 100.00" in result.output


class TestBaselinesCommand:
    """Tests for the baselines subcommand."""

    def test_baselines_with_sample_data(self, market_data):
        """Baselines command runs with actual market data."""
        with patch("bess_dispatch.data.loader.load_sample", return_value=market_data):
            runner = CliRunner()
            result = runner.invoke(main, ["baselines", "--use-sample"])

            assert result.exit_code == 0, result.output
            assert "Do Nothing" in result.output
            assert "Threshold" in result.output
            assert "Oracle" in result.output


class TestReportCommand:
    """Tests for the report subcommand."""

    def test_report_help(self):
        """Report command has correct options."""
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--use-sample" in result.output

    def test_report_with_sample_data(self, market_data, tmp_path):
        """Report command generates output files with sample data."""
        out_dir = str(tmp_path / "report_out")
        with patch("bess_dispatch.data.loader.load_sample", return_value=market_data):
            runner = CliRunner()
            result = runner.invoke(
                main, ["report", "--use-sample", "--output", out_dir]
            )

            assert result.exit_code == 0, result.output
            assert "Do Nothing" in result.output
            assert "Threshold" in result.output
            assert "Oracle" in result.output
            assert "Report saved to" in result.output

            # Check output files were created
            from pathlib import Path

            out_path = Path(out_dir)
            assert (out_path / "comparison.txt").exists()
            assert (out_path / "strategy_comparison.png").exists()
            assert (out_path / "do_nothing_dispatch.png").exists()
            assert (out_path / "threshold_dispatch.png").exists()
            assert (out_path / "oracle_dispatch.png").exists()
