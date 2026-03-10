"""Tests for performance metrics computation."""

from __future__ import annotations

import pytest

from bess_dispatch.agents.evaluate import EpisodeDetail, EvaluationResult
from bess_dispatch.analysis.metrics import StrategyMetrics, comparison_table, compute_metrics


def _make_episode(
    total_revenue: float = 100.0,
    total_degradation: float = 5.0,
    total_cycles: float = 2.0,
    total_reward: float = 95.0,
) -> EpisodeDetail:
    """Helper to build an EpisodeDetail with known values."""
    return EpisodeDetail(
        total_reward=total_reward,
        total_revenue=total_revenue,
        total_degradation=total_degradation,
        total_cycles=total_cycles,
        actions=[0.5, -0.5, 0.0],
        soc_trajectory=[0.5, 0.6, 0.5, 0.5],
        prices=[50.0, 60.0, 55.0],
        rewards=[30.0, 35.0, 30.0],
    )


def _make_eval_result(
    n_episodes: int = 1,
    total_revenue: float = 100.0,
    total_degradation: float = 5.0,
    total_cycles: float = 2.0,
    total_reward: float = 95.0,
) -> EvaluationResult:
    """Helper to build an EvaluationResult with known values."""
    episodes = [
        _make_episode(total_revenue, total_degradation, total_cycles, total_reward)
        for _ in range(n_episodes)
    ]
    return EvaluationResult(
        episodes=episodes,
        mean_reward=total_reward,
        std_reward=0.0,
        mean_revenue=total_revenue,
        mean_degradation=total_degradation,
        mean_cycles=total_cycles,
    )


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_returns_strategy_metrics(self):
        """compute_metrics returns a StrategyMetrics instance."""
        result = _make_eval_result()
        metrics = compute_metrics(result, "test")
        assert isinstance(metrics, StrategyMetrics)

    def test_correct_fields_from_known_inputs(self):
        """compute_metrics calculates correct values from known inputs."""
        result = _make_eval_result(
            n_episodes=2,
            total_revenue=100.0,
            total_degradation=5.0,
            total_cycles=2.0,
            total_reward=95.0,
        )
        metrics = compute_metrics(result, "Known", hours_per_episode=168)

        # 2 episodes: total_revenue = 200, total_degradation = 10, total_cycles = 4
        assert metrics.name == "Known"
        assert metrics.total_revenue == pytest.approx(200.0)
        assert metrics.total_degradation == pytest.approx(10.0)
        assert metrics.net_reward == pytest.approx(190.0)
        assert metrics.equivalent_cycles == pytest.approx(4.0)
        assert metrics.revenue_per_cycle == pytest.approx(50.0)  # 200 / 4
        assert metrics.revenue_per_degradation == pytest.approx(20.0)  # 200 / 10

    def test_annualized_revenue_extrapolation(self):
        """Annualized revenue correctly extrapolates from episode hours to a year."""
        result = _make_eval_result(
            n_episodes=1, total_revenue=168.0  # 1 EUR per hour
        )
        metrics = compute_metrics(result, "Annual", hours_per_episode=168)

        # total_hours = 168 * 1 = 168
        # rate = 168 / 168 = 1 EUR/h
        # annualized = 1 * 8760 = 8760
        assert metrics.annualized_revenue == pytest.approx(8760.0)

    def test_zero_cycles_no_division_error(self):
        """Zero cycles gives revenue_per_cycle=0, not a ZeroDivisionError."""
        result = _make_eval_result(total_cycles=0.0)
        metrics = compute_metrics(result, "NoCycles")
        assert metrics.revenue_per_cycle == 0.0

    def test_zero_degradation_no_division_error(self):
        """Zero degradation gives revenue_per_degradation=0, not a ZeroDivisionError."""
        result = _make_eval_result(total_degradation=0.0)
        metrics = compute_metrics(result, "NoDeg")
        assert metrics.revenue_per_degradation == 0.0

    def test_zero_hours_no_division_error(self):
        """Empty episodes list gives annualized_revenue=0."""
        result = EvaluationResult(
            episodes=[],
            mean_reward=0.0,
            std_reward=0.0,
            mean_revenue=0.0,
            mean_degradation=0.0,
            mean_cycles=0.0,
        )
        metrics = compute_metrics(result, "Empty")
        assert metrics.annualized_revenue == 0.0

    def test_multiple_episodes_sum_correctly(self):
        """Metrics sum across multiple episodes."""
        ep1 = _make_episode(total_revenue=50.0, total_degradation=2.0, total_cycles=1.0)
        ep2 = _make_episode(total_revenue=80.0, total_degradation=3.0, total_cycles=1.5)
        result = EvaluationResult(
            episodes=[ep1, ep2],
            mean_reward=0.0,
            std_reward=0.0,
            mean_revenue=65.0,
            mean_degradation=2.5,
            mean_cycles=1.25,
        )
        metrics = compute_metrics(result, "Multi", hours_per_episode=168)
        assert metrics.total_revenue == pytest.approx(130.0)
        assert metrics.total_degradation == pytest.approx(5.0)
        assert metrics.equivalent_cycles == pytest.approx(2.5)


class TestComparisonTable:
    """Tests for comparison_table function."""

    def test_returns_string_with_all_strategy_names(self):
        """Table includes all strategy names."""
        result1 = _make_eval_result(total_revenue=100.0)
        result2 = _make_eval_result(total_revenue=200.0)
        m1 = compute_metrics(result1, "DoNothing")
        m2 = compute_metrics(result2, "Oracle")

        table = comparison_table([m1, m2])
        assert isinstance(table, str)
        assert "DoNothing" in table
        assert "Oracle" in table

    def test_empty_list_returns_header_only(self):
        """Empty list still returns the header and separator."""
        table = comparison_table([])
        assert isinstance(table, str)
        assert "Strategy" in table
        # Header + separator = 2 lines
        lines = table.strip().split("\n")
        assert len(lines) == 2

    def test_table_has_header_separator_and_data_rows(self):
        """Table has header, separator, and one row per strategy."""
        m = compute_metrics(_make_eval_result(), "Test")
        table = comparison_table([m])
        lines = table.strip().split("\n")
        # header + separator + 1 data row
        assert len(lines) == 3

    def test_table_contains_numeric_values(self):
        """Table should contain formatted numeric values."""
        result = _make_eval_result(total_revenue=123.45, total_degradation=6.789)
        m = compute_metrics(result, "Nums")
        table = comparison_table([m])
        assert "123.45" in table
        assert "6.7890" in table
