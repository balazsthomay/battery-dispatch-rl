"""Tests for visualization/plotting functions."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import pytest

from bess_dispatch.agents.evaluate import EpisodeDetail
from bess_dispatch.analysis.metrics import StrategyMetrics
from bess_dispatch.analysis.plots import (
    plot_cross_market_heatmap,
    plot_dispatch_vs_price,
    plot_strategy_comparison,
)
import numpy as np


@pytest.fixture
def sample_episode() -> EpisodeDetail:
    """A small episode detail for testing plots."""
    return EpisodeDetail(
        total_reward=50.0,
        total_revenue=60.0,
        total_degradation=10.0,
        total_cycles=1.5,
        actions=[0.3, -0.5, 0.0, 0.8, -0.2],
        soc_trajectory=[0.5, 0.55, 0.45, 0.45, 0.6, 0.55],
        prices=[50.0, 40.0, 55.0, 70.0, 45.0],
        rewards=[10.0, 8.0, 12.0, 15.0, 5.0],
    )


@pytest.fixture
def sample_metrics_list() -> list[StrategyMetrics]:
    """A few StrategyMetrics for testing comparison plots."""
    return [
        StrategyMetrics(
            name="DoNothing",
            total_revenue=0.0,
            total_degradation=0.01,
            net_reward=-0.01,
            equivalent_cycles=0.0,
            revenue_per_cycle=0.0,
            annualized_revenue=0.0,
            revenue_per_degradation=0.0,
        ),
        StrategyMetrics(
            name="Threshold",
            total_revenue=100.0,
            total_degradation=5.0,
            net_reward=95.0,
            equivalent_cycles=3.0,
            revenue_per_cycle=33.33,
            annualized_revenue=5214.0,
            revenue_per_degradation=20.0,
        ),
        StrategyMetrics(
            name="Oracle",
            total_revenue=200.0,
            total_degradation=8.0,
            net_reward=192.0,
            equivalent_cycles=5.0,
            revenue_per_cycle=40.0,
            annualized_revenue=10428.0,
            revenue_per_degradation=25.0,
        ),
    ]


class TestPlotDispatchVsPrice:
    """Tests for plot_dispatch_vs_price."""

    def test_returns_figure(self, sample_episode):
        """Should return a matplotlib Figure."""
        fig = plot_dispatch_vs_price(sample_episode)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_prices_overlay(self, sample_episode):
        """Should handle prices overlay without error."""
        prices = np.array([50.0, 40.0, 55.0, 70.0, 45.0])
        fig = plot_dispatch_vs_price(sample_episode, prices=prices)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_custom_title(self, sample_episode):
        """Custom title is set on figure."""
        fig = plot_dispatch_vs_price(sample_episode, title="Custom Title")
        assert isinstance(fig, matplotlib.figure.Figure)
        # Title is on first axes
        ax_title = fig.axes[0].get_title()
        assert ax_title == "Custom Title"
        plt.close(fig)

    def test_can_save_to_file(self, sample_episode, tmp_path):
        """Figure can be saved to disk."""
        fig = plot_dispatch_vs_price(sample_episode)
        path = tmp_path / "dispatch.png"
        fig.savefig(path)
        assert path.exists()
        assert path.stat().st_size > 0
        plt.close(fig)

    def test_without_prices(self, sample_episode):
        """Works without prices (no twin axis)."""
        fig = plot_dispatch_vs_price(sample_episode, prices=None)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestPlotStrategyComparison:
    """Tests for plot_strategy_comparison."""

    def test_returns_figure(self, sample_metrics_list):
        """Should return a matplotlib Figure."""
        fig = plot_strategy_comparison(sample_metrics_list)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_correct_number_of_bar_groups(self, sample_metrics_list):
        """Should have bar groups matching number of strategies."""
        fig = plot_strategy_comparison(sample_metrics_list)
        ax = fig.axes[0]
        # 3 strategies * 3 metrics = 9 bars total
        bars = [p for p in ax.patches]
        assert len(bars) == 9
        plt.close(fig)

    def test_can_save_to_file(self, sample_metrics_list, tmp_path):
        """Figure can be saved to disk."""
        fig = plot_strategy_comparison(sample_metrics_list)
        path = tmp_path / "comparison.png"
        fig.savefig(path)
        assert path.exists()
        assert path.stat().st_size > 0
        plt.close(fig)

    def test_with_custom_title(self, sample_metrics_list):
        """Custom title is applied."""
        fig = plot_strategy_comparison(sample_metrics_list, title="My Title")
        assert fig.axes[0].get_title() == "My Title"
        plt.close(fig)


class TestPlotCrossMarketHeatmap:
    """Tests for plot_cross_market_heatmap."""

    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        data = {
            "DoNothing": {"DE_LU": 0.0, "ES": 0.0},
            "Oracle": {"DE_LU": 200.0, "ES": 150.0},
        }
        fig = plot_cross_market_heatmap(data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_can_save_to_file(self, tmp_path):
        """Figure can be saved to disk."""
        data = {
            "Strategy": {"Zone1": 10.0, "Zone2": 20.0},
        }
        fig = plot_cross_market_heatmap(data)
        path = tmp_path / "heatmap.png"
        fig.savefig(path)
        assert path.exists()
        assert path.stat().st_size > 0
        plt.close(fig)

    def test_with_custom_metric_name_and_title(self):
        """Custom metric name and title are used."""
        data = {"S1": {"Z1": 5.0}}
        fig = plot_cross_market_heatmap(
            data, metric_name="Revenue", title="Cross Test"
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_annotations_present(self):
        """Value annotations appear in the heatmap cells."""
        data = {"A": {"X": 42.5}}
        fig = plot_cross_market_heatmap(data)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert "42.5" in texts
        plt.close(fig)
