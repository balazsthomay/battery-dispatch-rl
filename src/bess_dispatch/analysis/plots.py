"""Visualization functions for BESS dispatch analysis."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from bess_dispatch.agents.evaluate import EpisodeDetail
from bess_dispatch.analysis.metrics import StrategyMetrics


def plot_dispatch_vs_price(
    episode: EpisodeDetail,
    prices: np.ndarray | None = None,
    title: str = "Dispatch vs Price",
) -> matplotlib.figure.Figure:
    """Plot battery actions and SoC against price signal.

    Creates a 2-panel figure:
    - Top: price + actions (as bars)
    - Bottom: SoC trajectory

    Args:
        episode: Single episode detail with actions and SoC trajectory.
        prices: Optional price array for overlay on the top panel.
        title: Figure title applied to the top panel.

    Returns:
        matplotlib Figure with the two-panel plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    steps = np.arange(len(episode.actions))

    # Top panel: price and actions
    if prices is not None and len(prices) >= len(steps):
        ax1_price = ax1.twinx()
        ax1_price.plot(
            steps, prices[: len(steps)], color="gray", alpha=0.5, label="Price"
        )
        ax1_price.set_ylabel("Price (EUR/MWh)")
        ax1_price.legend(loc="upper right")

    colors = [
        "green" if a < 0 else "red" if a > 0 else "gray" for a in episode.actions
    ]
    ax1.bar(steps, episode.actions, color=colors, alpha=0.7, width=1.0)
    ax1.set_ylabel("Action (- charge / + discharge)")
    ax1.set_title(title)
    ax1.axhline(y=0, color="black", linewidth=0.5)

    # Bottom panel: SoC
    ax2.plot(episode.soc_trajectory, color="blue", linewidth=1.5)
    ax2.set_ylabel("State of Charge")
    ax2.set_xlabel("Hour")
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax2.axhline(y=1, color="black", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    return fig


def plot_strategy_comparison(
    metrics_list: list[StrategyMetrics],
    title: str = "Strategy Comparison",
) -> matplotlib.figure.Figure:
    """Bar chart comparing strategies on key metrics.

    Args:
        metrics_list: List of StrategyMetrics to compare.
        title: Chart title.

    Returns:
        matplotlib Figure with grouped bar chart.
    """
    names = [m.name for m in metrics_list]
    revenues = [m.total_revenue for m in metrics_list]
    degradations = [m.total_degradation for m in metrics_list]
    net_rewards = [m.net_reward for m in metrics_list]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, revenues, width, label="Revenue", color="green", alpha=0.7)
    ax.bar(
        x,
        [-d for d in degradations],
        width,
        label="Degradation (neg)",
        color="red",
        alpha=0.7,
    )
    ax.bar(x + width, net_rewards, width, label="Net Reward", color="blue", alpha=0.7)

    ax.set_xlabel("Strategy")
    ax.set_ylabel("EUR")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    return fig


def plot_cross_market_heatmap(
    data: dict[str, dict[str, float]],
    metric_name: str = "Net Reward",
    title: str = "Cross-Market Performance",
) -> matplotlib.figure.Figure:
    """Heatmap of strategy performance across markets.

    Args:
        data: Nested dict {strategy_name: {zone: metric_value}}.
        metric_name: Name of the metric being shown.
        title: Plot title.

    Returns:
        matplotlib Figure with annotated heatmap.
    """
    strategies = list(data.keys())
    zones = list(next(iter(data.values())).keys()) if data else []

    matrix = np.array([[data[s].get(z, 0.0) for z in zones] for s in strategies])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(zones)))
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_xticklabels(zones)
    ax.set_yticklabels(strategies)

    # Add value annotations
    for i in range(len(strategies)):
        for j in range(len(zones)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center")

    ax.set_title(f"{title}\n({metric_name})")
    fig.colorbar(im)
    fig.tight_layout()
    return fig
