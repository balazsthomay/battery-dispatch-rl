"""Performance metrics for BESS dispatch strategies."""

from __future__ import annotations

from dataclasses import dataclass

from bess_dispatch.agents.evaluate import EvaluationResult


@dataclass
class StrategyMetrics:
    """Computed metrics for a single strategy."""

    name: str
    total_revenue: float  # EUR
    total_degradation: float
    net_reward: float  # revenue - degradation
    equivalent_cycles: float  # total |delta_soc| / 2 over evaluation
    revenue_per_cycle: float  # revenue / cycles (efficiency metric)
    annualized_revenue: float  # extrapolated to 1 year from episode duration
    revenue_per_degradation: float  # revenue / degradation (higher = better)


def compute_metrics(
    result: EvaluationResult,
    name: str,
    hours_per_episode: int = 168,
) -> StrategyMetrics:
    """Compute performance metrics from evaluation result.

    Args:
        result: Evaluation result with per-episode details.
        name: Human-readable strategy name.
        hours_per_episode: Number of hours in each episode (default 168 = 1 week).

    Returns:
        StrategyMetrics with aggregated and derived metrics.
    """
    total_hours = hours_per_episode * len(result.episodes)
    hours_per_year = 8760

    total_revenue = sum(ep.total_revenue for ep in result.episodes)
    total_degradation = sum(ep.total_degradation for ep in result.episodes)
    total_cycles = sum(ep.total_cycles for ep in result.episodes)
    net_reward = total_revenue - total_degradation

    revenue_per_cycle = total_revenue / total_cycles if total_cycles > 0 else 0.0
    annualized_revenue = (
        (total_revenue / total_hours * hours_per_year) if total_hours > 0 else 0.0
    )
    revenue_per_degradation = (
        total_revenue / total_degradation if total_degradation > 0 else 0.0
    )

    return StrategyMetrics(
        name=name,
        total_revenue=total_revenue,
        total_degradation=total_degradation,
        net_reward=net_reward,
        equivalent_cycles=total_cycles,
        revenue_per_cycle=revenue_per_cycle,
        annualized_revenue=annualized_revenue,
        revenue_per_degradation=revenue_per_degradation,
    )


def comparison_table(metrics_list: list[StrategyMetrics]) -> str:
    """Format a comparison table of multiple strategies.

    Args:
        metrics_list: List of StrategyMetrics to display.

    Returns:
        Formatted multi-line string table.
    """
    header = (
        f"{'Strategy':<15} {'Revenue':>10} {'Degradation':>12} "
        f"{'Net Reward':>12} {'Cycles':>8} {'Rev/Cycle':>10} {'Ann. Rev':>10}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for m in metrics_list:
        rows.append(
            f"{m.name:<15} {m.total_revenue:>10.2f} {m.total_degradation:>12.4f} "
            f"{m.net_reward:>12.2f} {m.equivalent_cycles:>8.2f} "
            f"{m.revenue_per_cycle:>10.2f} {m.annualized_revenue:>10.0f}"
        )
    return "\n".join(rows)
