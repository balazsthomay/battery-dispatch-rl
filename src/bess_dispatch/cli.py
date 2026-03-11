"""CLI entry point for BESS Dispatch."""

from pathlib import Path

import click


@click.group()
def main() -> None:
    """BESS Dispatch RL Optimizer CLI."""


@main.command()
@click.option(
    "--zone",
    default="DE_LU",
    type=click.Choice(["DE_LU", "ES", "NL", "PL"]),
)
@click.option("--year", multiple=True, type=int, default=[2023])
@click.option("--api-key", envvar="ENTSOE_API_KEY", default=None)
def download(zone: str, year: tuple[int, ...], api_key: str | None) -> None:
    """Download market data from ENTSO-E."""
    from bess_dispatch.config import DataConfig
    from bess_dispatch.data.client import EntsoeClient
    from bess_dispatch.data.loader import MarketDataLoader

    client = EntsoeClient(api_key=api_key)
    loader = MarketDataLoader(client=client)
    for y in year:
        click.echo(f"Downloading {zone} prices for {y}...")
        loader.load_prices(zone, y)
        click.echo(f"  Cached to data/{zone}/day_ahead_prices_{y}.parquet")


@main.command()
@click.option("--algorithm", default="DQN", type=click.Choice(["DQN", "SAC"]))
@click.option(
    "--zone",
    default="DE_LU",
    type=click.Choice(["DE_LU", "ES", "NL", "PL"]),
)
@click.option("--year", default=2023, type=int, help="Training data year")
@click.option("--timesteps", default=100_000, type=int)
@click.option("--save-dir", default="results/models", type=str)
@click.option("--use-sample", is_flag=True, help="Use committed sample data")
def train(
    algorithm: str,
    zone: str,
    year: int,
    timesteps: int,
    save_dir: str,
    use_sample: bool,
) -> None:
    """Train an RL agent."""
    from bess_dispatch.agents.train import train_dqn, train_sac
    from bess_dispatch.config import TrainingConfig
    from bess_dispatch.data.loader import MarketDataLoader, load_sample

    if use_sample:
        market_data = load_sample()
    else:
        loader = MarketDataLoader()
        market_data = loader.load_market_data(zone, year)

    tc = TrainingConfig(algorithm=algorithm, total_timesteps=timesteps)

    click.echo(f"Training {algorithm} on {zone} {year} for {timesteps} steps...")
    if algorithm == "DQN":
        model, venv = train_dqn(market_data, training_config=tc, save_dir=save_dir)
    else:
        model, venv = train_sac(market_data, training_config=tc, save_dir=save_dir)

    click.echo(f"Model saved to {save_dir}/")


@main.command()
@click.option("--model", "model_path", required=True, type=str)
@click.option("--algorithm", default="DQN", type=click.Choice(["DQN", "SAC"]))
@click.option("--zone", default="DE_LU")
@click.option("--year", default=2024, type=int, help="Evaluation data year")
@click.option("--use-sample", is_flag=True)
@click.option("--n-episodes", default=5, type=int)
def evaluate(
    model_path: str,
    algorithm: str,
    zone: str,
    year: int,
    use_sample: bool,
    n_episodes: int,
) -> None:
    """Evaluate a trained model."""
    from bess_dispatch.agents.evaluate import evaluate_policy
    from bess_dispatch.agents.train import load_model
    from bess_dispatch.data.loader import MarketDataLoader, load_sample

    if use_sample:
        market_data = load_sample()
    else:
        loader = MarketDataLoader()
        market_data = loader.load_market_data(zone, year)

    vecnorm_path = str(Path(model_path).parent / "vecnormalize.pkl")
    model, venv = load_model(model_path, vecnorm_path, algorithm, market_data)

    # Wrap model to normalize observations using training stats
    if venv is not None:
        from bess_dispatch.agents.evaluate import NormalizedPolicy
        policy = NormalizedPolicy(model, venv, discrete=(algorithm == "DQN"))
    else:
        policy = model

    result = evaluate_policy(policy, market_data, n_episodes=n_episodes)
    click.echo(f"Mean reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
    click.echo(f"Mean revenue: {result.mean_revenue:.2f}")
    click.echo(f"Mean degradation: {result.mean_degradation:.4f}")


@main.command()
@click.option("--zone", default="DE_LU")
@click.option("--year", default=2024, type=int, help="Evaluation data year")
@click.option("--use-sample", is_flag=True)
def baselines(zone: str, year: int, use_sample: bool) -> None:
    """Run baseline strategies and compare."""
    from bess_dispatch.agents.evaluate import evaluate_policy
    from bess_dispatch.baselines.do_nothing import DoNothingPolicy
    from bess_dispatch.baselines.oracle import OraclePolicy
    from bess_dispatch.baselines.threshold import ThresholdPolicy
    from bess_dispatch.data.loader import MarketDataLoader, load_sample

    if use_sample:
        market_data = load_sample()
    else:
        loader = MarketDataLoader()
        market_data = loader.load_market_data(zone, year)

    prices = market_data.prices.values

    # Do nothing
    dn = DoNothingPolicy()
    dn_result = evaluate_policy(dn, market_data)

    # Threshold
    th = ThresholdPolicy()
    th.fit(prices)
    th_result = evaluate_policy(th, market_data)

    # Oracle
    oracle = OraclePolicy(prices[:168])  # first episode
    oracle_result = evaluate_policy(oracle, market_data)

    click.echo(
        f"{'Strategy':<15} {'Reward':>10} {'Revenue':>10} {'Degradation':>12}"
    )
    click.echo("-" * 50)
    for name, result in [
        ("Do Nothing", dn_result),
        ("Threshold", th_result),
        ("Oracle", oracle_result),
    ]:
        click.echo(
            f"{name:<15} {result.mean_reward:>10.2f} "
            f"{result.mean_revenue:>10.2f} "
            f"{result.mean_degradation:>12.4f}"
        )


@main.command()
@click.option("--output", default="results/report", type=str)
@click.option("--use-sample", is_flag=True)
def report(output: str, use_sample: bool) -> None:
    """Generate analysis report with plots and metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from bess_dispatch.agents.evaluate import evaluate_policy
    from bess_dispatch.analysis.metrics import comparison_table, compute_metrics
    from bess_dispatch.analysis.plots import plot_dispatch_vs_price, plot_strategy_comparison
    from bess_dispatch.baselines.do_nothing import DoNothingPolicy
    from bess_dispatch.baselines.oracle import OraclePolicy
    from bess_dispatch.baselines.threshold import ThresholdPolicy
    from bess_dispatch.data.loader import MarketDataLoader, load_sample

    if use_sample:
        market_data = load_sample()
    else:
        loader = MarketDataLoader()
        market_data = loader.load_market_data()

    prices = market_data.prices.values
    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)

    # Evaluate all baselines
    strategies: dict[str, object] = {}

    dn = DoNothingPolicy()
    strategies["Do Nothing"] = evaluate_policy(dn, market_data)

    th = ThresholdPolicy()
    th.fit(prices)
    strategies["Threshold"] = evaluate_policy(th, market_data)

    oracle = OraclePolicy(prices[:168])
    strategies["Oracle"] = evaluate_policy(oracle, market_data)

    # Compute metrics
    metrics = [compute_metrics(r, name) for name, r in strategies.items()]

    # Print comparison table
    table = comparison_table(metrics)
    click.echo(table)

    # Save table
    (out_path / "comparison.txt").write_text(table)

    # Generate plots
    for name, result in strategies.items():
        ep = result.episodes[0]
        fig = plot_dispatch_vs_price(
            ep, prices[: len(ep.actions)], title=f"{name} Dispatch"
        )
        fig.savefig(
            out_path / f"{name.lower().replace(' ', '_')}_dispatch.png", dpi=150
        )
        plt.close(fig)

    fig = plot_strategy_comparison(metrics)
    fig.savefig(out_path / "strategy_comparison.png", dpi=150)
    plt.close(fig)

    click.echo(f"\nReport saved to {out_path}/")
