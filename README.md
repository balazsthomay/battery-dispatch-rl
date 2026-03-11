# BESS Dispatch RL Optimizer

Reinforcement learning agent for optimal battery energy storage system (BESS) dispatch against European day-ahead electricity markets. Trained on real ENTSO-E price data, the agent learns to charge during price dips and discharge during peaks while managing battery degradation.

## Results

**Cross-market performance** (trained on DE_LU 2023, evaluated on 2024 data):

| Strategy       | DE_LU | ES   | NL    | PL    |
|----------------|-------|------|-------|-------|
| Do Nothing     | 0.0   | 0.0  | 0.0   | 0.0   |
| Threshold      | 52.4  | 27.0 | 58.5  | 99.9  |
| DQN            | 75.0  | 45.5 | 119.8 | 122.8 |
| **SAC**        | **185.2** | **76.5** | **216.0** | **238.1** |
| Oracle (upper bound) | 359.1 | 264.9 | 419.2 | 472.5 |

*Mean reward (EUR/week) = revenue - degradation cost. Oracle uses perfect foresight.*

SAC captures ~50% of oracle performance across all markets. Both RL agents generalize to unseen markets (NL, PL) without retraining, outperforming the tuned threshold baseline everywhere.

![Cross-Market Heatmap](results/report/cross_market_heatmap.png)

**SAC learned dispatch behavior** (DE_LU, sample week):

![SAC Dispatch](results/report/sac_dispatch_de_lu.png)

The agent charges aggressively during low-price periods (green, hours 50-80) and discharges during price peaks (red). SoC management is coherent — the battery fills during cheap hours and depletes during expensive ones.

## Architecture

```
src/bess_dispatch/
├── config.py           # Battery, market, training parameters
├── data/               # ENTSO-E API client, parquet caching, data loading
├── env/
│   ├── battery.py      # Pure physics: efficiency, degradation, SoC dynamics
│   ├── bess_env.py     # Gymnasium env (32-dim obs, continuous actions)
│   └── wrappers.py     # Discrete wrapper for DQN, VecNormalize factory
├── baselines/          # Do-nothing, threshold, oracle (CVXPY QP)
├── agents/             # DQN/SAC training, evaluation harness
└── analysis/           # Metrics computation, matplotlib plots
```

**Key design decisions:**
- **Observation space (32 dims):** SoC + current price + 24h price history + cyclic time features + wind/solar forecasts
- **Degradation model:** Quadratic `|delta_soc|^2 * cost` + calendar aging — convex, Markovian, enables exact oracle via QP
- **Efficiency:** Symmetric eta=0.92 per direction (~84.6% round-trip)
- **Oracle:** Convex QP via CVXPY + CLARABEL solver (no binary variables needed — eta < 1 prevents simultaneous charge/discharge)

## Setup

```bash
# Clone and install
git clone https://github.com/balazsthomay/battery-dispatch-rl.git
cd battery-dispatch-rl
uv sync

# Run tests
uv run pytest

# Quick demo with sample data (no API key needed)
uv run bess-dispatch baselines --use-sample
uv run bess-dispatch train --algorithm DQN --use-sample --timesteps 5000
```

### With real ENTSO-E data

1. Register at [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and request API access
2. Create `.env` with your key:
   ```
   ENTSOE_API_KEY=your-token-here
   ```
3. Download data and run experiments:
   ```bash
   # Download price data
   bess-dispatch download --zone DE_LU --year 2023
   bess-dispatch download --zone DE_LU --year 2024

   # Train agents
   bess-dispatch train --algorithm SAC --zone DE_LU --year 2023 --timesteps 100000
   bess-dispatch train --algorithm DQN --zone DE_LU --year 2023 --timesteps 100000

   # Evaluate on held-out 2024 data
   bess-dispatch evaluate --model results/models/sac_de_lu/sac_model --algorithm SAC --year 2024

   # Run baselines
   bess-dispatch baselines --zone DE_LU --year 2024

   # Generate full report
   bess-dispatch report
   ```

## Technical Details

**Battery model:** 1 MW / 1 MWh lithium-ion BESS with:
- Round-trip efficiency: ~84.6% (0.92 per direction)
- Degradation: quadratic in cycle depth (`|delta_soc|^2 * 50 EUR`) + calendar aging
- SoC bounds: 0-100%, initial SoC: 50%

**Training:** 100K timesteps on DE_LU 2023, strict temporal split (eval on 2024). 4 parallel environments with VecNormalize. DQN uses 5 discrete actions {-1, -0.5, 0, 0.5, 1}; SAC uses continuous actions.

**Evaluation:** 10 random weekly episodes per zone, seeded for reproducibility.

## Requirements

- Python >= 3.12
- Key dependencies: gymnasium, stable-baselines3, cvxpy, entsoe-py, pandas, numpy, matplotlib
