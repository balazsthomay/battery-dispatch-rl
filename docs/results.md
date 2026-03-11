# Results Analysis

## Experiment Setup

- **Training data:** DE_LU (Germany-Luxembourg) day-ahead prices, full year 2023 (8,760 hours)
- **Evaluation data:** 2024 data across 4 European bidding zones (DE_LU, ES, NL, PL)
- **Battery:** 1 MW / 1 MWh, eta=0.92/direction, quadratic degradation (k=2, cost=50 EUR)
- **Episodes:** 168 hours (1 week), 10 random episodes per evaluation, seeded
- **Algorithms:** DQN (5 discrete actions), SAC (continuous), both 100K training steps

## Market Data Summary

| Zone  | Year | Hours | Price Range (EUR/MWh)   | Mean Price |
|-------|------|-------|-------------------------|------------|
| DE_LU | 2023 | 8,760 | [-500.00, 524.27]       | 95.17      |
| DE_LU | 2024 | 8,784 | [-135.45, 936.28]       | 78.51      |
| ES    | 2024 | 8,784 | [-2.00, 193.00]         | 63.04      |
| NL    | 2024 | 8,784 | [-200.00, 872.96]       | 77.28      |
| PL    | 2024 | 8,784 | [-61.64, 630.19]        | 96.26      |

DE_LU and NL have the most extreme negative prices (solar overproduction), creating strong charging opportunities. PL has the highest mean price and large positive spikes. ES has the narrowest spread.

## Strategy Comparison (DE_LU 2024)

| Strategy       | Revenue (EUR) | Degradation (EUR) | Net Reward (EUR) | Cycles | Ann. Revenue (EUR/yr) |
|----------------|---------------|--------------------|--------------------|--------|----------------------|
| Do Nothing     | 0.00          | 0.00               | 0.00               | 0.00   | 0                    |
| Threshold      | 587.39        | 63.45              | 523.94             | 6.83   | 3,063                |
| DQN            | 1,145.94      | 396.20             | 749.73             | 8.42   | 5,975                |
| SAC            | 2,835.08      | 982.85             | 1,852.23           | 28.80  | 14,783               |
| Oracle         | 4,976.38      | 1,385.57           | 3,590.81           | 64.98  | 25,948               |

*Totals across 10 weekly episodes (1,680 hours). Annualized revenue extrapolated from evaluation period.*

### Key Findings

1. **SAC outperforms DQN by 2.5x.** Continuous actions enable fine-grained power control, allowing SAC to manage the revenue-degradation tradeoff more precisely. DQN's coarse 5-action discretization limits its ability to modulate power levels.

2. **Degradation management is critical.** The default threshold strategy (full-power, 25/75 percentiles) actually *loses money* (-71 EUR/week) because quadratic degradation from aggressive cycling exceeds revenue. The tuned threshold uses only 10% power with wider percentiles (20/90), sacrificing revenue for dramatically lower degradation.

3. **SAC captures ~52% of oracle performance** on DE_LU (185 vs 359 EUR/week). The gap represents the value of perfect price foresight — knowing exactly when the best spreads occur.

4. **Oracle cycles aggressively (6.5 cycles/week)** because it knows which cycles are worth the degradation cost. SAC cycles less (2.9/week) but more selectively.

## Cross-Market Generalization

| Strategy  | DE_LU | ES   | NL    | PL    |
|-----------|-------|------|-------|-------|
| Threshold | 52.4  | 27.0 | 58.5  | 99.9  |
| DQN       | 75.0  | 45.5 | 119.8 | 122.8 |
| SAC       | 185.2 | 76.5 | 216.0 | 238.1 |
| Oracle    | 359.1 | 264.9| 419.2 | 472.5 |

*Mean reward EUR/week. All agents trained on DE_LU 2023 only.*

**The agents generalize without retraining.** SAC and DQN actually perform *better* on NL and PL than on DE_LU:
- **NL** has similar negative-price patterns to DE_LU (connected markets, solar dynamics) with even higher peak spreads
- **PL** has the highest price volatility (prices up to 630 EUR/MWh), creating large arbitrage opportunities
- **ES** shows the weakest performance — narrow price spreads (no negative prices beyond -2 EUR) and lower volatility limit opportunity

The consistent ranking (Oracle > SAC > DQN > Threshold > Do Nothing) across all markets validates that the learned policy captures general price-arbitrage patterns, not just DE_LU-specific quirks.

## SAC Behavior Analysis

The SAC agent's dispatch over a sample DE_LU week shows:

![SAC Dispatch](../results/report/sac_dispatch_de_lu.png)

- **Hours 0-50:** Mostly discharging — selling initial SoC during moderate-to-high prices
- **Hours 50-80:** Heavy charging (green bars) — prices drop to low levels, agent fills battery
- **Hours 80-100:** Strong discharge — selling stored energy at higher prices
- **Hours 100-168:** Mixed behavior with moderate price volatility — selective trading

The SoC trajectory confirms coherent energy management: the battery depletes, refills during cheap hours, then depletes again during peaks.

### Action Distribution

SAC uses the full continuous action range:
- Strong discharge (>0.5): 61% of time
- Light discharge (0-0.5): 21%
- Idle: 4%
- Charging: 14%

The heavy discharge bias reflects that starting at SoC=0.5 and selling stored energy is often profitable even without recharging, given the quadratic degradation cost of round-trip cycling.

## Revenue-Degradation Tradeoff

![Strategy Comparison](../results/report/strategy_comparison_de_lu.png)

The chart reveals the core tradeoff:
- **Threshold** minimizes degradation (63 EUR) but misses most revenue
- **SAC** generates high revenue (2,835 EUR) with substantial degradation (983 EUR), netting 1,852 EUR
- **Oracle** proves even more aggressive cycling is optimal with perfect information

Revenue-per-cycle: Threshold (86 EUR/cycle) > DQN (136 EUR/cycle) > SAC (98 EUR/cycle) > Oracle (77 EUR/cycle). More aggressive strategies earn less per cycle but make it up in volume.

## Limitations and What enspired Does Differently

### What this project does NOT capture

1. **Single market only.** This project trades day-ahead arbitrage. enspired trades simultaneously across day-ahead, intraday continuous, FCR, aFRR, and imbalance markets. The real optimization is cross-market capacity allocation — not captured here.

2. **No order book data.** Intraday continuous prices (not available from ENTSO-E) are where forecast errors create the highest-value short-term opportunities.

3. **Simplified degradation.** Real batteries have complex degradation functions: C-rate dependence, temperature effects, state-of-health feedback loops, calendar aging that depends on SoC level. Our quadratic model is a useful approximation but misses these interactions.

4. **No forecast uncertainty.** The agent observes true prices at decision time. Real trading requires acting on *forecasts* with confidence intervals — the value of information is fundamental to enspired's approach.

5. **Hourly resolution only.** enspired makes millisecond-level trading decisions. Higher temporal resolution changes the problem structure.

### What I'd build next with their data

- **Multi-market RL:** Extend the action space to include capacity allocation across wholesale + ancillary markets. The observation space would include order book features and reserve market prices.
- **Hierarchical policy:** Day-ahead capacity commitment at hourly resolution, intraday execution at 15-minute or finer resolution.
- **Distributional RL (IQN/QR-DQN):** Model return distributions to enable risk-aware dispatch — critical for managing warranty constraints.
- **Degradation-aware curriculum:** Train with progressively more realistic degradation models, starting simple and adding C-rate dependence and temperature.
