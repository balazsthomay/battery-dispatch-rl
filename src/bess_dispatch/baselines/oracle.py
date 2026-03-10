"""Perfect foresight oracle using convex optimization."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from bess_dispatch.config import BatteryConfig


@dataclass
class OracleResult:
    """Result of oracle optimization."""

    actions: np.ndarray  # optimal power levels per timestep [-1, 1]
    soc_trajectory: np.ndarray  # SoC at each timestep (length T+1)
    revenue: float
    degradation: float
    net_reward: float  # revenue - degradation
    status: str  # solver status


def solve_oracle(
    prices: np.ndarray,
    config: BatteryConfig | None = None,
    dt: float = 1.0,
) -> OracleResult:
    """Solve perfect-foresight dispatch via convex QP.

    Variables:
      - charge[t] >= 0: charging power (MW) at time t
      - discharge[t] >= 0: discharging power (MW) at time t
      - soc[t]: state of charge after step t

    Objective: maximize revenue - degradation
      revenue = sum(discharge[t] * eta * price[t] - charge[t] * price[t]) * dt
      degradation = sum(delta_soc[t]^k * degradation_cost) + calendar_aging * T

    Note: With eta < 1, simultaneous charge+discharge is never optimal
    (loses energy), so no binary variables needed.

    Args:
        prices: Array of electricity prices per timestep (EUR/MWh).
        config: Battery configuration. Uses defaults if None.
        dt: Timestep duration in hours.

    Returns:
        OracleResult with optimal dispatch schedule and metrics.
    """
    config = config or BatteryConfig()
    T = len(prices)

    # Variables
    charge = cp.Variable(T, nonneg=True)  # charging power (MW)
    discharge = cp.Variable(T, nonneg=True)  # discharging power (MW)
    soc = cp.Variable(T + 1)  # SoC trajectory

    # SoC change per step
    delta_soc_charge = charge * config.efficiency * dt / config.capacity_mwh
    delta_soc_discharge = discharge * dt / config.capacity_mwh

    # Revenue: sell at discharge*eta*dt*price, buy at charge*dt*price
    revenue = cp.sum(
        cp.multiply(discharge * config.efficiency * dt, prices)
        - cp.multiply(charge * dt, prices)
    )

    # Degradation: |delta_soc|^k * cost
    # Since charge, discharge >= 0, delta_soc per step is always non-negative
    delta_soc_total = delta_soc_charge + delta_soc_discharge

    if config.degradation_k == 2.0:
        degradation = config.degradation_cost * cp.sum(cp.square(delta_soc_total))
    else:
        # For other k, use power -- only works if k >= 1
        degradation = config.degradation_cost * cp.sum(
            cp.power(delta_soc_total, config.degradation_k)
        )

    calendar_deg = config.calendar_aging_per_hour * dt * T
    total_degradation = degradation + calendar_deg

    # Objective
    objective = cp.Maximize(revenue - total_degradation)

    # Constraints
    constraints = [
        soc[0] == config.initial_soc,
        charge <= config.max_power_mw,
        discharge <= config.max_power_mw,
    ]

    # SoC dynamics
    for t in range(T):
        constraints.append(
            soc[t + 1] == soc[t] + delta_soc_charge[t] - delta_soc_discharge[t]
        )

    # SoC bounds
    constraints += [
        soc >= config.min_soc,
        soc <= config.max_soc,
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return OracleResult(
            actions=np.zeros(T),
            soc_trajectory=np.full(T + 1, config.initial_soc),
            revenue=0.0,
            degradation=0.0,
            net_reward=0.0,
            status=problem.status,
        )

    # Build actions: positive=discharge, negative=charge (match env convention)
    actions = discharge.value - charge.value
    # Normalize to [-1, 1] range
    actions = actions / config.max_power_mw

    return OracleResult(
        actions=actions,
        soc_trajectory=soc.value,
        revenue=float(revenue.value),
        degradation=float(total_degradation.value),
        net_reward=float(problem.value),
        status=problem.status,
    )


class OraclePolicy:
    """Wraps oracle LP as a policy for the evaluation harness.

    NOTE: This is a non-causal policy -- it needs the full price series upfront.
    It precomputes optimal actions and replays them sequentially.
    """

    def __init__(
        self, prices: np.ndarray, config: BatteryConfig | None = None
    ) -> None:
        self.result = solve_oracle(prices, config)
        self._idx = 0

    def reset(self) -> None:
        """Reset the action index for a new episode."""
        self._idx = 0

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """Return precomputed action for current step.

        Args:
            obs: Observation (ignored -- actions are precomputed).
            deterministic: Ignored (always deterministic).

        Returns:
            (action, None) matching SB3 predict() interface.
        """
        if self._idx < len(self.result.actions):
            action = np.array([self.result.actions[self._idx]], dtype=np.float32)
            self._idx += 1
        else:
            action = np.array([0.0], dtype=np.float32)
        return action, None
