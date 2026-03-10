"""Pure battery physics functions — no side effects."""

from __future__ import annotations

from dataclasses import dataclass

from bess_dispatch.config import BatteryConfig


@dataclass
class BatteryState:
    """Mutable battery state."""

    soc: float  # State of charge [0, 1] as fraction of capacity
    total_degradation: float = 0.0
    total_revenue: float = 0.0
    total_cycles: float = 0.0


def apply_action(
    state: BatteryState,
    action: float,
    price: float,
    config: BatteryConfig,
    dt: float = 1.0,
) -> tuple[BatteryState, float, dict]:
    """Apply a charge/discharge action to the battery.

    Action convention:
      - action > 0: DISCHARGE (sell energy, earn revenue)
      - action < 0: CHARGE (buy energy, spend money)
      - action = 0: IDLE

    Efficiency convention (symmetric, per direction):
      - Charging: energy_stored = |power_from_grid| * eta * dt
      - Discharging: energy_to_grid = power_from_battery * eta * dt

    Returns:
        (new_state, reward, info_dict) where reward = revenue - degradation
    """
    # Clamp action to [-1, 1]
    action = max(-1.0, min(1.0, action))

    # Raw power (MW): positive = discharge, negative = charge
    raw_power = action * config.max_power_mw

    # Clamp power to respect SoC bounds
    if raw_power < 0:
        # Charging: energy stored = |power| * eta * dt
        # Max storable energy (MWh) before hitting max_soc
        max_energy_storable = (config.max_soc - state.soc) * config.capacity_mwh
        # Max power from grid that would store that energy
        if config.efficiency > 0:
            max_charge_power = max_energy_storable / (config.efficiency * dt)
        else:
            max_charge_power = 0.0
        # Clamp |raw_power| to max_charge_power (raw_power is negative)
        actual_power = max(raw_power, -max_charge_power)
    elif raw_power > 0:
        # Discharging: energy from battery = power * dt
        # Max extractable energy (MWh) before hitting min_soc
        max_energy_extractable = (state.soc - config.min_soc) * config.capacity_mwh
        max_discharge_power = max_energy_extractable / dt if dt > 0 else 0.0
        actual_power = min(raw_power, max_discharge_power)
    else:
        actual_power = 0.0

    # Compute SoC change and revenue
    if actual_power < 0:
        # Charging
        power_from_grid = abs(actual_power)
        energy_stored = power_from_grid * config.efficiency * dt
        soc_delta = energy_stored / config.capacity_mwh
        revenue = -(power_from_grid * dt * price)
    elif actual_power > 0:
        # Discharging
        power_from_battery = actual_power
        energy_to_grid = power_from_battery * config.efficiency * dt
        soc_delta = -(power_from_battery * dt / config.capacity_mwh)
        revenue = energy_to_grid * price
    else:
        soc_delta = 0.0
        revenue = 0.0

    # Update SoC
    new_soc = state.soc + soc_delta

    # Degradation: cycling + calendar
    cycle_degradation = abs(soc_delta) ** config.degradation_k * config.degradation_cost
    calendar_degradation = config.calendar_aging_per_hour * dt
    total_step_degradation = cycle_degradation + calendar_degradation

    # Cycles: |delta_soc| / 2 (full cycle = 0 -> 1 -> 0 = 2 half-cycles)
    cycles_added = abs(soc_delta) / 2.0

    # Reward
    reward = revenue - total_step_degradation

    # Build new state (do not mutate original)
    new_state = BatteryState(
        soc=new_soc,
        total_degradation=state.total_degradation + total_step_degradation,
        total_revenue=state.total_revenue + revenue,
        total_cycles=state.total_cycles + cycles_added,
    )

    info = {
        "revenue": revenue,
        "degradation": total_step_degradation,
        "soc": new_soc,
        "soc_delta": soc_delta,
        "actual_power": actual_power,
        "cycle_degradation": cycle_degradation,
        "calendar_degradation": calendar_degradation,
    }

    return new_state, reward, info
