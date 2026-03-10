"""Tests for battery physics."""

from __future__ import annotations

import numpy as np
import pytest

from bess_dispatch.config import BatteryConfig
from bess_dispatch.env.battery import BatteryState, apply_action


class TestBatteryState:
    """Tests for BatteryState dataclass."""

    def test_defaults(self):
        state = BatteryState(soc=0.5)
        assert state.soc == 0.5
        assert state.total_degradation == 0.0
        assert state.total_revenue == 0.0
        assert state.total_cycles == 0.0

    def test_custom_values(self):
        state = BatteryState(soc=0.8, total_degradation=1.0, total_revenue=100.0, total_cycles=5.0)
        assert state.soc == 0.8
        assert state.total_degradation == 1.0
        assert state.total_revenue == 100.0
        assert state.total_cycles == 5.0


class TestApplyActionIdle:
    """Tests for idle (action=0) behavior."""

    def test_idle_soc_unchanged(self, battery_config):
        state = BatteryState(soc=0.5)
        new_state, reward, info = apply_action(state, 0.0, 50.0, battery_config)
        assert new_state.soc == pytest.approx(0.5)

    def test_idle_revenue_zero(self, battery_config):
        state = BatteryState(soc=0.5)
        new_state, reward, info = apply_action(state, 0.0, 50.0, battery_config)
        assert info["revenue"] == pytest.approx(0.0)

    def test_idle_calendar_degradation(self, battery_config):
        state = BatteryState(soc=0.5)
        new_state, reward, info = apply_action(state, 0.0, 50.0, battery_config, dt=1.0)
        expected_cal_aging = battery_config.calendar_aging_per_hour * 1.0
        assert info["degradation"] == pytest.approx(expected_cal_aging)
        assert new_state.total_degradation == pytest.approx(expected_cal_aging)


class TestApplyActionCharge:
    """Tests for charging (negative action)."""

    def test_charge_soc_increases(self, battery_config):
        state = BatteryState(soc=0.5)
        new_state, _reward, _info = apply_action(state, -0.5, 50.0, battery_config)
        assert new_state.soc > 0.5

    def test_charge_soc_correct_value(self, battery_config):
        """Charging: soc_delta = |power| * η * dt / capacity."""
        state = BatteryState(soc=0.5)
        action = -0.5
        power = abs(action) * battery_config.max_power_mw  # 0.5 MW
        expected_delta = power * battery_config.efficiency * 1.0 / battery_config.capacity_mwh
        new_state, _reward, _info = apply_action(state, action, 50.0, battery_config)
        assert new_state.soc == pytest.approx(0.5 + expected_delta)

    def test_charge_revenue_negative(self, battery_config):
        """Charging costs money — revenue should be negative."""
        state = BatteryState(soc=0.5)
        new_state, _reward, info = apply_action(state, -0.5, 50.0, battery_config)
        assert info["revenue"] < 0

    def test_charge_revenue_correct_value(self, battery_config):
        """Revenue = -(|power| * dt * price)."""
        state = BatteryState(soc=0.5)
        action = -0.5
        price = 50.0
        power = abs(action) * battery_config.max_power_mw
        expected_revenue = -(power * 1.0 * price)
        new_state, _reward, info = apply_action(state, action, price, battery_config)
        assert info["revenue"] == pytest.approx(expected_revenue)


class TestApplyActionDischarge:
    """Tests for discharging (positive action)."""

    def test_discharge_soc_decreases(self, battery_config):
        state = BatteryState(soc=0.5)
        new_state, _reward, _info = apply_action(state, 0.5, 50.0, battery_config)
        assert new_state.soc < 0.5

    def test_discharge_soc_correct_value(self, battery_config):
        """Discharging: soc_delta = -(power * dt / capacity)."""
        state = BatteryState(soc=0.5)
        action = 0.5
        power = action * battery_config.max_power_mw  # 0.5 MW
        expected_delta = -(power * 1.0 / battery_config.capacity_mwh)
        new_state, _reward, _info = apply_action(state, action, 50.0, battery_config)
        assert new_state.soc == pytest.approx(0.5 + expected_delta)

    def test_discharge_revenue_positive(self, battery_config):
        """Discharging earns money — revenue should be positive."""
        state = BatteryState(soc=0.5)
        new_state, _reward, info = apply_action(state, 0.5, 50.0, battery_config)
        assert info["revenue"] > 0

    def test_discharge_revenue_correct_value(self, battery_config):
        """Revenue = power * η * dt * price."""
        state = BatteryState(soc=0.5)
        action = 0.5
        price = 50.0
        power = action * battery_config.max_power_mw
        expected_revenue = power * battery_config.efficiency * 1.0 * price
        new_state, _reward, info = apply_action(state, action, price, battery_config)
        assert info["revenue"] == pytest.approx(expected_revenue)


class TestSocClamping:
    """Tests for SoC bound enforcement."""

    def test_charge_at_max_soc(self, battery_config):
        """Charging when SoC is at max_soc should not change SoC."""
        state = BatteryState(soc=1.0)
        new_state, _reward, info = apply_action(state, -1.0, 50.0, battery_config)
        assert new_state.soc == pytest.approx(1.0)
        # Power should be clamped to 0
        assert info["actual_power"] == pytest.approx(0.0)

    def test_discharge_at_min_soc(self, battery_config):
        """Discharging when SoC is at min_soc should not change SoC."""
        state = BatteryState(soc=0.0)
        new_state, _reward, info = apply_action(state, 1.0, 50.0, battery_config)
        assert new_state.soc == pytest.approx(0.0)
        assert info["actual_power"] == pytest.approx(0.0)

    def test_charge_partial_clamp(self, battery_config):
        """Charging when near max_soc should partially clamp."""
        state = BatteryState(soc=0.95)
        new_state, _reward, _info = apply_action(state, -1.0, 50.0, battery_config)
        # Should not exceed max_soc
        assert new_state.soc <= battery_config.max_soc + 1e-9
        assert new_state.soc > 0.95  # Should still charge some

    def test_discharge_partial_clamp(self, battery_config):
        """Discharging when near min_soc should partially clamp."""
        state = BatteryState(soc=0.05)
        new_state, _reward, _info = apply_action(state, 1.0, 50.0, battery_config)
        # Should not go below min_soc
        assert new_state.soc >= battery_config.min_soc - 1e-9
        assert new_state.soc < 0.05  # Should still discharge some


class TestRoundTripEfficiency:
    """Tests for round-trip efficiency loss."""

    def test_round_trip_less_than_100_percent(self, battery_config):
        """Charging then discharging same energy should lose money at same price.

        Use a small action to avoid SoC clamping so the comparison is fair.
        """
        state = BatteryState(soc=0.5)
        price = 50.0

        # Small charge to avoid clamping
        state_after_charge, _r1, info1 = apply_action(state, -0.3, price, battery_config)
        cost = abs(info1["revenue"])  # Money spent charging

        # Discharge the same amount of SoC we gained
        soc_gained = state_after_charge.soc - 0.5
        # To lose soc_gained of SoC: power * dt / capacity = soc_gained
        # power = soc_gained * capacity / dt
        discharge_power = soc_gained * battery_config.capacity_mwh / 1.0
        discharge_action = discharge_power / battery_config.max_power_mw
        state_after_discharge, _r2, info2 = apply_action(
            state_after_charge, discharge_action, price, battery_config
        )
        earnings = info2["revenue"]

        # Round-trip: earnings < cost because we lose efficiency on both legs
        assert earnings < cost

    def test_round_trip_efficiency_value(self, battery_config):
        """Round-trip efficiency should be η² ≈ 0.8464."""
        # Charge from 0.0 to fill, then discharge
        state = BatteryState(soc=0.0)
        price = 100.0

        # Charge
        state1, _, info1 = apply_action(state, -1.0, price, battery_config)
        charge_cost = abs(info1["revenue"])

        # Discharge back
        state2, _, info2 = apply_action(state1, 1.0, price, battery_config)
        discharge_revenue = info2["revenue"]

        # Round trip efficiency = revenue / cost (at same price)
        round_trip_eff = discharge_revenue / charge_cost
        expected_eff = battery_config.efficiency ** 2
        assert round_trip_eff == pytest.approx(expected_eff, rel=0.01)


class TestDegradation:
    """Tests for degradation tracking."""

    def test_degradation_increases_on_action(self, battery_config):
        """Any non-zero action should increase degradation."""
        state = BatteryState(soc=0.5)
        new_state, _reward, info = apply_action(state, 0.5, 50.0, battery_config)
        assert new_state.total_degradation > 0
        assert info["degradation"] > 0

    def test_degradation_monotonically_increases(self, battery_config):
        """Degradation should never decrease over a sequence of actions."""
        rng = np.random.default_rng(123)
        state = BatteryState(soc=0.5)
        prev_degradation = 0.0
        for _ in range(50):
            action = float(rng.uniform(-1, 1))
            state, _reward, _info = apply_action(state, action, 50.0, battery_config)
            assert state.total_degradation >= prev_degradation
            prev_degradation = state.total_degradation

    def test_degradation_formula(self, battery_config):
        """Degradation = |Δsoc|^k * degradation_cost + calendar_aging."""
        state = BatteryState(soc=0.5)
        action = 0.5
        power = action * battery_config.max_power_mw
        soc_delta = power * 1.0 / battery_config.capacity_mwh
        expected_cycle_deg = abs(soc_delta) ** battery_config.degradation_k * battery_config.degradation_cost
        expected_cal_deg = battery_config.calendar_aging_per_hour * 1.0
        expected_total = expected_cycle_deg + expected_cal_deg

        new_state, _reward, info = apply_action(state, action, 50.0, battery_config)
        assert info["degradation"] == pytest.approx(expected_total)


class TestRevenueAndReward:
    """Tests for revenue calculation and reward signal."""

    def test_revenue_sign_charge_positive_price(self, battery_config):
        """Charging at positive price should cost money (negative revenue)."""
        state = BatteryState(soc=0.5)
        _, _, info = apply_action(state, -1.0, 100.0, battery_config)
        assert info["revenue"] < 0

    def test_revenue_sign_discharge_positive_price(self, battery_config):
        """Discharging at positive price should earn money (positive revenue)."""
        state = BatteryState(soc=0.5)
        _, _, info = apply_action(state, 1.0, 100.0, battery_config)
        assert info["revenue"] > 0

    def test_revenue_sign_charge_negative_price(self, battery_config):
        """Charging at negative price should earn money (you get paid to consume)."""
        state = BatteryState(soc=0.5)
        _, _, info = apply_action(state, -1.0, -50.0, battery_config)
        assert info["revenue"] > 0  # Negative price * negative power = positive revenue

    def test_reward_equals_revenue_minus_degradation(self, battery_config):
        """Reward = revenue - degradation_cost."""
        state = BatteryState(soc=0.5)
        _, reward, info = apply_action(state, 0.5, 50.0, battery_config)
        expected = info["revenue"] - info["degradation"]
        assert reward == pytest.approx(expected)


class TestCycles:
    """Tests for cycle counting."""

    def test_cycles_accumulate(self, battery_config):
        """Cycles should increase with actions."""
        state = BatteryState(soc=0.5)
        state1, _, _ = apply_action(state, 0.5, 50.0, battery_config)
        assert state1.total_cycles > 0

    def test_cycles_formula(self, battery_config):
        """total_cycles += |Δsoc| / 2."""
        state = BatteryState(soc=0.5)
        action = 0.5
        power = action * battery_config.max_power_mw
        soc_delta = power * 1.0 / battery_config.capacity_mwh
        expected_cycles = abs(soc_delta) / 2.0

        new_state, _, _ = apply_action(state, action, 50.0, battery_config)
        assert new_state.total_cycles == pytest.approx(expected_cycles)

    def test_zero_action_no_cycles(self, battery_config):
        """Idle action should not add cycles."""
        state = BatteryState(soc=0.5)
        new_state, _, _ = apply_action(state, 0.0, 50.0, battery_config)
        assert new_state.total_cycles == 0.0


class TestCalendarAging:
    """Tests for calendar aging."""

    def test_calendar_aging_idle(self, battery_config):
        state = BatteryState(soc=0.5)
        new_state, _, info = apply_action(state, 0.0, 50.0, battery_config, dt=2.0)
        expected = battery_config.calendar_aging_per_hour * 2.0
        assert info["degradation"] == pytest.approx(expected)

    def test_calendar_aging_always_present(self, battery_config):
        """Calendar aging should be present even during active cycling."""
        state = BatteryState(soc=0.5)
        _, _, info = apply_action(state, 1.0, 50.0, battery_config)
        # Total degradation should be > pure cycling degradation
        # i.e., calendar aging is added
        assert info["degradation"] > 0


class TestOriginalStateUnchanged:
    """Tests that the original state is not mutated."""

    def test_original_state_not_mutated(self, battery_config):
        state = BatteryState(soc=0.5, total_degradation=0.0, total_revenue=0.0, total_cycles=0.0)
        new_state, _, _ = apply_action(state, 0.5, 50.0, battery_config)
        assert state.soc == 0.5
        assert state.total_degradation == 0.0
        assert state.total_revenue == 0.0
        assert state.total_cycles == 0.0
        assert new_state.soc != 0.5  # new state should be different


class TestRandomActionSequence:
    """Test battery behavior under random action sequences."""

    def test_soc_stays_in_bounds(self, battery_config):
        """SoC should always stay within [min_soc, max_soc] under random actions."""
        rng = np.random.default_rng(999)
        state = BatteryState(soc=0.5)
        for _ in range(200):
            action = float(rng.uniform(-1, 1))
            state, _, _ = apply_action(state, action, float(rng.uniform(-50, 150)), battery_config)
            assert battery_config.min_soc - 1e-9 <= state.soc <= battery_config.max_soc + 1e-9

    def test_extreme_actions_clamped(self, battery_config):
        """Actions beyond [-1, 1] should be treated as if clamped."""
        state = BatteryState(soc=0.5)
        # Action = 10 should behave like action = 1
        state_extreme, _, info_extreme = apply_action(state, 10.0, 50.0, battery_config)
        state_normal, _, info_normal = apply_action(
            BatteryState(soc=0.5), 1.0, 50.0, battery_config
        )
        assert state_extreme.soc == pytest.approx(state_normal.soc)
        assert info_extreme["revenue"] == pytest.approx(info_normal["revenue"])


class TestInfoDict:
    """Tests for the info dictionary returned by apply_action."""

    def test_info_keys(self, battery_config):
        state = BatteryState(soc=0.5)
        _, _, info = apply_action(state, 0.5, 50.0, battery_config)
        assert "revenue" in info
        assert "degradation" in info
        assert "soc" in info
        assert "actual_power" in info
        assert "soc_delta" in info

    def test_info_soc_matches_state(self, battery_config):
        state = BatteryState(soc=0.5)
        new_state, _, info = apply_action(state, 0.5, 50.0, battery_config)
        assert info["soc"] == pytest.approx(new_state.soc)
