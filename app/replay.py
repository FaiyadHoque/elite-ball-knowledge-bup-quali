"""Independent replay of a finished schedule.

This mirrors what the judge does in Problem Statement section 11: it ignores how
the plan was produced and re-checks it hour by hour against the scenario and the
directives. The service runs it on every response before returning, so an
invalid plan is caught here rather than on the leaderboard.
"""
from __future__ import annotations

from typing import List, Sequence

from .optimizer import ConstraintSet
from .schemas import Battery, HourEntry, HourPlan

TOLERANCE = 0.01


def replay(
    hours: Sequence[HourEntry],
    battery: Battery,
    cs: ConstraintSet,
    plan: Sequence[HourPlan],
) -> List[str]:
    """Return a list of violations. An empty list means the plan is valid."""
    problems: List[str] = []

    if len(plan) != 24 or sorted(p.hour for p in plan) != list(range(24)):
        problems.append("hourly_plan must contain exactly one entry for each hour 0..23")
        return problems

    ordered = sorted(plan, key=lambda p: p.hour)
    energy = battery.initial_energy_kwh

    for h, entry in enumerate(ordered):
        hour = hours[h]

        for label, value in (
            ("grid_kwh", entry.grid_kwh),
            ("solar_used_kwh", entry.solar_used_kwh),
            ("battery_kwh", entry.battery_kwh),
        ):
            if value < -TOLERANCE:
                problems.append(f"hour {h}: {label} is negative")
            if value != value or value in (float("inf"), float("-inf")):
                problems.append(f"hour {h}: {label} is not finite")

        if entry.battery_action == "idle" and abs(entry.battery_kwh) > TOLERANCE:
            problems.append(f"hour {h}: battery_kwh must be 0 when idle")
        if entry.battery_action == "charge":
            if entry.battery_kwh > battery.max_charge_kwh_per_hour + TOLERANCE:
                problems.append(f"hour {h}: charge exceeds max_charge_kwh_per_hour")
            if not cs.charge_allowed[h]:
                problems.append(f"hour {h}: charging inside a no_charge_window")
        if entry.battery_action == "discharge":
            if entry.battery_kwh > battery.max_discharge_kwh_per_hour + TOLERANCE:
                problems.append(f"hour {h}: discharge exceeds max_discharge_kwh_per_hour")
            if not cs.discharge_allowed[h]:
                problems.append(f"hour {h}: discharging inside a no_discharge_window")

        if entry.solar_used_kwh > cs.effective_solar[h] + TOLERANCE:
            problems.append(f"hour {h}: solar_used_kwh exceeds effective solar")

        cap = cs.grid_cap[h]
        if cap is not None and entry.grid_kwh > cap + TOLERANCE:
            problems.append(f"hour {h}: grid_kwh exceeds the max_grid_window cap")

        charge = entry.battery_kwh if entry.battery_action == "charge" else 0.0
        discharge = entry.battery_kwh if entry.battery_action == "discharge" else 0.0

        balance = entry.grid_kwh + entry.solar_used_kwh + discharge - (hour.demand_kwh + charge)
        if abs(balance) > TOLERANCE:
            problems.append(f"hour {h}: energy balance is off by {balance:.4f} kWh")

        energy = energy + charge - discharge
        if abs(energy - entry.battery_energy_after_kwh) > TOLERANCE:
            problems.append(f"hour {h}: battery_energy_after_kwh does not follow the action")
        if entry.battery_energy_after_kwh > battery.capacity_kwh + TOLERANCE:
            problems.append(f"hour {h}: battery energy exceeds capacity")
        if entry.battery_energy_after_kwh < cs.min_level[h] - TOLERANCE:
            problems.append(f"hour {h}: battery energy is below the active minimum reserve")
        energy = entry.battery_energy_after_kwh

    if abs(energy - battery.initial_energy_kwh) > TOLERANCE:
        problems.append("end-of-day battery energy does not return to initial_energy_kwh")

    return problems


def totals(hours: Sequence[HourEntry], plan: Sequence[HourPlan]) -> dict:
    """Recompute the reported totals from the plan, which is the source of truth."""
    ordered = sorted(plan, key=lambda p: p.hour)
    total_grid = sum(p.grid_kwh for p in ordered)
    total_cost = sum(p.grid_kwh * hours[p.hour].tariff_bdt_per_kwh for p in ordered)
    peak = max((p.grid_kwh for p in ordered), default=0.0)
    return {
        "total_grid_kwh": round(total_grid, 4),
        "total_cost_bdt": round(total_cost, 4),
        "peak_grid_kwh": round(peak, 4),
    }
