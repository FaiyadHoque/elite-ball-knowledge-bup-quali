"""Paraphrase and robustness checks.

The hidden set paraphrases the same directives, so the interpreter is tested
against rewordings rather than the public sentences. Run with:

    python -m tests.test_robustness            # deterministic interpreter only
    LIVE_LLM=1 python -m tests.test_robustness # exercise the configured provider
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import fallback, llm  # noqa: E402
from app.directives import build_interpretation  # noqa: E402
from app.schemas import Battery  # noqa: E402

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

BATTERY = Battery(
    capacity_kwh=200,
    initial_energy_kwh=80,
    minimum_energy_kwh=20,
    max_charge_kwh_per_hour=50,
    max_discharge_kwh_per_hour=50,
)

# (note, expected directive_type, expected structured_adjustment)
CASES: List[Tuple[str, str, Optional[Dict[str, Any]]]] = [
    # --- solar_reduction, every phrasing the statement hints at -------------
    ("Solar output will drop to about 20% from 1 PM to 3 PM.",
     "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
    ("PV production will drop to about 20% between 13:00 and 15:00.",
     "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.",
     "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
    ("Cloud cover will leave about half of the forecast solar output from 10 AM until noon.",
     "solar_reduction", {"hours": [10, 11], "factor": 0.5}),
    ("Treat usable solar as roughly 25% of the forecast from noon until 2 PM.",
     "solar_reduction", {"hours": [12, 13], "factor": 0.25}),

    # --- no_charge_window ---------------------------------------------------
    ("Do not charge the battery between 2 PM and 4 PM.",
     "no_charge_window", {"hours": [14, 15]}),
    ("The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance.",
     "no_charge_window", {"hours": [2, 3, 4]}),
    ("The charging circuit will be unavailable from 2 PM until 4 PM.",
     "no_charge_window", {"hours": [14, 15]}),

    # --- no_discharge_window ------------------------------------------------
    ("For protection testing, the battery must not discharge from 6 PM until 8 PM.",
     "no_discharge_window", {"hours": [18, 19]}),
    ("Do not discharge the battery from 5 PM until 7 PM during relay testing.",
     "no_discharge_window", {"hours": [17, 18]}),

    # --- minimum_battery_reserve -------------------------------------------
    ("Keep at least 120 kWh in reserve from 6 PM until 9 PM.",
     "minimum_battery_reserve", {"hours": [18, 19, 20], "minimum_energy_kwh": 120}),
    ("Keep at least 50% of the battery capacity stored from 6 PM until 9 PM for emergency operations.",
     "minimum_battery_reserve", {"hours": [18, 19, 20], "minimum_energy_kwh": 100}),
    ("The data center requires at least 80 kWh to remain in the battery from 6 PM until 10 PM.",
     "minimum_battery_reserve", {"hours": [18, 19, 20, 21], "minimum_energy_kwh": 80}),

    # --- max_grid_window ----------------------------------------------------
    ("From 6 PM until 9 PM, campus grid import must not exceed 155 kWh in any hour.",
     "max_grid_window", {"hours": [18, 19, 20], "max_grid_kwh": 155}),
    ("Grid intake must stay at or below 190 kWh from 7 PM until 10 PM while the substation is constrained.",
     "max_grid_window", {"hours": [19, 20, 21], "max_grid_kwh": 190}),
    ("The evening transformer limit is 180 kWh of grid import from 7 PM until 9 PM.",
     "max_grid_window", {"hours": [19, 20], "max_grid_kwh": 180}),

    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.",
     "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
    ("Rooftop generation is expected at 40 percent of forecast between 9 AM and 11 AM.",
     "solar_reduction", {"hours": [9, 10], "factor": 0.4}),
    ("Battery charging is disabled from 11 AM until 1 PM while technicians inspect the charger.",
     "no_charge_window", {"hours": [11, 12]}),
    ("Hold a floor of 60 kWh in the battery reserve from 8 PM until 11 PM.",
     "minimum_battery_reserve", {"hours": [20, 21, 22], "minimum_energy_kwh": 60}),
    ("Cap grid import at 200 kWh per hour from 6 PM until 8 PM.",
     "max_grid_window", {"hours": [18, 19], "max_grid_kwh": 200}),
    ("Maintenance staff will repaint the corridor on Saturday.", "no_op", None),

    # --- no_op distractors --------------------------------------------------
    ("The cafeteria menu changes tomorrow.", "no_op", None),
    ("The sports office moved next month's registration deadline.", "no_op", None),
    ("The library is extending book-return hours next week.", "no_op", None),
    ("A seminar room booking was moved to next week.", "no_op", None),
    ("The student affairs office will publish club notices tomorrow.", "no_op", None),
]


def matches(entry: Any, expected_type: str, expected_adjustment: Optional[Dict[str, Any]]) -> bool:
    if entry.directive_type != expected_type:
        return False
    if expected_adjustment is None:
        return entry.structured_adjustment is None and entry.applies is False
    got = entry.structured_adjustment or {}
    if got.get("hours") != expected_adjustment["hours"]:
        return False
    for key, want in expected_adjustment.items():
        if key == "hours":
            continue
        value = got.get(key)
        if value is None or abs(float(value) - float(want)) > 0.01:
            return False
    return True


async def interpret_live(notes: List[str]) -> List[Tuple[str, Optional[Dict[str, Any]], str]]:
    candidates, provider = await llm.interpret(notes, BATTERY)
    print(f"provider: {provider}")
    return candidates


def main() -> int:
    notes = [case[0] for case in CASES]

    if os.getenv("LIVE_LLM"):
        try:
            candidates = asyncio.run(interpret_live(notes))
        except llm.InterpretationUnavailable as exc:
            print(f"{RED}live provider unavailable:{RESET} {exc}")
            return 2
    else:
        print("provider: deterministic-fallback (set LIVE_LLM=1 to test the model path)")
        candidates = fallback.interpret_all(notes, BATTERY)

    passed = 0
    for index, ((note, expected_type, expected_adjustment), candidate) in enumerate(zip(CASES, candidates)):
        entry = build_interpretation(index, candidate[0], candidate[1], candidate[2], BATTERY)
        ok = matches(entry, expected_type, expected_adjustment)
        passed += ok
        if not ok:
            print(f"{RED}FAIL{RESET} {note}")
            print(f"      want {expected_type} {expected_adjustment}")
            print(f"      got  {entry.directive_type} {entry.structured_adjustment}")

    print(f"\n{GREEN if passed == len(CASES) else RED}{passed}/{len(CASES)}{RESET} paraphrases interpreted correctly")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
