"""Harder, previously unseen paraphrases.

tests/test_robustness.py covers wordings close to the public pack. These are
deliberately awkward: indirect verbs, fractions written as words, 24-hour clock
mixed with plain English, and distractors that mention energy vocabulary without
imposing any constraint. Used to choose between candidate models, since hidden
notes will not reuse the public phrasing.

    LIVE_LLM=1 python -m tests.test_adversarial
    LIVE_LLM=1 GROQ_MODEL=openai/gpt-oss-120b python -m tests.test_adversarial
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from app import fallback, llm  # noqa: E402
from app.directives import build_interpretation  # noqa: E402
from app.service import normalise_window_end  # noqa: E402
from app.schemas import Battery  # noqa: E402

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

BATTERY = Battery(
    capacity_kwh=200,
    initial_energy_kwh=80,
    minimum_energy_kwh=20,
    max_charge_kwh_per_hour=50,
    max_discharge_kwh_per_hour=50,
)

CASES: List[Tuple[str, str, Optional[Dict[str, Any]]]] = [
    # Fractions as words, indirect phrasing.
    ("Between 9 and 11 in the morning, expect only a third of the usual PV yield.",
     "solar_reduction", {"hours": [9, 10], "factor": 1 / 3}),
    ("Solar will be knocked down by three quarters from 11 AM until 1 PM.",
     "solar_reduction", {"hours": [11, 12], "factor": 0.25}),
    ("From 14:00 to 17:00 the inverter will run at 60% of rated output.",
     "solar_reduction", {"hours": [14, 15, 16], "factor": 0.6}),

    # Charge/discharge expressed without the words "charge" or "discharge".
    ("The battery must not be topped up between 1 AM and 4 AM.",
     "no_charge_window", {"hours": [1, 2, 3]}),
    ("Hold back all battery output during the 5 PM to 7 PM peak test.",
     "no_discharge_window", {"hours": [17, 18]}),

    # Grid cap with unusual vocabulary.
    ("Do not let the meter pull more than 175 kWh in any hour from 8 PM to 11 PM.",
     "max_grid_window", {"hours": [20, 21, 22], "max_grid_kwh": 175}),

    # Reserve as a fraction of "pack capacity".
    ("Maintain a floor of one quarter of pack capacity from 7 PM through 10 PM.",
     "minimum_battery_reserve", {"hours": [19, 20, 21], "minimum_energy_kwh": 50}),
    ("Ensure 100 kWh minimum in storage 6-9 PM.",
     "minimum_battery_reserve", {"hours": [18, 19, 20], "minimum_energy_kwh": 100}),

    # End-exclusive traps: "through", "to" and dashes all EXCLUDE the end hour,
    # even though English often reads "through" as inclusive.
    ("No battery discharge from 7 PM through 10 PM during the load test.",
     "no_discharge_window", {"hours": [19, 20, 21]}),
    ("Charging is offline 8 AM-11 AM for the inspection.",
     "no_charge_window", {"hours": [8, 9, 10]}),
    ("Grid import must not exceed 150 kWh from 6 PM through 9 PM.",
     "max_grid_window", {"hours": [18, 19, 20], "max_grid_kwh": 150}),
    ("Keep 70 kWh in the battery from 9 PM to 11 PM.",
     "minimum_battery_reserve", {"hours": [21, 22], "minimum_energy_kwh": 70}),

    # Distractors that use energy vocabulary but impose no constraint.
    ("Grid usage is completely unrestricted today.", "no_op", None),
    ("The solar dashboard will get a new login page next month.", "no_op", None),
    ("A battery vendor representative visits campus on Thursday.", "no_op", None),
    ("The IT team will patch the servers overnight.", "no_op", None),
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
        if value is None or abs(float(value) - float(want)) > 0.02:
            return False
    return True


def main() -> int:
    notes = [case[0] for case in CASES]

    # Hidden scenarios carry 1-3 notes, so interpret in batches of three rather
    # than one oversized request that no real request would ever look like.
    if os.getenv("LIVE_LLM"):
        candidates = []
        providers = set()
        for start in range(0, len(notes), 3):
            batch = notes[start:start + 3]
            try:
                batch_candidates, provider = asyncio.run(llm.interpret(batch, BATTERY))
                model = os.getenv("GROQ_MODEL") if provider == "groq" else os.getenv("GEMINI_MODEL")
                providers.add(f"{provider} ({model})")
            except llm.InterpretationUnavailable as exc:
                print(f"{RED}live provider unavailable:{RESET} {exc}")
                return 2
            candidates.extend(batch_candidates)
        print(f"provider: {', '.join(sorted(providers))} (batches of 3)")
    else:
        print("provider: deterministic-fallback")
        candidates = fallback.interpret_all(notes, BATTERY)

    # Build the entries then run the same deterministic normalisation the
    # service applies, so the test exercises the shipped path end to end.
    entries = [
        build_interpretation(index, candidate[0], candidate[1], candidate[2], BATTERY)
        for index, candidate in enumerate(candidates)
    ]
    if os.getenv("LIVE_LLM"):
        normalise_window_end(entries, notes)

    passed = 0
    for index, ((note, expected_type, expected_adjustment), entry) in enumerate(zip(CASES, entries)):
        ok = matches(entry, expected_type, expected_adjustment)
        passed += ok
        if not ok:
            print(f"{RED}FAIL{RESET} {note}")
            print(f"      want {expected_type} {expected_adjustment}")
            print(f"      got  {entry.directive_type} {entry.structured_adjustment}")

    colour = GREEN if passed == len(CASES) else RED
    print(f"\n{colour}{passed}/{len(CASES)}{RESET} hard paraphrases interpreted correctly")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
