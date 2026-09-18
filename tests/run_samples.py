"""Run the 10 public sample cases through the pipeline and score them.

Two modes:
    python -m tests.run_samples              # in-process, no HTTP server needed
    python -m tests.run_samples --url URL    # against a running/deployed service

It reports the two things the judge scores separately: whether the structured
interpretation matches the published ground truth, and whether the returned
schedule survives an independent replay against that ground truth.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.optimizer import build_constraints  # noqa: E402
from app.replay import replay, totals  # noqa: E402
from app.schemas import (  # noqa: E402
    DirectiveInterpretation,
    HourPlan,
    ScenarioRequest,
)

TOLERANCE = 0.01
GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def find_case_file() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matches = glob.glob(os.path.join(here, "*Sample_Cases*.json"))
    if not matches:
        raise SystemExit("Could not find the public sample-cases JSON next to the project root.")
    return matches[0]


def compare_interpretation(actual: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> List[str]:
    """Compare against ground truth. Free-text explanation is deliberately ignored."""
    problems: List[str] = []
    if len(actual) != len(expected):
        return [f"expected {len(expected)} entries, got {len(actual)}"]

    for position, (got, want) in enumerate(zip(actual, expected)):
        if got.get("note_index") != position:
            problems.append(f"note {position}: note_index out of order")
        if bool(got.get("applies")) != bool(want.get("applies")):
            problems.append(f"note {position}: applies {got.get('applies')} != {want.get('applies')}")
        if got.get("directive_type") != want.get("directive_type"):
            problems.append(
                f"note {position}: type {got.get('directive_type')!r} != {want.get('directive_type')!r}"
            )
            continue

        got_adj, want_adj = got.get("structured_adjustment"), want.get("structured_adjustment")
        if want_adj is None:
            if got_adj is not None:
                problems.append(f"note {position}: expected a null structured_adjustment")
            continue
        if not isinstance(got_adj, dict):
            problems.append(f"note {position}: missing structured_adjustment")
            continue
        if got_adj.get("hours") != want_adj.get("hours"):
            problems.append(f"note {position}: hours {got_adj.get('hours')} != {want_adj.get('hours')}")
        for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if key in want_adj:
                got_value = got_adj.get(key)
                if got_value is None or abs(float(got_value) - float(want_adj[key])) > TOLERANCE:
                    problems.append(f"note {position}: {key} {got_value} != {want_adj[key]}")
    return problems


def score_case(case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    request = ScenarioRequest.model_validate(case["input"])
    hours = request.ordered_hours()
    expected = case["expected_output"]

    interp_problems = compare_interpretation(
        response.get("directive_interpretation", []), expected["directive_interpretation"]
    )

    # Replay the returned plan against GROUND-TRUTH directives, exactly as the
    # judge does: a plan built on a mis-read note must fail here.
    truth_entries = [DirectiveInterpretation.model_validate(e) for e in expected["directive_interpretation"]]
    truth_constraints = build_constraints(hours, request.battery, truth_entries)

    plan_problems: List[str] = []
    try:
        plan = [HourPlan.model_validate(p) for p in response.get("hourly_plan", [])]
    except Exception as exc:  # noqa: BLE001
        return {
            "interpretation": interp_problems,
            "plan": [f"hourly_plan did not validate: {type(exc).__name__}"],
            "cost": None,
            "reference_cost": expected.get("total_cost_bdt"),
            "quality": 0.0,
        }

    plan_problems += replay(hours, request.battery, truth_constraints, plan)

    recomputed = totals(hours, plan)
    for key in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
        reported = response.get(key)
        if reported is None or abs(float(reported) - recomputed[key]) > TOLERANCE:
            plan_problems.append(f"{key} reported {reported} but replays as {recomputed[key]}")

    if response.get("scenario_id") != case["input"]["scenario_id"]:
        plan_problems.append("scenario_id was not echoed")

    reference = float(expected["total_cost_bdt"])
    team_cost = recomputed["total_cost_bdt"]
    if plan_problems:
        quality = 0.0
    elif team_cost <= TOLERANCE:
        quality = 1.0
    else:
        quality = min(1.0, reference / team_cost)

    return {
        "interpretation": interp_problems,
        "plan": plan_problems,
        "cost": team_cost,
        "reference_cost": reference,
        "quality": quality,
    }


async def call_in_process(payload: Dict[str, Any]) -> Dict[str, Any]:
    from app import service

    request = ScenarioRequest.model_validate(payload)
    response, _ = await service.run(request)
    return json.loads(response.model_dump_json())


def call_http(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    import httpx

    reply = httpx.post(url.rstrip("/") + "/optimize-energy", json=payload, timeout=timeout)
    reply.raise_for_status()
    return reply.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Base URL of a running service. Omit to run in-process.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--case", help="Run a single case id, e.g. SAMPLE-03")
    args = parser.parse_args()

    pack = json.load(open(find_case_file(), encoding="utf-8"))
    cases = pack["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]

    interpretation_passes = 0
    plan_passes = 0
    qualities: List[float] = []
    latencies: List[float] = []

    for case in cases:
        started = time.perf_counter()
        try:
            if args.url:
                response = call_http(args.url, case["input"], args.timeout)
            else:
                response = asyncio.run(call_in_process(case["input"]))
        except Exception as exc:  # noqa: BLE001
            print(f"{RED}FAIL{RESET} {case['id']:<10} request error: {type(exc).__name__}: {exc}")
            qualities.append(0.0)
            continue
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        result = score_case(case, response)
        interp_ok = not result["interpretation"]
        plan_ok = not result["plan"]
        interpretation_passes += interp_ok
        plan_passes += plan_ok
        qualities.append(result["quality"])

        badge = f"{GREEN}PASS{RESET}" if interp_ok and plan_ok else f"{RED}FAIL{RESET}"
        cost = result["cost"]
        cost_text = f"{cost:.2f}" if cost is not None else "n/a"
        delta = ""
        if cost is not None:
            diff = cost - result["reference_cost"]
            colour = GREEN if diff <= TOLERANCE else YELLOW
            delta = f" {colour}({diff:+.2f} vs reference){RESET}"

        print(
            f"{badge} {case['id']:<10} {case['label'][:34]:<34} "
            f"interp={'ok' if interp_ok else 'BAD'} plan={'ok' if plan_ok else 'BAD'} "
            f"cost={cost_text}{delta} {DIM}{elapsed:.2f}s{RESET}"
        )
        for problem in result["interpretation"][:4]:
            print(f"      {YELLOW}interp:{RESET} {problem}")
        for problem in result["plan"][:4]:
            print(f"      {RED}plan:{RESET}   {problem}")

    total = len(cases) or 1
    average_quality = sum(qualities) / total
    print()
    print(f"interpretation exact-match : {interpretation_passes}/{total}")
    print(f"schedule valid vs truth    : {plan_passes}/{total}")
    print(f"optimization quality       : {average_quality:.4f}  -> {10 * average_quality:.2f}/10 rubric points")
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        print(f"latency mean/p95           : {sum(latencies) / len(latencies):.2f}s / {p95:.2f}s")

    return 0 if interpretation_passes == total and plan_passes == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
