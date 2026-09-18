"""Deterministic guardrails.

LLM output is untrusted structured data until it passes every check here
(Problem Statement section 08). Anything that fails is downgraded to no_op
rather than being passed to the optimizer or crashing the service.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .schemas import Battery, DirectiveInterpretation

REQUIRED_SHAPE: Dict[str, Tuple[str, ...]] = {
    "solar_reduction": ("hours", "factor"),
    "minimum_battery_reserve": ("hours", "minimum_energy_kwh"),
    "no_charge_window": ("hours",),
    "no_discharge_window": ("hours",),
    "max_grid_window": ("hours", "max_grid_kwh"),
}


class GuardrailError(ValueError):
    """Raised when a candidate directive violates a section-08 guardrail."""


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardrailError("numeric value is not a number")
    f = float(value)
    if not math.isfinite(f):
        raise GuardrailError("numeric value is not finite")
    return f


def clean_hours(raw: Any) -> List[int]:
    """Unique integers 0..23 in ascending order, per section 08."""
    if not isinstance(raw, (list, tuple)):
        raise GuardrailError("hours must be a list")
    out = set()
    for item in raw:
        if isinstance(item, bool):
            raise GuardrailError("hours entries must be integers")
        if isinstance(item, float):
            if not float(item).is_integer():
                raise GuardrailError("hours entries must be whole numbers")
            item = int(item)
        if isinstance(item, str) and item.strip().lstrip("-").isdigit():
            item = int(item.strip())
        if not isinstance(item, int):
            raise GuardrailError("hours entries must be integers")
        if not 0 <= item <= 23:
            raise GuardrailError("hours entries must be within 0..23")
        out.add(item)
    if not out:
        raise GuardrailError("hours must not be empty")
    return sorted(out)


def validate_adjustment(
    directive_type: str, adjustment: Any, battery: Battery
) -> Optional[Dict[str, Any]]:
    """Return a canonical structured_adjustment, or raise GuardrailError."""
    if directive_type == "no_op":
        return None
    if directive_type not in REQUIRED_SHAPE:
        raise GuardrailError(f"unsupported directive_type {directive_type!r}")
    if not isinstance(adjustment, dict):
        raise GuardrailError("structured_adjustment must be an object")

    hours = clean_hours(adjustment.get("hours"))
    canonical: Dict[str, Any] = {"hours": hours}

    if directive_type == "solar_reduction":
        factor = _finite(adjustment.get("factor"))
        if not 0.0 <= factor <= 1.0:
            raise GuardrailError("factor must be between 0 and 1 inclusive")
        canonical["factor"] = factor

    elif directive_type == "minimum_battery_reserve":
        reserve = _finite(adjustment.get("minimum_energy_kwh"))
        if reserve < 0:
            raise GuardrailError("minimum_energy_kwh must be non-negative")
        if reserve > battery.capacity_kwh:
            raise GuardrailError("minimum_energy_kwh exceeds battery capacity")
        canonical["minimum_energy_kwh"] = reserve

    elif directive_type == "max_grid_window":
        cap = _finite(adjustment.get("max_grid_kwh"))
        if cap < 0:
            raise GuardrailError("max_grid_kwh must be non-negative")
        canonical["max_grid_kwh"] = cap

    return canonical


NO_OP_EXPLANATION = "This note does not affect today's 24-hour energy schedule."


def no_op_entry(note_index: int, explanation: str = NO_OP_EXPLANATION) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation=explanation or NO_OP_EXPLANATION,
    )


def build_interpretation(
    note_index: int,
    directive_type: Any,
    adjustment: Any,
    explanation: Any,
    battery: Battery,
) -> DirectiveInterpretation:
    """Guardrail one candidate directive into a valid interpretation entry.

    Never raises: an unusable candidate becomes no_op, which is always a legal
    (if unscored) answer, so a bad model response can never invent a constraint.
    """
    text = explanation if isinstance(explanation, str) and explanation.strip() else ""
    if not isinstance(directive_type, str) or directive_type == "no_op":
        return no_op_entry(note_index, text)
    try:
        canonical = validate_adjustment(directive_type, adjustment, battery)
    except GuardrailError:
        return no_op_entry(note_index, NO_OP_EXPLANATION)
    return DirectiveInterpretation(
        note_index=note_index,
        applies=True,
        directive_type=directive_type,
        structured_adjustment=canonical,
        explanation=text or f"Interpreted as a {directive_type.replace('_', ' ')} directive.",
    )


def check_entry_invariants(entries: List[DirectiveInterpretation], note_count: int) -> None:
    """Final contract assertions before the response leaves the service."""
    if len(entries) != note_count:
        raise GuardrailError("one interpretation entry is required per operator note")
    for position, entry in enumerate(entries):
        if entry.note_index != position:
            raise GuardrailError("entries must be in note_index order 0..N-1")
        if entry.directive_type == "no_op":
            if entry.applies or entry.structured_adjustment is not None:
                raise GuardrailError("no_op must use applies=false and a null adjustment")
        else:
            if not entry.applies or entry.structured_adjustment is None:
                raise GuardrailError("non-no_op directives must use applies=true")
