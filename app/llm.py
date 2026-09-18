"""Language-model interpretation of operator notes.

This is the mandatory LLM stage: the model reads the natural-language notes and
emits structured candidate directives. It never performs arithmetic and never
sees the schedule. Everything it returns is treated as untrusted input by
app.directives before it can reach the optimizer.

Two free-tier providers are supported behind one flat response schema:
  1. Groq   (OpenAI-compatible chat completions, forced tool call)
  2. Gemini (generateContent with a responseSchema)
Both are called over plain HTTP so the deployment has no heavyweight vendor SDK.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .schemas import Battery, DIRECTIVE_TYPES

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


class InterpretationUnavailable(RuntimeError):
    """No configured provider produced usable structured output."""


# A single flat schema keeps both providers on the same contract. Type-specific
# fields are optional here and are enforced afterwards by the guardrails, which
# is where shape validation belongs anyway.
ITEM_PROPERTIES: Dict[str, Any] = {
    "note_index": {"type": "integer", "description": "Zero-based index of the note being interpreted."},
    "directive_type": {"type": "string", "enum": list(DIRECTIVE_TYPES)},
    "hours": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Affected hours 0-23, start inclusive and end exclusive. Omit for no_op.",
    },
    "factor": {"type": "number", "description": "solar_reduction only: usable fraction of solar REMAINING (0-1)."},
    "minimum_energy_kwh": {"type": "number", "description": "minimum_battery_reserve only: required kWh floor."},
    "max_grid_kwh": {"type": "number", "description": "max_grid_window only: hourly grid import cap in kWh."},
    "explanation": {"type": "string", "description": "One short sentence."},
}

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": ITEM_PROPERTIES,
                "required": ["note_index", "directive_type", "explanation"],
            },
        }
    },
    "required": ["interpretations"],
}

SYSTEM_PROMPT = """You convert campus energy operator notes into structured directives.

You output structure only. You never compute a schedule, a cost, or any energy number that is not stated in the note.

Directive types, and nothing else is allowed:
- solar_reduction        - usable solar is reduced for some hours. Needs hours + factor.
- minimum_battery_reserve- battery must stay at or above a kWh level for some hours. Needs hours + minimum_energy_kwh.
- no_charge_window       - the battery cannot CHARGE during some hours. Needs hours.
- no_discharge_window    - the battery cannot DISCHARGE during some hours. Needs hours.
- max_grid_window        - grid import is capped for some hours. Needs hours + max_grid_kwh.
- no_op                  - the note does not affect today's 24-hour electricity schedule. No other fields.

HOUR WINDOWS. Always whole hours 0-23, start INCLUSIVE, end EXCLUSIVE.
  "1 PM to 3 PM"        -> [13, 14]
  "6 PM until 9 PM"     -> [18, 19, 20]
  "2 AM until 5 AM"     -> [2, 3, 4]
  "noon until 2 PM"     -> [12, 13]
  "10 AM until noon"    -> [10, 11]
  "between 13:00 and 15:00" -> [13, 14]
List every hour explicitly, ascending, no duplicates.

SOLAR FACTOR is the fraction that REMAINS, never the fraction removed.
  "drops to about 20%"          -> factor 0.2
  "an 80% reduction"            -> factor 0.2
  "roughly one-fifth of normal" -> factor 0.2
  "about half the forecast"     -> factor 0.5
  "treated as 25% of forecast"  -> factor 0.25

PERCENTAGES OF THE BATTERY resolve against the capacity given in the scenario.
  "keep at least 50% of battery capacity" with capacity 200 -> minimum_energy_kwh 100

CHARGE vs DISCHARGE. "charger isolated", "charging circuit unavailable", "do not charge" are
no_charge_window. "must not discharge", "no discharging during relay testing" are no_discharge_window.
Read carefully; they are different directives.

no_op covers anything not about today's electricity schedule: menus, bookings, deadlines,
notices, room changes, events next week. When a note is irrelevant, return no_op rather than
inventing an energy rule. Do not invent demand, tariff, solar or battery numbers.

Return exactly one entry for every note, using its zero-based note_index."""


def build_user_prompt(notes: List[str], battery: Battery) -> str:
    listed = "\n".join(f"[{i}] {note}" for i, note in enumerate(notes))
    return (
        f"Battery for this scenario: capacity {battery.capacity_kwh} kWh, "
        f"current base minimum reserve {battery.minimum_energy_kwh} kWh, "
        f"starting energy {battery.initial_energy_kwh} kWh.\n"
        f"Planning horizon: hours 0 to 23 of a single day.\n\n"
        f"Operator notes ({len(notes)} total):\n{listed}\n\n"
        f"Return exactly {len(notes)} interpretation entries, one per note, note_index 0 to {len(notes) - 1}."
    )


def _normalise(payload: Any, note_count: int) -> List[Tuple[str, Optional[Dict[str, Any]], str]]:
    """Flatten provider output into (directive_type, adjustment, explanation) per note.

    Missing or duplicate entries are filled with no_op so the caller always gets
    exactly one candidate per note, preserving the note_index contract.
    """
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, list):
        payload = {"interpretations": payload}
    if not isinstance(payload, dict):
        raise InterpretationUnavailable("model response was not an object")

    items = payload.get("interpretations") or payload.get("directive_interpretation") or []
    if not isinstance(items, list):
        raise InterpretationUnavailable("interpretations was not a list")

    slots: List[Optional[Tuple[str, Optional[Dict[str, Any]], str]]] = [None] * note_count
    for raw in items:
        if not isinstance(raw, dict):
            continue
        index = raw.get("note_index")
        if isinstance(index, str) and index.strip().isdigit():
            index = int(index)
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        if not 0 <= index < note_count or slots[index] is not None:
            continue

        directive_type = raw.get("directive_type")
        explanation = raw.get("explanation") or ""

        adjustment: Optional[Dict[str, Any]] = None
        if directive_type != "no_op":
            nested = raw.get("structured_adjustment")
            source = nested if isinstance(nested, dict) else raw
            adjustment = {"hours": source.get("hours")}
            if directive_type == "solar_reduction":
                adjustment["factor"] = source.get("factor")
            elif directive_type == "minimum_battery_reserve":
                adjustment["minimum_energy_kwh"] = source.get("minimum_energy_kwh")
            elif directive_type == "max_grid_window":
                adjustment["max_grid_kwh"] = source.get("max_grid_kwh")

        slots[index] = (directive_type, adjustment, explanation)

    if all(slot is None for slot in slots):
        raise InterpretationUnavailable("model returned no usable entries")

    return [slot or ("no_op", None, "") for slot in slots]


async def _call_groq(
    client: httpx.AsyncClient, notes: List[str], battery: Battery, timeout: float
) -> List[Tuple[str, Optional[Dict[str, Any]], str]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise InterpretationUnavailable("GROQ_API_KEY is not configured")

    tool = {
        "type": "function",
        "function": {
            "name": "report_directives",
            "description": "Report the structured interpretation of every operator note.",
            "parameters": RESPONSE_SCHEMA,
        },
    }
    body = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(notes, battery)},
        ],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": "report_directives"}},
    }

    response = await client.post(
        GROQ_URL,
        json=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if calls:
        return _normalise(calls[0]["function"]["arguments"], len(notes))
    return _normalise(message.get("content") or "", len(notes))


async def _call_gemini(
    client: httpx.AsyncClient, notes: List[str], battery: Battery, timeout: float
) -> List[Tuple[str, Optional[Dict[str, Any]], str]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise InterpretationUnavailable("GEMINI_API_KEY is not configured")

    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": build_user_prompt(notes, battery)}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    response = await client.post(
        GEMINI_URL.format(model=model),
        json=body,
        headers={"x-goog-api-key": api_key},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _normalise(text, len(notes))


PROVIDERS = (("groq", _call_groq), ("gemini", _call_gemini))


def configured_providers() -> List[str]:
    available = []
    if os.getenv("GROQ_API_KEY"):
        available.append(f"groq:{os.getenv('GROQ_MODEL', DEFAULT_GROQ_MODEL)}")
    if os.getenv("GEMINI_API_KEY"):
        available.append(f"gemini:{os.getenv('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)}")
    return available


async def interpret(
    notes: List[str], battery: Battery
) -> Tuple[List[Tuple[str, Optional[Dict[str, Any]], str]], str]:
    """Try each configured provider in order.

    Returns the candidate directives and the name of the provider that answered.
    Raises InterpretationUnavailable when every provider fails, which is the
    caller's signal to fall back to the deterministic interpreter.
    """
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
    errors: List[str] = []

    async with httpx.AsyncClient() as client:
        for name, call in PROVIDERS:
            try:
                return await call(client, notes, battery, timeout), name
            except InterpretationUnavailable as exc:
                errors.append(f"{name}: {exc}")
            except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
                # Never surface provider payloads or credentials to the caller.
                errors.append(f"{name}: {type(exc).__name__}")

    raise InterpretationUnavailable("; ".join(errors) or "no provider configured")
