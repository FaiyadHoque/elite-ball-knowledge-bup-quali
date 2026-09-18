"""Find which Groq model works best with our forced tool call.

Reports status, latency and whether the structured interpretation is correct,
so the fastest accurate model can be pinned in .env.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import httpx

from app.llm import GROQ_URL, RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from app.schemas import Battery

CANDIDATES = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "groq/compound-mini",
    "groq/compound",
]

BATTERY = Battery(
    capacity_kwh=200,
    initial_energy_kwh=80,
    minimum_energy_kwh=20,
    max_charge_kwh_per_hour=50,
    max_discharge_kwh_per_hour=50,
)

NOTES = [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
    "Keep at least 50% of the battery capacity stored from 6 PM until 9 PM for emergency operations.",
    "The sports office moved next month's registration deadline.",
]

EXPECTED = [
    ("solar_reduction", [12, 13]),
    ("minimum_battery_reserve", [18, 19, 20]),
    ("no_op", None),
]

TOOL = {
    "type": "function",
    "function": {
        "name": "report_directives",
        "description": "Report the structured interpretation of every operator note.",
        "parameters": RESPONSE_SCHEMA,
    },
}


def evaluate(items: list) -> str:
    got = []
    for entry in items:
        source = entry.get("structured_adjustment") if isinstance(entry.get("structured_adjustment"), dict) else entry
        got.append((entry.get("directive_type"), source.get("hours")))
    if len(got) != len(EXPECTED):
        return f"wrong count: {got}"
    for index, (want_type, want_hours) in enumerate(EXPECTED):
        if got[index][0] != want_type or (got[index][1] or None) != want_hours:
            return f"wrong: {got}"
    return "CORRECT"


def main() -> None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(NOTES, BATTERY)},
    ]

    for model in CANDIDATES:
        for mode in ("tools", "json_object"):
            body = {"model": model, "temperature": 0, "messages": messages}
            if mode == "tools":
                body["tools"] = [TOOL]
                body["tool_choice"] = {"type": "function", "function": {"name": "report_directives"}}
            else:
                body["response_format"] = {"type": "json_object"}

            started = time.perf_counter()
            try:
                reply = httpx.post(
                    GROQ_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=40
                )
            except httpx.HTTPError as exc:
                print(f"{model:<24} {mode:<12} network error {type(exc).__name__}")
                continue
            elapsed = time.perf_counter() - started

            if reply.status_code != 200:
                try:
                    message = reply.json()["error"]["message"][:70]
                except Exception:  # noqa: BLE001
                    message = reply.text[:70]
                print(f"{model:<24} {mode:<12} {reply.status_code}  {message}")
                continue

            try:
                message = reply.json()["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                raw = calls[0]["function"]["arguments"] if calls else message.get("content")
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                items = parsed.get("interpretations") or parsed.get("directive_interpretation") or []
                print(f"{model:<24} {mode:<12} 200  {elapsed:5.2f}s  {evaluate(items)}")
            except Exception as exc:  # noqa: BLE001
                print(f"{model:<24} {mode:<12} 200  {elapsed:5.2f}s  unparseable: {type(exc).__name__}")


if __name__ == "__main__":
    main()
