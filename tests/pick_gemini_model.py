"""Find which Gemini flash model works with our structured-output request.

Tries each candidate with the real RESPONSE_SCHEMA and reports status and
latency, so the fastest working model can be pinned in .env.
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

from app.llm import GEMINI_URL, RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from app.schemas import Battery

CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
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


def main() -> None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set")

    body_base = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": build_user_prompt(NOTES, BATTERY)}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }

    for model in CANDIDATES:
        started = time.perf_counter()
        try:
            reply = httpx.post(
                GEMINI_URL.format(model=model),
                json=body_base,
                headers={"x-goog-api-key": key},
                timeout=30,
            )
        except httpx.HTTPError as exc:
            print(f"{model:<26} network error {type(exc).__name__}")
            continue
        elapsed = time.perf_counter() - started

        if reply.status_code != 200:
            message = ""
            try:
                message = reply.json()["error"]["message"][:80]
            except Exception:  # noqa: BLE001
                message = reply.text[:80]
            print(f"{model:<26} {reply.status_code}  {message}")
            continue

        try:
            text = reply.json()["candidates"][0]["content"]["parts"][0]["text"]
            items = json.loads(text)["interpretations"]
            got = [(i.get("directive_type"), i.get("hours")) for i in items]
            correct = all(
                got[n][0] == EXPECTED[n][0] and (got[n][1] or None) == EXPECTED[n][1]
                for n in range(min(len(got), len(EXPECTED)))
            ) and len(got) == len(EXPECTED)
            verdict = "CORRECT" if correct else f"wrong: {got}"
            print(f"{model:<26} 200  {elapsed:5.2f}s  {verdict}")
        except Exception as exc:  # noqa: BLE001
            print(f"{model:<26} 200  {elapsed:5.2f}s  unparseable: {type(exc).__name__}")


if __name__ == "__main__":
    main()
