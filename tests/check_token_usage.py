"""Measure tokens consumed per realistic request, to size the TPM budget."""
from __future__ import annotations

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

BATTERY = Battery(
    capacity_kwh=200, initial_energy_kwh=80, minimum_energy_kwh=20,
    max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50,
)
NOTES = [
    "Facilities will wash the rooftop solar panels from noon until 2 PM, leaving about 25% of forecast output.",
    "Keep at least 50% of the battery capacity stored from 6 PM until 9 PM.",
    "The sports office moved next month's registration deadline.",
]
TOOL = {
    "type": "function",
    "function": {
        "name": "report_directives",
        "description": "Report the structured interpretation of every operator note.",
        "parameters": RESPONSE_SCHEMA,
    },
}


def main() -> None:
    key = os.getenv("GROQ_API_KEY")
    for model in ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
        started = time.perf_counter()
        reply = httpx.post(
            GROQ_URL,
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(NOTES, BATTERY)},
                ],
                "tools": [TOOL],
                "tool_choice": {"type": "function", "function": {"name": "report_directives"}},
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=40,
        )
        elapsed = time.perf_counter() - started
        if reply.status_code != 200:
            print(f"{model:<24} status {reply.status_code}")
            continue
        usage = reply.json().get("usage", {})
        total = usage.get("total_tokens", 0)
        budget = 8000 // total if total else 0
        print(
            f"{model:<24} {elapsed:5.2f}s  prompt={usage.get('prompt_tokens'):>5} "
            f"completion={usage.get('completion_tokens'):>4} total={total:>5} "
            f"-> ~{budget} requests/min within the 8000 TPM cap"
        )


if __name__ == "__main__":
    main()
