"""Run sample_request_small.json live and print the real response, for the README."""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from app import service
from app.schemas import ScenarioRequest


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    payload = json.load(open(os.path.join(here, "sample_request_small.json"), encoding="utf-8"))
    request = ScenarioRequest.model_validate(payload)
    response, diagnostics = asyncio.run(service.run(request))

    out_path = os.path.join(here, "tests", "_readme_response.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(response.model_dump_json(indent=2))
    sys.stderr.write(f"provider: {diagnostics['interpretation_source']}\n")
    sys.stderr.write(f"written to: {out_path}\n")


if __name__ == "__main__":
    main()
