"""Report Groq rate-limit headers per model, so limits can be compared."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import httpx

from app.llm import GROQ_URL

MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def main() -> None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set")
    for model in MODELS:
        reply = httpx.post(
            GROQ_URL,
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        headers = {k: v for k, v in reply.headers.items() if k.lower().startswith("x-ratelimit")}
        print(f"--- {model} (status {reply.status_code}) ---")
        for name in sorted(headers):
            print(f"   {name}: {headers[name]}")


if __name__ == "__main__":
    main()
