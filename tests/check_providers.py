"""Connectivity check for the configured model providers.

Reports HTTP status and the provider's error message so a misconfiguration can
be diagnosed. Never prints key material.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import httpx

from app.llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    GEMINI_URL,
    GROQ_URL,
)


def redact(text: str) -> str:
    """Strip anything that looks like a key out of a provider error message."""
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        value = os.getenv(name)
        if value:
            text = text.replace(value, f"<{name}>")
    return text[:500]


def check_groq() -> None:
    key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    print(f"\n--- Groq ({model}) ---")
    if not key:
        print("  GROQ_API_KEY not set")
        return
    try:
        reply = httpx.post(
            GROQ_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "max_tokens": 5,
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
    except httpx.HTTPError as exc:
        print(f"  network error: {type(exc).__name__}")
        return
    print(f"  status: {reply.status_code}")
    if reply.status_code == 200:
        print(f"  reply : {reply.json()['choices'][0]['message']['content'][:60]!r}")
    else:
        print(f"  error : {redact(reply.text)}")


def check_gemini() -> None:
    key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    print(f"\n--- Gemini ({model}) ---")
    if not key:
        print("  GEMINI_API_KEY not set")
        return
    try:
        reply = httpx.post(
            GEMINI_URL.format(model=model),
            json={"contents": [{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}]},
            headers={"x-goog-api-key": key},
            timeout=20,
        )
    except httpx.HTTPError as exc:
        print(f"  network error: {type(exc).__name__}")
        return
    print(f"  status: {reply.status_code}")
    if reply.status_code == 200:
        text = reply.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"  reply : {text[:60]!r}")
    else:
        print(f"  error : {redact(reply.text)}")


def list_models() -> None:
    """Show which model ids each account can actually use."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        print("\n--- Groq available models ---")
        try:
            reply = httpx.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=20,
            )
            if reply.status_code == 200:
                ids = sorted(m["id"] for m in reply.json().get("data", []))
                for model_id in ids:
                    print(f"  {model_id}")
            else:
                print(f"  status {reply.status_code}: {redact(reply.text)}")
        except httpx.HTTPError as exc:
            print(f"  network error: {type(exc).__name__}")

    key = os.getenv("GEMINI_API_KEY")
    if key:
        print("\n--- Gemini available models ---")
        try:
            reply = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": key},
                timeout=20,
            )
            if reply.status_code == 200:
                for entry in reply.json().get("models", []):
                    if "generateContent" in entry.get("supportedGenerationMethods", []):
                        print(f"  {entry['name'].replace('models/', '')}")
            else:
                print(f"  status {reply.status_code}: {redact(reply.text)}")
        except httpx.HTTPError as exc:
            print(f"  network error: {type(exc).__name__}")


if __name__ == "__main__":
    check_groq()
    check_gemini()
    if "--models" in sys.argv:
        list_models()
