"""Rule-based operator-note interpreter.

IMPORTANT: this is NOT the primary interpreter. The Participant Guide is explicit
that hard-coded phrase matching as the sole interpreter is non-compliant, and the
language model is the required interpretation path. This module exists only as a
safe-failure net for when every model provider errors, times out, or returns
output that cannot pass the guardrails, so that the service degrades into a
controlled answer instead of a 5xx.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .schemas import Battery

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

FRACTION_WORDS = {
    "one-half": 0.5, "one half": 0.5, "a half": 0.5, "half": 0.5,
    "one-third": 1 / 3, "one third": 1 / 3, "a third": 1 / 3,
    "one-quarter": 0.25, "one quarter": 0.25, "a quarter": 0.25,
    "one-fifth": 0.2, "one fifth": 0.2, "a fifth": 0.2,
    "one-tenth": 0.1, "one tenth": 0.1, "a tenth": 0.1,
}

# The stated share is what is REMOVED.
REDUCTION_CUES = ("reduction", "reduce", "reduced by", "drop by", "decrease", "lower by", "cut by", "loss of")

# The stated share is what REMAINS.
REMAINING_CUES = (
    "drop to", "drops to", "fall to", "falls to", "down to", "treated as", "treat",
    "leave", "leaving", "remain", "of the forecast", "of normal", "of forecast",
    "of expected", "of the expected", "output of",
)

RANGE_SEPARATORS = ("until", "till", "til", "through", "thru", "to", "and", "-", "–", "—")

SOLAR_WORDS = ("solar", "pv", "panel", "photovoltaic", "rooftop", "inverter")
GRID_WORDS = ("grid", "import", "intake", "feeder", "substation", "transformer", "draw")

# A "time token" is (start_offset, hour_value, has_explicit_meridiem)
TimeToken = Tuple[int, int, bool]


def _to_24h(value: int, meridiem: str) -> int:
    if meridiem == "pm":
        return (value % 12) + 12
    return value % 12


def _parse_time_tokens(text: str) -> List[TimeToken]:
    tokens: List[TimeToken] = []
    blocked: List[Tuple[int, int]] = []  # spans already claimed by a richer match

    def claim(span: Tuple[int, int]) -> None:
        blocked.append(span)

    for match in re.finditer(r"\bnoon\b|\bmidday\b", text):
        tokens.append((match.start(), 12, True))
        claim(match.span())
    for match in re.finditer(r"\bmidnight\b", text):
        tokens.append((match.start(), 0, True))
        claim(match.span())

    # 24-hour clock: 13:00, 09:30
    for match in re.finditer(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text):
        tokens.append((match.start(), int(match.group(1)), True))
        claim(match.span())

    # 12-hour clock: 2 PM, 2pm, 2 p.m.
    for match in re.finditer(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)", text):
        meridiem = "am" if match.group(2).startswith("a") else "pm"
        tokens.append((match.start(), _to_24h(int(match.group(1)), meridiem), True))
        claim(match.span())

    def is_blocked(span: Tuple[int, int]) -> bool:
        return any(span[0] < end and start < span[1] for start, end in blocked)

    # Bare digits. Reject anything that is plainly a quantity rather than a clock
    # reading: percentages, kWh amounts, and the minutes half of a clock time.
    for match in re.finditer(r"(?<![\d:.])(\d{1,2})(?![\d:])", text):
        if is_blocked(match.span()):
            continue
        trailing = text[match.end():match.end() + 12]
        if re.match(r"\s*(%|percent|kwh|kw\b|kilowatt|bdt|taka)", trailing):
            continue
        value = int(match.group(1))
        if 0 <= value <= 23:
            tokens.append((match.start(), value, False))

    for word, value in WORD_NUMBERS.items():
        for match in re.finditer(rf"\b{word}\b", text):
            if is_blocked(match.span()):
                continue
            trailing = text[match.end():match.end() + 12]
            if re.match(r"\s*(%|percent|kwh|kw\b|hour)", trailing):
                continue
            tokens.append((match.start(), value, False))

    tokens.sort(key=lambda t: t[0])
    return tokens


def _separator_between(text: str, left: TimeToken, right: TimeToken) -> bool:
    """Is there a range word between these two time tokens?"""
    gap = text[left[0]:right[0]].lower()
    # Strip the left token's own words so "two" in "two until three" is not a separator.
    if len(gap) > 40:
        return False
    for separator in RANGE_SEPARATORS:
        if separator in ("-", "–", "—"):
            if separator in gap:
                return True
        elif re.search(rf"\b{re.escape(separator)}\b", gap):
            return True
    return False


def extract_window(text: str) -> Optional[List[int]]:
    """Best-effort start-inclusive / end-exclusive hour window."""
    lowered = text.lower()
    tokens = _parse_time_tokens(lowered)
    if len(tokens) < 2:
        return None

    start_token = end_token = None
    for left, right in zip(tokens, tokens[1:]):
        if _separator_between(lowered, left, right):
            start_token, end_token = left, right
            break
    if start_token is None or end_token is None:
        start_token, end_token = tokens[0], tokens[1]

    start_hour, end_hour = start_token[1], end_token[1]
    afternoon_context = any(w in lowered for w in ("evening", "afternoon", "tonight", "pm"))

    # "1-3 PM": the trailing meridiem governs the bare start too.
    if not start_token[2] and end_token[2] and end_hour >= 13 and start_hour < 12:
        if start_hour + 12 < end_hour or start_hour >= end_hour:
            start_hour += 12

    # Neither end carries a meridiem: "from one until three".
    if not start_token[2] and not end_token[2]:
        implied_pm = afternoon_context or any(w in lowered for w in SOLAR_WORDS)
        if implied_pm and start_hour < 12 and end_hour <= 12:
            start_hour += 12
            end_hour += 12

    if not (0 <= start_hour <= 23 and 0 <= end_hour <= 24):
        return None
    if end_hour <= start_hour:
        return None
    return list(range(start_hour, min(end_hour, 24)))


def _percentage(text: str) -> Optional[float]:
    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", text)
    if match:
        return float(match.group(1)) / 100.0
    for phrase, value in FRACTION_WORDS.items():
        if phrase in text:
            return value
    return None


def _kwh_value(text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*kwh", text)
    return float(match.group(1)) if match else None


def interpret_note(note: str, battery: Battery) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """Return (directive_type, structured_adjustment, explanation)."""
    text = note.lower()
    window = extract_window(text)
    negated = any(w in text for w in (
        "not", "no ", "disabl", "unavail", "isolat", "prohibit", "block", "offline", "off-line", "cannot", "avoid",
    ))

    # Order matters: the substring "charge" also appears inside "discharge".
    if window and "discharg" in text and negated:
        return "no_discharge_window", {"hours": window}, "Battery discharging is unavailable during the stated window."

    if window and re.search(r"\bcharg", text) and "discharg" not in text and negated:
        return "no_charge_window", {"hours": window}, "Battery charging is unavailable during the stated window."

    if window and any(w in text for w in SOLAR_WORDS):
        share = _percentage(text)
        if share is not None:
            remaining = share
            if any(cue in text for cue in REDUCTION_CUES) and not any(cue in text for cue in REMAINING_CUES):
                remaining = max(0.0, 1.0 - share)
            return (
                "solar_reduction",
                {"hours": window, "factor": round(min(max(remaining, 0.0), 1.0), 6)},
                "Usable solar is reduced during the stated window.",
            )

    reserve_cues = ("at least", "keep", "reserve", "remain in the battery", "stored in the battery", "requires")
    if window and any(w in text for w in reserve_cues) and "batter" in text or (window and "reserve" in text):
        reserve = _kwh_value(text)
        if reserve is None:
            share = _percentage(text)
            if share is not None and "capacit" in text:
                reserve = share * battery.capacity_kwh
        if reserve is not None and window:
            return (
                "minimum_battery_reserve",
                {"hours": window, "minimum_energy_kwh": round(min(reserve, battery.capacity_kwh), 6)},
                "A raised battery reserve is required during the stated window.",
            )

    cap_cues = ("not exceed", "at or below", "no more than", "must stay", "cap", "limit", "maximum", "max")
    if window and any(w in text for w in GRID_WORDS) and any(w in text for w in cap_cues):
        cap = _kwh_value(text)
        if cap is not None:
            return (
                "max_grid_window",
                {"hours": window, "max_grid_kwh": round(cap, 6)},
                "Grid import is capped during the stated window.",
            )

    return "no_op", None, "This note does not affect today's 24-hour energy schedule."


def interpret_all(notes: List[str], battery: Battery) -> List[Tuple[str, Optional[Dict[str, Any]], str]]:
    return [interpret_note(note, battery) for note in notes]
