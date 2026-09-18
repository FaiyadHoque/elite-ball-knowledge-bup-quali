"""Measure the deterministic window parser against every test note.

If extract_window() is reliable on notes that carry a time range, it can be used
to normalise the hours a model returns, which the Participant Guide permits as
deterministic post-processing. This script decides whether that is safe.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.fallback import extract_window  # noqa: E402
from tests.test_adversarial import CASES as HARD_CASES  # noqa: E402
from tests.test_robustness import CASES as EASY_CASES  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def main() -> int:
    agree = disagree = absent = skipped = 0

    for note, directive_type, adjustment in list(EASY_CASES) + list(HARD_CASES):
        if adjustment is None:  # no_op notes carry no window to check
            window = extract_window(note.lower())
            if window is not None:
                print(f"{YELLOW}window found on a no_op note{RESET}: {note}\n      -> {window}")
            skipped += 1
            continue

        expected = adjustment["hours"]
        got = extract_window(note.lower())
        if got is None:
            absent += 1
            print(f"{YELLOW}NO WINDOW{RESET} {note}\n      expected {expected}")
        elif got == expected:
            agree += 1
        else:
            disagree += 1
            print(f"{RED}WRONG{RESET}    {note}\n      expected {expected}\n      got      {got}")

    total = agree + disagree + absent
    print(f"\nnotes with a time window : {total}")
    print(f"  {GREEN}parsed correctly{RESET}       : {agree}")
    print(f"  {RED}parsed incorrectly{RESET}     : {disagree}")
    print(f"  {YELLOW}no window extracted{RESET}    : {absent}")
    print(f"no_op notes skipped      : {skipped}")
    print(
        "\nSafe to use as a normaliser only if 'parsed incorrectly' is 0: "
        "a wrong window would overwrite a correct model answer."
    )
    return 0 if disagree == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
