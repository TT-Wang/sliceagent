"""Eval cases. Each verifier is INDEPENDENT of whatever tests the agent writes for
itself (we check behavior directly), so the agent can't pass by gaming its own tests."""
from __future__ import annotations

import os
import subprocess

from .runner import EvalCase


def _py(workdir: str, code: str) -> subprocess.CompletedProcess:
    return subprocess.run(["python3", "-c", code], cwd=workdir, capture_output=True, text=True, timeout=30)


def _tail(p: subprocess.CompletedProcess) -> str:
    out = (p.stdout + p.stderr).strip()
    return out.splitlines()[-1] if out else "(no output)"


def _mkscratch(w: str) -> None:
    os.makedirs(os.path.join(w, "scratch"), exist_ok=True)


# ---- 1. build a small library (independent behavior check) ----

def _verify_strutils(w: str):
    code = (
        "import sys; sys.path.insert(0,'scratch'); import strutils as m\n"
        "assert m.reverse_string('abc')=='cba'\n"
        "assert m.is_palindrome('A man a plan a canal Panama')\n"
        "assert not m.is_palindrome('hello')\n"
        "assert m.count_vowels('Education')==5\n"
        "print('OK')"
    )
    p = _py(w, code)
    return ("OK" in p.stdout, _tail(p))


STRUTILS = EvalCase(
    name="strutils_build",
    prompt=(
        "Create scratch/strutils.py with three functions: "
        "reverse_string(s) returns s reversed; "
        "is_palindrome(s) returns True if s reads the same forwards and backwards, ignoring case and spaces; "
        "count_vowels(s) returns the number of vowels (aeiou, case-insensitive). "
        "Verify your work by running python, then give a one-line summary."
    ),
    setup=_mkscratch,
    verify=_verify_strutils,
)

# ---- 2. fix planted bugs until a provided test passes (test = the oracle) ----

_MATHLIB_BUGGY = (
    "def add(a, b):\n    return a + b\n\n"
    "def sub(a, b):\n    return a + b\n\n"        # BUG
    "def square(n):\n    return n * 2\n\n"         # BUG
    "def is_even(n):\n    return n % 2 == 1\n"     # BUG
)
_MATHLIB_TEST = (
    "from mathlib import add, sub, square, is_even\n"
    "assert add(2, 3) == 5\n"
    "assert sub(5, 2) == 3\n"
    "assert square(4) == 16\n"
    "assert is_even(4) is True\n"
    "assert is_even(3) is False\n"
    "print('ALL TESTS PASSED')\n"
)


def _setup_mathlib(w: str) -> None:
    d = os.path.join(w, "scratch")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "mathlib.py"), "w") as f:
        f.write(_MATHLIB_BUGGY)
    with open(os.path.join(d, "test_mathlib.py"), "w") as f:
        f.write(_MATHLIB_TEST)


def _verify_mathlib(w: str):
    d = os.path.join(w, "scratch")
    if open(os.path.join(d, "test_mathlib.py")).read() != _MATHLIB_TEST:
        return (False, "test file was modified (integrity fail)")
    p = subprocess.run(["python3", "test_mathlib.py"], cwd=d, capture_output=True, text=True, timeout=30)
    return ("ALL TESTS PASSED" in p.stdout, _tail(p))


MATHLIB = EvalCase(
    name="mathlib_debug",
    prompt=(
        "scratch/ contains mathlib.py (which has bugs) and test_mathlib.py. "
        "Fix the bugs in scratch/mathlib.py ONLY so the tests pass. "
        "Run `cd scratch && python3 test_mathlib.py`, fix failures, repeat until it prints ALL TESTS PASSED. "
        "Do not edit test_mathlib.py."
    ),
    setup=_setup_mathlib,
    verify=_verify_mathlib,
)

# ---- 3. multi-step logic: an arithmetic evaluator with precedence + error handling ----

def _verify_calc(w: str):
    code = (
        "import sys; sys.path.insert(0,'scratch')\n"
        "from calc import calc\n"
        "assert calc('2 + 3') == 5\n"
        "assert calc('2 + 3 * 4') == 14\n"
        "assert calc('(2 + 3) * 4') == 20\n"
        "raised = False\n"
        "try:\n    calc('1 / 0')\nexcept Exception:\n    raised = True\n"
        "assert raised, 'div by zero should raise'\n"
        "print('OK')"
    )
    p = _py(w, code)
    return ("OK" in p.stdout, _tail(p))


CALC = EvalCase(
    name="calc_eval",
    prompt=(
        "Create scratch/calc.py with a function calc(expr: str) that evaluates an arithmetic expression "
        "supporting + - * / with correct operator precedence and parentheses, and raises an exception on "
        "division by zero. Verify by running python, then give a one-line summary."
    ),
    setup=_mkscratch,
    verify=_verify_calc,
)


CASES = [STRUTILS, MATHLIB, CALC]
