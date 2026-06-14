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


# ---- 4. REAL REPO: find & fix a bug in a multi-module package (exercises CodeIndex) ----
# A small but realistic package. The bug is in ONE module (reporting.py) among several
# distractors, so the agent must DISCOVER the right file (the RELATED CODE tier earns its
# keep) rather than create from scratch.

_LEDGER = {
    "ledger/__init__.py": (
        "from .transactions import Transaction\n"
        "from .accounts import Account\n"
        "from .reporting import balance, summary\n"
    ),
    "ledger/transactions.py": (
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\n"
        "class Transaction:\n"
        "    amount: float  # positive = credit, negative = debit\n"
        "    category: str = 'general'\n"
        "    note: str = ''\n"
    ),
    "ledger/accounts.py": (
        "class Account:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "        self.transactions = []\n\n"
        "    def add(self, txn):\n"
        "        self.transactions.append(txn)\n"
    ),
    "ledger/validation.py": (  # distractor (correct)
        "def is_valid_amount(x):\n"
        "    return isinstance(x, (int, float)) and x == x\n"
    ),
    "ledger/formatting.py": (  # distractor (correct)
        "def money(x):\n"
        "    return f'${x:,.2f}'\n"
    ),
    "ledger/reporting.py": (   # <-- THE BUG lives here
        "def balance(transactions):\n"
        "    # BUG: should sum ALL amounts; this drops debits (only sums credits)\n"
        "    return sum(t.amount for t in transactions if t.amount > 0)\n\n\n"
        "def summary(transactions):\n"
        "    credits = sum(t.amount for t in transactions if t.amount > 0)\n"
        "    debits = sum(t.amount for t in transactions if t.amount < 0)\n"
        "    return {'credits': credits, 'debits': debits, 'balance': balance(transactions)}\n"
    ),
}

_LEDGER_TEST = (
    "import os\n"
    "import sys\n"
    "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
    "from ledger import Transaction, balance, summary\n\n"
    "def make():\n"
    "    return [Transaction(100), Transaction(-30), Transaction(-20), Transaction(50)]\n\n"
    "assert balance(make()) == 100, f'balance was {balance(make())}'\n"
    "s = summary(make())\n"
    "assert s['credits'] == 150, s\n"
    "assert s['debits'] == -50, s\n"
    "assert s['balance'] == 100, s\n"
    "print('ALL TESTS PASSED')\n"
)


def _setup_ledger(w: str) -> None:
    for rel, body in _LEDGER.items():
        path = os.path.join(w, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(body)
    tdir = os.path.join(w, "tests")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "test_reporting.py"), "w") as f:
        f.write(_LEDGER_TEST)


def _verify_ledger(w: str):
    # integrity: the agent must not have edited the test to make it pass
    if open(os.path.join(w, "tests", "test_reporting.py")).read() != _LEDGER_TEST:
        return (False, "test file was modified (integrity fail)")
    # independent behavior check with FRESH inputs (ungameable by the visible test)
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from ledger import Transaction, balance, summary\n"
        "txns = [Transaction(200), Transaction(-50), Transaction(-25), Transaction(75)]\n"
        "assert balance(txns) == 200, f'balance={balance(txns)}'\n"
        "s = summary(txns)\n"
        "assert s['credits'] == 275 and s['debits'] == -75 and s['balance'] == 200, s\n"
        "print('OK')"
    )
    p = _py(w, code)
    return ("OK" in p.stdout, _tail(p))


LEDGER = EvalCase(
    name="repo_fix",
    prompt=(
        "This repository (a 'ledger' Python package with several modules) has a failing test. "
        "Run `python3 tests/test_reporting.py`; it fails. Find which module has the bug, fix it so "
        "the test prints ALL TESTS PASSED, and do NOT modify anything under tests/. "
        "Give a one-line summary when done."
    ),
    setup=_setup_ledger,
    verify=_verify_ledger,
    max_steps=20,
    use_code_index=True,
)


CASES = [STRUTILS, MATHLIB, CALC, LEDGER]
