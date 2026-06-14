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


# ================= STRESS BATTERY (each probes a distinct generic dimension) =================

# ---- 5. LARGE FILE: bug in the middle of a long file (stresses OPEN FILES truncation) ----

def _setup_bigmod(w: str) -> None:
    lines = ['"""A large module of small operations."""', ""]
    for i in range(60):
        lines.append(f"def op_{i:02d}(x):")
        if i == 30:
            lines.append(f"    return x + {i} + 1  # subtle off-by-one")  # BUG
        else:
            lines.append(f"    return x + {i}")
        lines.append("")
    with open(os.path.join(w, "bigmod.py"), "w") as f:
        f.write("\n".join(lines))
    test = (
        "import bigmod\n"
        "assert bigmod.op_05(100) == 105\n"
        "assert bigmod.op_30(100) == 130, f'op_30={bigmod.op_30(100)}'\n"
        "assert bigmod.op_59(0) == 59\n"
        "print('ALL TESTS PASSED')\n"
    )
    with open(os.path.join(w, "test_bigmod.py"), "w") as f:
        f.write(test)


_BIGMOD_TEST_HASH = None  # set after setup style below; we re-read in verify


def _verify_bigmod(w: str):
    code = (
        "import sys; sys.path.insert(0,'.')\n"
        "import bigmod\n"
        "assert bigmod.op_30(7) == 37, f'op_30(7)={bigmod.op_30(7)}'\n"
        "assert bigmod.op_29(0) == 29 and bigmod.op_31(0) == 31\n"
        "print('OK')"
    )
    p = _py(w, code)
    return ("OK" in p.stdout, _tail(p))


BIGMOD = EvalCase(
    name="large_file_fix",
    prompt=(
        "bigmod.py has ~60 functions; exactly one returns a wrong value, which makes "
        "`python3 test_bigmod.py` fail. Find the single buggy function, fix it in bigmod.py, "
        "and rerun until the test prints ALL TESTS PASSED. Do not edit test_bigmod.py."
    ),
    setup=_setup_bigmod,
    verify=_verify_bigmod,
    max_steps=20,
    use_code_index=True,
)

# ---- 6. MULTI-FILE: two bugs in two modules; both must be fixed (working-set + coordination) ----

_SHOP = {
    "shop/__init__.py": "from .checkout import total\n",
    "shop/discount.py": (
        "def apply_discount(price):\n"
        "    # 10% off\n"
        "    return price  # BUG: discount not applied\n"
    ),
    "shop/tax.py": (
        "def add_tax(price):\n"
        "    return round(price * 1.8, 2)  # BUG: should be 1.08 (8% tax)\n"
    ),
    "shop/checkout.py": (
        "from .discount import apply_discount\n"
        "from .tax import add_tax\n\n\n"
        "def total(price):\n"
        "    return add_tax(apply_discount(price))\n"
    ),
}
_SHOP_TEST = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
    "from shop import total\n"
    "assert round(total(100), 2) == 97.2, f'total(100)={total(100)}'\n"
    "print('ALL TESTS PASSED')\n"
)


def _setup_shop(w: str) -> None:
    for rel, body in _SHOP.items():
        p = os.path.join(w, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(body)
    td = os.path.join(w, "tests")
    os.makedirs(td, exist_ok=True)
    with open(os.path.join(td, "test_total.py"), "w") as f:
        f.write(_SHOP_TEST)


def _verify_shop(w: str):
    if open(os.path.join(w, "tests", "test_total.py")).read() != _SHOP_TEST:
        return (False, "test file was modified (integrity fail)")
    code = (
        "import sys; sys.path.insert(0,'.')\n"
        "from shop.discount import apply_discount\n"
        "from shop.tax import add_tax\n"
        "from shop import total\n"
        "assert round(apply_discount(100),2)==90.0, f'discount={apply_discount(100)}'\n"
        "assert round(add_tax(100),2)==108.0, f'tax={add_tax(100)}'\n"
        "assert round(total(200),2)==194.4, f'total(200)={total(200)}'\n"
        "print('OK')"
    )
    p = _py(w, code)
    return ("OK" in p.stdout, _tail(p))


SHOP = EvalCase(
    name="multi_file_fix",
    prompt=(
        "The `shop` package computes an order total as tax applied to a discounted price. "
        "`python3 tests/test_total.py` fails. There are bugs in MORE THAN ONE module; find and "
        "fix all of them so the test prints ALL TESTS PASSED. Do not edit anything under tests/."
    ),
    setup=_setup_shop,
    verify=_verify_shop,
    max_steps=20,
    use_code_index=True,
)

# ---- 7. CROSS-LANGUAGE: a JavaScript bug (stresses CodeIndex/tooling language-agnosticism) ----

_MATHOPS_JS = (
    "function add(a, b) { return a + b; }\n"
    "function multiply(a, b) { return a + b; }  // BUG: should be a * b\n"
    "function isEven(n) { return n % 2 === 0; }\n"
    "module.exports = { add, multiply, isEven };\n"
)
_MATHOPS_TEST_JS = (
    "const { add, multiply, isEven } = require('./mathops');\n"
    "const assert = require('assert');\n"
    "assert.strictEqual(add(2, 3), 5);\n"
    "assert.strictEqual(multiply(3, 4), 12);\n"
    "assert.strictEqual(isEven(4), true);\n"
    "assert.strictEqual(isEven(3), false);\n"
    "console.log('ALL TESTS PASSED');\n"
)


def _setup_js(w: str) -> None:
    with open(os.path.join(w, "mathops.js"), "w") as f:
        f.write(_MATHOPS_JS)
    with open(os.path.join(w, "test.js"), "w") as f:
        f.write(_MATHOPS_TEST_JS)


def _verify_js(w: str):
    if open(os.path.join(w, "test.js")).read() != _MATHOPS_TEST_JS:
        return (False, "test file was modified (integrity fail)")
    chk = "const m=require('./mathops'); if(m.multiply(6,7)===42 && m.add(2,2)===4) console.log('OK');"
    p = subprocess.run(["node", "-e", chk], cwd=w, capture_output=True, text=True, timeout=30)
    return ("OK" in p.stdout, _tail(p))


JSFIX = EvalCase(
    name="js_fix",
    prompt=(
        "This is a Node.js project. `node test.js` fails. Find and fix the bug in mathops.js so "
        "the test prints ALL TESTS PASSED. Do not edit test.js. Give a one-line summary when done."
    ),
    setup=_setup_js,
    verify=_verify_js,
    max_steps=15,
    use_code_index=True,
)

# ---- 8. FEATURE ADD: implement a new function that REUSES existing helpers across modules ----

_TEXTKIT = {
    "textkit/__init__.py": "from .tokens import tokenize\nfrom .casing import title_case\n",
    "textkit/tokens.py": (
        "def tokenize(s):\n"
        "    '''split on whitespace into words'''\n"
        "    return [w for w in s.split() if w]\n"
    ),
    "textkit/casing.py": (
        "def title_case(word):\n"
        "    '''capitalize first letter, lowercase the rest'''\n"
        "    return word[:1].upper() + word[1:].lower() if word else word\n"
    ),
}
# verifier expects headline() to REUSE tokenize + title_case (not reimplement)


def _setup_textkit(w: str) -> None:
    for rel, body in _TEXTKIT.items():
        p = os.path.join(w, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(body)


def _verify_textkit(w: str):
    code = (
        "import sys; sys.path.insert(0,'.')\n"
        "from textkit.headline import headline\n"
        "assert headline('the QUICK brown FOX') == 'The Quick Brown Fox', headline('the QUICK brown FOX')\n"
        "assert headline('  hello   WORLD  ') == 'Hello World', repr(headline('  hello   WORLD  '))\n"
        "import inspect, textkit.headline as h\n"
        "src = inspect.getsource(h)\n"
        "assert 'tokenize' in src and 'title_case' in src, 'must reuse existing helpers'\n"
        "print('OK')"
    )
    p = _py(w, code)
    return ("OK" in p.stdout, _tail(p))


TEXTKIT = EvalCase(
    name="feature_add",
    prompt=(
        "The `textkit` package has helpers for tokenizing and casing text. Add a new module "
        "textkit/headline.py with a function headline(s) that turns a string into a headline: "
        "each word title-cased and joined by single spaces. REUSE the existing helpers in the "
        "package rather than reimplementing them. Verify by running python, then summarize."
    ),
    setup=_setup_textkit,
    verify=_verify_textkit,
    max_steps=15,
    use_code_index=True,
)


# ---- 9. WIDE: sweep the same fix across many files (stresses the K=4 working-set bound) ----

_WIDE_N = 6


def _wide_test() -> str:
    lines = ["import os, sys",
             "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"]
    for i in range(_WIDE_N):
        lines.append(f"from pkg.seg_{i} import value as v{i}; assert v{i}() == {i}0, f'seg_{i}={{v{i}()}}'")
    lines.append("print('ALL TESTS PASSED')")
    return "\n".join(lines) + "\n"


def _setup_wide(w: str) -> None:
    d = os.path.join(w, "pkg")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "__init__.py"), "w") as f:
        f.write("")
    for i in range(_WIDE_N):
        with open(os.path.join(d, f"seg_{i}.py"), "w") as f:
            f.write(f"def value():\n    return {i}0 + 1  # BUG: should be {i}0\n")
    td = os.path.join(w, "tests")
    os.makedirs(td, exist_ok=True)
    with open(os.path.join(td, "test_segs.py"), "w") as f:
        f.write(_wide_test())


def _verify_wide(w: str):
    if open(os.path.join(w, "tests", "test_segs.py")).read() != _wide_test():
        return (False, "test file was modified (integrity fail)")
    checks = "import sys; sys.path.insert(0,'.')\n"
    for i in range(_WIDE_N):
        checks += f"from pkg.seg_{i} import value as v{i}; assert v{i}() == {i}0, 'seg_{i}'\n"
    checks += "print('OK')"
    p = _py(w, checks)
    return ("OK" in p.stdout, _tail(p))


WIDE = EvalCase(
    name="wide_fix",
    prompt=(
        "Every module pkg/seg_0.py .. pkg/seg_5.py has the SAME kind of bug (an off-by-one in "
        "value()), making `python3 tests/test_segs.py` fail. Fix the bug in ALL of them so the "
        "test prints ALL TESTS PASSED. Do not edit anything under tests/."
    ),
    setup=_setup_wide,
    verify=_verify_wide,
    max_steps=20,
    use_code_index=True,
)


CASES = [STRUTILS, MATHLIB, CALC, LEDGER]
STRESS_CASES = [BIGMOD, SHOP, JSFIX, TEXTKIT, WIDE]
ALL_CASES = CASES + STRESS_CASES
