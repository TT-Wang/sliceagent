"""fuzzy_find_unique — indentation-tolerant unique-span finder for str_replace.
No model, no pytest. Run: python tests/test_fuzzy.py

Covers sec-5 test_fuzzy.py cases: line-trim span; indent-flex span; None on >1
line-trim match; None on no match; None on empty old; byte-correct round-trip
content[:s]+new+content[e:].
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.fuzzy import fuzzy_find_unique          # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def empty_old_returns_none():
    assert fuzzy_find_unique("def foo():\n    pass\n", "") is None


@check
def empty_content_with_nonempty_old_returns_none():
    assert fuzzy_find_unique("", "x") is None


@check
def no_match_returns_none():
    content = "def foo():\n    return 1\n"
    assert fuzzy_find_unique(content, "def nonexistent():") is None


@check
def line_trim_span_found():
    # Pattern differs only by trailing whitespace per line -> line-trim strategy.
    content = "def foo():\n    return 1\n"
    old = "def foo():   \n    return 1   "          # trailing spaces, exact match fails
    span = fuzzy_find_unique(content, old)
    assert span is not None
    s, e = span
    assert content[s:e] == "def foo():\n    return 1"


@check
def line_trim_single_line_span():
    content = "alpha\nbeta\ngamma\n"
    span = fuzzy_find_unique(content, "  beta  ")    # leading+trailing ws, line-trim
    assert span is not None
    s, e = span
    assert content[s:e] == "beta"


@check
def indent_flex_span_found():
    # Pattern has DIFFERENT leading indentation than the file -> indent-flexible.
    # (line-trim would also match here; both strategies agree, span is correct.)
    content = "class A:\n        def m(self):\n            return 2\n"
    old = "def m(self):\n    return 2"               # 0/4 indent vs file's 8/12
    span = fuzzy_find_unique(content, old)
    assert span is not None
    s, e = span
    assert content[s:e] == "        def m(self):\n            return 2"


@check
def more_than_one_line_trim_match_returns_none():
    # Two identical blocks -> ambiguous -> uniqueness gate returns None.
    content = "x = 1\nx = 1\ndone\n"
    assert fuzzy_find_unique(content, "x = 1") is None


@check
def exact_match_single_returns_span():
    content = "first line\nsecond line\nthird line\n"
    span = fuzzy_find_unique(content, "second line")
    assert span is not None
    s, e = span
    assert content[s:e] == "second line"


@check
def multiline_exact_block_span():
    content = "head\nfoo()\nbar()\ntail\n"
    old = "foo()\nbar()"
    span = fuzzy_find_unique(content, old)
    assert span is not None
    s, e = span
    assert content[s:e] == "foo()\nbar()"


@check
def round_trip_replacement_is_byte_correct():
    # The load-bearing contract: caller splices content[:s] + new + content[e:].
    content = "def foo():\n    return 1\n"
    old = "    return 1   "                          # trailing ws -> line-trim
    new = "    return 42"
    span = fuzzy_find_unique(content, old)
    assert span is not None
    s, e = span
    rebuilt = content[:s] + new + content[e:]
    assert rebuilt == "def foo():\n    return 42\n", repr(rebuilt)


@check
def round_trip_multiline_byte_correct():
    content = "a\n        b\n        c\nd\n"
    old = "b\nc"                                      # indent-flexible
    new = "B\nC"
    span = fuzzy_find_unique(content, old)
    assert span is not None
    s, e = span
    rebuilt = content[:s] + new + content[e:]
    assert rebuilt == "a\nB\nC\nd\n", repr(rebuilt)


@check
def returns_first_strategy_winner_not_indent_when_line_trim_unique():
    # line-trim is tried first; a unique line-trim hit short-circuits.
    content = "    keep_me = True\n"
    span = fuzzy_find_unique(content, "keep_me = True")   # indent differs
    assert span is not None
    s, e = span
    assert content[s:e] == "    keep_me = True"


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
