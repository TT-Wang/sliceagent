import os

# Seed project: a NAIVE interval store over half-open integer intervals
# [start, end). This is the starting repo BEFORE turn 1. It is small and
# working for the features it has: add() simply APPENDS the raw (start,end)
# pair (no merging), raw() returns them in insertion order, and contains(x)
# scans them. The 10 user turns replace this with a canonical merged form
# and build a full interval algebra on top of it.

SEED_ISET = '''\
"""A tiny algebra over HALF-OPEN integer intervals [start, end).

Half-open semantics (READ THIS before changing behavior):
  * An interval [start, end) covers x iff ``start <= x < end``.
  * An interval with ``start >= end`` is EMPTY and is ignored.

Seed design notes:
  * This seed is NAIVE / UNMERGED: ``add(start, end)`` simply APPENDS the raw
    ``(start, end)`` pair to an internal list -- it does no merging at all.
  * ``raw()`` returns the appended pairs in insertion order.
  * ``contains(x)`` returns True iff ANY appended interval covers x.

Later turns introduce a CANONICAL MERGED form (a sorted list of
non-overlapping intervals) and an algebra on top of it. Keep the half-open
[start, end) convention everywhere.

Public API (do not rename without updating callers):
  add(start, end), raw(), contains(x)
"""


class IntervalSet:
    def __init__(self):
        # Naive store: raw appended (start, end) pairs, insertion order.
        self._raw = []

    def add(self, start, end):
        """Append the raw interval. Empty intervals (start >= end) are ignored."""
        if start >= end:
            return
        self._raw.append((start, end))

    def raw(self):
        """Return the appended (start, end) pairs in insertion order."""
        return list(self._raw)

    def contains(self, x):
        """True iff any appended interval covers x (start <= x < end)."""
        for s, e in self._raw:
            if s <= x < e:
                return True
        return False
'''

SEED_README = '''\
# intervalset

A tiny algebra over half-open integer intervals `[start, end)`, used as a
teaching toy.

Current capabilities (seed):
  * `add(start, end)` -- naive, just appends the raw pair (no merging)
  * `raw()` -- the appended pairs in insertion order
  * `contains(x)` -- half-open membership test

See `intervalset/iset.py` for the design notes. Later work replaces the naive
store with a canonical merged form; keep the half-open `[start, end)`
convention everywhere.
'''

SEED_TEST = '''\
"""Smoke tests that ship with the seed. These exercise only the seed feature
set (the naive unmerged store). Keep them green."""
from intervalset.iset import IntervalSet


def test_add_and_contains():
    s = IntervalSet()
    s.add(1, 3)
    assert s.contains(0) is False
    assert s.contains(2) is True
    assert s.contains(3) is False  # half-open: end is exclusive


def test_raw_insertion_order():
    s = IntervalSet()
    s.add(1, 3)
    s.add(5, 7)
    assert s.raw() == [(1, 3), (5, 7)]


def test_empty_interval_ignored():
    s = IntervalSet()
    s.add(4, 4)   # empty, ignored
    s.add(6, 5)   # empty, ignored
    assert s.raw() == []
    assert s.contains(4) is False


if __name__ == "__main__":
    test_add_and_contains()
    test_raw_insertion_order()
    test_empty_interval_ignored()
    print("seed tests ok")
'''


def setup(workdir):
    pkg = os.path.join(workdir, "intervalset")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write("from .iset import IntervalSet\n")
    with open(os.path.join(pkg, "iset.py"), "w") as f:
        f.write(SEED_ISET)
    with open(os.path.join(workdir, "README.md"), "w") as f:
        f.write(SEED_README)
    tests = os.path.join(workdir, "tests")
    os.makedirs(tests, exist_ok=True)
    with open(os.path.join(tests, "test_seed.py"), "w") as f:
        f.write(SEED_TEST)
