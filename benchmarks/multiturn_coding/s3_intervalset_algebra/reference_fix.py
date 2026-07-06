import os

# Correct, full implementation of the interval algebra after all ten turns.
# VALIDATION ONLY -- never shown to the benchmarked agents.
#
# Core invariant that makes the two regressions vanish:
#   State is ALWAYS a canonical list of sorted, non-overlapping, and
#   NON-ADJACENT (from turn 6 on) half-open intervals. Every mutating op
#   rebuilds that canonical form via a single _normalize() pass that coalesces
#   both overlapping AND touching (end == next start) intervals.
#   remove() is expressed as a per-interval clip that emits the left remainder
#   [s, rs) and the right remainder [re, e) independently, so an interior hole
#   naturally splits one interval into two. difference() is defined in terms of
#   the same remove logic, so it stays consistent by construction.

REFERENCE = '''\
"""A tiny algebra over HALF-OPEN integer intervals [start, end).

Half-open semantics:
  * An interval [start, end) covers x iff ``start <= x < end``.
  * An interval with ``start >= end`` is EMPTY and is ignored.

Canonical form:
  * State is a SORTED list of non-overlapping, non-adjacent intervals.
  * ``_normalize`` coalesces both OVERLAPPING and merely-TOUCHING intervals
    (where one's end equals the next's start), since [1,3) and [3,5) together
    cover exactly [1,5) under half-open semantics.
"""


class IntervalSet:
    def __init__(self):
        self._ivals = []  # canonical: sorted, non-overlapping, non-adjacent

    # ----- canonicalization ----------------------------------------------
    @staticmethod
    def _normalize(pairs):
        clean = [(s, e) for (s, e) in pairs if s < e]
        clean.sort()
        merged = []
        for s, e in clean:
            if merged and s <= merged[-1][1]:
                # overlap OR adjacency (s == prev end) -> coalesce
                ps, pe = merged[-1]
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    # ----- turn 1: canonical merged form ---------------------------------
    def add(self, start, end):
        if start >= end:
            return
        self._ivals = self._normalize(self._ivals + [(start, end)])

    def intervals(self):
        return list(self._ivals)

    def raw(self):
        # Back-compat with the seed API; the canonical form is the raw form now.
        return list(self._ivals)

    def contains(self, x):
        for s, e in self._ivals:
            if s <= x < e:
                return True
        return False

    # ----- turn 2: length -------------------------------------------------
    def length(self):
        return sum(e - s for s, e in self._ivals)

    # ----- turn 3: overlaps ----------------------------------------------
    def overlaps(self, s, e):
        if s >= e:
            return False
        for a, b in self._ivals:
            if s < b and a < e:
                return True
        return False

    # ----- turn 4 / turn 9: remove (interior split) ----------------------
    def remove(self, s, e):
        if s >= e:
            return
        out = []
        for a, b in self._ivals:
            if e <= a or b <= s:
                # no overlap with the removed region
                out.append((a, b))
                continue
            # left remainder [a, s) if the interval starts before removal
            if a < s:
                out.append((a, s))
            # right remainder [e, b) if the interval ends after removal
            if e < b:
                out.append((e, b))
        self._ivals = self._normalize(out)

    # ----- turn 5: pure shift --------------------------------------------
    def shift(self, delta):
        new = IntervalSet()
        new._ivals = self._normalize([(s + delta, e + delta) for s, e in self._ivals])
        return new

    # ----- turn 7: union / intersection ----------------------------------
    def union(self, other):
        new = IntervalSet()
        new._ivals = self._normalize(list(self._ivals) + list(other._ivals))
        return new

    def intersection(self, other):
        out = []
        for a, b in self._ivals:
            for c, d in other._ivals:
                lo = max(a, c)
                hi = min(b, d)
                if lo < hi:
                    out.append((lo, hi))
        new = IntervalSet()
        new._ivals = self._normalize(out)
        return new

    # ----- turn 8 / turn 9: difference (consistent with remove) ----------
    def difference(self, other):
        new = IntervalSet()
        new._ivals = list(self._ivals)
        for s, e in other._ivals:
            new.remove(s, e)
        return new

    # ----- turn 10: gaps within a window ---------------------------------
    def gaps(self, lo, hi):
        if lo >= hi:
            return []
        out = []
        cur = lo
        for a, b in self._ivals:
            if b <= lo or a >= hi:
                continue
            a = max(a, lo)
            b = min(b, hi)
            if a > cur:
                out.append((cur, a))
            cur = max(cur, b)
        if cur < hi:
            out.append((cur, hi))
        return out
'''


def apply(workdir):
    mod = os.path.join(workdir, "intervalset", "iset.py")
    with open(mod, "w") as f:
        f.write(REFERENCE)
