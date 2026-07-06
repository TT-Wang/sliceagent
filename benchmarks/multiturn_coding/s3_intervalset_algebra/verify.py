"""Independent behavioral oracle for the interval-algebra scenario.

This file is NOT given to the benchmarked agent to edit. It imports the agent's
final ``intervalset.iset.IntervalSet`` in a fresh subprocess and exercises the
cumulative behavior demanded by all ten turns, plus the two regression
scenarios that an agent which lost earlier context would get wrong.

Run via subprocess so a crashing/looping import cannot take down the parent and
so import caching never masks a broken module.
"""
import os
import subprocess
import sys
import textwrap

PY = sys.executable


# The probe runs inside the agent's workdir. It uses ONLY the public API the
# turns established: add/intervals/contains (t1), length (t2), overlaps (t3),
# remove (t4/t9), shift (t5), union/intersection (t7), difference (t8),
# gaps (t10). Each check prints a tag on failure.
PROBE = textwrap.dedent('''
    import sys
    from intervalset.iset import IntervalSet

    fails = []
    def check(cond, tag):
        if not cond:
            fails.append(tag)

    def mk(*pairs):
        s = IntervalSet()
        for a, b in pairs:
            s.add(a, b)
        return s

    # ---- turn 1: canonical merged form --------------------------------
    s = mk((5, 8), (1, 3))
    check(s.intervals() == [(1, 3), (5, 8)], "t1_sorted")
    s = mk((1, 5), (3, 8))          # strictly overlapping -> coalesce
    check(s.intervals() == [(1, 8)], "t1_overlap_merge")
    s = mk((1, 3), (10, 12), (2, 11))  # bridge everything
    check(s.intervals() == [(1, 12)], "t1_bridge")
    check(s.contains(2) and not s.contains(0) and not s.contains(12), "t1_contains")
    s = mk((4, 4))                  # empty ignored
    check(s.intervals() == [], "t1_empty_ignored")

    # ---- turn 2: length() ---------------------------------------------
    s = mk((1, 3), (5, 8))
    check(s.length() == 5, "t2_length")           # 2 + 3
    check(mk().length() == 0, "t2_length_empty")
    s = mk((1, 4), (2, 6))                          # merges to [1,6)
    check(s.length() == 5, "t2_length_merged")

    # ---- turn 3: overlaps(s, e) ---------------------------------------
    s = mk((1, 3), (5, 8))
    check(s.overlaps(2, 6) is True, "t3_overlaps_true")
    check(s.overlaps(3, 5) is False, "t3_overlaps_gap")   # [3,5) sits in the gap
    check(s.overlaps(8, 20) is False, "t3_overlaps_touch_only")  # touches end 8, no cover
    check(s.overlaps(7, 9) is True, "t3_overlaps_partial")
    check(s.overlaps(5, 5) is False, "t3_overlaps_empty_query")

    # ---- turn 4: remove(s, e) -----------------------------------------
    s = mk((0, 10))
    s.remove(0, 3)                                  # trim left end
    check(s.intervals() == [(3, 10)], "t4_trim_left")
    s = mk((0, 10))
    s.remove(7, 10)                                 # trim right end
    check(s.intervals() == [(0, 7)], "t4_trim_right")
    s = mk((0, 5), (10, 15))
    s.remove(-5, 20)                                # delete everything
    check(s.intervals() == [], "t4_delete_all")
    s = mk((0, 5))
    s.remove(3, 3)                                  # empty removal -> no change
    check(s.intervals() == [(0, 5)], "t4_empty_removal")

    # ---- turn 5: shift(delta) is pure ---------------------------------
    s = mk((1, 3), (5, 8))
    t = s.shift(10)
    check(t.intervals() == [(11, 13), (15, 18)], "t5_shift_result")
    check(s.intervals() == [(1, 3), (5, 8)], "t5_shift_pure")     # original unchanged
    check(s.shift(-1).intervals() == [(0, 2), (4, 7)], "t5_shift_neg")

    # ---- turn 6: REGRESSION -- adjacency coalescing -------------------
    s = mk((1, 3), (3, 5))                          # touching -> single [1,5)
    check(s.intervals() == [(1, 5)], "t6_adjacent_merge")
    check(s.length() == 4, "t6_adjacent_length")
    s = mk((0, 2), (2, 4), (4, 6))                  # chain of touches
    check(s.intervals() == [(0, 6)], "t6_adjacent_chain")
    s = mk((1, 3), (4, 6))                          # a real gap must NOT merge
    check(s.intervals() == [(1, 3), (4, 6)], "t6_gap_not_merged")

    # ---- turn 7: union / intersection ---------------------------------
    a = mk((1, 5), (10, 15))
    b = mk((3, 12))
    u = a.union(b)
    check(u.intervals() == [(1, 15)], "t7_union")                 # all bridged
    check(a.intervals() == [(1, 5), (10, 15)], "t7_union_pure_a") # a untouched
    check(b.intervals() == [(3, 12)], "t7_union_pure_b")         # b untouched
    i = a.intersection(b)
    check(i.intervals() == [(3, 5), (10, 12)], "t7_intersection")
    # adjacency coalescing in union results
    u2 = mk((0, 3)).union(mk((3, 6)))
    check(u2.intervals() == [(0, 6)], "t7_union_adjacent")
    # disjoint intersection is empty
    check(mk((0, 2)).intersection(mk((5, 7))).intervals() == [], "t7_intersection_empty")

    # ---- turn 8: difference -------------------------------------------
    a = mk((0, 10))
    b = mk((3, 6))
    d = a.difference(b)
    check(d.intervals() == [(0, 3), (6, 10)], "t8_difference_hole")
    check(a.intervals() == [(0, 10)], "t8_difference_pure_a")     # a untouched
    check(b.intervals() == [(3, 6)], "t8_difference_pure_b")     # b untouched
    d2 = mk((0, 5), (10, 15)).difference(mk((-5, 20)))
    check(d2.intervals() == [], "t8_difference_all")
    d3 = mk((0, 5)).difference(mk((10, 20)))
    check(d3.intervals() == [(0, 5)], "t8_difference_none")

    # ---- turn 9: REGRESSION -- interior split on remove ---------------
    s = mk((0, 10))
    s.remove(4, 6)                                  # interior hole -> split
    check(s.intervals() == [(0, 4), (6, 10)], "t9_interior_split")
    check(s.length() == 8, "t9_interior_split_length")
    # multiple intervals, remove spanning a boundary and cutting interiors
    s = mk((0, 5), (10, 20))
    s.remove(3, 13)                                 # trims right of first, left of second
    check(s.intervals() == [(0, 3), (13, 20)], "t9_span_boundary")
    # remove a hole then the two halves are independent
    s = mk((0, 20))
    s.remove(5, 8)
    s.remove(12, 15)
    check(s.intervals() == [(0, 5), (8, 12), (15, 20)], "t9_two_holes")
    # difference must stay consistent with the fixed interior-split remove
    d = mk((0, 10)).difference(mk((4, 6)))
    check(d.intervals() == [(0, 4), (6, 10)], "t9_difference_consistent")

    # ---- turn 10: gaps(lo, hi) ----------------------------------------
    s = mk((1, 3), (5, 8))
    check(s.gaps(0, 10) == [(0, 1), (3, 5), (8, 10)], "t10_gaps_basic")
    check(s.gaps(3, 5) == [(3, 5)], "t10_gaps_window_in_gap")
    check(s.gaps(1, 3) == [], "t10_gaps_fully_covered")
    check(s.gaps(5, 5) == [], "t10_gaps_empty_window")
    check(mk().gaps(0, 4) == [(0, 4)], "t10_gaps_empty_set")
    # window clips the complement
    check(s.gaps(2, 6) == [(3, 5)], "t10_gaps_clip")

    # ---- integration: mix of everything -------------------------------
    a = mk((0, 4), (8, 12))
    b = mk((2, 10))
    s = a.union(b)                                  # -> [0,12)
    s.remove(5, 7)                                  # interior hole -> [0,5)+[7,12)
    check(s.intervals() == [(0, 5), (7, 12)], "int_build")
    check(s.length() == 10, "int_length")
    check(s.overlaps(4, 8) is True, "int_overlaps")
    check(s.gaps(0, 14) == [(5, 7), (12, 14)], "int_gaps")
    diff = s.difference(mk((0, 3)))
    check(diff.intervals() == [(3, 5), (7, 12)], "int_difference")

    if fails:
        print("FAILS:" + ",".join(fails))
        sys.exit(1)
    print("ALL_OK")
    sys.exit(0)
''')


def verify(workdir):
    mod = os.path.join(workdir, "intervalset", "iset.py")
    if not os.path.isfile(mod):
        return (False, "intervalset/iset.py is missing")
    try:
        proc = subprocess.run(
            [PY, "-c", PROBE],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return (False, "probe timed out (possible infinite loop in module)")
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 and "ALL_OK" in out:
        return (True, "all cumulative + regression checks passed")
    # Surface the most useful slice of output.
    detail = out.strip().splitlines()
    tail = "\n".join(detail[-15:]) if detail else "(no output)"
    return (False, "probe failed:\n" + tail)
