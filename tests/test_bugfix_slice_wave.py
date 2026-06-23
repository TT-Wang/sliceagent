"""Regression tests for the slice/regions wave: #17 render_regions is robust to empty/gap slots (no
KeyError, no silently-dropped region), #33 MememMemory.close() releases the FTS5 connection idempotently.
No model, no pytest. Run: PYTHONPATH=src python tests/test_bugfix_slice_wave.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import memagent.regions as regions  # noqa: E402
from memagent.memory import MememMemory, NullMemory  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def render_regions_robust_to_gap_and_empty_slots():  # #17
    # a layout where the FIRST slot is empty (would KeyError on slots[0]) and a region sits at gap slot 5
    # (would be silently dropped by the old literal index list).
    fake_order = [
        ("alpha", "x", lambda ctx: "A", 1),
        ("gap5", "x", lambda ctx: "AT-SLOT-5", 5),
        ("tail", "x", lambda ctx: "T", 7),
    ]
    saved = regions.REGION_ORDER
    regions.REGION_ORDER = fake_order
    try:
        out = regions.render_regions({})
        assert "AT-SLOT-5" in out, "a region at a gap slot must NOT be dropped (#17)"
        assert "A" in out and "T" in out
        # empty leading slot 0 → "" (blank line), not a KeyError
        assert out.startswith("\n") or out.splitlines()[0] == "", out.splitlines()[:1]
    finally:
        regions.REGION_ORDER = saved
    # empty order → "" not a crash
    regions.REGION_ORDER = []
    try:
        assert regions.render_regions({}) == ""
    finally:
        regions.REGION_ORDER = saved


@check
def memory_close_is_idempotent():  # #33
    NullMemory().close()                          # no-op, no raise
    m = MememMemory.__new__(MememMemory)           # bypass memem import
    m.close()                                     # never opened (_idx unset) → safe
    closed = {"n": 0}

    class _Idx:
        def close(self):
            closed["n"] += 1
    m._idx = _Idx()
    m.close()
    m.close()                                     # idempotent — second call must not re-close
    assert closed["n"] == 1, closed
    assert m._idx is None


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
