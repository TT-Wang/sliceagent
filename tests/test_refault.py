"""Refault self-tuning + I/O telemetry (the "cache, not log" loop).
A re-read of a file still in the recency (ghost) ring is a REFAULT — the kernel grants ITSELF a brief
reclaim-protection (no model involvement). hit/miss/refault/evict are counted so the moat is measured.
No model, no pytest. Run: PYTHONPATH=src python tests/test_refault.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.slice import (HOT_CEILING, HOT_TTL, READ_BUDGET, Slice,  # noqa: E402
                            render_slice, touch_file)
from memagent.swap import _DEFAULT_SWAP  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ── telemetry (#6) ────────────────────────────────────────────────────────────────────────────────
@check
def io_counts_hit_miss_evict():
    s = Slice(); s.reset("t")
    touch_file(s, "a.py")                       # first sight → miss
    touch_file(s, "a.py")                       # already resident → hit
    for i in range(READ_BUDGET + 3):            # floods past the budget → evictions
        touch_file(s, f"r{i}.py")
    assert s.io["miss"] >= 1 and s.io["hit"] == 1 and s.io["evict"] > 0, s.io


# ── refault → kernel-granted promotion (#1) ─────────────────────────────────────────────────────────
@check
def refault_promotes_and_the_file_survives_eviction():
    s = Slice(); s.reset("t")
    for i in range(5):                          # f0..f4: READ_BUDGET keeps f1..f4, f0 evicted → ghost
        touch_file(s, f"f{i}.py")
    assert "f0.py" in [g["ref"] for g in s.ghosts] and s.io["evict"] >= 1
    touch_file(s, "f0.py")                      # re-read a ghost → REFAULT → promote
    assert s.io["refault"] >= 1 and "f0.py" in s.hot
    for i in range(5, 5 + READ_BUDGET + 3):     # flood that would normally evict an old read
        touch_file(s, f"f{i}.py")
    assert "f0.py" in s.active_files, "a refault-promoted (hot) file must survive plain-read eviction"


@check
def promotion_is_bounded_by_hot_ceiling():
    s = Slice(); s.reset("t")
    for i in range(HOT_CEILING + 4):
        _DEFAULT_SWAP._promote(s, f"h{i}.py")
    assert len(s.hot) <= HOT_CEILING, len(s.hot)
    assert "h0.py" not in s.hot, "the least-recent soft-pin is force-dropped past the ceiling"


@check
def hot_decays_after_ttl_steps():
    s = Slice(); s.reset("t")
    _DEFAULT_SWAP._promote(s, "x.py")
    assert "x.py" in s.hot
    for _ in range(HOT_TTL):                     # prefetch runs once per step → ages the soft-pin
        _DEFAULT_SWAP.prefetch(s)
    assert "x.py" not in s.hot, "a kernel soft-pin must expire after HOT_TTL steps (never accumulates)"


# ── no refault → byte-identical old behavior (moat-safe) ────────────────────────────────────────────
@check
def no_refault_reduces_to_the_old_eviction_rule():
    s = Slice(); s.reset("t")
    touch_file(s, "a.py", edited=True)
    for i in range(READ_BUDGET + 3):            # all FRESH reads → no refault, no promotion
        touch_file(s, f"r{i}.py")
    reads = [p for p in s.active_files if p != "a.py"]
    assert "a.py" in s.active_files and len(reads) == READ_BUDGET, "no promotions → exactly the old rule"
    assert s.hot == {} and s.io["refault"] == 0


@check
def kernel_internal_state_never_renders_into_the_slice():
    s = Slice(); s.reset("t")
    _DEFAULT_SWAP._promote(s, "kernel_only.py")
    s.io["refault"] = 99
    out = render_slice(s, "(open files)")
    assert "kernel_only.py" not in out and "refault" not in out, "io/hot are kernel-internal, not the model's slice"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
