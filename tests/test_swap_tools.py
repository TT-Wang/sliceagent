"""pin / view — the active-asker MM syscalls (kernel Step D).
No model, no pytest. Run: PYTHONPATH=src python tests/test_swap_tools.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.slice import PIN_CEILING, READ_BUDGET, Slice, touch_file  # noqa: E402
from memagent.swap import _DEFAULT_SWAP  # noqa: E402
from memagent.swap_tools import make_pin_tool, make_view_tool  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ── pin: deliberate growth, reclaim-protected ────────────────────────────────────────────────────
@check
def pin_keeps_a_plain_read_resident_against_eviction():
    s = Slice(); s.reset("t")
    _DEFAULT_SWAP.pin(s, "core/contract.py")          # pin a read-only file (not edited)
    assert "core/contract.py" in s.pinned
    for i in range(READ_BUDGET + 5):                   # flood with unrelated reads
        touch_file(s, f"scratch{i}.py")
    assert "core/contract.py" in s.active_files, "a PINNED file must survive plain-read eviction"


@check
def unpin_lets_the_file_page_out_again():
    s = Slice(); s.reset("t")
    _DEFAULT_SWAP.pin(s, "core/contract.py")
    _DEFAULT_SWAP.unpin(s, "core/contract.py")
    assert "core/contract.py" not in s.pinned
    for i in range(READ_BUDGET + 5):
        touch_file(s, f"scratch{i}.py")
    assert "core/contract.py" not in s.active_files, "an UNPINNED file reverts to evictable residue"


@check
def pinned_set_is_bounded_by_pin_ceiling():
    s = Slice(); s.reset("t")
    for i in range(PIN_CEILING + 4):
        _DEFAULT_SWAP.pin(s, f"f{i}.py")
    assert len(s.pinned) <= PIN_CEILING, "pins are force-compacted past the disaster ceiling"
    assert "f0.py" not in s.pinned, "the LEAST-RECENT pin is the one force-compacted"


@check
def pin_does_not_break_the_empty_case():
    # no pins => evict behaves exactly as before (the co-residency contract is unchanged)
    s = Slice(); s.reset("t")
    touch_file(s, "a.py", edited=True)
    for i in range(READ_BUDGET + 3):
        touch_file(s, f"r{i}.py")
    reads = [p for p in s.active_files if p != "a.py"]
    assert "a.py" in s.active_files and len(reads) == READ_BUDGET, "empty pinned => old behavior"


# ── the tools (registry wiring + handlers) ───────────────────────────────────────────────────────
@check
def pin_tool_pins_and_unpins_via_handler():
    s = Slice(); s.reset("t")
    tool = make_pin_tool(lambda: s)
    assert tool.name == "pin"
    out = tool.handler({"path": "auth/login.py"})
    assert "auth/login.py" in s.pinned and "Pinned" in out
    out2 = tool.handler({"path": "auth/login.py", "unpin": True})
    assert "auth/login.py" not in s.pinned and "Unpinned" in out2


@check
def pin_tool_needs_a_path():
    s = Slice(); s.reset("t")
    tool = make_pin_tool(lambda: s)
    assert "pass a 'path'" in tool.handler({})


@check
def view_mem_reports_headroom_with_pin_count():
    s = Slice(); s.reset("t")
    _DEFAULT_SWAP.pin(s, "x.py")
    tool = make_view_tool(lambda: s)
    assert tool.name == "view"
    out = tool.handler({"kind": "mem"})
    assert "headroom" in out and f"pinned 1/{PIN_CEILING}" in out


@check
def view_maps_lists_resident_pages_and_tags():
    s = Slice(); s.reset("t")
    touch_file(s, "edited.py", edited=True)
    _DEFAULT_SWAP.pin(s, "pinned.py")
    out = make_view_tool(lambda: s).handler({"kind": "maps"})
    assert "edited.py" in out and "edited" in out
    assert "pinned.py" in out and "pinned" in out


@check
def view_is_read_only():
    s = Slice(); s.reset("t")
    touch_file(s, "a.py")
    before = list(s.active_files), list(s.pinned)
    make_view_tool(lambda: s).handler({"kind": "mem"})
    make_view_tool(lambda: s).handler({"kind": "maps"})
    assert (list(s.active_files), list(s.pinned)) == before, "view must not mutate the slice"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
