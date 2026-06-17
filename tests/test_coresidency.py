"""Multi-file co-residency + prompt-cache locality (the h2h s3 fix + cache/wall improvements).
No model, no pytest. Run: PYTHONPATH=src python tests/test_coresidency.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.slice import (DEP_CEILING, READ_BUDGET, Slice, build_artifacts,  # noqa: E402
                            render_slice, touch_file)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _FakeTools:
    """Minimal ToolHost stub: stable content per path, so build_artifacts is deterministic."""
    def read_text(self, p):
        return f"# {p}\nvalue = 1\n"
    def root(self):
        return "."


# ── dependency co-residency (the s3 multi-file root cause) ──────────────────────────────────────
@check
def dependency_of_edited_file_survives_eviction():
    s = Slice(); s.reset("t")
    touch_file(s, "eventbus/dispatcher.py", edited=True)      # the change set
    s.protected_deps = {"eventbus/context.py"}                # a contract dep (from the code graph)
    touch_file(s, "eventbus/context.py")                      # read the dependency
    for i in range(READ_BUDGET + 3):                          # flood with unrelated reads
        touch_file(s, f"other{i}.py")
    assert "eventbus/dispatcher.py" in s.active_files, "edited file must never evict"
    assert "eventbus/context.py" in s.active_files, "a DEPENDENCY of an edited file must stay co-resident"
    assert "other0.py" not in s.active_files, "unrelated old reads still evict (no bloat)"


@check
def plain_reads_still_evict_no_bloat():
    s = Slice(); s.reset("t")
    touch_file(s, "a.py", edited=True)
    touch_file(s, "plain.py")                                 # a plain read, NOT a dep
    for i in range(READ_BUDGET + 3):
        touch_file(s, f"r{i}.py")
    assert "plain.py" not in s.active_files, "non-dependency reads must still be bounded by READ_BUDGET"


@check
def protected_deps_are_bounded():
    s = Slice(); s.reset("t")
    touch_file(s, "core.py", edited=True)
    s.protected_deps = {f"dep{i}.py" for i in range(DEP_CEILING + 4)}
    for i in range(DEP_CEILING + 4):
        touch_file(s, f"dep{i}.py")
    for i in range(READ_BUDGET + 2):
        touch_file(s, f"x{i}.py")
    kept = [p for p in s.active_files if p.startswith("dep")]
    assert len(kept) <= DEP_CEILING, f"co-resident deps must be bounded by DEP_CEILING, kept {len(kept)}"


@check
def no_dep_graph_reduces_to_old_rule():
    s = Slice(); s.reset("t")  # protected_deps stays empty (NullRetriever / no graph)
    touch_file(s, "a.py", edited=True)
    for i in range(READ_BUDGET + 3):
        touch_file(s, f"r{i}.py")
    reads = [p for p in s.active_files if p != "a.py"]
    assert "a.py" in s.active_files and len(reads) == READ_BUDGET, "empty deps → exactly the old behavior"


# ── prompt-cache locality (drives cache-hit% and wall time) ──────────────────────────────────────
@check
def open_files_lead_volatile_tiers():
    s = Slice(); s.reset("t")
    s.last_error = "boom"
    out = render_slice(s, "ARTIFACTS")
    # stable bulk (OPEN FILES) must precede the volatile, recency-salient tail (RECENT / CURRENT ERROR)
    assert out.index("# OPEN FILES") < out.index("# RECENT"), "OPEN FILES must lead RECENT (cache prefix)"
    assert out.index("# OPEN FILES") < out.index("# CURRENT ERROR"), "OPEN FILES must lead CURRENT ERROR"


@check
def open_files_render_is_byte_stable_under_reread():
    # a re-read reorders the working set by recency; OPEN FILES must still render byte-identically so
    # the prompt-cache prefix stays warm (this was the silent cache-buster).
    a = Slice(); a.reset("t"); a.active_files = ["a.py", "b.py", "c.py"]
    b = Slice(); b.reset("t"); b.active_files = ["c.py", "a.py", "b.py"]   # different touch order, same set
    assert build_artifacts(a, _FakeTools()) == build_artifacts(b, _FakeTools()), \
        "OPEN FILES must be byte-stable regardless of touch/recency order"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
