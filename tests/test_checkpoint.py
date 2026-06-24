"""F1 — the deterministic CHECKPOINT (consolidate_checkpoint): a bounded re-projection of the carried task
state into one snapshot, used as the F2 overflow-rebuild artifact. No model, no pytest.
Run: PYTHONPATH=src python tests/test_checkpoint.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.slice import Slice, consolidate_checkpoint, record_note  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def checkpoint_distills_carried_state():
    s = Slice(); s.reset("Fix the auth bug in login.py")
    s.requirements = [{"text": "keep the public API stable", "done": False},
                      {"text": "already satisfied one", "done": True}]
    record_note(s, "decided to use a retry wrapper because the gateway is rate-limited", source="claim")
    record_note(s, "root cause: missing await in login()", source="claim")
    s.edited_files = {"login.py", "auth.py"}
    s.open_report = "tests still failing on timeout"
    full = consolidate_checkpoint(s, compact=False)
    assert "intent: Fix the auth bug" in full
    assert "keep the public API stable" in full and "already satisfied one" not in full  # OPEN reqs only
    assert "decisions:" in full and "retry wrapper" in full                              # typed decision
    assert "login.py" in full and "auth.py" in full                                      # change-set
    assert "missing await" in full                                                       # findings digest (full)
    assert "open: tests still failing" in full


@check
def checkpoint_compact_excludes_findings_digest():
    s = Slice(); s.reset("g")
    record_note(s, "a plain established fact about the parser", source="claim")
    record_note(s, "decided to go with plan A for the cache", source="claim")
    compact = consolidate_checkpoint(s, compact=True)
    full = consolidate_checkpoint(s, compact=False)
    assert "decided to go with plan A" in compact, "decisions appear in compact"
    assert "a plain established fact" not in compact, "findings digest is FULL-only (no dup with the tier)"
    assert "a plain established fact" in full


@check
def checkpoint_self_suppresses_when_empty():
    assert consolidate_checkpoint(Slice()) == "", "a fresh slice → no checkpoint bytes"


@check
def checkpoint_is_bounded():
    s = Slice(); s.reset("x" * 1000)
    for i in range(60):
        record_note(s, f"established fact number {i} " + "y" * 300, source="claim")
    s.edited_files = {f"file_{i}.py" for i in range(40)}
    full = consolidate_checkpoint(s, compact=False)
    assert len(full) < 4000, f"checkpoint must stay bounded, got {len(full)}"
    assert "(+" in full, "a large change-set is summarized with a (+N) overflow count"


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
