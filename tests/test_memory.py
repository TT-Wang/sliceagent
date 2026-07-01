"""Memory tier — relevance gate + retrieval-feedback wiring. No model, no pytest.
memem is monkeypatched so the test never touches/mutates the real vault.
Run: python tests/test_memory.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.neocortex import _memory_relevant   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def relevance_gate_is_word_level():
    assert _memory_relevant("how to add two numbers", ["calc", "add"])       # shares 'add'
    assert not _memory_relevant("async feed fetching via gather", ["calc", "add"])  # shares nothing
    assert _memory_relevant("anything at all", [])                            # no terms → can't gate, keep
    assert not _memory_relevant("the address book module", ["add"])          # whole-word: add != address
    assert _memory_relevant("we add items to the store", ["add"])


@check
def recall_gates_noise_and_reinforces_relevant():
    try:
        import memem.obsidian_store
        import memem.retrieve
        from memagent.memory import MememMemory
    except Exception:
        print("  (skip: memem not importable)"); return

    cap = {"scope": None, "writeback": None, "bumped": []}

    def fake_retrieve(query, k=8, log_call_type="hook_auto", scope_id="", writeback=True, paths_context=None):
        cap["scope"], cap["writeback"] = scope_id, writeback
        return [
            {"id": "m1", "path": "a.md", "title": "calc helper", "body": "how to add two numbers", "score": 2.0},
            {"id": "m2", "path": "b.md", "title": "TechFeed", "body": "async feed fetching via gather", "score": 0.1},
        ]

    orig_r, orig_b = memem.retrieve.retrieve, memem.obsidian_store.bump_access
    memem.retrieve.retrieve = fake_retrieve
    memem.obsidian_store.bump_access = lambda mid: cap["bumped"].append(mid)
    try:
        m = MememMemory()
        out = m.recall("create calc.py with an add function")
        assert len(out) == 1 and out[0].path == "a.md"        # m2 (noise) gated out
        assert cap["bumped"] == ["m1"]                          # only the surfaced/relevant one reinforced
        assert cap["scope"] == m._scope and cap["writeback"] is False  # scope passed; we own the writeback
    finally:
        memem.retrieve.retrieve, memem.obsidian_store.bump_access = orig_r, orig_b


@check
def mark_used_delegates_to_bump_access():
    try:
        import memem.obsidian_store
        from memagent.memory import MememMemory
    except Exception:
        print("  (skip: memem not importable)"); return
    got = []
    orig = memem.obsidian_store.bump_access
    memem.obsidian_store.bump_access = lambda mid: got.append(mid)
    try:
        m = MememMemory()
        m.mark_used("abc123")
        m.mark_used("")                # empty id → no-op
        assert got == ["abc123"]
    finally:
        memem.obsidian_store.bump_access = orig


@check
def nullmemory_recall_and_mark_used_safe():
    from memagent.memory import NullMemory
    m = NullMemory()
    assert m.recall("x") == []
    m.mark_used("anything")            # no-op, no crash


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
