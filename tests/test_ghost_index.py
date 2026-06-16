"""Core-review #4 — GHOST INDEX: recovery pointers to recently paged-out files/skills (refs only).
No model, no pytest. Run: PYTHONPATH=src python tests/test_ghost_index.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.slice import (MAX_ACTIVE_SKILLS, MAX_GHOSTS, READ_BUDGET, Slice,  # noqa: E402
                            add_skill, render_ghosts, render_slice, touch_file)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def evicted_reads_become_ghosts_and_clear_on_reread():
    s = Slice(); s.reset("t")
    for i in range(6):                          # 6 reads, READ_BUDGET keeps the last 4
        touch_file(s, f"f{i}.py")
    refs = [g["ref"] for g in s.ghosts]
    assert "f0.py" in refs and "f1.py" in refs, refs
    assert len(s.active_files) == READ_BUDGET
    touch_file(s, "f0.py")                       # bring an evicted file back
    assert "f0.py" not in [g["ref"] for g in s.ghosts], "re-read must clear its ghost"
    assert "f0.py" in s.active_files


@check
def edited_files_are_never_ghosted():
    s = Slice(); s.reset("t")
    for i in range(3):
        touch_file(s, f"e{i}.py", edited=True)   # change set — protected
    for i in range(6):
        touch_file(s, f"r{i}.py")                # exploratory reads — evictable
    for i in range(3):
        assert f"e{i}.py" in s.active_files, "edited files must never be evicted"
    assert not any(g["ref"].startswith("e") for g in s.ghosts), "edited files must never be ghosted"
    assert any(g["ref"].startswith("r") for g in s.ghosts)


@check
def evicted_skills_become_ghosts_and_clear_on_reload():
    s = Slice(); s.reset("t")
    for i in range(MAX_ACTIVE_SKILLS + 1):
        add_skill(s, f"skill{i}", "body")
    assert any(g["kind"] == "skill" and g["ref"] == "skill0" for g in s.ghosts), s.ghosts
    add_skill(s, "skill0", "body")               # reload it
    assert not any(g["kind"] == "skill" and g["ref"] == "skill0" for g in s.ghosts)


@check
def ghost_ring_is_bounded():
    s = Slice(); s.reset("t")
    for i in range(30):
        touch_file(s, f"g{i}.py")
    assert len(s.ghosts) <= MAX_GHOSTS, len(s.ghosts)


@check
def render_shows_pointers_and_is_suppressed_when_empty():
    assert render_ghosts(Slice()) == ""
    s = Slice(); s.reset("t")
    for i in range(6):
        touch_file(s, f"x{i}.py")
    body = render_ghosts(s)
    assert "read_file" in body and "x0.py" in body, body
    assert "GHOST INDEX" in render_slice(s, "(open files)")
    assert "GHOST INDEX" not in render_slice(Slice(), "(open files)")


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
