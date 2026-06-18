"""CHANGE-SET CLOSURE — symbol-aware 'verify before done'. SwapManager.prefetch flags dependents whose
CURRENT tokens still reference a symbol an edit removed/moved (a dangling call-site); render_closure shows
the UNOPENED ones, outranks the done-nudge, and self-extinguishes when opened. Silent on feature-adds
(nothing removed) so it never inflates non-refactor tasks. Deterministic — no model.
Run: PYTHONPATH=src python tests/test_closure.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.regions import render_closure, render_convergence, STOP_NUDGE_AFTER, CLOSURE_MAX_SHOWN  # noqa: E402
from memagent.slice import Slice                                                                      # noqa: E402
from memagent.swap import SwapManager                                                                 # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _state(edited, stale, active, since_edit=STOP_NUDGE_AFTER, err=""):
    s = Slice(); s.reset("coordinated refactor")
    s.edited_files = set(edited)
    s.stale_deps = set(stale)          # what prefetch would have computed
    s.active_files = list(active)
    s.since_edit = since_edit
    s.last_error = err
    return s


# ── render_closure: shows unopened dangling deps, outranks the done-nudge, self-extinguishes ──
@check
def fires_on_unopened_dangling_dep():
    s = _state(edited=["pkg/a.py"], stale=["pkg/b.py", "pkg/c.py"], active=["pkg/a.py"])
    out = render_closure(s)
    assert "CHANGE-SET CLOSURE" in out and "pkg/b.py" in out and "pkg/c.py" in out, out
    assert render_convergence(s) == "", "closure must suppress the STOP nudge while a dangling dep is open"


@check
def self_extinguishes_when_opened():
    s = _state(edited=["pkg/a.py"], stale=["pkg/b.py"], active=["pkg/a.py", "pkg/b.py"])
    assert render_closure(s) == "", "opening the stale dep clears the nudge"
    assert "CONVERGENCE CHECK" in render_convergence(s), "done-nudge returns once closed"


@check
def edited_dep_counts_as_reached():
    s = _state(edited=["pkg/a.py", "pkg/b.py"], stale=["pkg/b.py"], active=["pkg/a.py"])
    assert render_closure(s) == ""


@check
def silent_when_nothing_stale():            # feature-add / no-graph host -> empty stale set
    s = _state(edited=["pkg/a.py"], stale=[], active=["pkg/a.py"])
    assert render_closure(s) == ""
    assert "CONVERGENCE CHECK" in render_convergence(s), "no dangling dep -> normal convergence"


@check
def silent_while_error_open():
    s = _state(edited=["pkg/a.py"], stale=["pkg/b.py"], active=["pkg/a.py"], err="Boom")
    assert render_closure(s) == ""


@check
def silent_before_settle():
    s = _state(edited=["pkg/a.py"], stale=["pkg/b.py"], active=["pkg/a.py"], since_edit=STOP_NUDGE_AFTER - 1)
    assert render_closure(s) == ""


@check
def bounded_locator_count():
    stale = [f"pkg/d{i}.py" for i in range(CLOSURE_MAX_SHOWN + 5)]
    s = _state(edited=["pkg/a.py"], stale=stale, active=["pkg/a.py"])
    shown = sum(1 for d in stale if d in render_closure(s))
    assert shown == CLOSURE_MAX_SHOWN, f"bounded to {CLOSURE_MAX_SHOWN}, showed {shown}"


# ── prefetch: the SYMBOL-AWARE staleness computation (the actual fix) ──
class _FakeRetriever:
    """def_names/ref_tokens/deps over a tiny fixed graph. A.py used to define {foo, bar}; after the edit
    it defines only {foo} (bar moved). B references bar (dangling); C references baz (fine)."""
    def deps(self, path, limit=6): return ["pkg/b.py", "pkg/c.py"]
    def def_names(self, path): return {"foo"} if path == "pkg/a.py" else set()
    def ref_tokens(self, path): return {"bar"} if path == "pkg/b.py" else {"baz"}


@check
def prefetch_flags_only_the_dangling_dependent():
    s = Slice(); s.reset("move bar out of a.py")
    s.pre_defs = {"pkg/a.py": {"foo", "bar"}}      # snapshot BEFORE the edit
    s.edited_files = {"pkg/a.py"}
    s.active_files = ["pkg/a.py"]
    SwapManager(_FakeRetriever()).prefetch(s)
    assert s.stale_deps == {"pkg/b.py"}, f"only the dep referencing the removed 'bar' is stale: {s.stale_deps}"


@check
def prefetch_silent_when_nothing_removed():     # feature-add: current defs >= pre defs
    s = Slice(); s.reset("add a feature to a.py")
    s.pre_defs = {"pkg/a.py": {"foo"}}            # nothing removed (a.py still defines foo)
    s.edited_files = {"pkg/a.py"}
    s.active_files = ["pkg/a.py"]
    SwapManager(_FakeRetriever()).prefetch(s)
    assert s.stale_deps == set(), f"no removed symbol -> no stale deps: {s.stale_deps}"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
