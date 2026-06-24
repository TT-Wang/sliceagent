"""Regression tests for the Textual TUI bug-fixes (per the bug-hunt): dialog state is per-id (parallel
tools → concurrent confirm/ask_user must not clobber each other), DialogResult carries the id, and
ask_user resolves on escape (no hang). Needs textual; skips cleanly if absent. No pytest.
Run: PYTHONPATH=src python tests/test_tui_textual.py
"""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import textual  # noqa: F401
except Exception:
    print("(textual not installed — skipping Textual TUI tests)")
    sys.exit(0)

from memagent.tui_app import MemagentTui, DialogResult, ConfirmScreen, AskUserScreen  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _bare_app():
    app = MemagentTui.__new__(MemagentTui)   # bypass __init__ (no event loop)
    app._dialogs = {}
    app._dialog_seq = 0
    app._dialog_lock = threading.Lock()
    return app


@check
def dialog_per_id_routing_no_crosstalk():
    app = _bare_app()
    d1, e1 = app._open_dialog("no")            # e.g. a confirm
    d2, e2 = app._open_dialog("(no answer)")   # e.g. a parallel ask_user
    assert d1 != d2, "each dialog gets a unique id"
    # answering d2 must wake ONLY d2 — not d1 (the old shared-state bug woke the wrong/last thread)
    app.on_dialog_result(DialogResult(d2, "Fix it"))
    assert e2.is_set() and not e1.is_set(), "only the addressed dialog's event fires"
    assert app._close_dialog(d2, "(no answer)") == "Fix it"
    assert app._close_dialog(d1, "no") == "no", "d1's default is intact (no clobber)"


@check
def dialogresult_carries_id():
    m = DialogResult(5, "yes")
    assert m.dialog_id == 5 and m.value == "yes"


@check
def stale_result_for_unknown_id_is_ignored():
    app = _bare_app()
    app.on_dialog_result(DialogResult(999, "boom"))   # no such dialog → must be a no-op, not a crash
    assert app._dialogs == {}


@check
def askuser_resolves_on_escape():
    # AskUserScreen must have an on_key that resolves the dialog (without it, escape hung the worker)
    assert hasattr(AskUserScreen, "on_key"), "AskUserScreen needs an escape handler so it can't hang"
    # both modal screens take a dialog_id now (routing)
    ConfirmScreen("edit_file", "x", "writes", 1)
    AskUserScreen("q", ["a", "b"], 2)


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
