"""Keyboard-MENU selection tests — drive run_selector (the picker behind /model and /mode) headlessly with
a prompt_toolkit pipe input, feeding real arrow/Enter/Esc key bytes and asserting the chosen index. Together
with test_tui_widgets (the arrow permission-confirm) this covers every keyboard-selection surface offline.

No model, no pty. Run: PYTHONPATH=src python tests/test_tui_menus.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


ROWS = [("alpha", "first"), ("bravo", "second"), ("charlie", "third")]


def _drive(keys: str, current: int = 0):
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from memagent.tui import run_selector
    with create_pipe_input() as pinp:
        pinp.send_text(keys)
        return run_selector("pick", ROWS, current=current, pt_input=pinp, pt_output=DummyOutput())


@check
def enter_picks_current():
    assert _drive("\r", current=0) == 0, "Enter chooses the highlighted (current) row"


@check
def down_enter_picks_next():
    assert _drive("\x1b[B\r", current=0) == 1, "Down then Enter picks the next row"


@check
def down_down_enter_picks_third():
    assert _drive("\x1b[B\x1b[B\r", current=0) == 2, "two Downs then Enter picks row 2"


@check
def up_wraps_to_last():
    assert _drive("\x1b[A\r", current=0) == 2, "Up from the top wraps to the last row"


@check
def ctrl_n_moves_down():
    assert _drive("\x0e\r", current=0) == 1, "Ctrl-N moves down (emacs binding)"


@check
def escape_cancels_to_none():
    assert _drive("\x1b") is None, "Esc cancels the menu (returns None)"


def main():
    try:
        import prompt_toolkit  # noqa: F401
        from prompt_toolkit.input.defaults import create_pipe_input  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"SKIP test_tui_menus: prompt_toolkit unavailable ({type(e).__name__})")
        return
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
