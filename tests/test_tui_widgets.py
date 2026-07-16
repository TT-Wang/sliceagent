"""PTY widget tests — drive the rich-TUI keyboard selector with REAL key bytes through a pseudo-terminal
and assert the result. Covers the generic `_arrow_select` widget: ←/→ move, Enter
chooses, hotkeys jump, app-cursor arrows, combined arrow+Enter. This is the keyboard-selection path that
regressed three times; only a real pty + real escape bytes tests it faithfully.

Run: PYTHONPATH=src python tests/test_tui_widgets.py   (skips cleanly where no pty is available)
"""
import os
import select
import subprocess
import sys
import time

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _SRC)

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


_CODE = (
    "import sys; from sliceagent.tui import _arrow_select; import os; "
    "r=_arrow_select(['Yes','No','Always']); os._exit(r if isinstance(r,int) and r>=0 else 9)"
)


def _drive(presses) -> int:
    """Spawn _arrow_select on a pty slave; send each keypress (with a gap → its own read); return the index."""
    import pty
    master, slave = pty.openpty()
    env = {**os.environ, "PYTHONPATH": os.path.abspath(_SRC)}
    p = subprocess.Popen([sys.executable, "-c", _CODE], stdin=slave, stdout=slave,
                         stderr=subprocess.DEVNULL, env=env)
    os.close(slave)
    import threading
    stop = threading.Event()
    ready = threading.Event()

    def drain():
        while not stop.is_set():
            try:
                r, _, _ = select.select([master], [], [], 0.1)
            except OSError:
                break
            if r:
                try:
                    if os.read(master, 4096):
                        ready.set()
                except OSError:
                    break
    drain_thread = threading.Thread(target=drain, daemon=True)
    drain_thread.start()
    # Wait for the selector's first draw instead of assuming every subprocess
    # reaches raw mode within a fixed delay. The old 0.6s sleep became flaky
    # late in the full standalone battery on a loaded machine.
    ready.wait(timeout=3.0)
    for chunk in presses:
        os.write(master, chunk)
        time.sleep(0.15)                  # gap → each keypress arrives as its own os.read (like a human)
    try:
        p.wait(timeout=8)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        rc = -1
    stop.set()
    drain_thread.join(timeout=0.5)
    try:
        os.close(master)
    except OSError:
        pass
    return rc


@check
def enter_selects_default_yes():
    assert _drive([b"\r"]) == 0, "Enter alone should choose the default (Yes=0)"


@check
def right_then_enter_selects_no():
    assert _drive([b"\x1b[C", b"\r"]) == 1, "Right then Enter should choose No=1"


@check
def two_rights_enter_selects_always():
    assert _drive([b"\x1b[C", b"\x1b[C", b"\r"]) == 2, "Right x2 then Enter should choose Always=2"


@check
def left_wraps_to_always():
    assert _drive([b"\x1b[D", b"\r"]) == 2, "Left from Yes should wrap to Always=2"


@check
def app_cursor_right_selects_no():
    assert _drive([b"\x1bOC", b"\r"]) == 1, "Application-cursor Right (ESC O C) should choose No=1"


@check
def hotkey_n_then_enter_selects_no():
    assert _drive([b"n", b"\r"]) == 1, "hotkey 'n' then Enter should choose No=1"


@check
def combined_hotkey_and_enter_one_read_selects_no():
    assert _drive([b"n\r"]) == 1, "hotkey+Enter delivered in one read should choose No=1"


@check
def combined_arrow_and_enter_one_read_selects_no():
    assert _drive([b"\x1b[C\r"]) == 1, "arrow+Enter delivered in one read should still choose No=1"


def main():
    try:
        import pty  # noqa: F401
    except Exception:  # noqa: BLE001
        print("SKIP test_tui_widgets: no pty on this platform")
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
