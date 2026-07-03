"""PTY tests for _menu_select — the wizard's VERTICAL arrow menu. Regression for the live wizard bug:
the single-line _arrow_select wrapped with 6 long provider labels and its one-line redraw stacked copies
of itself down the screen. The vertical menu must (a) return the arrow-chosen index, (b) cancel on Esc,
and (c) redraw IN PLACE — cursor-up over exactly its own N rows (the anti-stacking invariant).
Run: PYTHONPATH=src python tests/test_menu_select.py   (skips cleanly where no pty is available)
"""
import os
try:
    import pty
except ImportError:  # Windows: no pty — this drives a real POSIX terminal
    print("SKIP: no pty module on this platform")
    import sys as _sys
    _sys.exit(0)
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


_OPTS = [f"Provider number {i} with a deliberately long label to force the old wrap bug ({i})"
         for i in range(6)]

_CHILD = (
    "import os, sys\n"
    "from sliceagent.tui import _menu_select\n"
    f"opts = {_OPTS!r}\n"
    "idx = _menu_select(opts, default=0)\n"
    "os._exit(100 if idx is None else (99 if idx == -1 else 42 + idx))\n"
)


def _run(presses: list) -> tuple:
    """Run the child on a pty; return (exit_code, output_bytes)."""
    try:
        master, slave = pty.openpty()
    except OSError:
        return None, b""
    env = {**os.environ, "PYTHONPATH": os.path.abspath(_SRC), "COLUMNS": "120", "LINES": "40"}
    p = subprocess.Popen([sys.executable, "-c", _CHILD], stdin=slave, stdout=slave,
                         stderr=subprocess.DEVNULL, env=env, start_new_session=True)
    os.close(slave)
    buf = bytearray()
    deadline = time.monotonic() + 15
    sent = 0
    while time.monotonic() < deadline and p.poll() is None:
        r, _, _ = select.select([master], [], [], 0.1)
        if r:
            try:
                buf.extend(os.read(master, 4096))
            except OSError:
                break
        # feed keys only once the menu has rendered (its last row is on screen)
        if sent < len(presses) and _OPTS[-1][:20].encode() in buf:
            os.write(master, presses[sent]); sent += 1
            time.sleep(0.15)
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()
    try:
        os.close(master)
    except OSError:
        pass
    return p.returncode, bytes(buf)


@check
def down_down_enter_picks_the_third_option():
    rc, _out = _run([b"\x1b[B", b"\x1b[B", b"\r"])
    if rc is None:
        return   # no pty here — skip
    assert rc == 44, f"expected idx 2 (exit 44), got {rc}"


@check
def esc_cancels():
    rc, _out = _run([b"\x1b"])
    if rc is None:
        return
    assert rc == 99, f"expected cancel (exit 99), got {rc}"


@check
def redraw_is_in_place_not_stacked():
    rc, out = _run([b"\x1b[B", b"\r"])
    if rc is None:
        return
    assert rc == 43, f"expected idx 1 (exit 43), got {rc}"
    # the anti-stacking invariant: every redraw walks the cursor UP over its own 6 rows
    assert b"\x1b[6A" in out, "redraw must cursor-up over exactly its own rows (in-place, not stacking)"
    # and rows never exceed the terminal width (wrap is what broke the single-line selector)
    import re
    plain = re.sub(rb"\x1b\[[0-9;]*[A-Za-z]", b"", out)
    for line in plain.split(b"\r\n"):
        assert len(line) <= 120, f"a rendered row exceeds the terminal width ({len(line)} cols) — wrap risk"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
