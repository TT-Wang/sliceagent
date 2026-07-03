"""PTY tests for _EscSentinel — Esc-to-interrupt in RICH/default mode. Drives a REAL pty with real key
bytes (same pattern as test_tui_widgets.py, which regressed 3 times without one): a lone Esc must deliver
a real SIGINT (interrupting even a simulated hung blocking call, which an Event-only design would miss); an
arrow/CSI sequence must NOT false-fire; physical Ctrl-C must still work while the sentinel holds raw mode
(tty.setraw disables ISIG, so the sentinel must re-implement Ctrl-C detection itself or it goes silent —
caught by this suite during development); confirm()'s arrow-key selector must be unaffected by a paused
sentinel; termios/threads must be fully cleaned up on stop().

Run: PYTHONPATH=src python tests/test_esc_sentinel.py   (skips cleanly where no pty is available)
"""
import os
try:
    import pty
except ImportError:  # Windows: no pty — the Esc sentinel is POSIX-only by design (inert on Windows)
    print("SKIP: no pty module on this platform")
    import sys as _sys
    _sys.exit(0)
import subprocess
import sys
import time

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _SRC)

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _run_child(code: str, presses: list, settle=0.35, timeout=8) -> int:
    """Spawn `code` on a pty slave, wait `settle`s, then walk `presses`: a bytes item is written to the
    child's stdin (small gap after); a str item is a MARKER — block until the child has printed it (≤3s)
    before continuing. Markers make key delivery deterministic (no fixed-sleep race against child startup
    or a still-active sentinel window). Returns the child's exit code (or -1 on timeout)."""
    master, slave = pty.openpty()
    env = {**os.environ, "PYTHONPATH": os.path.abspath(_SRC)}
    p = subprocess.Popen([sys.executable, "-c", code], stdin=slave, stdout=slave,
                         stderr=subprocess.DEVNULL, env=env)
    os.close(slave)
    import select
    import threading
    stop = threading.Event()
    buf = bytearray()                    # everything the child has written, for marker waits
    buf_lock = threading.Lock()

    def drain():
        while not stop.is_set():
            try:
                r, _, _ = select.select([master], [], [], 0.1)
                if r:
                    data = os.read(master, 4096)
                    with buf_lock:
                        buf.extend(data)
            except OSError:      # master closed by the main thread while select()/read() was in flight
                break
    threading.Thread(target=drain, daemon=True).start()

    def wait_marker(marker: bytes, deadline: float = 3.0) -> None:
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            with buf_lock:
                if marker in buf:
                    return
            time.sleep(0.02)
        # fall through on timeout — the exit-code assertion reports the real failure

    time.sleep(settle)
    for chunk in presses:
        if isinstance(chunk, str):       # marker: wait until the child says it's ready
            wait_marker(chunk.encode())
            continue
        try:
            os.write(master, chunk)
        except OSError:
            break
        time.sleep(0.15)
    try:
        p.wait(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        rc = -1
    finally:
        stop.set()
        try:
            os.close(master)
        except OSError:
            pass
    return rc


def _skip_if_no_pty() -> bool:
    try:
        m, s = pty.openpty()
        os.close(m); os.close(s)
        return False
    except Exception:  # noqa: BLE001
        return True


# ── bare Esc delivers SIGINT, promptly, even during a simulated blocking call ──────────────────────────
_CODE_ESC_ABORTS = (
    "import time, os\n"
    "from sliceagent.tui import make_esc_sentinel\n"
    "s = make_esc_sentinel(); s.start()\n"
    "t0 = time.monotonic()\n"
    "try:\n"
    "    time.sleep(20)\n"          # simulates a long blocking LLM/tool call
    "    os._exit(1)\n"             # reached WITHOUT interrupt -> fail
    "except KeyboardInterrupt:\n"
    "    dt = time.monotonic() - t0\n"
    "    s.stop()\n"
    "    os._exit(42 if dt < 3.0 else 43)\n"   # 42 = aborted promptly; 43 = aborted but too slow
)


@check
def bare_esc_delivers_sigint_promptly_even_mid_blocking_call():
    if _skip_if_no_pty():
        return
    rc = _run_child(_CODE_ESC_ABORTS, [b"\x1b"])
    assert rc == 42, f"expected prompt SIGINT-abort (42), got {rc} (43=too slow, 1=never interrupted)"


# ── an arrow/CSI escape sequence must NOT be mistaken for a bare Esc ───────────────────────────────────
_CODE_NO_FALSE_FIRE = (
    "import time, os\n"
    "from sliceagent.tui import make_esc_sentinel\n"
    "s = make_esc_sentinel(); s.start()\n"
    "try:\n"
    "    time.sleep(2.0)\n"
    "    s.stop()\n"
    "    os._exit(50)\n"            # reached the end untouched -> correct
    "except KeyboardInterrupt:\n"
    "    os._exit(51)\n"            # false-fired on a CSI sequence -> bug
)


@check
def arrow_csi_sequence_does_not_false_fire():
    if _skip_if_no_pty():
        return
    rc = _run_child(_CODE_NO_FALSE_FIRE, [b"\x1b[C"])   # a right-arrow CSI sequence, not bare Esc
    assert rc == 50, f"an arrow/CSI sequence must not trigger an abort, got {rc}"


# ── physical Ctrl-C must still work WHILE the sentinel holds raw mode (tty.setraw disables ISIG, so the
# tty driver's own auto-SIGINT-on-Ctrl-C is off — the sentinel must re-implement it itself) ────────────
_CODE_CTRLC_STILL_WORKS = _CODE_ESC_ABORTS  # identical child; only the driven byte differs


@check
def physical_ctrl_c_still_aborts_while_sentinel_holds_raw_mode():
    if _skip_if_no_pty():
        return
    rc = _run_child(_CODE_CTRLC_STILL_WORKS, [b"\x03"])   # real Ctrl-C byte
    assert rc == 42, (f"Ctrl-C must still abort promptly while the sentinel is active (raw mode disables "
                      f"the tty driver's own SIGINT-on-Ctrl-C — the sentinel must handle \\x03 itself), "
                      f"got {rc}")


# ── confirm()'s arrow-key selector is UNAFFECTED by a paused sentinel: pause -> _arrow_select (real arrow
# keys) -> resume -> a LATER bare Esc still aborts. This is the central "UX preserved, not traded" claim ──
_CODE_PAUSE_RESUME_ROUNDTRIP = (
    "import time, os\n"
    "from sliceagent.tui import make_esc_sentinel, _arrow_select\n"
    "s = make_esc_sentinel(); s.start()\n"
    "time.sleep(0.3)\n"
    "s.pause()\n"                                  # mirrors _pause_active_live() before a confirm() prompt
    "print('MARK_SELECT_READY', flush=True)\n"     # handshake: selector owns the tty from here
    "idx = _arrow_select(['Yes', 'No', 'Always'], default=0)\n"
    "s.resume()\n"                                  # mirrors _resume_active_esc_sentinel() after confirm()
    "print('MARK_RESUMED', flush=True)\n"          # handshake: sentinel re-armed, Esc may fire now
    "if idx != 1:\n"
    "    os._exit(60 + (idx if isinstance(idx, int) else 9))\n"   # arrow-select result wrong -> sentinel interfered
    "try:\n"
    "    time.sleep(10)\n"
    "    os._exit(1)\n"
    "except KeyboardInterrupt:\n"
    "    s.stop()\n"
    "    os._exit(42)\n"                            # arrow-select worked AND post-resume Esc still aborts
)


@check
def confirm_arrow_select_unaffected_by_a_paused_sentinel_and_resumes_after():
    if _skip_if_no_pty():
        return
    # drive, handshake-gated: wait for the selector to own the tty -> right-arrow (default=0 -> idx=1) ->
    # Enter (choose) -> wait for the sentinel to re-arm -> bare Esc. The markers eliminate the startup /
    # sentinel-window races (keys can otherwise land while the sentinel still owns stdin and be consumed).
    rc = _run_child(_CODE_PAUSE_RESUME_ROUNDTRIP,
                    ["MARK_SELECT_READY", b"\x1b[C", b"\r", "MARK_RESUMED", b"\x1b"])
    assert rc == 42, (f"expected: arrow-select returns idx=1 (60+1=61 on mismatch) AND a later Esc still "
                      f"aborts (42) — got {rc}")


# ── stop() after a normal (non-aborted) turn leaves termios as it was, and the thread is gone ──
# lflag (index 3) is masked against PENDIN (0x20000000, macOS's "retype pending input" kernel bit): a raw
# os.read()-driven raw-mode enter/exit cycle can leave THIS bit differing even on a byte-perfect tcsetattr
# restore — verified the EXISTING, already-shipped _arrow_select has the identical discrepancy after its
# own restore, so this is a harmless macOS pty kernel quirk shared by the codebase's proven raw-mode idiom,
# not something introduced here. Every OTHER termios field (iflag/oflag/cflag/speeds/cc[]) must match exactly.
_CODE_CLEAN_STOP = (
    "import termios, sys, time, threading, os\n"
    "from sliceagent.tui import make_esc_sentinel\n"
    "fd = sys.stdin.fileno()\n"
    "before = termios.tcgetattr(fd)\n"
    "s = make_esc_sentinel(); s.start()\n"
    "time.sleep(0.3)\n"
    "alive_during = s._thread is not None and s._thread.is_alive()\n"
    "s.stop()\n"
    "time.sleep(0.15)\n"
    "alive_after = s._thread is not None and s._thread.is_alive()\n"
    "after = termios.tcgetattr(fd)\n"
    "PENDIN = termios.PENDIN\n"
    "lflag_matches = (before[3] | PENDIN) == (after[3] | PENDIN)\n"
    "other_fields_match = before[:3] == after[:3] and before[4:] == after[4:]\n"
    "ok = lflag_matches and other_fields_match and alive_during and not alive_after\n"
    "os._exit(42 if ok else 70)\n"
)


@check
def stop_restores_termios_and_leaves_no_thread():
    if _skip_if_no_pty():
        return
    rc = _run_child(_CODE_CLEAN_STOP, [])
    assert rc == 42, f"termios must be restored and the thread must be gone after stop(), got {rc}"


# ── while the sentinel holds the tty, OUTPUT must stay cooked (OPOST on) so the turn's reply doesn't
#    staircase — but INPUT must stay raw (ICANON off) so Esc/Ctrl-C read as bytes. Regression: tty.setraw
#    clears OPOST, which made every reply drift right (a today regression from the Esc-interrupt feature).
_CODE_OUTPUT_STAYS_COOKED = (
    "import time, os, sys, termios\n"
    "from sliceagent.tui import make_esc_sentinel\n"
    "s = make_esc_sentinel(); s.start()\n"
    "time.sleep(0.3)\n"                                  # let the sentinel enter raw mode
    "a = termios.tcgetattr(sys.stdin.fileno())\n"
    "opost_on = bool(a[1] & termios.OPOST)\n"            # output cooked -> \\n maps to \\r\\n -> no staircase
    "icanon_off = not (a[3] & termios.ICANON)\n"         # input raw -> Esc/Ctrl-C arrive as bytes
    "s.stop()\n"
    "os._exit(60 if (opost_on and icanon_off) else 61)\n"
)


@check
def output_stays_cooked_while_input_is_raw():
    if _skip_if_no_pty():
        return
    rc = _run_child(_CODE_OUTPUT_STAYS_COOKED, [])
    assert rc == 60, f"OPOST must stay ON (cooked output, no staircase) with ICANON OFF (raw input), got {rc}"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
