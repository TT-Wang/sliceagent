"""Interactive PTY session tools (terminal_*) — drive REPLs/games, hold shell+env across turns.
The other half of the live-process gap. Deterministic-ish (uses real PTYs + the `wait` expect
primitive with generous timeouts), no model, no pytest.
Run: PYTHONPATH=src python tests/test_terminal.py
"""
import os
import shlex
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.tools import LocalToolHost  # noqa: E402

PY = shlex.quote(sys.executable)
CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _host():
    wd = tempfile.mkdtemp(prefix="term-")
    return wd, LocalToolHost(root=wd)


@check
def tools_registered():
    _, h = _host()
    names = {s["function"]["name"] for s in h.schemas()}
    for t in ("terminal_open", "terminal_send", "terminal_read", "terminal_wait", "terminal_close"):
        assert t in names, f"{t} not registered"


@check
def shell_env_persists_across_sends():
    """cd/export must survive between tool calls — the persistent-shell use case."""
    _, h = _host()
    h.run("terminal_open", {"session": "s"})
    h.run("terminal_send", {"session": "s", "input": "export X=hello"})
    h.run("terminal_send", {"session": "s", "input": "echo MARKER:$X:END"})
    out = h.run("terminal_wait", {"session": "s", "until": r"MARKER:hello:END", "timeout": 5})
    assert "matched" in out, f"env did not persist across sends: {out!r}"
    h.run("terminal_close", {"session": "s"})


@check
def drive_a_repl():
    """Open a live Python REPL and interact with it (the zork/REPL pattern)."""
    _, h = _host()
    h.run("terminal_open", {"session": "py", "command": f"{PY} -i -q -u"})
    h.run("terminal_send", {"session": "py", "input": "print(6*7)"})
    out = h.run("terminal_wait", {"session": "py", "until": r"\b42\b", "timeout": 5})
    assert "matched" in out, f"REPL did not answer: {out!r}"
    h.run("terminal_close", {"session": "py"})


@check
def successive_prompts():
    """A program that asks two questions in turn — read live, answer, read the next (text game)."""
    wd, h = _host()
    open(os.path.join(wd, "quiz.py"), "w").write(
        "import sys\n"
        "print('Q1?', flush=True); a = sys.stdin.readline().strip()\n"
        "print(f'got:{a}', flush=True)\n"
        "print('Q2?', flush=True); b = sys.stdin.readline().strip()\n"
        "print(f'done:{b}', flush=True)\n")
    h.run("terminal_open", {"session": "q", "command": f"{PY} quiz.py"})
    assert "matched" in h.run("terminal_wait", {"session": "q", "until": "Q1", "timeout": 5})
    h.run("terminal_send", {"session": "q", "input": "alpha"})
    assert "matched" in h.run("terminal_wait", {"session": "q", "until": "got:alpha", "timeout": 5})
    h.run("terminal_send", {"session": "q", "input": "beta"})
    assert "matched" in h.run("terminal_wait", {"session": "q", "until": "done:beta", "timeout": 5})
    h.run("terminal_close", {"session": "q"})


@check
def close_then_use_errors():
    _, h = _host()
    h.run("terminal_open", {"session": "z"})
    h.run("terminal_close", {"session": "z"})
    out = h.run("terminal_send", {"session": "z", "input": "echo hi"})
    assert "unknown session" in out, out


@check
def cleanup_closes_all():
    _, h = _host()
    h.run("terminal_open", {"session": "a"})
    h.run("terminal_open", {"session": "b"})
    h.terminals.cleanup()
    assert "unknown session" in h.run("terminal_read", {"session": "a"}).lower() \
        or "unknown session" in h.run("terminal_read", {"session": "a"})


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
