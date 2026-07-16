"""Windows-footgun lint gate + platform_compat POSIX-identity checks.

Two guarantees: (1) scripts/check_windows_footguns.py stays clean, so the Unix-only bug CLASS
(cp1252 open, os.kill(pid,0), bare killpg/setsid, signal.SIGKILL, stray shell=True) can't creep
back in; (2) on POSIX, platform_compat's helpers return EXACTLY what the call sites inlined before
the seam existed — the zero-Linux/macOS-impact contract, pinned as a test.
No model. Run: PYTHONPATH=src python tests/test_windows_footguns.py
"""
import os
import signal
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def footgun_lint_is_clean():
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "check_windows_footguns.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    assert r.returncode == 0, f"footgun lint found violations:\n{r.stdout}"


@check
def posix_sh_is_identical_to_old_inline_call():
    from sliceagent import platform_compat as pc
    if pc.IS_WINDOWS:
        return  # POSIX-identity contract only applies off-Windows
    assert pc.sh("echo hi") == {"args": "echo hi", "shell": True}


@check
def posix_group_kwargs_identical():
    from sliceagent import platform_compat as pc
    if pc.IS_WINDOWS:
        return
    assert pc.popen_group_kwargs() == {"start_new_session": True}
    assert pc.SIG_KILL == signal.SIGKILL


@check
def posix_kill_tree_kills_whole_group():
    """kill_tree must take down a child AND its grandchild (the old killpg behavior)."""
    from sliceagent import platform_compat as pc
    if pc.IS_WINDOWS:
        return
    p = subprocess.Popen("sleep 30 & wait", shell=True, start_new_session=True)
    pc.kill_tree(p, pc.SIG_KILL)
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise AssertionError("kill_tree did not terminate the process group")


@check
def terminal_module_imports_even_without_pty():
    """terminal.py must import (not crash) when pty/fcntl are absent — the Windows import path."""
    import importlib
    import sliceagent.terminal as t
    importlib.reload(t)
    assert hasattr(t, "SessionManager")
    # simulate the Windows state: pty gated off → open() must refuse with a clear error
    old = t.pty
    try:
        t.pty = None
        sm = t.SessionManager()
        try:
            sm.open("x", cwd="/tmp")
            raise AssertionError("open() should refuse when pty is unavailable")
        except ValueError as e:
            assert "PTY sessions aren't available" in str(e)
    finally:
        t.pty = old


@check
def native_windows_docker_fails_before_building_invalid_mounts():
    import sliceagent.sandbox as sandbox
    old = sandbox.IS_WINDOWS
    try:
        sandbox.IS_WINDOWS = True
        try:
            sandbox.make_sandbox("docker")
            raise AssertionError("native Windows Docker backend must fail early")
        except ValueError as exc:
            message = str(exc)
            assert "native Windows" in message and "AGENT_SANDBOX=local" in message and "WSL2" in message
    finally:
        sandbox.IS_WINDOWS = old


if __name__ == "__main__":
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)
