"""Regression tests for the tools/sandbox/exec wave: #27/#28 factory validation, #31 secret-dir reach
exclusion, #19 terminal-open clean failure (no fd leak), #20 wait-pattern bounds. No model, no pytest.
Run: PYTHONPATH=src python tests/test_bugfix_tools_wave.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.sandbox import make_sandbox  # noqa: E402
from sliceagent.policy import make_policy  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402
from sliceagent.terminal import SessionManager  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def factories_reject_unknown_values():  # #27 / #28
    make_sandbox("local"); make_sandbox("docker")          # valid → no raise
    make_policy("guard"); make_policy("readonly"); make_policy("ask"); make_policy("allow")
    for bad in ("dokcer", "host", "none"):
        try:
            make_sandbox(bad); assert False, f"unknown backend {bad} must raise"
        except ValueError:
            pass
    for bad in ("redonly", "permissive", "off"):
        try:
            make_policy(bad); assert False, f"unknown policy {bad} must raise"
        except ValueError:
            pass


@check
def shell_path_grant_skips_secret_dirs():  # #31
    home = os.path.expanduser("~")
    base = tempfile.mkdtemp(dir=home, prefix=".sliceagent-sectest-")
    try:
        secret = os.path.join(base, ".aws"); os.makedirs(secret)
        open(os.path.join(secret, "credentials"), "w").write("x")
        normal = os.path.join(base, "data"); os.makedirs(normal)
        open(os.path.join(normal, "f.txt"), "w").write("x")
        host = LocalToolHost(tempfile.mkdtemp(prefix="ws-"))
        host._grant_shell_paths(f"cat '{secret}/credentials' '{normal}/f.txt'")
        roots = [os.path.realpath(r) for r in host._extra_roots]
        assert os.path.realpath(secret) not in roots, "must NOT auto-grant reach into a secret dir (#31)"
        assert os.path.realpath(normal) in roots, "a normal HOME dir IS granted (reach=action)"
    finally:
        shutil.rmtree(base, ignore_errors=True)


@check
def terminal_open_failure_is_clean():  # #19
    sm = SessionManager()
    raised = False
    try:
        sm.open("x", cwd="/no/such/dir/zzz-sliceagent")   # bad cwd → Popen raises
    except Exception:  # noqa: BLE001
        raised = True
    assert raised, "open() with a bad cwd must raise, not hang"
    assert "x" not in sm._s, "a failed open must not register a half-built session"


@check
def wait_pattern_is_bounded():  # #20
    sm = SessionManager()
    wd = tempfile.mkdtemp(prefix="term-")
    sm.open("s", cwd=wd)
    try:
        try:
            sm.wait("s", "(", timeout=0.3); assert False, "invalid regex must raise ValueError"
        except ValueError:
            pass
        try:
            sm.wait("s", "a" * 600, timeout=0.3); assert False, "over-long pattern must raise ValueError"
        except ValueError:
            pass
    finally:
        sm.cleanup()


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
