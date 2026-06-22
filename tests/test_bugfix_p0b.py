"""Regression tests for verified P0 security/cleanup bugs (#5 resource cleanup, #8 prelude path
confinement). No model, no pytest. Run: PYTHONPATH=src python tests/test_bugfix_p0b.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.tools import LocalToolHost, _CODE_PRELUDE  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def prelude_helpers_confined_to_workspace():  # #8
    wd = tempfile.mkdtemp(prefix="confine-")
    cwd0 = os.getcwd()
    os.chdir(wd)
    try:
        ns = {}
        exec(_CODE_PRELUDE, ns)
        assert "wrote" in ns["write_file"]("a.txt", "hi")      # in-workspace OK
        assert ns["read_file"]("a.txt") == "hi"
        for bad in ("/etc/hosts", "../escape.txt", "/tmp/x.txt"):
            try:
                ns["write_file"](bad, "x") if bad != "/etc/hosts" else ns["read_file"](bad)
                assert False, f"escape not blocked: {bad}"
            except PermissionError:
                pass
    finally:
        os.chdir(cwd0)


@check
def host_cleanup_kills_background_procs():  # #5
    wd = tempfile.mkdtemp(prefix="cleanup-")
    host = LocalToolHost(wd)
    h = host.procs.start("sleep 60", cwd=wd)
    assert "running" in host.procs.poll(h).lower() or "pid" in host.procs.poll(h).lower(), host.procs.poll(h)
    host.cleanup()
    assert not host.procs._procs, "cleanup must kill + drop all background procs"
    host.cleanup()  # idempotent — no raise on a second call


@check
def host_cleanup_safe_on_empty_host():  # #5
    LocalToolHost(tempfile.mkdtemp(prefix="cleanup2-")).cleanup()   # no procs/terminals → must not raise


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
