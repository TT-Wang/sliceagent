"""Regression tests for the P0 trust-boundary fixes: #6/#7 plugins are opt-in (host-priv code must not
auto-load), #4 proc/terminal warn when running outside a non-local sandbox. No model, no pytest.
Run: PYTHONPATH=src python tests/test_bugfix_p0c.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.plugins import load_plugins  # noqa: E402
from sliceagent.registry import ToolRegistry  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _plugin_dir():
    d = tempfile.mkdtemp(prefix="plug-")
    p = os.path.join(d, "evil")
    os.makedirs(p)
    with open(os.path.join(p, "plugin.toml"), "w") as f:
        f.write('name = "evil"\n')
    # a plugin whose import would have a side effect (writes a marker) — proves it did/didn't execute
    marker = os.path.join(d, "EXECUTED")
    with open(os.path.join(p, "__init__.py"), "w") as f:
        f.write(f"open({marker!r}, 'w').write('x')\ndef register(ctx):\n    pass\n")
    return d, marker


@check
def plugins_do_not_load_without_optin():  # #6/#7
    d, marker = _plugin_dir()
    os.environ.pop("AGENT_ALLOW_PLUGINS", None)
    logs = []
    mcp, hooks = load_plugins(ToolRegistry(), None, [d], root=d, config=None, on_log=logs.append)
    assert not os.path.exists(marker), "plugin code must NOT execute without AGENT_ALLOW_PLUGINS"
    assert mcp == {} and hooks == []
    assert any("NOT loaded" in m for m in logs), logs


@check
def plugins_load_with_explicit_optin():  # #6/#7
    d, marker = _plugin_dir()
    os.environ["AGENT_ALLOW_PLUGINS"] = "1"
    logs = []
    try:
        load_plugins(ToolRegistry(), None, [d], root=d, config=None, on_log=logs.append)
        assert os.path.exists(marker), "with opt-in, the plugin executes"
        assert any("host privileges" in m for m in logs), logs
    finally:
        os.environ.pop("AGENT_ALLOW_PLUGINS", None)


@check
def proc_terminal_warn_under_nonlocal_sandbox():  # #4
    class _FakeDocker:
        scrub_secrets = True
    host = LocalToolHost(tempfile.mkdtemp(prefix="hw-"), sandbox=_FakeDocker())
    out = host.run("proc_start", {"command": "sleep 30"})
    assert "HOST" in out and "isolation does not apply" in out, out
    host.cleanup()
    # local sandbox → no warning
    host2 = LocalToolHost(tempfile.mkdtemp(prefix="hw2-"))   # default LocalSandbox
    out2 = host2.run("proc_start", {"command": "sleep 30"})
    assert "HOST" not in out2, out2
    host2.cleanup()


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
