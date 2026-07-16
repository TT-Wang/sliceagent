"""Regression tests for the P0 trust-boundary fixes: #6/#7 plugins are opt-in (host-priv code must not
auto-load), #4 proc/terminal warn when running outside a non-local sandbox. No model, no pytest.
Run: PYTHONPATH=src python tests/test_bugfix_p0c.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.plugins import load_plugins  # noqa: E402
from sliceagent.registry import ToolEntry, ToolRegistry  # noqa: E402
from sliceagent.skills import SkillManager  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402
from sliceagent.cli import _plugin_tool_names  # noqa: E402

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
    mcp = load_plugins(ToolRegistry(), None, [d], root=d, config=None, on_log=logs.append)
    assert not os.path.exists(marker), "plugin code must NOT execute without AGENT_ALLOW_PLUGINS"
    assert mcp == {}
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


def _write_plugin(manifest: str, body: str, extra_files: dict[str, str] | None = None):
    root = tempfile.mkdtemp(prefix="plugin-atomic-")
    plugin = os.path.join(root, "demo")
    os.makedirs(plugin)
    with open(os.path.join(plugin, "plugin.toml"), "w", encoding="utf-8") as stream:
        stream.write(manifest)
    with open(os.path.join(plugin, "__init__.py"), "w", encoding="utf-8") as stream:
        stream.write(body)
    for name, content in (extra_files or {}).items():
        with open(os.path.join(plugin, name), "w", encoding="utf-8") as stream:
            stream.write(content)
    return root


@check
def failed_plugin_registration_rolls_back_tools_overrides_skills_and_mcp():
    root = _write_plugin('name = "demo"\n', '''
def register(ctx):
    ctx.registry.entry("kept").handler = lambda _args: "mutated-in-place"
    ctx.registry.entry("kept").schema["function"]["description"] = "mutated-in-place"
    ctx.skills._skills["kept-skill"].body = "mutated-in-place"
    ctx.register_tool("kept", "replacement", lambda _args: "new", override=True)
    ctx.register_tool("partial", "partial", lambda _args: "partial")
    ctx.register_skill("partial-skill", "partial body")
    ctx.register_mcp_server("partial", {"command": "partial-server"})
    ctx.registry._tools = None
    ctx.skills._skills = None
    raise RuntimeError("boom after registration")
''')
    registry = ToolRegistry()
    original = ToolEntry(
        name="kept",
        schema={"type": "function", "function": {"name": "kept", "description": "old",
                "parameters": {"type": "object", "properties": {}}}},
        handler=lambda _args: "old",
    )
    registry.register(original)
    generation = registry.generation
    skills = SkillManager([])
    skills.add("kept-skill", "original body")
    logs = []
    os.environ["AGENT_ALLOW_PLUGINS"] = "1"
    try:
        mcp = load_plugins(registry, skills, [root], root=root, config=None, on_log=logs.append)
    finally:
        os.environ.pop("AGENT_ALLOW_PLUGINS", None)
    assert registry.entry("kept") is original, "a failed override must restore the original ToolEntry"
    assert registry.run("kept", {}) == "old"
    assert registry.entry("kept").schema["function"]["description"] == "old"
    assert not registry.has("partial") and registry.generation == generation
    assert skills.names() == ["kept-skill"] and skills.load("kept-skill") == "original body"
    assert mcp == {} and any("failed: boom after registration" in item for item in logs), logs


@check
def plugin_init_is_a_real_package_and_keeps_lazy_relative_imports_available():
    root = _write_plugin('name = "relative-demo"\n', '''
from .helper import PREFIX

def register(ctx):
    def handler(_args):
        from .lazy import SUFFIX
        return PREFIX + SUFFIX
    ctx.register_tool("relative_import_tool", "relative import", handler)
''', {"helper.py": 'PREFIX = "package-"\n', "lazy.py": 'SUFFIX = "ok"\n'})
    registry = ToolRegistry(); logs = []
    os.environ["AGENT_ALLOW_PLUGINS"] = "1"
    try:
        load_plugins(registry, SkillManager([]), [root], root=root, config=None, on_log=logs.append)
    finally:
        os.environ.pop("AGENT_ALLOW_PLUGINS", None)
    assert registry.run("relative_import_tool", {}) == "package-ok", logs


@check
def malformed_manifest_name_and_systemexit_cannot_crash_or_partially_register():
    malformed = _write_plugin('name = 123\n', '''
def register(ctx):
    ctx.register_tool("coerced_name_tool", "ok", lambda _args: "ok")
''')
    registry = ToolRegistry(); logs = []
    os.environ["AGENT_ALLOW_PLUGINS"] = "1"
    try:
        load_plugins(registry, SkillManager([]), [malformed], root=malformed,
                     config=None, on_log=logs.append)
    finally:
        os.environ.pop("AGENT_ALLOW_PLUGINS", None)
    assert registry.has("coerced_name_tool")
    assert any("non-string manifest name" in item for item in logs), logs

    exiting = _write_plugin('name = "exiting"\n', '''
def register(ctx):
    ctx.register_tool("must_rollback", "partial", lambda _args: "partial")
    raise SystemExit(7)
''')
    registry = ToolRegistry(); logs = []
    os.environ["AGENT_ALLOW_PLUGINS"] = "1"
    try:
        load_plugins(registry, SkillManager([]), [exiting], root=exiting,
                     config=None, on_log=logs.append)
    finally:
        os.environ.pop("AGENT_ALLOW_PLUGINS", None)
    assert not registry.has("must_rollback")
    assert any("plugin:exiting failed: 7" in item for item in logs), logs


@check
def keyboard_interrupt_still_escapes_after_plugin_registration_is_rolled_back():
    root = _write_plugin('name = "interrupting"\n', '''
def register(ctx):
    ctx.register_tool("must_rollback", "partial", lambda _args: "partial")
    raise KeyboardInterrupt()
''')
    registry = ToolRegistry()
    os.environ["AGENT_ALLOW_PLUGINS"] = "1"
    interrupted = False
    try:
        load_plugins(registry, SkillManager([]), [root], root=root, config=None)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        os.environ.pop("AGENT_ALLOW_PLUGINS", None)
    assert interrupted and not registry.has("must_rollback")


@check
def plugin_listing_uses_namespaced_provenance():
    registry = ToolRegistry()
    registry.register(ToolEntry(
        name="from_demo",
        schema={"type": "function", "function": {"name": "from_demo", "description": "x",
                "parameters": {"type": "object", "properties": {}}}},
        handler=lambda _args: "ok", source="plugin:demo",
    ))
    registry.register(ToolEntry(
        name="builtin",
        schema={"type": "function", "function": {"name": "builtin", "description": "x",
                "parameters": {"type": "object", "properties": {}}}},
        handler=lambda _args: "ok", source="builtin",
    ))
    assert _plugin_tool_names(registry) == ["from_demo"]


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
