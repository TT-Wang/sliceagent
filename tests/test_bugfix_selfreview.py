"""Regression tests for the bugs memagent's self-review found (real misses in the earlier #8/#35 work):
  #1 execute_code's list_files() bypassed workspace confinement
  #2 episodic redaction missed the top-level record fields (title/note/markdown/meta)
  #3 read_only classification treated unknown/plugin/MCP tools as read-only (optimistic)
No model, no pytest. Run: PYTHONPATH=src python tests/test_bugfix_selfreview.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.tools import _CODE_PRELUDE  # noqa: E402
from memagent.memory import MememMemory  # noqa: E402
from memagent.agents import AgentSpec, READ_ONLY_TOOLS  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def prelude_list_files_is_confined():  # #1
    wd = tempfile.mkdtemp(prefix="confine-")
    cwd0 = os.getcwd()
    os.chdir(wd)
    try:
        os.mkdir("sub"); open("a.txt", "w").write("x")
        ns = {}
        exec(_CODE_PRELUDE, ns)
        assert "a.txt" in ns["list_files"]("."), "in-workspace listing still works"
        for bad in ("/tmp", "/etc", "..", "/"):
            try:
                ns["list_files"](bad)
                assert False, f"list_files must not escape the workspace: {bad}"
            except PermissionError:
                pass
    finally:
        os.chdir(cwd0)


@check
def episodic_redaction_covers_top_level_fields():  # #2
    m = MememMemory.__new__(MememMemory)
    secret = "sk-" + "A" * 40   # looks like an API key → redact_text should scrub it
    rec = {
        "title": f"working on {secret}",
        "note": f"the key is {secret}",
        "markdown": f"# turn\nused {secret} in a call",
        "meta": {"files": [f"/tmp/{secret}.txt"], "stop_reason": "end_turn"},
        "steps": [{"observation": [f"output {secret}"], "action": [{"args": {"k": secret}}]}],
    }
    out = m._clamp_record(rec)
    blob = repr(out)
    assert secret not in blob, f"a secret must not survive in ANY record field; leaked in: {blob[:200]}"
    # structure preserved (still a usable record)
    assert "title" in out and "note" in out and "markdown" in out and "steps" in out


@check
def read_only_is_pessimistic_for_unknown_tools():  # #3
    assert AgentSpec(name="x", tools=("dangerous_plugin_tool",)).read_only is False, \
        "an unknown/plugin tool must NOT be assumed read-only"
    assert AgentSpec(name="x", tools=("read_file", "dangerous_plugin_tool")).read_only is False, \
        "a mix with any unknown tool is writable"
    assert AgentSpec(name="x", tools=tuple(READ_ONLY_TOOLS)).read_only is True, \
        "the known read-only surface is still read-only"
    assert AgentSpec(name="x", tools=("read_file", "grep")).read_only is True
    assert AgentSpec(name="x", tools=("edit_file",)).read_only is False
    assert AgentSpec(name="x", tools=None).read_only is False, "full surface (None) is writable"


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
