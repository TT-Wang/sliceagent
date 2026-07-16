"""Regression tests for the subagent/dispatch wave: #44 the universal `note` arg is stripped before the
handler (but kept on the event for capture), #42/#43 a child's allowlist is enforced at RUNTIME (incl.
no read-only→writable escalation via spawn), #59 empty 'task' → clear error. No model, no pytest.
Run: PYTHONPATH=src python tests/test_bugfix_subagent_wave.py
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.loop import run_tool_batch  # noqa: E402
from sliceagent.events import ToolResult  # noqa: E402
from sliceagent.subagent import SubagentHost  # noqa: E402
from sliceagent.agents import BUILTIN_AGENTS  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Hooks:
    def preflight_tool(self, name, args):
        return NS(stop=False, reason="")
    def transform_tool_result(self, name, args, out):
        return None


@check
def note_stripped_from_handler_but_kept_on_event():  # #44
    seen = {}

    class _Tools:
        def accesses(self, name, args):
            seen["access_args"] = dict(args)
            return []
        def run(self, name, args):
            seen["handler_args"] = dict(args)
            return "ok"

    events = []
    tcs = [NS(name="mcp_tool", args={"path": "x", "note": "a durable finding"})]
    run_tool_batch(tcs, _Tools(), events.append, _Hooks())
    assert seen["handler_args"] == {"path": "x"}, seen["handler_args"]   # handler must NOT see note
    assert "note" not in seen["access_args"], "accesses() must also get clean args"
    tr = [e for e in events if isinstance(e, ToolResult)][0]
    assert tr.args.get("note") == "a durable finding", "note must still ride the event for FINDINGS capture"


def _host(spec):
    class _Inner:
        def run(self, name, args):
            return f"inner-ran:{name}"
        def accesses(self, name, args):
            return []
        def schemas(self):
            return []
    return SubagentHost(_Inner(), llm=None, retriever=None, memory=None,
                        max_depth=2, max_steps=5, depth=0, notify=lambda m: None,
                        spec=spec, agents=BUILTIN_AGENTS)


@check
def explorer_allowlist_enforced_at_runtime():  # #42
    h = _host(BUILTIN_AGENTS["explorer"])
    assert "not available" in h.run("edit_file", {"path": "a", "content": "x"}), "explorer must not edit"
    assert h.run("read_file", {"path": "a"}) == "inner-ran:read_file", "an allowed read tool passes through"


@check
def explorer_cannot_escalate_via_spawn():  # #43
    h = _host(BUILTIN_AGENTS["explorer"])
    out = h.run("spawn_subagent", {"task": "go write files"})
    assert "not available" in out, f"a read-only explorer must NOT spawn a writable child: {out!r}"


@check
def spawn_requires_nonempty_task():  # #59
    h = _host(BUILTIN_AGENTS["general"])   # general → allowlist skipped, reaches the task check
    assert "non-empty 'task'" in h.run("spawn_subagent", {}), "empty args must give a clear error"
    assert "non-empty 'task'" in h.run("spawn_subagent", {"task": "   "})


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
