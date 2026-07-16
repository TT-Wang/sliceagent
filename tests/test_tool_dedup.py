"""Same-step exact-call dedup.

Lossless by construction: a duplicate (name, args) read-only call in ONE batch reuses the first call's
result instead of executing the tool twice, and every tool_call_id still gets a (byte-identical) reply.
Mutating/unknown tools are never deduped. No LLM, no pytest.
Run: PYTHONPATH=src python tests/test_tool_dedup.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.registry import ToolText                     # noqa: E402
from sliceagent.loop import run_tool_batch                   # noqa: E402
from sliceagent.hooks import Hooks                           # noqa: E402
from sliceagent.tool_identity import canonical_tool_args      # noqa: E402
from sliceagent.events import ToolResult, ToolStarted        # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _TC:
    def __init__(self, name, args, id):
        self.name = name; self.args = args; self.id = id


class _CountHost:
    """Records every real run(); output embeds the per-name execution count so a NON-deduped second call
    would be observably different from the first (proving dedup when they are byte-identical)."""
    def __init__(self):
        self.calls = []
    def accesses(self, n, a):
        return []
    def run(self, n, a):
        self.calls.append((n, dict(a) if a else {}))
        nth = sum(1 for c in self.calls if c[0] == n)
        return ToolText(f"OUT[{n}] {canonical_tool_args(a or {})} #{nth}", ok=True)


def _batch(tcs, host, sink=None):
    return run_tool_batch(tcs, host, (sink or (lambda e: None)), Hooks())


@check
def identical_readonly_calls_execute_once_and_return_identical():
    host = _CountHost()
    tcs = [_TC("read_file", {"path": "a.py"}, "c1"), _TC("read_file", {"path": "a.py"}, "c2")]
    blocked, results = _batch(tcs, host)
    assert len(host.calls) == 1, f"read_file must run ONCE, ran {len(host.calls)}"
    assert results[0]["output"] == results[1]["output"], "dup must be byte-identical to the original"
    assert results[0]["id"] == "c1" and results[1]["id"] == "c2", "every tool_call_id still gets a reply"
    assert blocked == 0


@check
def note_only_difference_still_dedups():
    # `note` is stripped from the dedup identity (canonical_tool_args), so reads differing only in note dedup.
    host = _CountHost()
    tcs = [_TC("read_file", {"path": "a.py", "note": "first"}, "c1"),
           _TC("read_file", {"path": "a.py", "note": "second"}, "c2")]
    _, results = _batch(tcs, host)
    assert len(host.calls) == 1
    assert results[0]["output"] == results[1]["output"]


@check
def different_args_do_not_dedup():
    host = _CountHost()
    tcs = [_TC("read_file", {"path": "a.py"}, "c1"), _TC("read_file", {"path": "b.py"}, "c2")]
    _, results = _batch(tcs, host)
    assert len(host.calls) == 2, "different paths are different calls"
    assert results[0]["output"] != results[1]["output"]


@check
def mutating_tool_is_never_deduped():
    host = _CountHost()
    tcs = [_TC("run_command", {"cmd": "make"}, "c1"), _TC("run_command", {"cmd": "make"}, "c2")]
    _, _ = _batch(tcs, host)
    assert len(host.calls) == 2, "run_command may carry intended side effects — must run twice"


@check
def unknown_or_mcp_tool_is_never_deduped():
    host = _CountHost()
    tcs = [_TC("mcp__db__query", {"q": "1"}, "c1"), _TC("mcp__db__query", {"q": "1"}, "c2")]
    _, _ = _batch(tcs, host)
    assert len(host.calls) == 2, "tools outside the dedup-safe set are never deduped"


@check
def dedup_propagates_errors_identically():
    class _ErrHost(_CountHost):
        def run(self, n, a):
            self.calls.append((n, dict(a) if a else {}))
            return ToolText("Error: boom", ok=False)
    host = _ErrHost()
    tcs = [_TC("grep", {"pattern": "x"}, "c1"), _TC("grep", {"pattern": "x"}, "c2")]
    _, results = _batch(tcs, host)
    assert len(host.calls) == 1
    assert results[0]["output"] == results[1]["output"] == "Error: boom"
    assert results[0]["failing"] is True and results[1]["failing"] is True, "the failing flag carries to the dup too"


@check
def duplicate_has_one_physical_start_and_one_logical_outcome_each():
    # Only one tool physically starts, but every provider invocation gets an auditable durable outcome.
    # The duplicate outcome is explicitly non-reducing because the source effects were already applied.
    events = []
    host = _CountHost()
    tcs = [_TC("read_file", {"path": "a.py"}, "c1"), _TC("read_file", {"path": "a.py"}, "c2")]
    _batch(tcs, host, sink=events.append)
    results = [e for e in events if isinstance(e, ToolResult)]
    n_res = len(results)
    n_start = sum(1 for e in events if isinstance(e, ToolStarted))
    assert n_res == 2, f"every logical invocation must dispatch one ToolResult; got {n_res}"
    assert n_start == 1, f"deduped dup must not dispatch a 2nd ToolStarted; got {n_start}"
    assert [event.invocation_id for event in results] == ["c1", "c2"]
    assert results[0].apply_effects is True and results[1].apply_effects is False


@check
def three_identical_dedup_to_one_execution():
    host = _CountHost()
    tcs = [_TC("glob", {"pat": "*.py"}, f"c{i}") for i in range(3)]
    _, results = _batch(tcs, host)
    assert len(host.calls) == 1, "N identical read-only calls collapse to one execution"
    assert results[0]["output"] == results[1]["output"] == results[2]["output"]
    assert [r["id"] for r in results] == ["c0", "c1", "c2"], "all three ids still get a reply"


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
