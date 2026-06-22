"""M: structured tool-result success flag (ToolText.ok) replaces prose-match failure inference.

A tool that returns NORMALLY is success — even if its text begins with "Error"/"Exit code" (a grep hit,
a log line, a docstring). Only a raised handler, an unknown tool, or an explicit ToolText(ok=False)
(e.g. a nonzero exit code, a not-unique str_replace) is a failure. run_tool_batch reads .ok, with the
old prose-match kept ONLY as a fallback for plain strings (a plugin transform, an MCP tool, a test fake).

Why it matters for pass rate: a false "failing" flag pollutes the anti-loop REPEATED/FAILING tally,
downgrades a finding's provenance to "claim", flags the episode, and feeds the STUCK floor — so a
correct step that merely PRINTS "Error..." could push the agent toward a wrong/parked outcome.

No LLM. Run: PYTHONPATH=src python tests/test_tool_result_ok.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.registry import ToolEntry, ToolRegistry, ToolText  # noqa: E402
from memagent.loop import run_tool_batch                          # noqa: E402
from memagent.hooks import Hooks                                  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _entry(name, handler):
    return ToolEntry(name=name,
                     schema={"type": "function", "function": {"name": name, "parameters": {}}},
                     handler=handler)


# --- registry.run() level: ok is assigned correctly ------------------------------------------------
@check
def registry_success_with_error_text_is_ok():
    r = ToolRegistry()
    r.register(_entry("grep", lambda a: "Error: connection refused\nError: timeout"))  # a legit grep HIT
    out = r.run("grep", {})
    assert isinstance(out, ToolText) and out.ok is True, \
        "a normally-returning tool is SUCCESS even if its output starts with 'Error'"


@check
def registry_raising_handler_is_failure():
    def boom(_a):
        raise RuntimeError("disk full")
    r = ToolRegistry()
    r.register(_entry("x", boom))
    out = r.run("x", {})
    assert out.ok is False and out.startswith("Error")


@check
def registry_explicit_fail_passthrough():
    r = ToolRegistry()
    r.register(_entry("cmd", lambda a: ToolText("Exit code 1\nboom", ok=False)))
    assert r.run("cmd", {}).ok is False


@check
def registry_unknown_tool_is_failure():
    assert ToolRegistry().run("nope", {}).ok is False


# --- run_tool_batch level: the loop's `failing` flag ----------------------------------------------
class _TC:
    def __init__(self, name="t"):
        self.name = name; self.args = {}; self.id = "c1"


class _Host:
    def __init__(self, out):
        self._out = out
    def accesses(self, n, a):
        return []
    def run(self, n, a):
        return self._out


def _failing(out):
    _, results = run_tool_batch([_TC()], _Host(out), lambda e: None, Hooks())
    return results[0]["failing"]


@check
def batch_tooltext_ok_with_error_text_not_failing():
    # THE M fix: a successful tool whose output starts with "Error" is NOT failing.
    assert _failing(ToolText("Error: 3 matches found in app.log", ok=True)) is False


@check
def batch_tooltext_notok_is_failing():
    assert _failing(ToolText("Exit code 2\nsegfault", ok=False)) is True


@check
def batch_plain_str_falls_back_to_prose_match():
    # plain str has no .ok (MCP / plugin transform / fake) → best-effort prose-match fallback survives.
    assert _failing("Error: something broke") is True
    assert _failing("all good, 5 files written") is False


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
