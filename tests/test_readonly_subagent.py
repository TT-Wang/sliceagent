"""Read-only EXPLORE child (W6) — schema filtering + bounded summary.
No model, no pytest. Run: python tests/test_readonly_subagent.py

Asserts the moat-safe read-only delegation contract:
  * read_only_schemas drops edit/shell/spawn tools, keeps read/search/recall.
  * SubagentHost(read_only=True).schemas() exposes NO spawn/edit even when depth<max_depth.
  * read_only=False schemas are unchanged (inner tools + spawn_subagent + spawn_explore).
  * run_subagent(read_only=True) still returns a bounded one-line summary string.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.subagent import (                       # noqa: E402
    SubagentHost,
    read_only_schemas,
    run_subagent,
)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _fn_schema(name):
    return {"type": "function",
            "function": {"name": name, "description": name,
                         "parameters": {"type": "object", "properties": {}, "required": []}}}


# Full builtin-ish tool surface: 6 read-only-allowed, 4 mutating/shell, 1 spawn.
_KEEP = ["read_file", "list_files", "grep", "glob", "skill", "recall_history"]
_DROP = ["edit_file", "str_replace", "run_command", "execute_code", "spawn_subagent"]
_ALL_SCHEMAS = [_fn_schema(n) for n in _KEEP + _DROP]


def _names(schemas):
    return [s.get("function", {}).get("name") for s in schemas]


class _InnerHost:
    """Minimal ToolHost stand-in: only schemas() is exercised by these checks."""
    def schemas(self):
        return [_fn_schema(n) for n in _KEEP + _DROP[:-1]]  # inner has no spawn_subagent of its own


# ---- read_only_schemas (pure filter) ----------------------------------------

@check
def read_only_schemas_keeps_read_search_drops_edit_and_spawn():
    out = read_only_schemas(_ALL_SCHEMAS)
    names = _names(out)
    assert set(names) == set(_KEEP), names                 # exactly the allowlist
    for dropped in _DROP:
        assert dropped not in names, dropped               # edit + shell + spawn all gone


@check
def read_only_schemas_handles_malformed_entries():
    # an entry with no "function" key must not crash; just be excluded
    out = read_only_schemas([{"type": "function"}, _fn_schema("read_file")])
    assert _names(out) == ["read_file"]


# ---- SubagentHost.schemas() under read_only ---------------------------------

@check
def host_read_only_has_no_spawn_or_edit_even_with_depth_left():
    # depth (0) < max_depth (1): a WRITABLE host would append spawn here. read_only must not.
    host = SubagentHost(_InnerHost(), llm=None, retriever=None, memory=None, policy=None,
                        max_depth=1, depth=0, read_only=True)
    names = _names(host.schemas())
    assert "spawn_subagent" not in names, names
    assert "spawn_explore" not in names, names
    assert "edit_file" not in names and "str_replace" not in names, names
    assert "run_command" not in names and "execute_code" not in names, names
    assert set(names) == set(_KEEP), names                 # only the read-only allowlist survives


@check
def host_writable_schemas_unchanged_regression():
    # read_only=False (default) must still surface the inner tools verbatim PLUS the two spawn tools.
    inner = _InnerHost()
    host = SubagentHost(inner, llm=None, retriever=None, memory=None, policy=None,
                        max_depth=1, depth=0, read_only=False)
    names = _names(host.schemas())
    inner_names = _names(inner.schemas())
    # every inner tool preserved, nothing dropped
    for n in inner_names:
        assert n in names, n
    # delegation tools appended (depth<max_depth)
    assert names == inner_names + ["spawn_subagent", "spawn_explore"], names


@check
def host_writable_no_spawn_at_depth_floor():
    # at depth==max_depth the writable host offers no delegation (existing depth gate intact).
    host = SubagentHost(_InnerHost(), llm=None, retriever=None, memory=None, policy=None,
                        max_depth=1, depth=1, read_only=False)
    names = _names(host.schemas())
    assert "spawn_subagent" not in names and "spawn_explore" not in names, names


# ---- run_subagent(read_only=True) returns a bounded summary -----------------

class _Resp:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []
        self.finish_reason = "stop"
        self.usage = {}


class _FakeLLM:
    """Ends the child turn in one step with a final assistant text (its summary).

    When `expect_read_only` is set it also verifies the child's tool surface is gated:
    no edit/shell/spawn tool ever reaches the model (the gate is upstream, in SubagentHost)."""
    def __init__(self, text, *, expect_read_only=False):
        self._text = text
        self._expect_read_only = expect_read_only
        self.calls = 0
        self.last_tool_names = []
    def complete(self, messages, schemas):
        self.calls += 1
        names = _names(schemas)
        self.last_tool_names = names
        if self._expect_read_only:
            assert "edit_file" not in names and "spawn_subagent" not in names, names
            assert "run_command" not in names and "execute_code" not in names, names
        return _Resp(self._text)


class _Retriever:
    def retrieve(self, *a, **k):
        return []


class _Tools:
    """Wrapped (real) host; SubagentHost(read_only=True) gates which schemas reach the LLM."""
    def schemas(self):
        return [_fn_schema(n) for n in _KEEP + _DROP[:-1]]
    def root(self):
        return "/tmp/ws"
    def accesses(self, name, args):
        return []
    def run(self, name, args):
        return ""
    def read_text(self, path):
        return ""


@check
def run_subagent_read_only_returns_bounded_summary():
    llm = _FakeLLM("Found the parser entry point in pkg/parse.py", expect_read_only=True)
    child_tools = SubagentHost(_Tools(), llm=llm, retriever=_Retriever(), memory=None,
                               policy=None, max_depth=1, depth=1, read_only=True)
    out = run_subagent("locate the parser entry point", tools=child_tools, llm=llm,
                       retriever=_Retriever(), memory=None, policy=None, max_steps=3,
                       depth=1, read_only=True)
    assert isinstance(out, str) and out, repr(out)
    assert llm.calls >= 1                                   # the child actually ran
    assert out.startswith("[explore "), out                # labelled as a read-only run
    assert "parser entry point" in out or "parse.py" in out, out
    assert len(out) < 600, len(out)                         # bounded — not a transcript


@check
def run_subagent_writable_label_distinct_from_explore():
    # regression: a writable run is labelled [subagent ...], not [explore ...]
    llm = _FakeLLM("done")
    child_tools = SubagentHost(_Tools(), llm=llm, retriever=_Retriever(), memory=None,
                               policy=None, max_depth=1, depth=1, read_only=False)
    out = run_subagent("do a thing", tools=child_tools, llm=llm, retriever=_Retriever(),
                       memory=None, policy=None, max_steps=3, depth=1, read_only=False)
    assert out.startswith("[subagent "), out


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
