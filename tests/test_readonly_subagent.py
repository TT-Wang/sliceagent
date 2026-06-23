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
from memagent.tools import LocalToolHost              # noqa: E402
from memagent.slice import Slice, make_build_slice    # noqa: E402
from memagent.memory import NullMemory                # noqa: E402
from memagent.retriever import NullRetriever          # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _fn_schema(name):
    return {"type": "function",
            "function": {"name": name, "description": name,
                         "parameters": {"type": "object", "properties": {}, "required": []}}}


# Full builtin-ish tool surface: 6 read-only-allowed, 4 mutating/shell, 1 spawn.
_KEEP = ["read_file", "list_files", "grep", "skill", "recall_history"]   # no 'glob' — no such tool is registered
_DROP = ["edit_file", "str_replace", "run_command", "execute_code", "spawn_subagent"]
_ALL_SCHEMAS = [_fn_schema(n) for n in _KEEP + _DROP]


def _names(schemas):
    return [s.get("function", {}).get("name") for s in schemas]


@check
def subagent_host_faithfully_projects_the_wrapped_host():
    # REGRESSION (the "agent can't see its own folder" bug): SubagentHost must delegate every
    # non-overridden host attr to the wrapped host. If root() is dropped, make_build_slice gets
    # cwd="" and the WORKING DIRECTORY / cwd / WORKSPACE / git ENVIRONMENT tier silently vanishes
    # whenever subagents are enabled.
    import tempfile
    wd = LocalToolHost(tempfile.mkdtemp(prefix="subhost-"))
    sh = SubagentHost(wd, llm=None, retriever=NullRetriever(), memory=NullMemory(),
                      policy=None, max_depth=1)
    assert sh.root() == wd.root(), "SubagentHost must forward root() to the wrapped host"
    s = Slice(); s.reset("hi")
    sysmsg = make_build_slice(s, sh, NullRetriever(), NullMemory(), "hi")()[0]["content"]
    assert "# WORKING DIRECTORY" in sysmsg, "cwd/env tier must survive the SubagentHost wrapper"
    assert f"Working directory (cwd): {sh.root()}" in sysmsg, "the real cwd must reach the slice"


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
    # delegation tools appended (depth<max_depth): the two built-ins + the generic registry tool
    assert names == inner_names + ["spawn_subagent", "spawn_explore", "spawn_agent"], names


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
        self._calls = [0]            # a list so a shallow copy (the explorer profile's per-child llm view) shares it
        self.last_tool_names = []
    @property
    def calls(self):
        return self._calls[0]
    def complete(self, messages, schemas):
        self._calls[0] += 1
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


@check
def explorer_keeps_reads_resident_no_eviction_churn():
    # 'explore stuck' fix: a read-only explorer must keep its WHOLE exploration resident. With the
    # default READ_BUDGET the early reads evict → the model re-reads the paged-out files → the anti-loop
    # guard flags the re-reads as no-progress → the child goes "stuck" before it can summarize. The
    # explorer budget (EXPLORER_READ_BUDGET) holds the exploration, so there is no eviction churn.
    from memagent.subagent import EXPLORER_READ_BUDGET
    from memagent.slice import Slice, touch_file
    from memagent.swap import READ_BUDGET
    assert EXPLORER_READ_BUDGET > READ_BUDGET
    # explorer budget: 10 distinct reads ALL stay resident (no eviction → no re-read churn)
    s = Slice(); s.reset("explore"); s.read_budget = s.read_ceiling = EXPLORER_READ_BUDGET
    for i in range(10):
        touch_file(s, f"pkg/f{i}.py")
    assert len(s.active_files) == 10, f"explorer evicted reads (churn): {len(s.active_files)} resident"
    # default budget: the SAME 10 reads churn down to READ_BUDGET (the bug condition)
    d = Slice(); d.reset("task")
    for i in range(10):
        touch_file(d, f"pkg/g{i}.py")
    assert len(d.active_files) == READ_BUDGET, f"default kept {len(d.active_files)}, expected {READ_BUDGET}"


@check
def explorer_guard_does_not_stuck_on_repeated_reads():
    # the explorer guard relaxes the READ axes: a repeated idempotent read (same result) must NOT be
    # hard-blocked (which is what drove review children to "stuck"); the DEFAULT guard still blocks it.
    from memagent.guardrails import ToolCallGuardrailConfig
    from memagent.hooks import GuardrailHook
    relaxed = GuardrailHook(ToolCallGuardrailConfig(no_progress_block_after=10**6, result_repeat_block_after=10**6))
    for _ in range(8):                                  # same read, same result, many times
        assert relaxed.authorize_tool("read_file", {"path": "a.py"}).allow, "explorer guard blocked a re-read"
        relaxed.transform_tool_result("read_file", {"path": "a.py"}, "same content")
    default = GuardrailHook()
    blocked = False
    for _ in range(8):
        if not default.authorize_tool("read_file", {"path": "a.py"}).allow:
            blocked = True
            break
        default.transform_tool_result("read_file", {"path": "a.py"}, "same content")
    assert blocked, "default guard should block a repeated no-progress read (explorer must differ)"


@check
def explorer_profile_runs_fast_reasoning_without_mutating_parent():
    # EXPLORER profile: a read-only child runs at fast reasoning via a per-child llm VIEW; the shared parent
    # llm is never mutated and a writable child uses the parent unchanged.
    from types import SimpleNamespace
    from memagent.subagent import _profile_llm
    parent = SimpleNamespace(reasoning="full")
    view = _profile_llm(parent, "fast")
    assert view.reasoning == "fast", view.reasoning
    assert parent.reasoning == "full", "parent llm must NOT be mutated (per-child view only)"
    assert view is not parent, "a reasoning override must get its OWN llm view"
    assert _profile_llm(parent, None) is parent, "no reasoning override → parent llm unchanged"
    # already at target → no needless copy
    fast = SimpleNamespace(reasoning="fast")
    assert _profile_llm(fast, "fast") is fast


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
