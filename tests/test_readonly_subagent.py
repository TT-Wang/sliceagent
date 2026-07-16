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

from sliceagent.subagent import (                       # noqa: E402
    SubagentHost,
    read_only_schemas,
    run_subagent,
)
from sliceagent.tools import LocalToolHost              # noqa: E402
from sliceagent.pfc import Slice  # noqa: E402
from sliceagent.seed import make_build_slice  # noqa: E402
from sliceagent.memory import NullMemory                # noqa: E402
from sliceagent.retriever import NullRetriever          # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _fn_schema(name):
    return {"type": "function",
            "function": {"name": name, "description": name,
                         "parameters": {"type": "object", "properties": {}, "required": []}}}


# Full builtin-ish tool surface: 6 read-only-allowed, 4 mutating/shell, 1 spawn.
_KEEP = ["read_file", "list_files", "grep", "skill", "search_history"]   # no 'glob' — no such tool is registered
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
    sh = SubagentHost(wd, llm=None, retriever=NullRetriever(), memory=NullMemory(), max_depth=1)
    assert sh.root() == wd.root(), "SubagentHost must forward root() to the wrapped host"
    s = Slice(); s.reset("hi")
    sysmsg = make_build_slice(s, sh, NullRetriever(), NullMemory(), "hi")()[0]["content"]
    assert "# CURRENT PROJECT & REACH" in sysmsg, "cwd/env tier must survive the SubagentHost wrapper"
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
    host = SubagentHost(_InnerHost(), llm=None, retriever=None, memory=None,
                        max_depth=1, depth=0, read_only=True)
    names = _names(host.schemas())
    assert "spawn_subagent" not in names, names
    assert "spawn_explore" not in names, names
    assert "edit_file" not in names and "str_replace" not in names, names
    assert "run_command" not in names and "execute_code" not in names, names
    # a CHILD must not get search_history — it is bound to the PARENT session and previews the parent's turns
    # (isolation: children couple only through the two seals). So the child allowlist is _KEEP MINUS search_history.
    assert "search_history" not in names, names
    assert set(names) == set(_KEEP) - {"search_history"}, names


@check
def child_schema_does_not_advertise_parent_contextfs():
    import tempfile

    wd = LocalToolHost(tempfile.mkdtemp(prefix="subhost-contextfs-"))
    try:
        assert "@sliceagent/index.md" in str(wd.schemas())
        child = SubagentHost(
            wd, llm=None, retriever=NullRetriever(), memory=NullMemory(),
            max_depth=1, depth=1, read_only=True,
        )
        assert "@sliceagent" not in str(child.schemas()), child.schemas()
        blocked = child.run("read_file", {"path": "@sliceagent/index.md"})
        assert "parent's private namespaces" in blocked and "@sliceagent/" in blocked, blocked
    finally:
        wd.cleanup()


@check
def host_writable_schemas_unchanged_regression():
    # read_only=False (default) must still surface the inner tools verbatim PLUS the ONE delegation tool.
    # (spawn_explore/spawn_subagent were collapsed into spawn_agent — measured fan-out parity.)
    inner = _InnerHost()
    host = SubagentHost(inner, llm=None, retriever=None, memory=None,
                        max_depth=1, depth=0, read_only=False)
    names = _names(host.schemas())
    inner_names = _names(inner.schemas())
    # every inner tool preserved, nothing dropped
    for n in inner_names:
        assert n in names, n
    # exactly ONE delegation tool appended (depth<max_depth): spawn_agent subsumes the old aliases
    assert names == inner_names + ["spawn_agent"], names
    assert "spawn_explore" not in names and "spawn_subagent" not in names, names


@check
def host_writable_no_spawn_at_depth_floor():
    # at depth==max_depth the writable host offers no delegation (existing depth gate intact).
    host = SubagentHost(_InnerHost(), llm=None, retriever=None, memory=None,
                        max_depth=1, depth=1, read_only=False)
    names = _names(host.schemas())
    assert "spawn_agent" not in names, names


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
    import sliceagent.subagent as subagent_module

    class GroundedLLM(_FakeLLM):
        def complete(self, messages, schemas):
            self._calls[0] += 1
            names = _names(schemas)
            self.last_tool_names = names
            if self._expect_read_only:
                assert "edit_file" not in names and "run_command" not in names, names
            if self._calls[0] == 1:
                response = _Resp("inspect parser source")
                response.finish_reason = "tool_calls"
                response.tool_calls = [type("Call", (), {
                    "name": "read_file", "args": {"path": "pkg/parse.py"}, "id": "read-1",
                })()]
                return response
            return _Resp(self._text)

    original = subagent_module.EXPLORER_REASONING
    subagent_module.EXPLORER_REASONING = "full"
    try:
        llm = GroundedLLM("Found the parser entry point in pkg/parse.py", expect_read_only=True)
        child_tools = SubagentHost(_Tools(), llm=llm, retriever=_Retriever(), memory=None,
                                   max_depth=1, depth=1, read_only=True)
        out = run_subagent("locate the parser entry point", tools=child_tools, llm=llm,
                           retriever=_Retriever(), memory=None, max_steps=3,
                           depth=1, read_only=True)
    finally:
        subagent_module.EXPLORER_REASONING = original
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
                               max_depth=1, depth=1, read_only=False)
    out = run_subagent("do a thing", tools=child_tools, llm=llm, retriever=_Retriever(),
                       memory=None, max_steps=3, depth=1, read_only=False)
    assert out.startswith("[subagent "), out


@check
def explorer_keeps_reads_resident_no_eviction_churn():
    # A read-only explorer keeps its whole exploration resident. With the default READ_BUDGET the early reads
    # evict and invite needless re-reads; EXPLORER_READ_BUDGET holds the exploration without churn.
    from sliceagent.subagent import EXPLORER_READ_BUDGET
    from sliceagent.pfc import Slice, touch_file
    from sliceagent.swap import READ_BUDGET
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
def explorer_profile_runs_fast_reasoning_without_mutating_parent():
    # EXPLORER profile: a read-only child runs at fast reasoning via a per-child llm VIEW; the shared parent
    # llm is never mutated and a writable child uses the parent unchanged.
    from types import SimpleNamespace
    from sliceagent.subagent import _profile_llm
    parent = SimpleNamespace(reasoning="full")
    view = _profile_llm(parent, "fast")
    assert view.reasoning == "fast", view.reasoning
    assert parent.reasoning == "full", "parent llm must NOT be mutated (per-child view only)"
    assert view is not parent, "a reasoning override must get its OWN llm view"
    assert view._on_delta is None, "child view disconnects the parent's streaming delta sink (S7)"
    # S7: a child ALWAYS gets its OWN copy — even when reasoning is inherited or already matches — so a child's
    # model/_fellback mutation can't switch the PARENT and child streaming can't leak to the parent UI.
    inherited = _profile_llm(parent, None)
    assert inherited is not parent, "child gets its own view even with inherited reasoning (S7)"
    assert inherited._on_delta is None
    fast = SimpleNamespace(reasoning="fast", _on_delta=lambda *_: None)
    fv = _profile_llm(fast, "fast")
    assert fv is not fast and fv._on_delta is None, "always an isolated view with streaming off (S7)"


@check
def child_private_progress_sinks_never_reuse_parent_renderer():
    from types import SimpleNamespace
    from sliceagent.subagent import _profile_llm

    parent_delta = lambda *_: None
    child_delta = lambda *_: None
    child_activity = lambda *_: None
    parent = SimpleNamespace(reasoning="full", _on_delta=parent_delta, _transport_activity="parent")
    view = _profile_llm(
        parent, "fast", delta_sink=child_delta, activity_sink=child_activity,
    )
    assert view is not parent and view.reasoning == "fast"
    assert view._on_delta is child_delta and view._transport_activity is child_activity
    assert parent._on_delta is parent_delta and parent._transport_activity == "parent"


@check
def child_progress_uses_typed_transport_phases_without_token_text():
    from sliceagent.events import ApiRetry, AssistantText, ModelCallPrepared, ToolStarted
    from sliceagent.subagent import _nested_sink

    updates = []
    sink = _nested_sink(
        updates.append, depth=1, agent_id="child-1", parent_turn_id="turn-1",
        launch_ordinal=1, kind="explorer", objective="inspect parser",
    )
    sink(ModelCallPrepared(step=1, attempt=1, messages=[]))
    sink.on_activity("awaiting_model", {"transport": "sse"})
    sink.on_activity("first_byte", {"transport": "sse"})
    sink.on_activity("reasoning", {"transport": "sse"})
    sink.on_activity("writing", {"transport": "sse"})
    sink(ApiRetry(attempt=1, error="timeout", delay_s=0.4, max_attempts=3))
    sink(ToolStarted("read_file", {"path": "pkg/parse.py"}))
    assert [update.phase for update in updates] == [
        "awaiting_model", "model_active", "reasoning", "writing", "retry_wait", "running_tool",
    ]
    assert updates[0].attempt == 1, "transport hint must not erase ModelCallPrepared attempt identity"
    assert updates[4].attempt == 2 and updates[4].max_attempts == 3
    assert updates[4].retry_delay_s == 0.4
    assert updates[-1].tool_name == "read_file" and updates[-1].tool_count == 1
    assert all("private token" not in update.detail for update in updates)
    before = len(updates)
    sink(AssistantText("Done — no summary to add.", final=True, synthetic=True))
    assert len(updates) == before, "host fallback must not claim the model is writing a report"
    sink(AssistantText("real final report", final=True))
    assert updates[-1].phase == "writing" and len(updates) == before + 1


@check
def child_evidence_pressure_keeps_pairing_recent_bytes_and_provenance():
    from sliceagent.subagent import _compact_child_evidence

    messages = [{"role": "system", "content": "explore"}]
    originals = {}
    for index in range(6):
        call_id = f"call-{index}"
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"pkg/f%d.py"}' % index,
                },
            }],
        })
        body = (chr(65 + index) * 7000) + f"\nEND-{index}"
        originals[call_id] = body
        messages.append({"role": "tool", "tool_call_id": call_id, "content": body})

    compacted = _compact_child_evidence(
        messages, soft_bytes=1, target_bytes=1, view_bytes=1024, keep_recent=2,
    )
    assert compacted is not None and len(compacted) == len(messages)
    tool_rows = [row for row in compacted if row.get("role") == "tool"]
    assert [row["tool_call_id"] for row in tool_rows] == list(originals)
    for row in tool_rows[:-2]:
        source = originals[row["tool_call_id"]]
        assert "older child evidence pressure view" in row["content"]
        assert "original_sha256" in row["content"] and "pkg/f" in row["content"]
        assert source[:100] in row["content"] and source[-100:] in row["content"]
    assert tool_rows[-2]["content"] == originals["call-4"]
    assert tool_rows[-1]["content"] == originals["call-5"]
    assert messages[2]["content"] == originals["call-0"], "canonical input must not be mutated"


@check
def staged_explorer_uses_fast_navigation_then_one_full_tool_free_synthesis():
    import sliceagent.subagent as subagent_module

    class _StageLLM:
        def __init__(self):
            self.reasoning = "full"
            self.shared = {"profiles": [], "schemas": []}

        def complete(self, messages, schemas):
            self.shared["profiles"].append(self.reasoning)
            self.shared["schemas"].append(_names(schemas))
            call = len(self.shared["profiles"])
            if call == 1:
                response = _Resp("locating parser evidence")
                response.tool_calls = [type("Call", (), {
                    "name": "read_file", "args": {"path": "pkg/parse.py"}, "id": "read-parser",
                })()]
                response.finish_reason = "tool_use"
                return response
            return _Resp("navigation handoff" if call == 2 else "FINAL grounded report")

    original = subagent_module.EXPLORER_REASONING
    subagent_module.EXPLORER_REASONING = "staged"
    try:
        llm = _StageLLM()
        child_tools = SubagentHost(
            _Tools(), llm=llm, retriever=_Retriever(), memory=None,
            max_depth=1, depth=1, read_only=True,
        )
        out = run_subagent(
            "inspect parser", tools=child_tools, llm=llm, retriever=_Retriever(),
            memory=None, max_steps=4, depth=1, read_only=True,
        )
    finally:
        subagent_module.EXPLORER_REASONING = original
    assert llm.shared["profiles"] == ["fast", "fast", "full"], llm.shared
    assert llm.shared["schemas"][0] and llm.shared["schemas"][1] \
        and llm.shared["schemas"][2] == [], llm.shared
    assert "FINAL grounded report" in out, out


@check
def truncated_child_output_is_preserved_without_wrapper_recovery_call():
    import sliceagent.subagent as subagent_module

    class _LengthLLM:
        def __init__(self):
            self.reasoning = "full"
            self.shared = {"calls": 0}

        def complete(self, messages, schemas):
            self.shared["calls"] += 1
            if self.shared["calls"] == 1:
                response = _Resp("inspect parser source")
                response.finish_reason = "tool_calls"
                response.tool_calls = [type("Call", (), {
                    "name": "read_file", "args": {"path": "pkg/parse.py"}, "id": "read-1",
                })()]
                return response
            response = _Resp("partial evidence-backed report")
            response.finish_reason = "length"
            return response

    original = subagent_module.EXPLORER_REASONING
    subagent_module.EXPLORER_REASONING = "full"
    try:
        llm = _LengthLLM()
        child_tools = SubagentHost(
            _Tools(), llm=llm, retriever=_Retriever(), memory=None,
            max_depth=1, depth=1, read_only=True,
        )
        out = run_subagent(
            "inspect parser", tools=child_tools, llm=llm, retriever=_Retriever(),
            memory=None, max_steps=4, depth=1, read_only=True,
        )
    finally:
        subagent_module.EXPLORER_REASONING = original
    assert llm.shared["calls"] == 2, "max_tokens must not trigger a wrapper-level model recovery"
    assert out.startswith("Error: subagent did not finish cleanly:")
    assert "partial evidence-backed report" in out


@check
def exhausted_provider_attempts_do_not_start_a_fourth_report_recovery_call():
    import sliceagent.errors as errors_module
    import sliceagent.subagent as subagent_module

    class _TimeoutLLM:
        def __init__(self):
            self.reasoning = "full"
            self.shared = {"calls": 0}

        def is_retryable(self, _error):
            return True

        def complete(self, messages, schemas):
            self.shared["calls"] += 1
            raise TimeoutError("provider timeout")

    original_profile = subagent_module.EXPLORER_REASONING
    original_backoff = errors_module.jittered_backoff
    subagent_module.EXPLORER_REASONING = "staged"
    errors_module.jittered_backoff = lambda _attempt: 0.0
    try:
        llm = _TimeoutLLM()
        child_tools = SubagentHost(
            _Tools(), llm=llm, retriever=_Retriever(), memory=None,
            max_depth=1, depth=1, read_only=True,
        )
        out = run_subagent(
            "inspect parser", tools=child_tools, llm=llm, retriever=_Retriever(),
            memory=None, max_steps=4, depth=1, read_only=True,
        )
    finally:
        subagent_module.EXPLORER_REASONING = original_profile
        errors_module.jittered_backoff = original_backoff
    assert llm.shared["calls"] == 3, "the shared retry owner gets 3 attempts; wrapper must not add a fourth"
    assert out.startswith("Error: subagent did not finish cleanly:")


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
