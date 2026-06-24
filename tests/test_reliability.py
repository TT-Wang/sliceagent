"""Reliability hardening: hook-safety (advisory fails-open, authorize fails-CLOSED), streaming-assembly
resilience (skip a bad chunk, salvage a mid-stream break, re-roll when nothing assembled), and the opt-in
per-tool scheduler timeout. No model, no pytest. Run: PYTHONPATH=src python tests/test_reliability.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.access import none                                   # noqa: E402
from memagent.loop import _safe_advisory, _safe_authorize, run_tool_batch  # noqa: E402
from memagent.scheduler import run_scheduled                       # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ---- E1: hook safety ---------------------------------------------------------
@check
def advisory_hook_failure_degrades_to_default():
    assert _safe_advisory("x", lambda: 1 / 0, default="D") == "D"      # raises → default
    assert _safe_advisory("x", lambda: {"stop_turn": True}) == {"stop_turn": True}  # ok → value


@check
def authorize_hook_fails_closed():
    class _Boom:
        def authorize_tool(self, n, a):
            raise RuntimeError("permission backend down")
    d = _safe_authorize(_Boom(), "run_command", {"command": "rm -rf /"})
    assert d.allow is False, "a crashing permission hook MUST deny (fail closed), never allow"
    assert "denied" in (d.reason or "")


class _TC:
    def __init__(self, name, args, _id):
        self.name, self.args, self.id = name, args, _id


class _Tools:
    def accesses(self, n, a):
        return none()
    def run(self, n, a):
        return "RAN"


@check
def run_tool_batch_blocks_when_authorize_raises():
    class _RaisingHooks:
        def authorize_tool(self, n, a):
            raise RuntimeError("boom")
        def transform_tool_result(self, n, a, o):
            return None
    blocked, results = run_tool_batch([_TC("run_command", {}, "1")], _Tools(), lambda e: None, _RaisingHooks())
    assert blocked == 1, "a raising authorize must count as a block"
    assert results[0]["failing"] is True
    assert "blocked by policy" in results[0]["output"], results[0]["output"]
    assert "RAN" not in results[0]["output"], "the tool must NOT have executed"


@check
def run_tool_batch_allows_when_authorize_ok():
    class _OkHooks:
        def authorize_tool(self, n, a):
            from types import SimpleNamespace
            return SimpleNamespace(allow=True, reason="")
        def transform_tool_result(self, n, a, o):
            return None
    _, results = run_tool_batch([_TC("read_file", {"path": "x"}, "1")], _Tools(), lambda e: None, _OkHooks())
    assert results[0]["output"] == "RAN" and results[0]["failing"] is False


# ---- E3: streaming-assembly resilience --------------------------------------
from types import SimpleNamespace as NS  # noqa: E402


def _chunk(content=None, finish=None):
    return NS(usage=None, choices=[NS(finish_reason=finish, delta=NS(content=content, tool_calls=None,
                                                                     reasoning_content=None))])


def _mk_llm():
    from memagent.llm import OpenAILLM
    llm = object.__new__(OpenAILLM)          # bypass __init__ (no key needed) — the supported test-stub path
    llm._on_delta = None
    llm.model = "test-model"
    llm._base_url = ""
    return llm


@check
def stream_assembles_normal_completion():
    llm = _mk_llm()
    chunks = [_chunk("hel"), _chunk("lo"), _chunk(finish="stop")]
    llm.client = NS(chat=NS(completions=NS(create=lambda **k: iter(chunks))))
    resp = llm._stream_assemble({})
    assert resp.choices[0].message.content == "hello"
    assert resp.choices[0].finish_reason == "stop"


@check
def stream_skips_a_malformed_chunk():
    llm = _mk_llm()
    class _BadChunk:
        @property
        def usage(self):
            return None
        @property
        def choices(self):
            raise ValueError("corrupt chunk")
    chunks = [_chunk("good "), _BadChunk(), _chunk("tail", finish="stop")]
    llm.client = NS(chat=NS(completions=NS(create=lambda **k: iter(chunks))))
    resp = llm._stream_assemble({})
    assert resp.choices[0].message.content == "good tail", "a bad chunk must be skipped, not fatal"


@check
def stream_salvages_partial_on_midstream_break():
    llm = _mk_llm()
    def _gen(**k):
        yield _chunk("partial answer")
        raise ConnectionError("stream dropped")
    llm.client = NS(chat=NS(completions=NS(create=_gen)))
    resp = llm._stream_assemble({})
    assert resp.choices[0].message.content == "partial answer"
    assert resp.choices[0].finish_reason == "length", "a salvaged partial is marked incomplete"


@check
def stream_reraises_when_nothing_assembled():
    llm = _mk_llm()
    def _gen(**k):
        raise ConnectionError("dropped before any content")
        yield  # pragma: no cover
    llm.client = NS(chat=NS(completions=NS(create=_gen)))
    try:
        llm._stream_assemble({})
        assert False, "an empty broken stream must re-raise so with_retry can re-roll"
    except ConnectionError:
        pass


# ---- E4: scheduler per-tool timeout -----------------------------------------
@check
def scheduler_no_timeout_is_unchanged():
    assert run_scheduled([(none(), lambda: "a"), (none(), lambda: "b")]) == ["a", "b"]


@check
def scheduler_reaps_an_overrunning_task():
    def slow():
        time.sleep(0.6); return "SLOW"
    out = run_scheduled([(none(), lambda: "fast"), (none(), slow)], timeout=0.2)
    assert out[0] == "fast", "the fast task must still return its real result"
    assert "timed out" in out[1], f"the overrunning task must be reaped: {out[1]!r}"


# ---- review round 1: incomplete tool-call salvage must be dropped (llm.py) ----
@check
def stream_drops_incomplete_tool_call():
    # a tool_call delta with id but NO name, then the stream ends → the nameless call must NOT survive
    # (else run_tool_batch sees tc.name=None). It's dropped; surviving content is kept.
    llm = _mk_llm()
    def _gen(**k):
        tc = NS(index=0, id="call_1", function=NS(name=None, arguments='{"a":1}'))
        yield NS(usage=None, choices=[NS(finish_reason=None,
                 delta=NS(content=None, tool_calls=[tc], reasoning_content=None))])
        yield NS(usage=None, choices=[NS(finish_reason="stop",
                 delta=NS(content="answer", tool_calls=None, reasoning_content=None))])
    llm.client = NS(chat=NS(completions=NS(create=_gen)))
    resp = llm._stream_assemble({})
    assert resp.choices[0].message.tool_calls == [], "an incomplete (name=None) tool call must be dropped"
    assert resp.choices[0].message.content == "answer"


# ---- review round 1: budget accounting fails CLOSED (loop.py) -----------------
@check
def crashing_budget_hook_parks_the_turn_fail_closed():
    from memagent.events import make_dispatcher
    from memagent.hooks import Hooks
    from memagent.loop import run_turn

    class _BudgetCrash(Hooks):
        def record_step_usage(self, usage):       # the budget accountant blows up
            raise RuntimeError("budget backend down")

    def build_slice():
        return [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    class _LLM:
        def complete(self, messages, schemas):
            return NS(content="hi", tool_calls=[], finish_reason="stop",
                      usage={"prompt_tokens": 1, "completion_tokens": 1})

    class _Tools:
        def schemas(self):
            return []

    res = run_turn(build_slice=build_slice, llm=_LLM(), tools=_Tools(),
                   dispatch=make_dispatcher(lambda e: None), hooks=_BudgetCrash(), max_steps=3)
    assert res.stop_reason == "token_budget", (
        f"a crashing budget hook must fail CLOSED (park 'token_budget'), got {res.stop_reason!r}")


# ---- crash-recovery WAL ------------------------------------------------------
@check
def wal_record_pending_clear_roundtrip():
    import tempfile
    from memagent import recovery
    os.environ["MEMAGENT_CACHE_DIR"] = tempfile.mkdtemp(prefix="wal-cache-")
    try:
        root = tempfile.mkdtemp(prefix="ws-")
        assert recovery.pending(root) is None
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "working on it"}]
        recovery.record(root, goal="fix the bug", messages=msgs, step=2)
        p = recovery.pending(root)
        assert p and p["goal"] == "fix the bug" and p["step"] == 2, p
        assert recovery.last_assistant(p) == "working on it"
        recovery.clear(root)
        assert recovery.pending(root) is None, "clear() must remove the WAL (clean exit ⇒ no crash record)"
        # per-workspace isolation
        other = tempfile.mkdtemp(prefix="ws2-")
        recovery.record(root, goal="g", messages=msgs, step=1)
        assert recovery.pending(other) is None and recovery.pending(root) is not None
    finally:
        os.environ.pop("MEMAGENT_CACHE_DIR", None)


@check
def wal_strips_image_base64():
    import json
    import tempfile
    from memagent import recovery
    os.environ["MEMAGENT_CACHE_DIR"] = tempfile.mkdtemp(prefix="wal-img-")
    try:
        root = tempfile.mkdtemp(prefix="ws-img-")
        msgs = [{"role": "user", "content": [{"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,HUGEB64BLOB"}}]}]
        recovery.record(root, goal="g", messages=msgs, step=1)
        blob = json.dumps(recovery.pending(root))
        assert "HUGEB64BLOB" not in blob, "image base64 must be stripped from the WAL (size + privacy)"
        assert "[image attached]" in blob
    finally:
        os.environ.pop("MEMAGENT_CACHE_DIR", None)


@check
def wal_record_never_raises_when_mkstemp_fails():
    import tempfile
    from memagent import recovery
    orig = tempfile.mkstemp
    tempfile.mkstemp = lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))
    try:
        recovery.record(tempfile.gettempdir(), goal="g", messages=[{"role": "user", "content": "u"}], step=1)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"record() must swallow a mkstemp failure (no unbound tmp NameError): {e!r}")
    finally:
        tempfile.mkstemp = orig


@check
def run_turn_checkpoints_before_each_llm_call():
    from memagent.events import make_dispatcher
    from memagent.hooks import Hooks
    from memagent.loop import run_turn
    calls = []

    def build_slice():
        return [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    class _LLM:
        def complete(self, messages, schemas):
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    class _Tools:
        def schemas(self):
            return []

    run_turn(build_slice=build_slice, llm=_LLM(), tools=_Tools(),
             dispatch=make_dispatcher(lambda e: None), hooks=Hooks(), max_steps=3,
             checkpoint=lambda m, s: calls.append((len(m), s)))
    assert calls, "checkpoint must fire (crash-recovery WAL needs the in-flight state)"
    assert calls[0][1] == 1, "the first checkpoint is step 1, before the LLM call"


@check
def run_turn_survives_a_crashing_checkpoint():
    # a broken checkpoint (best-effort WAL) must NEVER break the turn
    from memagent.events import make_dispatcher
    from memagent.hooks import Hooks
    from memagent.loop import run_turn

    class _LLM:
        def complete(self, messages, schemas):
            return NS(content="ok", tool_calls=[], finish_reason="stop", usage={})

    class _Tools:
        def schemas(self):
            return []

    res = run_turn(build_slice=lambda: [{"role": "user", "content": "u"}], llm=_LLM(), tools=_Tools(),
                   dispatch=make_dispatcher(lambda e: None), hooks=Hooks(), max_steps=3,
                   checkpoint=lambda m, s: (_ for _ in ()).throw(OSError("disk full")))
    assert res.stop_reason == "end_turn", "a crashing checkpoint must not abort the turn"


# ---- loop guard: deduped reads must NOT kill a long run (only real spin counts as STUCK) -------
@check
def deduped_read_block_is_not_stuck():
    from memagent.hooks import GuardrailHook
    g = GuardrailHook()
    for _ in range(8):
        d = g.authorize_tool("read_file", {"path": "a.py"})
        if not d.allow:
            assert d.counts_as_stuck is False, "a deduped idempotent-read block must NOT count toward STUCK"
            return
        g.transform_tool_result("read_file", {"path": "a.py"}, "same content")   # success, same result
    raise AssertionError("the default guard should eventually block a repeated no-progress read")


@check
def repeated_failing_call_is_stuck():
    from memagent.hooks import GuardrailHook
    g = GuardrailHook()
    for _ in range(8):
        d = g.authorize_tool("run_command", {"command": "x"})
        if not d.allow:
            assert d.counts_as_stuck is True, "a repeated FAILING call MUST count toward STUCK"
            return
        g.transform_tool_result("run_command", {"command": "x"}, "Error: boom")   # failing result
    raise AssertionError("the default guard should block a repeated exact failure")


@check
def run_tool_batch_counts_only_hard_blocks():
    from memagent.hooks import ToolDecision

    class _Soft:
        def authorize_tool(self, n, a):
            return ToolDecision(False, "this read returned the same result", counts_as_stuck=False)
        def transform_tool_result(self, n, a, o):
            return None

    class _Hard:
        def authorize_tool(self, n, a):
            return ToolDecision(False, "real spin", counts_as_stuck=True)
        def transform_tool_result(self, n, a, o):
            return None

    soft, rs = run_tool_batch([_TC("read_file", {"path": "a"}, "1")], _Tools(), lambda e: None, _Soft())
    hard, _ = run_tool_batch([_TC("run_command", {}, "1")], _Tools(), lambda e: None, _Hard())
    assert soft == 0, "a soft (deduped) block must not count toward the STUCK floor"
    assert hard == 1, "a hard (spin) block must count toward the STUCK floor"
    assert rs[0]["failing"] is True and "same result" in rs[0]["output"], "soft-blocked call still nudges"


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
