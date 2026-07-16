"""Reliability hardening: hook-safety (extension failures fail open), streaming-assembly
resilience (skip a bad chunk, salvage a mid-stream break, re-roll when nothing assembled), and the opt-in
per-tool scheduler timeout. No model, no pytest. Run: PYTHONPATH=src python tests/test_reliability.py
"""
import os
import subprocess
import sys
import textwrap
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.access import none                                   # noqa: E402
from sliceagent.loop import _safe_advisory, _safe_preflight, run_tool_batch  # noqa: E402
from sliceagent.scheduler import run_scheduled                       # noqa: E402

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
def preflight_hook_failure_degrades_to_proceed():
    class _Boom:
        def preflight_tool(self, n, a):
            raise RuntimeError("extension backend down")
    result = _safe_preflight(_Boom(), "run_command", {"command": "pytest"})
    assert result.stop is False, "a crashing lifecycle hook must not strand ordinary work"


@check
def composite_preflight_failure_does_not_skip_later_catastrophic_floor():
    from sliceagent.hooks import CatastrophicSafeguardHook, CompositeHooks, Hooks

    class _None(Hooks):
        def preflight_tool(self, _name, _args):
            return None

    class _Boom(Hooks):
        def preflight_tool(self, _name, _args):
            raise RuntimeError("optional extension unavailable")

    for first in (_None(), _Boom()):
        result = _safe_preflight(
            CompositeHooks(first, CatastrophicSafeguardHook()),
            "run_command", {"command": "rm -rf /"},
        )
        assert result.stop and result.kind == "catastrophic"


@check
def composite_hook_failure_never_skips_sibling_budget_lifecycle():
    from sliceagent.hooks import BudgetHook, CompositeHooks, Hooks

    class _Broken(Hooks):
        def __getattribute__(self, name):
            if name in {
                "before_step", "record_step_usage", "remaining_token_budget", "reset_for_turn",
            }:
                return lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(name))
            return super().__getattribute__(name)

    budget = BudgetHook(5)
    budget.spent = 99
    hooks = CompositeHooks(_Broken(), budget)
    hooks.reset_for_turn()
    assert budget.spent == 0 and hooks.remaining_token_budget() == 5
    assert hooks.record_step_usage({"prompt_tokens": 2, "completion_tokens": 1}) is None
    assert budget.spent == 3 and hooks.remaining_token_budget() == 2
    assert hooks.record_step_usage({"prompt_tokens": 2, "completion_tokens": 1}) == {
        "stop_turn": True,
    }
    assert budget.spent == 6
    assert hooks.before_step(2).get("stop_turn") is True


class _TC:
    def __init__(self, name, args, _id):
        self.name, self.args, self.id = name, args, _id


class _Tools:
    def accesses(self, n, a):
        return none()
    def run(self, n, a):
        return "RAN"


@check
def run_tool_batch_runs_ordinary_work_when_preflight_raises():
    class _RaisingHooks:
        def preflight_tool(self, n, a):
            raise RuntimeError("boom")
        def transform_tool_result(self, n, a, o):
            return None
    blocked, results = run_tool_batch([_TC("run_command", {}, "1")], _Tools(), lambda e: None, _RaisingHooks())
    assert blocked == 0
    assert results[0]["failing"] is False
    assert results[0]["output"] == "RAN", "extension failure must fail open for ordinary work"


@check
def run_tool_batch_runs_when_preflight_proceeds():
    class _OkHooks:
        def preflight_tool(self, n, a):
            from types import SimpleNamespace
            return SimpleNamespace(stop=False, reason="")
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
    from sliceagent.llm import OpenAILLM
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
def scheduler_timeout_returns_after_bounded_grace():
    release = threading.Event()
    finished = threading.Event()

    def slow():
        try:
            release.wait()
            return "SLOW"
        finally:
            finished.set()

    started = time.monotonic()
    try:
        out = run_scheduled([(none(), lambda: "fast"), (none(), slow)], timeout=0.05)
        elapsed = time.monotonic() - started
        assert out[0] == "fast", "the fast task must still return its real result"
        assert "still running" in out[1], f"the unresolved reader must be explicit: {out[1]!r}"
        assert elapsed < 0.35, "deadline + bounded grace must return without awaiting the hung reader"
    finally:
        release.set()
        assert finished.wait(1), "the daemon fixture must settle and release its global reader slot"


@check
def delegation_wave_is_bounded_by_the_lifecycle_ceiling():
    """A spawned child is PURE_READ but timeout_safe=False (exempt from the SHORT per-tool reader deadline so
    it can SEAL its report). Before the fix that exemption also removed ALL wall-clock bounding, so a child
    whose loop never returned froze the parent turn forever (wave_timeout=None). The generous lifecycle
    ceiling must bound it: a never-returning timeout_safe=False PURE_READ wave returns INDETERMINATE within
    lifecycle_timeout + grace, EVEN when the ordinary reader timeout is off (None) — proving the ceiling, not
    the reader deadline, is what cuts it."""
    from sliceagent.execution import ToolInvocation, ToolOutcome, ToolPurity, ToolStatus
    from sliceagent.scheduler import ScheduledTool, run_ordered

    release = threading.Event()
    finished = threading.Event()
    inv = ToolInvocation("wedged-child", "spawn_agent", {}, 0)

    def wedged():
        try:
            release.wait()
            return ToolOutcome(inv, ToolStatus.SUCCEEDED, "sealed")
        finally:
            finished.set()

    task = ScheduledTool(inv, ToolPurity.PURE_READ, wedged, timeout_safe=False)
    started = time.monotonic()
    try:
        # timeout=None → the SHORT reader deadline is OFF, so ONLY the lifecycle ceiling can cut the wave.
        out = run_ordered([task], timeout=None, lifecycle_timeout=0.05)
        elapsed = time.monotonic() - started
        assert out[0].status is ToolStatus.INDETERMINATE, f"a wedged child must not hang the turn: {out[0].status}"
        assert elapsed < 0.35, f"lifecycle ceiling + grace must return without awaiting the child: {elapsed:.2f}s"
    finally:
        release.set()
        assert finished.wait(1), "the daemon child must settle and release its lifecycle reader slot"


@check
def timed_out_reader_does_not_block_process_exit():
    code = textwrap.dedent("""
        import threading
        from sliceagent.execution import ToolInvocation, ToolOutcome, ToolPurity, ToolStatus
        from sliceagent.scheduler import ScheduledTool, run_ordered

        gate = threading.Event()
        invocation = ToolInvocation("exit-hang", "read_file", {}, 0)
        task = ScheduledTool(
            invocation,
            ToolPurity.PURE_READ,
            lambda: (gate.wait(), ToolOutcome(invocation, ToolStatus.SUCCEEDED, "late"))[1],
        )
        assert run_ordered([task], timeout=0.01)[0].status is ToolStatus.INDETERMINATE
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "src"))
    completed = subprocess.run(
        [sys.executable, "-c", code], env=env, timeout=2, check=False,
    )
    assert completed.returncode == 0, "a detached timed-out reader must not participate in Python exit joins"


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


# ---- extension usage observers cannot strand a completed turn -----------------
@check
def crashing_usage_observer_degrades_to_no_opinion():
    from sliceagent.events import make_dispatcher
    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn

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
    assert res.stop_reason == "end_turn", \
        f"a crashing extension observer must not invent a budget stop, got {res.stop_reason!r}"


# ---- crash-recovery WAL ------------------------------------------------------
@check
def wal_record_pending_clear_roundtrip():
    import tempfile
    from sliceagent import recovery
    os.environ["SLICEAGENT_CACHE_DIR"] = tempfile.mkdtemp(prefix="wal-cache-")
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
        os.environ.pop("SLICEAGENT_CACHE_DIR", None)


@check
def wal_strips_image_base64():
    import json
    import tempfile
    from sliceagent import recovery
    os.environ["SLICEAGENT_CACHE_DIR"] = tempfile.mkdtemp(prefix="wal-img-")
    try:
        root = tempfile.mkdtemp(prefix="ws-img-")
        msgs = [{"role": "user", "content": [{"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,HUGEB64BLOB"}}]}]
        recovery.record(root, goal="g", messages=msgs, step=1)
        blob = json.dumps(recovery.pending(root))
        assert "HUGEB64BLOB" not in blob, "image base64 must be stripped from the WAL (size + privacy)"
        assert "[image attached]" in blob
    finally:
        os.environ.pop("SLICEAGENT_CACHE_DIR", None)


@check
def wal_record_never_raises_when_mkstemp_fails():
    import tempfile
    from sliceagent import recovery
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
    from sliceagent.events import make_dispatcher
    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn
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
    from sliceagent.events import make_dispatcher
    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn

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


@check
def interrupt_mid_wave_harvests_settled_siblings_before_propagating():
    """The Esc-during-fan-out incident: 7 finished children were re-labelled indeterminate because the
    interrupted wave re-raised without surfacing settled outcomes. A SIGINT mid-wave must publish the REAL
    outcomes of already-finished jobs (their side effects exist on disk) and lose only the true stragglers."""
    if os.name == "nt":
        return  # POSIX-signal-driven scenario
    import signal as _signal
    from sliceagent.execution import ToolInvocation, ToolOutcome, ToolPurity, ToolStatus
    from sliceagent.scheduler import ScheduledTool, run_ordered

    hang = threading.Event()
    started = threading.Event()

    def fast(inv):
        return lambda: ToolOutcome(inv, ToolStatus.SUCCEEDED, f"done-{inv.id}")

    def slow(inv):
        def run():
            started.set()
            hang.wait(20)                      # the straggler that held the wave hostage
            return ToolOutcome(inv, ToolStatus.SUCCEEDED, "late")
        return run

    invs = [ToolInvocation(f"iv-{n}", "read_file", {"path": f"f{n}"}, n) for n in range(3)]
    tasks = [
        ScheduledTool(invs[0], ToolPurity.PURE_READ, fast(invs[0])),
        ScheduledTool(invs[1], ToolPurity.PURE_READ, fast(invs[1])),
        ScheduledTool(invs[2], ToolPurity.PURE_READ, slow(invs[2])),
    ]
    published: list = []

    def sigint_when_ready():
        started.wait(5)
        time.sleep(0.3)                        # let the two fast jobs settle
        os.kill(os.getpid(), _signal.SIGINT)   # delivered to the main thread blocked in condition.wait

    threading.Thread(target=sigint_when_ready, daemon=True).start()
    interrupted = False
    try:
        run_ordered(tasks, on_outcomes=lambda wave: published.extend(wave))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        hang.set()                             # release the daemon straggler
    assert interrupted, "the interrupt must still propagate (the turn parks)"
    texts = {out.invocation.id: out.text for out in published}
    assert texts.get("iv-0") == "done-iv-0" and texts.get("iv-1") == "done-iv-1", \
        f"settled siblings must be harvested with their REAL outcomes, got {texts}"
    assert "iv-2" not in texts, "the genuine straggler must NOT get a fabricated success"


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
