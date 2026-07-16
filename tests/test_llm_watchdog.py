"""Off-main-thread LLM hard-timeout watchdog (the TB ThreadPoolExecutor hang fix).

sliceagent's wall-clock backstop for a wedged LLM connection was a SIGALRM, which only arms on the MAIN
thread. Terminal-Bench (and any concurrent host) runs the agent in a ThreadPoolExecutor worker, where
SIGALRM cannot arm — so a silently-stalled completion hung the turn forever. _create_watchdog enforces
the same deadline with a futures watchdog, on any thread. No network. Run:
  PYTHONPATH=src python tests/test_llm_watchdog.py
"""
import copy
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.errors import IndeterminateModelCallError  # noqa: E402
from sliceagent.llm import (OpenAILLM, _PhysicalCallGate, _provider_call_capacity,  # noqa: E402
                            _provider_call_gate)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Completions:
    def __init__(self, delay, value="ok"):
        self._delay = delay; self._value = value
    def create(self, **kwargs):
        time.sleep(self._delay)
        return self._value


class _Chat:
    def __init__(self, comp):
        self.completions = comp


class _Client:
    def __init__(self, delay, value="ok"):
        self.chat = _Chat(_Completions(delay, value))


class _Stub:
    """Duck-typed stand-in exposing only what _create_watchdog touches."""
    def __init__(self, delay, hard=1):
        self._hard_timeout = hard
        self._base_url = "http://local"
        self.client = _Client(delay)


@check
def watchdog_aborts_a_wedged_call_off_main_thread():
    stub = _Stub(delay=10, hard=1)        # SDK call would take 10s; deadline is 1s
    t0 = time.time()
    raised = False
    try:
        OpenAILLM._create_watchdog(stub, {"model": "x", "messages": []})
    except IndeterminateModelCallError:
        raised = True
    dt = time.time() - t0
    assert raised, "watchdog did not raise APITimeoutError on a wedged call"
    assert dt < 4, f"watchdog took {dt:.1f}s — did not enforce the ~1s deadline"


@check
def watchdog_passes_through_a_fast_call():
    stub = _Stub(delay=0.0, hard=5)
    out = OpenAILLM._create_watchdog(stub, {"model": "x", "messages": []})
    assert out == "ok", out


@check
def watchdog_timeout_is_not_retried_while_the_abandoned_socket_may_be_live():
    # Python cannot cancel the timed-out SDK worker. Retrying would overlap it and make latency/spend opaque.
    llm_is_retryable = OpenAILLM.is_retryable
    err = None
    try:
        OpenAILLM._create_watchdog(_Stub(delay=10, hard=1), {"model": "x", "messages": []})
    except IndeterminateModelCallError as e:
        err = e
    assert err is not None, "watchdog did not raise a timeout error"
    # is_retryable is an instance method; call it on an OpenAILLM instance (no network).
    llm = OpenAILLM.__new__(OpenAILLM)
    assert llm.is_retryable(err) is False


@check
def abandoned_watchdog_call_holds_the_one_slot_lease_until_it_physically_closes():
    gate = _PhysicalCallGate(1)
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    lock = threading.Lock()
    audit = {"calls": 0, "active": 0, "max_active": 0}

    class AuditedCompletions:
        def create(self, **_kwargs):
            with lock:
                audit["calls"] += 1
                call_number = audit["calls"]
                audit["active"] += 1
                audit["max_active"] = max(audit["max_active"], audit["active"])
            try:
                if call_number == 1:
                    first_started.set()
                    assert release_first.wait(2), "test failed to release first physical call"
                    return "first-late-result"
                second_started.set()
                return "second-result"
            finally:
                with lock:
                    audit["active"] -= 1

    stub = _Stub(delay=0, hard=0.05)
    stub.client = type("Client", (), {
        "chat": type("Chat", (), {"completions": AuditedCompletions()})(),
    })()
    stub._provider_call_gate = gate

    try:
        OpenAILLM._create_watchdog(stub, {"model": "x", "messages": []})
        assert False, "first logical call must time out while its physical worker remains alive"
    except IndeterminateModelCallError:
        pass
    assert first_started.is_set() and gate.active == 1

    stub._hard_timeout = 1
    box = {}

    def second_logical_call():
        try:
            box["result"] = OpenAILLM._create_watchdog(stub, {"model": "x", "messages": []})
        except Exception as error:  # noqa: BLE001
            box["error"] = error

    second = threading.Thread(target=second_logical_call); second.start()
    time.sleep(0.1)
    assert not second_started.is_set(), "second physical request bypassed the occupied one-slot lease"
    assert audit["max_active"] == 1 and gate.active == 1
    release_first.set(); second.join(2)
    assert not second.is_alive() and box == {"result": "second-result"}, box
    assert second_started.is_set() and audit["max_active"] == 1 and gate.active == 0


@check
def provider_physical_call_cap_defaults_and_rejects_invalid_values():
    old = os.environ.get("LLM_PROVIDER_MAX_INFLIGHT")
    try:
        os.environ.pop("LLM_PROVIDER_MAX_INFLIGHT", None)
        assert _provider_call_capacity() == 4
        os.environ["LLM_PROVIDER_MAX_INFLIGHT"] = "7"
        assert _provider_call_capacity() == 7
        for invalid in ("0", "-2", "not-a-number"):
            os.environ["LLM_PROVIDER_MAX_INFLIGHT"] = invalid
            assert _provider_call_capacity() == 4, invalid
    finally:
        if old is None:
            os.environ.pop("LLM_PROVIDER_MAX_INFLIGHT", None)
        else:
            os.environ["LLM_PROVIDER_MAX_INFLIGHT"] = old


@check
def provider_gate_identity_cannot_be_bypassed_by_runtime_cap_or_equivalent_url():
    old = os.environ.get("LLM_PROVIDER_MAX_INFLIGHT")
    try:
        os.environ["LLM_PROVIDER_MAX_INFLIGHT"] = "1"
        first = _provider_call_gate(("unique-audit-key", "HTTPS://API.DEEPSEEK.COM:443", "none", 60))
        os.environ["LLM_PROVIDER_MAX_INFLIGHT"] = "7"
        equivalent = _provider_call_gate(("unique-audit-key", "https://api.deepseek.com/v1/", "none", 60))
        assert first is equivalent and first.capacity == 1
    finally:
        if old is None:
            os.environ.pop("LLM_PROVIDER_MAX_INFLIGHT", None)
        else:
            os.environ["LLM_PROVIDER_MAX_INFLIGHT"] = old


@check
def main_alarm_is_armed_only_after_the_capacity_lease_is_owned():
    events = []

    class Lease:
        def __init__(self, gate): self.gate = gate; self.released = False
        def release(self):
            if not self.released:
                self.released = True; self.gate.active -= 1; events.append("released")

    class Gate:
        active = 0
        def new_lease(self):
            return Lease(self)
        def acquire_lease(self, lease, **_kwargs):
            self.active += 1; events.append("acquired"); return lease

    class FakeSignal:
        SIGALRM = 1
        def alarm(self, _seconds): events.append("alarm-cancelled")
        def signal(self, _sig, _handler): events.append("handler-restored")

    class Stub:
        _hard_timeout = 1
        _base_url = "http://local"
        _transport_spec = ("x", "http://local", "none", 1)
        _provider_call_gate = Gate()
        def _arm_hard_alarm(self, _remaining):
            assert self._provider_call_gate.active == 1
            events.append("armed")
            return FakeSignal(), object(), "alarm"

    stub = Stub()
    result = OpenAILLM._create(stub, {}, caller=lambda _kwargs: events.append("called") or "ok")
    assert result == "ok" and stub._provider_call_gate.active == 0
    assert events[:3] == ["acquired", "armed", "called"], events
    assert "released" in events


@check
def main_alarm_cleanup_and_capacity_retirement_survive_interrupt_during_release():
    gate = _PhysicalCallGate(1)
    events = []
    original_release = gate._release
    interruptions = [0]

    def interrupt_once(lease):
        interruptions[0] += 1
        if interruptions[0] == 1:
            raise KeyboardInterrupt("deadline edge during lease retirement")
        return original_release(lease)

    gate._release = interrupt_once

    class FakeSignal:
        SIGALRM = 1
        ITIMER_REAL = 0

        def setitimer(self, _kind, value):
            events.append(("timer", value))

        def signal(self, _sig, handler):
            events.append(("handler", handler))

    previous = object()

    class Stub:
        _hard_timeout = 1
        _provider_call_gate = gate

        def _arm_hard_alarm(self, _remaining):
            return FakeSignal(), previous, "itimer"

    try:
        OpenAILLM._create(Stub(), {}, caller=lambda _kwargs: "ok")
        assert False, "injected interruption must propagate after cleanup"
    except KeyboardInterrupt:
        pass
    assert gate.active == 0
    assert ("timer", 0) in events and ("handler", previous) in events


@check
def unavailable_main_alarm_retires_caller_lease_before_watchdog_entry():
    gate = _PhysicalCallGate(1)
    provider_starts = []

    class Stub:
        _hard_timeout = 1
        _provider_call_gate = gate

        def _arm_hard_alarm(self, _remaining):
            return None

        def _create_watchdog(self, _kwargs, _caller, **_options):
            assert gate.active == 0, "main-thread fallback transferred a live lease across the call boundary"
            raise KeyboardInterrupt("interrupt at watchdog entry")

    try:
        OpenAILLM._create(
            Stub(), {}, caller=lambda _kwargs: provider_starts.append(True),
        )
        assert False, "test interruption must propagate"
    except KeyboardInterrupt:
        pass
    assert gate.active == 0 and provider_starts == []


@check
def interrupted_thread_start_cancels_before_worker_can_open_provider_request():
    gate = _PhysicalCallGate(1)
    provider_started = threading.Event()
    original_thread = threading.Thread

    class NativeStartThenInterrupt:
        def __init__(self, *args, **kwargs):
            self.inner = original_thread(*args, **kwargs)

        @property
        def ident(self):
            return self.inner.ident

        def start(self):
            self.inner.start()
            raise KeyboardInterrupt("after native thread start")

        def is_alive(self):
            return self.inner.is_alive()

    stub = _Stub(delay=0, hard=1)
    stub._provider_call_gate = gate
    threading.Thread = NativeStartThenInterrupt
    try:
        try:
            OpenAILLM._create_watchdog(
                stub, {}, caller=lambda _kwargs: provider_started.set() or "impossible",
            )
            assert False, "test interruption must propagate"
        except KeyboardInterrupt:
            pass
    finally:
        threading.Thread = original_thread
    deadline = time.monotonic() + 1
    while gate.active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not provider_started.is_set(), "worker entered provider I/O before ownership handoff committed"
    assert gate.active == 0


@check
def watchdog_done_acknowledgement_outweighs_remaining_thread_teardown_bytecodes():
    original_thread = threading.Thread

    class PessimisticThread:
        def __init__(self, *args, **kwargs):
            self.inner = original_thread(*args, **kwargs)

        @property
        def ident(self):
            return self.inner.ident

        def start(self):
            return self.inner.start()

        def is_alive(self):
            return True  # old code falsely treated this teardown window as a live provider call

    threading.Thread = PessimisticThread
    try:
        result = OpenAILLM._create_watchdog(
            _Stub(delay=0, hard=1), {}, caller=lambda _kwargs: "completed",
        )
    finally:
        threading.Thread = original_thread
    assert result == "completed"


@check
def watchdog_capacity_wait_and_provider_execution_share_one_absolute_deadline():
    gate = _PhysicalCallGate(1)
    occupied = gate.acquire(timeout=1)
    release_provider = threading.Event()
    provider_started = threading.Event()

    class BlockingCompletions:
        def create(self, **_kwargs):
            provider_started.set()
            release_provider.wait(2)
            return "late"

    stub = _Stub(delay=0, hard=0.2)
    stub._provider_call_gate = gate
    stub.client = type("Client", (), {
        "chat": type("Chat", (), {"completions": BlockingCompletions()})(),
    })()
    releaser = threading.Thread(target=lambda: (time.sleep(0.12), occupied.release()), daemon=True)
    releaser.start()
    started = time.monotonic()
    try:
        OpenAILLM._create_watchdog(stub, {"model": "x", "messages": []})
        assert False, "provider worker should consume only the deadline remainder"
    except IndeterminateModelCallError:
        pass
    elapsed = time.monotonic() - started
    try:
        assert provider_started.is_set()
        assert elapsed < 0.27, f"capacity wait reset the watchdog deadline: {elapsed:.3f}s"
    finally:
        release_provider.set()
    releaser.join(1)


def _close(llm):
    try:
        llm.client.close()
    except Exception:  # noqa: BLE001 — test cleanup only
        pass


@check
def absolute_deadline_matches_completion_budget_across_providers_and_sdk_stays_one_shot():
    old = os.environ.pop("LLM_HARD_TIMEOUT_SEC", None)
    old_tokens = os.environ.pop("AGENT_COMPLETION_TOKENS", None)
    deepseek = other = None
    try:
        deepseek = OpenAILLM(model="deepseek-reasoner", api_key="test", proxy="none",
                             base_url="https://api.deepseek.com/v1", timeout=60)
        other = OpenAILLM(model="gpt-5", api_key="test", proxy="none",
                          base_url="https://api.openai.com/v1", timeout=60)
        assert deepseek._hard_timeout == 286, deepseek._hard_timeout
        assert other._hard_timeout == 286, other._hard_timeout
        assert deepseek._timeout == other._timeout == 60
        assert deepseek.client.max_retries == other.client.max_retries == 0
    finally:
        _close(deepseek) if deepseek is not None else None
        _close(other) if other is not None else None
        if old is not None:
            os.environ["LLM_HARD_TIMEOUT_SEC"] = old
        if old_tokens is not None:
            os.environ["AGENT_COMPLETION_TOKENS"] = old_tokens


@check
def explicit_absolute_deadline_survives_live_provider_switch():
    old = os.environ.get("LLM_HARD_TIMEOUT_SEC")
    llm = None
    try:
        os.environ["LLM_HARD_TIMEOUT_SEC"] = "211"
        llm = OpenAILLM(model="deepseek-reasoner", api_key="test", proxy="none",
                        base_url="https://api.deepseek.com/v1", timeout=60)
        assert llm._hard_timeout == 211
        llm.switch(model="gpt-5", base_url="", api_key="test-2")
        assert llm._hard_timeout == 211, "operator override must not be discarded by /model"
        assert llm.client.max_retries == 0
    finally:
        _close(llm) if llm is not None else None
        if old is None:
            os.environ.pop("LLM_HARD_TIMEOUT_SEC", None)
        else:
            os.environ["LLM_HARD_TIMEOUT_SEC"] = old


@check
def defaults_recompute_on_switch_and_stay_isolated_in_shallow_child_view():
    old = os.environ.pop("LLM_HARD_TIMEOUT_SEC", None)
    old_tokens = os.environ.pop("AGENT_COMPLETION_TOKENS", None)
    parent = None
    try:
        parent = OpenAILLM(model="gpt-5", api_key="test", proxy="none",
                           base_url="https://api.openai.com/v1", timeout=60)
        child = copy.copy(parent)
        assert child._hard_timeout == parent._hard_timeout == 286
        child.max_tokens = 16_384
        child.switch(model="deepseek-reasoner")
        assert child._hard_timeout == 542
        assert parent._hard_timeout == 286 and parent.model == "gpt-5", \
            "a child model switch must not mutate the parent's deadline/model"
        parent.switch(model="deepseek-reasoner", base_url="https://api.deepseek.com/v1", api_key="test")
        assert parent._hard_timeout == 286
        parent.switch(model="gpt-5", base_url="", api_key="test")
        assert parent._hard_timeout == 286
    finally:
        _close(parent) if parent is not None else None
        if old is not None:
            os.environ["LLM_HARD_TIMEOUT_SEC"] = old
        if old_tokens is not None:
            os.environ["AGENT_COMPLETION_TOKENS"] = old_tokens


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
