"""Off-main-thread LLM hard-timeout watchdog (the TB ThreadPoolExecutor hang fix).

sliceagent's wall-clock backstop for a wedged LLM connection was a SIGALRM, which only arms on the MAIN
thread. Terminal-Bench (and any concurrent host) runs the agent in a ThreadPoolExecutor worker, where
SIGALRM cannot arm — so a silently-stalled completion hung the turn forever. _create_watchdog enforces
the same deadline with a futures watchdog, on any thread. No network. Run:
  PYTHONPATH=src python tests/test_llm_watchdog.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.llm import OpenAILLM, _import_api_timeout_error  # noqa: E402

APITimeoutError = _import_api_timeout_error()

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
    except APITimeoutError:
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
def watchdog_timeout_is_retryable():
    # the raised error must be classified retryable so with_retry retries (then the loop parks, not hangs)
    llm_is_retryable = OpenAILLM.is_retryable
    err = APITimeoutError(request=None) if False else None
    try:
        OpenAILLM._create_watchdog(_Stub(delay=10, hard=1), {"model": "x", "messages": []})
    except APITimeoutError as e:
        err = e
    assert err is not None, "watchdog did not raise a timeout error"
    # is_retryable is an instance method; call it on an OpenAILLM instance (no network).
    llm = OpenAILLM.__new__(OpenAILLM)
    assert llm.is_retryable(err) is True


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
