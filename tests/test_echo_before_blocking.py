"""Regression guard for the Enter→echo latency bug. The INVARIANT: the user's message is echoed to the
log BEFORE any blocking work — above all before route_topic's LLM round-trip.

We assert it at the source level (the loop is an I/O-heavy function not worth mocking end-to-end): in each
path the echo call must textually precede the route_topic call. A future edit that reintroduces the bug
(routing before echo) fails here. No model, no pytest. Run: PYTHONPATH=src python tests/test_echo_before_blocking.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def inline_repl_echoes_before_route_topic():
    # cli.main(): _tui.user_echo(...) must appear BEFORE route_topic(...) in the source.
    from sliceagent import cli
    # scope to the REPL while-loop — the _run_one_turn helper (LIVE path, echoes in run_live) also calls
    # route(llm) and is defined BEFORE the loop, so a whole-source search would match the wrong call.
    repl = inspect.getsource(cli.main)
    repl = repl[repl.find("while True:"):]
    i_echo = repl.find("user_echo")
    i_route = repl.find("route(llm")         # the actual routing call (not the 'route_topic' in comments)
    assert i_echo != -1, "the inline REPL no longer echoes the user message"
    assert i_route != -1, "the inline REPL no longer routes topics (test stale?)"
    assert i_echo < i_route, (
        "REGRESSION: topic routing precedes the user-message echo in the inline REPL — "
        "the Enter→echo latency bug is back. Echo BEFORE routing.")


@check
def inline_repl_routing_has_a_spinner():
    # the silent-gap fix: routing must be covered by a status spinner so the UI isn't frozen+silent during an
    # AGENT_ROUTER=llm round-trip (RichSink only spins on SliceBuilt, which fires later inside run_turn).
    from sliceagent import cli
    repl = inspect.getsource(cli.main)
    repl = repl[repl.find("while True:"):]
    assert "routing…" in repl or "routing..." in repl, (
        "routing is not covered by a 'routing…' spinner — the UI will freeze silently in AGENT_ROUTER=llm mode")
    assert repl.find("_console.status") < repl.find("route(llm"), "the routing spinner must wrap the route() call"


@check
def live_path_echoes_before_dispatching_the_turn():
    # the LIVE composer (build_live_app): user_echo must precede spawning the turn worker thread, so the
    # message paints the instant Enter is pressed (same invariant as the REPL + Textual paths).
    try:
        from sliceagent import tui
    except Exception as e:  # noqa: BLE001
        print(f"  (skip live check: {type(e).__name__})")
        return
    src = inspect.getsource(tui.build_live_app)
    i_echo = src.find("user_echo")
    i_thread = src.find("threading.Thread")
    assert i_echo != -1 and i_thread != -1, "build_live_app shape changed (test stale?)"
    assert i_echo < i_thread, "REGRESSION: the live composer spawns the turn worker before echoing the message"


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
