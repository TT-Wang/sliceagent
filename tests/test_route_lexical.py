"""The lexical topic router (route_topic_lexical) + the env dispatcher (route). Deterministic, no model.
Contract: continue is the default (zero cost), resume fires only on an explicit parked id or a resume-cue
+ title match, and the host NEVER guesses 'new' (deferred to the agent's new_topic tool — a false-'new'
would wrongly discard the active working set, so it's avoided by design).

Run: PYTHONPATH=src python tests/test_route_lexical.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.session import route, route_topic_lexical   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Topic:
    def __init__(self, task_id, title):
        self.task_id, self.title, self.goal = task_id, title, title


class _Sess:
    def __init__(self, active_goal, parked=()):
        self._active = _Topic("active", active_goal) if active_goal is not None else None
        self.active_id = "active" if active_goal is not None else None
        self._parked = [_Topic(t, ti) for t, ti in parked]

    def active(self):
        return self._active

    def open_threads(self, include_active=False):
        return list(self._parked)


@check
def first_message_is_new():
    assert route_topic_lexical("do the thing", _Sess(None)) == ("new", "")


@check
def follow_up_defaults_to_continue():
    for msg in ["also handle the empty-password case", "now add backoff", "run the tests",
                "looks good, add type hints", "that didn't work, try a lock", "ok ship it"]:
        assert route_topic_lexical(msg, _Sess("fix the auth bug")) == ("continue", ""), msg


@check
def explicit_parked_id_resumes():
    s = _Sess("add caching", parked=[("topic-7", "refactor the parser")])
    assert route_topic_lexical("switch to topic-7", s) == ("resume", "topic-7")


@check
def resume_cue_plus_title_match_resumes():
    s = _Sess("write docs", parked=[("auth", "fix the auth login bug"), ("ci", "set up CI pipeline")])
    assert route_topic_lexical("let's go back to the auth bug", s) == ("resume", "auth")
    assert route_topic_lexical("resume the CI pipeline work", s) == ("resume", "ci")


@check
def resume_cue_without_title_match_stays_continue():
    # "go back" is a cue but nothing matches a parked title → must NOT false-resume (false resume is harmful)
    s = _Sess("fix auth", parked=[("ci", "set up CI")])
    assert route_topic_lexical("let's go back to basics and rewrite from scratch", s) == ("continue", "")


@check
def new_task_is_deferred_to_continue_not_guessed():
    # the host never returns 'new' on its own — the agent's new_topic tool handles a real new task
    for msg in ["separate thing — set up CI", "unrelated: optimize the SQL queries",
                "new task: build a CSV export CLI", "completely different feature: dark mode"]:
        action, _ = route_topic_lexical(msg, _Sess("fix the parser"))
        assert action == "continue", f"host guessed 'new' (risk: false reset): {msg!r}"


@check
def dispatcher_defaults_to_lexical_no_llm_call():
    # AGENT_ROUTER unset → lexical; the llm must NOT be called (a sentinel that explodes if touched)
    class _Boom:
        def complete(self, *a, **k):
            raise AssertionError("dispatcher called the LLM in lexical (default) mode")
    os.environ.pop("AGENT_ROUTER", None)
    assert route(_Boom(), "now add backoff", _Sess("fix auth")) == ("continue", "")


@check
def dispatcher_llm_mode_uses_the_classifier():
    # AGENT_ROUTER=llm → route_topic (the LLM path) is invoked
    class _Resp:
        content = '{"action":"new","task_id":""}'
        def __init__(self): self.tool_calls = []; self.finish_reason = "stop"; self.usage = {}
    class _LLM:
        def __init__(self): self.called = 0
        def complete(self, messages, schemas):
            self.called += 1
            return _Resp()
    os.environ["AGENT_ROUTER"] = "llm"
    try:
        llm = _LLM()
        action, _ = route(llm, "build a totally different thing", _Sess("fix auth"))
        assert llm.called == 1, "llm mode did not call the classifier"
        assert action == "new", action
    finally:
        os.environ.pop("AGENT_ROUTER", None)


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
