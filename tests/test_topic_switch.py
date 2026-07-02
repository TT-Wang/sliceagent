"""Topic-switch tests (MEMORY-SPEC step 3 acceptance spec). No model, no pytest.
Run: python tests/test_topic_switch.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.memory import NullMemory   # noqa: E402
from sliceagent.session import Session     # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def fresh():
    return Session(NullMemory(), "s-test")


@check
def new_topic_is_isolated():
    sess = fresh()
    sess.new_topic("task A")
    a = sess.active()
    a.findings = ["A finding"]; a.active_files = ["a.py"]; a.edited_files = {"a.py"}; a.last_error = "A err"
    b_id = sess.new_topic("task B")
    b = sess.active()
    assert b.goal == "task B" and sess.active_id == b_id
    assert b.findings == [] and b.active_files == [] and b.edited_files == set() and b.last_error == ""


@check
def switch_back_restores():
    sess = fresh()
    a_id = sess.new_topic("task A")
    a = sess.active(); a.findings = ["A finding"]; a.active_files = ["a.py"]; a.edited_files = {"a.py"}
    sess.new_topic("task B"); sess.active().findings = ["B finding"]
    sa = sess.switch_topic(a_id)
    assert sa.goal == "task A" and sa.findings == ["A finding"]
    assert sa.active_files == ["a.py"] and sa.edited_files == {"a.py"}
    assert "B finding" not in sa.findings


@check
def independent_evolution():
    sess = fresh()
    a_id = sess.new_topic("A"); sess.active().findings = ["a1"]
    b_id = sess.new_topic("B"); sess.active().findings = ["b1"]
    sess.switch_topic(a_id); sess.active().findings.append("a2")
    sess.switch_topic(b_id)
    assert sess.active().findings == ["b1"]            # B untouched by A's later edit
    sess.switch_topic(a_id)
    assert sess.active().findings == ["a1", "a2"]


@check
def in_session_switch_is_lossless():
    # within a session, switching keeps the SAME slice object — findings/action_log preserved
    sess = fresh()
    a_id = sess.new_topic("A")
    sess.active().findings = ["a distilled finding"]
    sess.active().action_log = {"sig": {"count": 1}}
    sess.new_topic("B")
    sa = sess.switch_topic(a_id)
    assert sa.findings == ["a distilled finding"]
    assert sa.action_log == {"sig": {"count": 1}}


@check
def open_threads_index():
    sess = fresh()
    a_id = sess.new_topic("task A")
    b_id = sess.new_topic("task B")
    c_id = sess.new_topic("task C")                    # C active
    threads = sess.open_threads()                      # excludes the active one (C)
    assert sorted(t.task_id for t in threads) == sorted([a_id, b_id])
    assert all(t.status == "parked" for t in threads)
    allt = {t.task_id: t for t in sess.open_threads(include_active=True)}
    assert allt[c_id].status == "active" and allt[a_id].title == "task A"


@check
def switch_unknown_raises():
    sess = fresh(); sess.new_topic("A")
    try:
        sess.switch_topic("t-nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


@check
def cross_session_resume_if_memem():
    try:
        from sliceagent.memory import MememMemory
        m = MememMemory()
    except Exception:
        print("  (skip: memem not importable)")
        return
    m._vault = tempfile.mkdtemp()
    s1 = Session(m, "sess-1")
    a_id = s1.new_topic("durable task A")
    s1.active().findings = ["persisted finding"]; s1.active().active_files = ["x.py"]
    s1.active().since_edit = 5
    s1.new_topic("task B")                              # parks A → durable checkpoint to vault
    s2 = Session(m, "sess-2")                           # fresh session, no live tasks
    sa = s2.switch_topic(a_id)                          # a_id not live → load_task from vault
    assert sa.goal == "durable task A" and sa.findings == ["persisted finding"]
    assert sa.active_files == ["x.py"]
    assert sa.since_edit == 0                           # cross-session resume = fresh action epoch


@check
def build_renders_other_threads_and_follows_active():
    from sliceagent.retriever import NullRetriever
    from sliceagent.seed import make_build_slice
    from sliceagent.tools import LocalToolHost
    sess = fresh()
    a_id = sess.new_topic("fix the parser")
    b_id = sess.new_topic("write the docs")        # B active, A parked
    tools = LocalToolHost(tempfile.mkdtemp())
    build = make_build_slice(sess, tools, NullRetriever(), NullMemory(), "write the docs")
    system, user = (m["content"] for m in build())
    assert "OTHER OPEN THREADS" in user and a_id in user and "fix the parser" in user
    assert "write the docs" in user                 # 2B: CURRENT REQUEST = the ACTIVE topic's goal (user, not system)
    assert "write the docs" not in system           # goal no longer rides the byte-stable system message
    sess.switch_topic(a_id)                          # build follows the active topic
    system2, user2 = (m["content"] for m in build())
    assert "fix the parser" in user2 and b_id in user2     # now A active (CURRENT REQUEST), B listed as a thread


@check
def topic_tools_drive_routing():
    from sliceagent.session import make_topic_tools
    sess = fresh()
    by = {t.name: t for t in make_topic_tools(sess)}
    a_id = sess.new_topic("task A")
    sess.new_topic("task B")                        # B active
    assert "Switched" in by["switch_topic"].handler({"task_id": a_id}) and sess.active_id == a_id
    assert by["new_topic"].handler({"goal": "task C"}) and sess.active().goal == "task C"
    assert "Error" in by["switch_topic"].handler({"task_id": "t-nope"})   # unknown → no crash


class FakeLLM:
    def __init__(self, content):
        self._c = content
    def complete(self, messages, tools):
        from sliceagent.interfaces import AssistantMessage
        return AssistantMessage(content=self._c, tool_calls=[], usage={}, finish_reason="stop")


@check
def continue_topic_preserves_context():
    sess = fresh()
    sess.new_topic("implement add()")
    s = sess.active()
    s.findings = ["added add()"]; s.active_files = ["calc.py"]; s.edited_files = {"calc.py"}
    s.last_error = "boom"; s.since_edit = 5
    s.action_log = {"sig": {"count": 3, "failing": True, "last": "boom"}}
    sess.continue_topic("now add a docstring")
    s2 = sess.active()
    assert s2.goal == "now add a docstring"                # new directive
    assert s2.findings == ["added add()"] and s2.active_files == ["calc.py"] and s2.edited_files == {"calc.py"}
    assert s2.last_error == "" and s2.since_edit == 0     # fresh error/convergence epoch
    # I3 WS2 — the anti-loop epoch is DEMOTED, not cleared: counts survive (a genuinely repeated
    # command still trips REPEATED-with-no-progress) but the stale failing flag is dropped.
    assert s2.action_log == {"sig": {"count": 3, "failing": False, "last": "boom"}}


@check
def router_classifies():
    from sliceagent.session import route_topic
    sess = fresh()
    assert route_topic(FakeLLM('{"action":"continue"}'), "hi", sess) == ("new", "")   # no active → new, no call
    a_id = sess.new_topic("task A")
    assert route_topic(FakeLLM('{"action":"continue"}'), "more", sess) == ("continue", "")
    assert route_topic(FakeLLM('{"action":"new"}'), "different", sess) == ("new", "")
    sess.new_topic("task B")                                # B active, A parked
    assert route_topic(FakeLLM(f'{{"action":"resume","task_id":"{a_id}"}}'), "go back", sess) == ("resume", a_id)
    assert route_topic(FakeLLM('{"action":"resume","task_id":"t-nope"}'), "x", sess) == ("continue", "")   # bad id
    assert route_topic(FakeLLM('garbage, no json here'), "x", sess) == ("continue", "")   # parse fail → safe


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
