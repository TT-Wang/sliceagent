"""Offline tests for consolidation (MEMORY-SPEC step 4 — cache→memory). No model, no pytest.
Run: python tests/test_consolidate.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.consolidate import (  # noqa: E402
    promote_episodes, promote_procedures, render_skill)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def rec(task, turn, obs, note="", failing=False, stop="tool_use", files=None):
    return {"task_id": task, "turn": turn, "record": {
        "steps": [{"slice": "", "action": [], "observation": obs}],
        "note": note, "meta": {"failing": failing, "stop_reason": stop, "files": files or []}}}


def act(name, **args):
    return {"name": name, "args": args, "failing": False}


def prec(task, turn, actions, *, stop="end_turn", failing=False, files=None, title=""):
    """A procedure-shaped record: a turn with real actions."""
    return {"task_id": task, "turn": turn, "record": {
        "title": title,
        "steps": [{"slice": "", "action": actions, "observation": ["ok"] * len(actions)}],
        "note": "", "meta": {"failing": failing, "stop_reason": stop, "files": files or []}}}


@check
def corrective_episode_promotes():
    recs = [rec("t1", 1, ["Error: boom"], failing=True, files=["a.py"]),
            rec("t1", 2, ["ok"], note="fixed by using to_native_string", stop="end_turn", files=["a.py"])]
    lessons = promote_episodes(recs)
    assert len(lessons) == 1
    c = lessons[0]["content"]
    assert "boom" in c and "to_native_string" in c and "a.py" in c
    assert "python" in lessons[0]["tags"]


@check
def no_error_no_lesson():
    assert promote_episodes([rec("t1", 1, ["ok"], note="done", stop="end_turn", files=["a.py"])]) == []


@check
def unresolved_no_lesson():
    recs = [rec("t1", 1, ["Error: boom"], failing=True),
            rec("t1", 2, ["Exit code 1"], failing=True, stop="max_steps")]   # never ended clean
    assert promote_episodes(recs) == []


@check
def dedupe_same_pitfall():
    recs = [rec("t1", 1, ["Error: same boom"], failing=True), rec("t1", 2, ["ok"], stop="end_turn"),
            rec("t2", 1, ["Error: same boom"], failing=True), rec("t2", 2, ["ok"], stop="end_turn")]
    assert len(promote_episodes(recs)) == 1


@check
def secret_excluded():
    recs = [rec("t1", 1, ["Error: api_key=sk-abc123 rejected"], failing=True),
            rec("t1", 2, ["ok"], stop="end_turn")]
    assert promote_episodes(recs) == []


@check
def consolidate_reads_cache_if_memem():
    try:
        from memagent.memory import MememMemory
        m = MememMemory()
    except Exception:
        print("  (skip: memem not importable)")
        return
    m._vault = tempfile.mkdtemp()
    captured = []
    m.remember = lambda content, *, title="", scope="default", tags="": captured.append((title, content, tags))
    m.append_episode("s1", "t1", 1, {"steps": [{"slice": "", "action": [], "observation": ["Error: boom"]}],
                                     "note": "", "meta": {"failing": True, "stop_reason": "tool_use", "files": ["a.py"]}})
    m.append_episode("s1", "t1", 2, {"steps": [{"slice": "", "action": [], "observation": ["ok"]}],
                                     "note": "fixed it", "meta": {"failing": False, "stop_reason": "end_turn", "files": ["a.py"]}})
    m.consolidate("s1")
    assert len(captured) == 1 and "boom" in captured[0][1] and "fixed it" in captured[0][1]
    captured.clear()
    m.consolidate("s-none")                      # no cache file → no-op, no crash
    assert captured == []


@check
def fact_is_frequency_weighted():
    recs = [rec("t1", 1, ["Error: boom"], failing=True), rec("t1", 2, ["ok"], note="fix1", stop="end_turn", files=["a.py"]),
            rec("t2", 1, ["Error: boom"], failing=True), rec("t2", 2, ["ok"], note="fix2", stop="end_turn", files=["b.py"])]
    facts = promote_episodes(recs)
    assert len(facts) == 1 and facts[0]["kind"] == "fact" and facts[0]["freq"] == 2
    assert "recurred 2×" in facts[0]["content"]


@check
def procedure_from_clean_multistep_workflow():
    recs = [prec("t1", 1, [act("read_file", path="calc.py"),
                           act("str_replace", path="calc.py", old_string="x"),
                           act("run_command", command="python calc.py")],
                 files=["calc.py"], title="add sub() to the Calculator class")]
    procs = promote_procedures(recs)
    assert len(procs) == 1 and procs[0]["kind"] == "procedure"
    assert "calculator" in procs[0]["name"] and len(procs[0]["steps"]) == 3
    md = render_skill(procs[0])
    assert md.startswith("---\nname:") and "## Process" in md and "read_file" in md and "calc.py" in md


@check
def corrective_workflow_is_fact_not_procedure():
    recs = [prec("t1", 1, [act("run_command", command="pytest")], failing=True, stop="tool_use", files=["a.py"]),
            prec("t1", 2, [act("str_replace", path="a.py"), act("read_file", path="a.py"),
                           act("run_command", command="pytest")], stop="end_turn", files=["a.py"])]
    assert promote_procedures(recs) == []          # had a failure → it's a fact, not a procedure


@check
def short_or_single_kind_workflow_no_procedure():
    assert promote_procedures([prec("t1", 1, [act("write_file", path="x.py")], title="trivial")]) == []  # <3 actions
    three_reads = [act("read_file", path=f"{i}.py") for i in range(3)]                                    # 1 distinct kind
    assert promote_procedures([prec("t1", 1, three_reads, title="just reading")]) == []


@check
def procedures_dedup_by_shape_repeated_first():
    A = [act("read_file", path="a.py"), act("str_replace", path="a.py"), act("run_command", command="t")]
    B = [act("write_file", path="b.py"), act("append_to_file", path="b.py"), act("run_command", command="t")]
    recs = [prec("t1", 1, A, files=["a.py"], title="A one"),
            prec("t2", 1, A, files=["a.py"], title="A two"),   # same shape → freq 2
            prec("t3", 1, B, files=["b.py"], title="B")]
    procs = promote_procedures(recs)
    assert len(procs) == 2 and procs[0]["freq"] == 2 and procs[1]["freq"] == 1   # repeated shape first


@check
def consolidate_routes_facts_and_procedures_if_memem():
    try:
        from memagent.memory import MememMemory
    except Exception:
        print("  (skip: memem not importable)"); return
    m = MememMemory(); m._vault = tempfile.mkdtemp()
    sk = tempfile.mkdtemp(); os.environ["MEMAGENT_SKILLS_DIR"] = sk
    captured = []
    m.remember = lambda content, *, title="", scope="default", tags="": captured.append(title)
    try:
        # a corrective FACT episode
        m.append_episode("s1", "t1", 1, {"title": "fix", "steps": [{"slice": "", "action": [], "observation": ["Error: boom"]}],
                                         "note": "", "meta": {"failing": True, "stop_reason": "tool_use", "files": ["a.py"]}})
        m.append_episode("s1", "t1", 2, {"title": "fix", "steps": [{"slice": "", "action": [], "observation": ["ok"]}],
                                         "note": "fixed it", "meta": {"failing": False, "stop_reason": "end_turn", "files": ["a.py"]}})
        # a clean multi-step PROCEDURE episode (different task)
        acts = [{"name": "read_file", "args": {"path": "b.py"}, "failing": False},
                {"name": "str_replace", "args": {"path": "b.py"}, "failing": False},
                {"name": "run_command", "args": {"command": "python b.py"}, "failing": False}]
        m.append_episode("s1", "t2", 3, {"title": "build the parser", "steps": [{"slice": "", "action": acts, "observation": ["ok", "ok", "ok"]}],
                                         "note": "", "meta": {"failing": False, "stop_reason": "end_turn", "files": ["b.py"]}})
        m.consolidate("s1")
        assert len(captured) == 1                                   # one FACT remembered
        skills = [d for d in os.listdir(sk) if os.path.isdir(os.path.join(sk, d))]
        assert skills, "expected a procedure skill written"
        body = open(os.path.join(sk, skills[0], "SKILL.md")).read()  # one PROCEDURE skill
        assert body.startswith("---\nname:") and "## Process" in body and "read_file" in body
    finally:
        os.environ.pop("MEMAGENT_SKILLS_DIR", None)


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
