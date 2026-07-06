"""Offline tests for consolidation (MEMORY-SPEC step 4 — cache→memory). No model, no pytest.
Run: python tests/test_consolidate.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.neocortex import (  # noqa: E402
    build_learn_prompt, promote_episodes, promote_procedures, render_skill, render_skill_llm)
from sliceagent.memory import make_write_skill_tool  # noqa: E402

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


def prec(task, turn, actions, *, stop="end_turn", failing=False, files=None, title="", requirements_open=0):
    """A procedure-shaped record: a turn with real actions."""
    meta = {"failing": failing, "stop_reason": stop, "files": files or []}
    if requirements_open:
        meta["requirements_open"] = requirements_open
    return {"task_id": task, "turn": turn, "record": {
        "title": title,
        "steps": [{"slice": "", "action": actions, "observation": ["ok"] * len(actions)}],
        "note": "", "meta": meta}}


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
        from sliceagent.memory import MememMemory
        m = MememMemory()
    except Exception:
        print("  (skip: memem not importable)")
        return
    m._vault = tempfile.mkdtemp()
    captured = []
    m.remember = lambda content, *, title="", scope="default", tags="", paths=None: captured.append((title, content, tags))
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
def incomplete_task_with_open_requirements_no_procedure():
    # a clean multistep workflow, but the task DECLARED standing requirements and left one OPEN at its
    # final turn → an incomplete task, so no skill is mined from it (task-outcome gate, #3).
    workflow = [act("read_file", path="calc.py"), act("str_replace", path="calc.py", old_string="x"),
                act("run_command", command="python calc.py")]
    assert promote_procedures([prec("t1", 1, workflow, files=["calc.py"], title="add sub()",
                                    requirements_open=1)]) == []
    # all requirements met (==0 / absent) → still promoted (the gate is the OPEN count, not mere presence)
    assert len(promote_procedures([prec("t1", 1, workflow, files=["calc.py"], title="add sub()",
                                        requirements_open=0)])) == 1


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
        from sliceagent.memory import MememMemory
    except Exception:
        print("  (skip: memem not importable)"); return
    m = MememMemory(); m._vault = tempfile.mkdtemp()
    sk = tempfile.mkdtemp(); os.environ["SLICEAGENT_SKILLS_DIR"] = sk
    captured = []
    m.remember = lambda content, *, title="", scope="default", tags="", paths=None: captured.append(title)
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
        os.environ.pop("SLICEAGENT_SKILLS_DIR", None)


# --- R1: lessons tagged with files (paths_context bonus) + paths threaded read-side ---------------
@check
def promote_episodes_tags_lesson_with_files():
    recs = [rec("t1", 1, ["Error: boom"], failing=True, files=["pkg/core.py"]),
            rec("t1", 2, ["ok"], note="fixed", stop="end_turn", files=["pkg/core.py"])]
    assert promote_episodes(recs)[0]["files"] == ["pkg/core.py"], "R1: lesson must carry its files for paths"


@check
def lookup_threads_paths_to_recall():
    from sliceagent.pagetable import PageTable
    captured = {}
    class _M:
        def recall(self, query, k=6, paths=None):
            captured["paths"] = paths
            return []
    PageTable(memory=_M()).lookup("fix the parser", kind="memory-lessons", k=6, paths=["a.py", "b.py"])
    assert captured["paths"] == ["a.py", "b.py"], "R1: paths must reach recall (→ retrieve paths_context)"


# --- R3: near-duplicate-GOAL workflows collapse to one skill (lexical cluster, no memem) ----------
@check
def near_dup_goal_procedures_collapse_to_one():
    A = [act("read_file", path="a.py"), act("str_replace", path="a.py"), act("run_command", command="t")]
    B = [act("write_file", path="a.py"), act("append_to_file", path="a.py"), act("run_command", command="t")]
    recs = [prec("t1", 1, A, files=["a.py"], title="add pagination to the users endpoint"),
            prec("t2", 1, B, files=["a.py"], title="add pagination to users endpoint handler")]
    assert len(promote_procedures(recs)) == 1, "same-intent workflows (diff shapes) must collapse to one skill"
    # distinct intents must NOT collapse
    recs2 = [prec("t1", 1, A, files=["a.py"], title="add pagination to the users endpoint"),
             prec("t2", 1, B, files=["b.py"], title="parse the markdown changelog into json")]
    assert len(promote_procedures(recs2)) == 2, "distinct intents must stay separate"


# --- mining behaviour moved from the removed live miner (now owned by promote_episodes) ----------
@check
def self_inflicted_episode_mines_nothing():
    # the agent hit its OWN sandbox (confinement) — not an engineering pitfall → mine NOTHING
    recs = [rec("t1", 1, ["Error: path escapes the boundary (/repo): /etc/x — File tools are confined"],
                failing=True, files=["x.py"]),
            rec("t1", 2, ["ok"], note="moved it", stop="end_turn", files=["x.py"])]
    assert promote_episodes(recs) == [], "a self-inflicted confinement error must not mine a lesson"


@check
def real_pitfall_chosen_over_self_inflicted():
    # a turn with BOTH a confinement error AND a real error must mine the REAL one
    recs = [rec("t1", 1, ["Error: path escapes the boundary (/repo): /etc/x",
                          "Error: ImportError: cannot import bar"], failing=True, files=["imp.py"]),
            rec("t1", 2, ["ok"], note="added the missing import", stop="end_turn", files=["imp.py"])]
    lessons = promote_episodes(recs)
    assert len(lessons) == 1
    assert "ImportError" in lessons[0]["content"], f"mined the wrong pitfall: {lessons[0]['content']!r}"
    assert "escapes the boundary" not in lessons[0]["content"], "self-inflicted error leaked into the lesson"


@check
def fact_title_leads_with_note_or_pitfall_never_the_goal():
    # the records carry no user goal; the title is the resolution NOTE (or the pitfall), body leads "Pitfall:"
    recs = [rec("t1", 1, ["Error: ModuleNotFoundError: feedparser"], failing=True, files=["agg.py"]),
            rec("t1", 2, ["ok"], note="installed feedparser", stop="end_turn", files=["agg.py"])]
    lessons = promote_episodes(recs)
    assert len(lessons) == 1
    title, content = lessons[0]["title"], lessons[0]["content"]
    assert "installed feedparser" in title or "ModuleNotFoundError" in title, f"weak title: {title!r}"
    assert content.startswith("Pitfall:"), "the lesson body must LEAD with the pitfall"


# --- B1: LLM-generalized skill body (render_skill_llm) + F1 scan-first ---------------------------
class _StubLLM:
    def complete(self, msgs, schemas):
        class R:
            content = "## When to use\nwhen parsing input\n## Process\n1. read it\n2. transform\n## Pitfalls\nnone known"
        return R()


@check
def render_skill_llm_generalizes_falls_back_and_scans_first():
    proc = {"name": "build-parser", "description": "build the parser",
            "steps": ["read x", "edit y", "run z"], "files": ["p.py"], "freq": 2}
    g = render_skill_llm(proc, _StubLLM())
    assert g.startswith("---\nname:") and "## Process" in g and "Generalized" in g, "LLM body not used"
    assert "## Process (observed)" in render_skill_llm(proc, None), "no-llm must fall back to recorded"
    sproc = dict(proc, description="use api_key=sk-zzz123456 to build")          # F1
    assert "## Process (observed)" in render_skill_llm(sproc, _StubLLM()), "secret must skip the LLM"


# --- B2: /learn prompt (transcript→skill, cache-sourced) + the foreground write_skill tool --------
@check
def learn_prompt_drives_write_skill_from_the_cache():
    p = build_learn_prompt("")
    assert "write_skill" in p and "history/index.md" in p and "## Process" in p
    assert "workflow we just went through" in p                                  # default source
    assert "deploy.md" in build_learn_prompt("the deploy steps in deploy.md")    # honors the user's source


@check
def write_skill_tool_writes_user_provenance_and_validates():
    sk = tempfile.mkdtemp(); os.environ["SLICEAGENT_SKILLS_DIR"] = sk
    try:
        tool = make_write_skill_tool()
        out = tool.handler({"name": "Deploy Flow", "description": "deploy the app to staging",
                            "body": "## When to use\nbefore a release\n## Process\n1. build\n2. push"})
        assert "saved" in out.lower(), out
        import glob
        body = open(glob.glob(os.path.join(sk, "*", "SKILL.md"))[0]).read()
        assert "provenance: user" in body and "deploy the app to staging" in body
        assert "deploy-flow" in body[:80]                                        # name slugged
        assert "need a name" in tool.handler({"name": "x"}).lower()              # validation
    finally:
        os.environ.pop("SLICEAGENT_SKILLS_DIR", None)


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
