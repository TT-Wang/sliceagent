"""Independent behavioral oracle for the task-DAG scheduler scenario.

This file is NOT given to the benchmarked agent to edit. It imports the agent's
final ``taskdag.scheduler.Scheduler`` in a fresh subprocess and exercises the
cumulative behavior demanded by all ten turns, plus the two regression
scenarios that an agent which lost earlier context would get wrong.

Run via subprocess so a crashing/looping import cannot take down the parent and
so import caching never masks a broken module.
"""
import os
import subprocess
import sys
import textwrap

PY = sys.executable


# The probe runs inside the agent's workdir. It uses ONLY the public API the
# turns established: add_task/add_dependency/tasks/dependencies (seed),
# topo_order, has_cycle, ready, waves, remove_task, run, to_dict/from_dict.
# Each check prints a tag on failure.
PROBE = textwrap.dedent('''
    import sys
    from taskdag.scheduler import Scheduler

    fails = []
    def check(cond, tag):
        if not cond:
            fails.append(tag)

    def before(order, x, y):
        # x appears before y in order
        return order.index(x) < order.index(y)

    # ---- seed behavior still intact -----------------------------------
    s = Scheduler()
    s.add_task("a"); s.add_task("b")
    s.add_dependency("b", "a")
    check(s.tasks() == ["a", "b"], "seed_tasks")
    check(s.dependencies("b") == ["a"], "seed_deps")
    s.add_task("b")  # idempotent, must not wipe edge
    check(s.dependencies("b") == ["a"], "seed_idempotent")

    # ---- turn 1: topo_order with alphabetical Kahn tie-break ----------
    s = Scheduler()
    # c depends on a and b; d depends on c
    s.add_dependency("c", "a")
    s.add_dependency("c", "b")
    s.add_dependency("d", "c")
    order = s.topo_order()
    check(sorted(order) == ["a", "b", "c", "d"], "t1_all_present")
    check(before(order, "a", "c"), "t1_a_before_c")
    check(before(order, "b", "c"), "t1_b_before_c")
    check(before(order, "c", "d"), "t1_c_before_d")
    # deterministic alphabetical frontier: a and b are both free initially,
    # a must come before b
    check(before(order, "a", "b"), "t1_alpha_tiebreak")

    # a broader tie-break: three independent roots emit alphabetically
    s = Scheduler()
    s.add_task("z"); s.add_task("m"); s.add_task("a")
    check(s.topo_order() == ["a", "m", "z"], "t1_roots_sorted")

    # ---- turn 2: cycle safety -----------------------------------------
    s = Scheduler()
    s.add_dependency("b", "a")
    raised = False
    try:
        s.add_dependency("a", "b")   # would create a<->b cycle
    except ValueError:
        raised = True
    check(raised, "t2_cycle_raises")
    # graph unchanged after the rejected edge
    check(s.dependencies("a") == [], "t2_graph_unchanged")
    check(s.has_cycle() is False, "t2_no_cycle_after_reject")
    # self dependency
    raised = False
    try:
        s.add_dependency("x", "x")
    except ValueError:
        raised = True
    check(raised, "t2_self_dep_raises")
    # a legal longer chain then a back-edge
    s = Scheduler()
    s.add_dependency("b", "a")
    s.add_dependency("c", "b")
    raised = False
    try:
        s.add_dependency("a", "c")   # a->c would close a->...->a cycle
    except ValueError:
        raised = True
    check(raised, "t2_transitive_cycle_raises")

    # ---- turn 3: ready(done) ------------------------------------------
    s = Scheduler()
    s.add_dependency("c", "a")
    s.add_dependency("c", "b")
    s.add_dependency("d", "c")
    check(s.ready([]) == ["a", "b"], "t3_ready_empty")
    check(s.ready(["a"]) == ["b"], "t3_ready_partial")
    check(s.ready(["a", "b"]) == ["c"], "t3_ready_c")
    check(s.ready(["a", "b", "c"]) == ["d"], "t3_ready_d")
    # completed tasks are never returned
    check(s.ready(["a", "b", "c", "d"]) == [], "t3_ready_all_done")

    # ---- turn 4: waves() ----------------------------------------------
    s = Scheduler()
    s.add_dependency("c", "a")
    s.add_dependency("c", "b")
    s.add_dependency("d", "c")
    s.add_task("e")             # independent root
    check(s.waves() == [["a", "b", "e"], ["c"], ["d"]], "t4_waves")
    # every task in exactly one wave
    flat = [n for lvl in s.waves() for n in lvl]
    check(sorted(flat) == ["a", "b", "c", "d", "e"], "t4_waves_partition")
    check(len(flat) == len(set(flat)), "t4_waves_no_dupes")

    # ---- turn 5: remove_task ------------------------------------------
    s = Scheduler()
    s.add_dependency("b", "a")
    s.add_dependency("c", "b")
    s.remove_task("b")
    check(s.tasks() == ["a", "c"], "t5_removed_from_tasks")
    # c no longer depends on the removed b
    check(s.dependencies("c") == [], "t5_edge_purged")
    raised = False
    try:
        s.remove_task("nope")
    except KeyError:
        raised = True
    check(raised, "t5_missing_keyerror")

    # ---- turn 6: run() plain (all ok) ---------------------------------
    s = Scheduler()
    s.add_dependency("c", "a")
    s.add_dependency("c", "b")
    s.add_dependency("d", "c")
    calls = []
    st = s.run(lambda n: calls.append(n))
    check(st == {"a": "ok", "b": "ok", "c": "ok", "d": "ok"}, "t6_all_ok")
    # runner called once per task, in a valid topo order
    check(sorted(calls) == ["a", "b", "c", "d"], "t6_called_all")
    check(before(calls, "a", "c") and before(calls, "c", "d"), "t6_topo_calls")

    # ---- turn 7: REGRESSION -- remove_task leaves no dangling edge -----
    # A naive remove_task that only deletes its own entry leaves c pointing at
    # the deleted b, so topo_order()/waves() would KeyError or resurface b.
    s = Scheduler()
    s.add_dependency("b", "a")
    s.add_dependency("c", "b")
    s.add_dependency("d", "c")
    s.remove_task("c")          # d still "wants" c until edge is purged
    to = s.topo_order()         # must NOT raise, must not contain c
    check("c" not in to, "t7_removed_absent_from_topo")
    check(sorted(to) == ["a", "b", "d"], "t7_topo_after_remove")
    w = s.waves()               # must NOT raise
    flatw = [n for lvl in w for n in lvl]
    check(sorted(flatw) == ["a", "b", "d"], "t7_waves_after_remove")
    check("c" not in flatw, "t7_removed_absent_from_waves")
    # d now has no surviving dependency, so it is ready immediately
    check("d" in s.ready([]), "t7_dangling_gone_from_ready")
    check(s.dependencies("d") == [], "t7_d_edge_purged")

    # ---- turn 8: run() with failure, direct skip ----------------------
    s = Scheduler()
    s.add_dependency("b", "a")     # b depends on a
    s.add_task("x")                # independent
    def runner_fail_a(n):
        if n == "a":
            raise RuntimeError("boom")
    st = s.run(runner_fail_a)
    check(st["a"] == "failed", "t8_a_failed")
    check(st["b"] == "skipped", "t8_b_skipped")
    check(st["x"] == "ok", "t8_independent_ok")
    # skipped task's runner must NOT have been called
    called = []
    def runner_track(n):
        called.append(n)
        if n == "a":
            raise RuntimeError("boom")
    st = s.run(runner_track)
    check("b" not in called, "t8_skipped_not_called")
    check("a" in called and "x" in called, "t8_others_called")

    # ---- turn 9: REGRESSION -- transitive skip ------------------------
    # chain a -> b -> c; a fails. A shallow implementation skips b but runs c.
    s = Scheduler()
    s.add_dependency("b", "a")
    s.add_dependency("c", "b")
    s.add_task("z")                # independent, should stay ok
    called = []
    def runner2(n):
        called.append(n)
        if n == "a":
            raise ValueError("boom")
    st = s.run(runner2)
    check(st["a"] == "failed", "t9_a_failed")
    check(st["b"] == "skipped", "t9_b_skipped")
    check(st["c"] == "skipped", "t9_c_transitively_skipped")
    check(st["z"] == "ok", "t9_z_independent_ok")
    check("c" not in called, "t9_c_not_called")
    check("b" not in called, "t9_b_not_called")

    # ---- turn 10: serialization round-trip ----------------------------
    s = Scheduler()
    s.add_dependency("c", "a")
    s.add_dependency("c", "b")
    s.add_dependency("d", "c")
    d = s.to_dict()
    check(d["tasks"] == ["a", "b", "c", "d"], "t10_to_dict_tasks")
    check(d["deps"]["c"] == ["a", "b"], "t10_to_dict_deps_c")
    check(d["deps"]["a"] == [], "t10_to_dict_deps_a")
    rebuilt = Scheduler.from_dict(d)
    check(rebuilt.to_dict() == s.to_dict(), "t10_roundtrip_equal")
    # rebuilt graph behaves the same
    check(rebuilt.topo_order() == s.topo_order(), "t10_roundtrip_topo")

    # ---- final integration: diamond + chain with a mid-graph failure ---
    # Graph:
    #   build depends on fetch
    #   test  depends on build
    #   lint  depends on fetch
    #   deploy depends on test AND lint
    #   docs  (independent)
    s = Scheduler()
    s.add_dependency("build", "fetch")
    s.add_dependency("test", "build")
    s.add_dependency("lint", "fetch")
    s.add_dependency("deploy", "test")
    s.add_dependency("deploy", "lint")
    s.add_task("docs")

    # waves of the whole graph
    check(
        s.waves() == [["docs", "fetch"], ["build", "lint"], ["test"], ["deploy"]],
        "int_waves",
    )
    # topo prefix constraints
    order = s.topo_order()
    check(before(order, "fetch", "build"), "int_fetch_before_build")
    check(before(order, "build", "test"), "int_build_before_test")
    check(before(order, "fetch", "lint"), "int_fetch_before_lint")
    check(before(order, "test", "deploy"), "int_test_before_deploy")
    check(before(order, "lint", "deploy"), "int_lint_before_deploy")

    # plant a failure at the mid-graph task "build":
    #   build fails -> test skipped (depends on build)
    #                -> deploy skipped (depends on test, transitively)
    #   fetch ok, lint ok (only depends on fetch), docs ok
    def runner_build_fails(n):
        if n == "build":
            raise RuntimeError("compile error")
    st = s.run(runner_build_fails)
    check(
        st == {
            "fetch": "ok",
            "build": "failed",
            "test": "skipped",
            "lint": "ok",
            "deploy": "skipped",
            "docs": "ok",
        },
        "int_status_map",
    )

    # serialize the integration graph and confirm idempotent round-trip
    check(
        Scheduler.from_dict(s.to_dict()).to_dict() == s.to_dict(),
        "int_roundtrip",
    )

    if fails:
        print("FAILS:" + ",".join(fails))
        sys.exit(1)
    print("ALL_OK")
    sys.exit(0)
''')


def verify(workdir):
    mod = os.path.join(workdir, "taskdag", "scheduler.py")
    if not os.path.isfile(mod):
        return (False, "taskdag/scheduler.py is missing")
    try:
        proc = subprocess.run(
            [PY, "-c", PROBE],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return (False, "probe timed out (possible infinite loop in scheduler)")
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 and "ALL_OK" in out:
        return (True, "all cumulative + regression checks passed")
    # Surface the most useful slice of output.
    detail = out.strip().splitlines()
    tail = "\n".join(detail[-15:]) if detail else "(no output)"
    return (False, "probe failed:\n" + tail)
