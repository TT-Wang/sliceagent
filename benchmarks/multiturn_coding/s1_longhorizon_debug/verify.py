"""Independent behavioral oracle for the long-horizon KV-store scenario.

This file is NOT given to the benchmarked agent to edit. It imports the agent's
final ``tinykv.store.KVStore`` in a fresh subprocess and exercises the cumulative
behavior demanded by all six turns, plus the regression scenarios that an agent
which lost earlier context would get wrong.

Run via subprocess so a crashing/looping import cannot take down the parent and
so import caching never masks a broken module.
"""
import os
import subprocess
import sys
import textwrap

PY = sys.executable


# The probe runs inside the agent's workdir. It uses ONLY the public API the
# turns established: set/get/delete/begin/commit/rollback/in_transaction,
# keys/items, numincr, snapshot. Each check prints a tag on failure.
PROBE = textwrap.dedent('''
    import sys
    from tinykv.store import KVStore

    fails = []
    def check(cond, tag):
        if not cond:
            fails.append(tag)

    # ---- seed behavior still intact -----------------------------------
    s = KVStore()
    s.set("a", 1)
    check(s.get("a") == 1, "seed_set_get")
    check(s.get("nope", "d") == "d", "seed_default")
    s.begin(); s.set("a", 2); s.commit()
    check(s.get("a") == 2, "seed_commit")
    s.begin(); s.set("a", 5); s.rollback()
    check(s.get("a") == 2, "seed_rollback")

    # ---- turn 1: nestable transactions --------------------------------
    s = KVStore()
    s.set("x", "base")
    s.begin()                 # layer A
    s.set("x", "A")
    s.begin()                 # layer B (nested)
    s.set("x", "B")
    check(s.get("x") == "B", "t1_nested_visible")
    s.commit()                # B folds into A, not base
    check(s.get("x") == "B", "t1_inner_commit_to_parent")
    check(s.in_transaction(), "t1_still_in_tx")
    s.rollback()              # discard A entirely
    check(s.get("x") == "base", "t1_outer_rollback")
    check(not s.in_transaction(), "t1_closed")

    # nested commit all the way down reaches base
    s = KVStore()
    s.begin(); s.begin(); s.begin()
    s.set("deep", 7)
    s.commit(); s.commit(); s.commit()
    check(s.get("deep") == 7, "t1_deep_commit_to_base")
    check(not s.in_transaction(), "t1_deep_closed")

    # ---- turn 2: nesting-aware deletes / tombstones -------------------
    s = KVStore()
    s.set("k", "v0")
    s.begin()                 # A
    s.delete("k")
    check(s.get("k") is None, "t2_del_hidden_in_A")
    s.begin()                 # B nested under A
    check(s.get("k") is None, "t2_del_hidden_in_B")
    s.rollback()              # discard B; k must remain deleted by A
    check(s.get("k") is None, "t2_del_survives_inner_rollback")
    s.commit()                # A commits tombstone to base
    check(s.get("k") is None, "t2_del_committed_to_base")
    check("k" not in s.snapshot(), "t2_del_gone_from_base")

    # tombstone must not leak into a sibling opened AFTER rollback
    s = KVStore()
    s.set("k", "v0")
    s.begin()                 # A
    s.delete("k")
    s.rollback()              # k restored
    check(s.get("k") == "v0", "t2_del_rolledback_restores")
    s.begin()                 # sibling A2 opened after
    check(s.get("k") == "v0", "t2_no_tombstone_leak_to_sibling")
    s.rollback()

    # commit of a nested delete propagates the tombstone to the PARENT layer
    s = KVStore()
    s.set("k", "v0")
    s.begin()                 # A
    s.begin()                 # B
    s.delete("k")
    s.commit()                # B -> A : A must now hide k
    check(s.get("k") is None, "t2_nested_del_commit_to_parent")
    s.rollback()              # discard A : k visible again
    check(s.get("k") == "v0", "t2_parent_rollback_restores")

    # ---- turn 3: keys() / items() honoring tombstones ----------------
    s = KVStore()
    s.set("a", 1); s.set("b", 2); s.set("c", 3)
    s.begin()
    s.delete("b")
    s.set("d", 4)
    check(s.keys() == ["a", "c", "d"], "t3_keys_view")
    check(s.items() == [("a", 1), ("c", 3), ("d", 4)], "t3_items_view")
    s.begin()
    s.delete("a")
    s.set("b", 22)            # resurrect b in nested layer
    check(s.keys() == ["b", "c", "d"], "t3_keys_nested")
    check(s.items() == [("b", 22), ("c", 3), ("d", 4)], "t3_items_nested")
    s.rollback(); s.rollback()
    check(s.keys() == ["a", "b", "c"], "t3_keys_after_rollback")

    # ---- turn 4: numincr participating in transactions ----------------
    s = KVStore()
    s.numincr("n")            # missing -> 0 + 1
    check(s.get("n") == 1, "t4_incr_from_missing")
    s.numincr("n", 4)
    check(s.get("n") == 5, "t4_incr_by")
    s.begin()
    s.numincr("n", 10)
    check(s.get("n") == 15, "t4_incr_in_tx")
    s.rollback()
    check(s.get("n") == 5, "t4_incr_rolledback")
    s.set("bad", "str")
    raised = False
    try:
        s.numincr("bad")
    except TypeError:
        raised = True
    check(raised, "t4_incr_typeerror")
    # numincr writes at innermost level only
    s = KVStore()
    s.set("c", 1)
    s.begin()                 # A
    s.begin()                 # B
    s.numincr("c", 9)         # -> 10 at B
    check(s.get("c") == 10, "t4_incr_innermost")
    s.rollback()              # discard B
    check(s.get("c") == 1, "t4_incr_inner_rollback")
    s.commit()                # A empty
    check(s.get("c") == 1, "t4_incr_base_intact")

    # ---- turn 5: REGRESSION (nested abort must not corrupt parent) ----
    # Scenario 1 from the prompt: delete in A, set in B, rollback B.
    s = KVStore()
    s.set("k", "OLD")
    s.begin()                 # A
    s.delete("k")             # A tombstones k
    s.begin()                 # B
    s.set("k", "NEW")         # B writes k
    check(s.get("k") == "NEW", "t5_B_sees_new")
    s.rollback()              # discard B ONLY
    check(s.get("k") is None, "t5_after_B_rollback_stays_deleted")
    s.commit()                # A commits tombstone to base
    check(s.get("k") is None, "t5_after_A_commit_deleted")
    check("k" not in s.snapshot(), "t5_base_forgot_k")

    # Scenario 2: delete in A, numincr in B, rollback B, commit A.
    s = KVStore()
    s.set("k", 100)
    s.begin()                 # A
    s.delete("k")             # tombstone in A
    s.begin()                 # B
    s.numincr("k", 1)         # B treats missing-as-0 -> writes 1 in B
    check(s.get("k") == 1, "t5_B_incr_visible")
    s.rollback()              # discard B; A tombstone must survive
    check(s.get("k") is None, "t5_after_incr_rollback_deleted")
    s.commit()                # A -> base : k must be gone, NOT 100
    check(s.get("k") is None, "t5_no_base_leak_after_commit")
    check("k" not in s.snapshot(), "t5_base_truly_forgot_k")

    # Scenario 3: parent writes must be untouched by inner abort.
    s = KVStore()
    s.set("p", "base")
    s.begin()                 # A
    s.set("p", "A_val")
    s.set("q", "A_q")
    s.begin()                 # B
    s.set("p", "B_val")
    s.delete("q")
    s.rollback()              # discard B
    check(s.get("p") == "A_val", "t5_parent_value_intact")
    check(s.get("q") == "A_q", "t5_parent_other_intact")
    s.commit()
    check(s.get("p") == "A_val", "t5_parent_commit_p")
    check(s.get("q") == "A_q", "t5_parent_commit_q")

    # ---- turn 6: snapshot() of committed base only -------------------
    s = KVStore()
    s.set("a", 1); s.set("b", 2)
    s.begin()
    s.set("a", 999)           # uncommitted
    s.set("z", 0)             # uncommitted new key
    s.delete("b")             # uncommitted delete
    snap = s.snapshot()
    check(snap == {"a": 1, "b": 2}, "t6_snapshot_committed_only")
    # snapshot is a copy
    snap["a"] = "mutated"
    snap["new"] = 1
    check(s.get("a", "_t") == 999, "t6_snapshot_copy_no_effect_open")
    s.rollback()
    check(s.get("a") == 1 and s.get("b") == 2, "t6_after_rollback")
    snap2 = s.snapshot()
    check(snap2 == {"a": 1, "b": 2}, "t6_snapshot_after_rollback")
    # no sentinel objects leak into snapshot
    for v in snap2.values():
        check(not (type(v).__name__ == "object" and v is not None and not isinstance(v, (int, str, float, bool, list, dict, tuple))), "t6_no_sentinel")

    # ---- cross-feature interplay: everything together ----------------
    s = KVStore()
    s.set("ctr", 10)
    s.set("name", "init")
    s.begin()                 # A
    s.numincr("ctr", 5)       # ctr -> 15 in A
    s.delete("name")          # tombstone name in A
    s.begin()                 # B
    s.numincr("ctr", 100)     # ctr -> 115 in B
    s.set("name", "revived")  # name back in B
    check(s.keys() == ["ctr", "name"], "x_keys_with_revive")
    check(s.get("ctr") == 115, "x_ctr_in_B")
    s.rollback()              # discard B
    check(s.get("ctr") == 15, "x_ctr_back_to_A")
    check(s.get("name") is None, "x_name_still_deleted")
    check(s.keys() == ["ctr"], "x_keys_after_B_rollback")
    s.commit()                # A -> base
    check(s.snapshot() == {"ctr": 15}, "x_final_snapshot")

    if fails:
        print("FAILS:" + ",".join(fails))
        sys.exit(1)
    print("ALL_OK")
    sys.exit(0)
''')


def verify(workdir):
    store = os.path.join(workdir, "tinykv", "store.py")
    if not os.path.isfile(store):
        return (False, "tinykv/store.py is missing")
    try:
        proc = subprocess.run(
            [PY, "-c", PROBE],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return (False, "probe timed out (possible infinite loop in store)")
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 and "ALL_OK" in out:
        return (True, "all cumulative + regression checks passed")
    # Surface the most useful slice of output.
    detail = out.strip().splitlines()
    tail = "\n".join(detail[-15:]) if detail else "(no output)"
    return (False, "probe failed:\n" + tail)
