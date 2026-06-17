import os
import sys
import json
import subprocess


# Independent oracle. We do NOT import the agent's mini_grep.py or trust any test
# the agent can see. We import the (possibly-edited) cliparse.py in a FRESH
# subprocess and exercise nargs behavior directly on argparse's own API with
# inputs the agent never saw, so the bug cannot be papered over in the app or by
# special-casing the README's exact example.

_CHILD = r'''
import json, sys
import cliparse as ap

results = {}

def run(tag, build, argv):
    try:
        p = build()
        ns = p.parse_args(argv)
        results[tag] = {"ok": True, "ns": {k: v for k, v in vars(ns).items()}}
    except SystemExit as e:
        results[tag] = {"ok": False, "err": "SystemExit:%s" % e.code}
    except BaseException as e:
        results[tag] = {"ok": False, "err": "%s:%s" % (type(e).__name__, e)}


# Case 1: optional nargs='*' must greedily collect ALL following values; '--'
# cleanly ends the list before the required positional (fresh values).
def b1():
    p = ap.ArgumentParser(prog="t1", exit_on_error=False)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("needle")
    p.add_argument("files", nargs="*")
    return p
run("star_multi_dd", b1, ["--exclude", "p1", "p2", "p3", "--", "HIT", "a.txt", "b.txt"])

# Case 1b: same parser, exclude given AFTER the positionals (no '--' needed).
run("star_after_pos", b1, ["HIT", "a.txt", "b.txt", "--exclude", "p1", "p2", "p3"])

# Case 2: optional nargs='*' with a SINGLE value, then '--' and a positional.
run("star_single", b1, ["--exclude", "only", "--", "NEEDLE", "z.txt"])

# Case 3: optional nargs='*' with ZERO values (must still parse, gives []).
run("star_zero", b1, ["--exclude", "--", "NEEDLE2"])

# Case 4: optional nargs='+' must collect ALL following values.
def b4():
    p = ap.ArgumentParser(prog="t4", exit_on_error=False)
    p.add_argument("--items", nargs="+", type=int)
    p.add_argument("tail", nargs="*")
    return p
run("plus_multi", b4, ["--items", "10", "20", "30", "40"])

# Case 4b: nargs='+' collects all values up to the '--' separator.
run("plus_then_dd", b4, ["--items", "10", "20", "30", "--", "x"])

# Case 5: optional nargs='+' with only ONE value still works.
run("plus_one", b4, ["--items", "7"])

# Case 6: REGRESSION GUARD - fixed-count nargs=2 must stay exactly two.
def b6():
    p = ap.ArgumentParser(prog="t6", exit_on_error=False)
    p.add_argument("--pair", nargs=2, type=int)
    p.add_argument("rest", nargs="*")
    return p
run("fixed_two", b6, ["--pair", "3", "4", "leftover1", "leftover2"])

# Case 7: REGRESSION GUARD - nargs='?' optional still takes at most one.
def b7():
    p = ap.ArgumentParser(prog="t7", exit_on_error=False)
    p.add_argument("--maybe", nargs="?", const="C", default="D")
    p.add_argument("tail", nargs="*")
    return p
run("opt_qmark", b7, ["--maybe", "X", "t1", "t2"])

# Case 8: REGRESSION GUARD - positional nargs='*' (non-option side) intact.
def b8():
    p = ap.ArgumentParser(prog="t8", exit_on_error=False)
    p.add_argument("--flag", action="store_true")
    p.add_argument("words", nargs="*")
    return p
run("pos_star", b8, ["--flag", "alpha", "beta", "gamma"])

sys.stdout.write(json.dumps(results))
'''


def _expect(results, tag, ok, ns_expect=None):
    r = results.get(tag)
    if r is None:
        return False, "missing case %r" % tag
    if r.get("ok") != ok:
        return False, "case %r: ok=%r (%s)" % (tag, r.get("ok"), r.get("err", ""))
    if ok and ns_expect is not None:
        for k, v in ns_expect.items():
            got = r["ns"].get(k)
            if got != v:
                return False, "case %r: %s=%r, expected %r" % (tag, k, got, v)
    return True, ""


def verify(workdir):
    cli = os.path.join(workdir, "cliparse.py")
    if not os.path.isfile(cli):
        return False, "cliparse.py not found in workdir"

    # Defensively drop any cached bytecode so we always test the CURRENT source
    # (an edit landing in the same wall-clock second can otherwise leave a stale
    # .pyc that import would reuse). The child also runs with -B (no .pyc writes).
    pycache = os.path.join(workdir, "__pycache__")
    if os.path.isdir(pycache):
        for fn in os.listdir(pycache):
            if fn.startswith("cliparse.") and fn.endswith(".pyc"):
                try:
                    os.remove(os.path.join(pycache, fn))
                except OSError:
                    pass

    proc = subprocess.run(
        [sys.executable, "-B", "-c", _CHILD],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return False, "child crashed (rc=%d): %s" % (
            proc.returncode, (proc.stderr or proc.stdout)[-600:])
    try:
        results = json.loads(proc.stdout)
    except Exception as e:
        return False, "could not parse child output: %s :: %r" % (e, proc.stdout[-400:])

    checks = [
        ("star_multi_dd", True, {"exclude": ["p1", "p2", "p3"], "needle": "HIT",
                                 "files": ["a.txt", "b.txt"]}),
        ("star_after_pos", True, {"exclude": ["p1", "p2", "p3"], "needle": "HIT",
                                  "files": ["a.txt", "b.txt"]}),
        ("star_single", True, {"exclude": ["only"], "needle": "NEEDLE",
                               "files": ["z.txt"]}),
        ("star_zero", True, {"exclude": [], "needle": "NEEDLE2", "files": []}),
        ("plus_multi", True, {"items": [10, 20, 30, 40], "tail": []}),
        ("plus_then_dd", True, {"items": [10, 20, 30], "tail": ["x"]}),
        ("plus_one", True, {"items": [7], "tail": []}),
        ("fixed_two", True, {"pair": [3, 4], "rest": ["leftover1", "leftover2"]}),
        ("opt_qmark", True, {"maybe": "X", "tail": ["t1", "t2"]}),
        ("pos_star", True, {"flag": True, "words": ["alpha", "beta", "gamma"]}),
    ]
    for tag, ok, ns in checks:
        passed, detail = _expect(results, tag, ok, ns)
        if not passed:
            return False, "FAIL " + detail

    return True, ("all 10 fresh nargs cases pass (multi '*'/'+' collect every "
                  "value; fixed-count and '?' regressions intact)")
