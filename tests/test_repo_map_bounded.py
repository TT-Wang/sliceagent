"""The repo-map walk must be BOUNDED — it can never hang the first slice build, no matter what the
workspace root is (a huge monorepo, or a home dir that a stray package.json turned into a 'project').
Found live: launching from /Users/<u> with HOME overridden made os.walk crawl the entire home dir for
minutes; ctrl-c then killed the REPL with a traceback. Invariant: repo_map completes within its walk
budget on ANY root, and the OS-account home never counts as a project root regardless of $HOME.
Run: PYTHONPATH=src python tests/test_repo_map_bounded.py
"""
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent import sensory_cortex as sc  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _make_tree(root: pathlib.Path, n_dirs: int) -> None:
    for i in range(n_dirs):
        d = root / f"pkg{i:03d}"
        d.mkdir(parents=True)
        (d / "mod.py").write_text("x = 1\n")


@check
def walk_stops_at_the_dir_budget():
    import tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    _make_tree(root, 60)
    out = sc.repo_map(str(root), max_dirs=10)
    # bounded: only what the budget allowed got mapped (root dir itself may count as one visit)
    mapped = [ln for ln in out.splitlines() if ln.strip()]
    assert 0 < len(mapped) <= 10, f"expected ≤10 mapped dirs under a 10-dir budget, got {len(mapped)}"


@check
def bounded_walk_is_fast_even_on_a_big_tree():
    import tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    _make_tree(root, 300)
    t0 = time.monotonic()
    out = sc.repo_map(str(root), max_dirs=50)
    dt = time.monotonic() - t0
    assert out, "map must still be produced within the budget"
    assert dt < 5.0, f"bounded walk took {dt:.1f}s — the budget is not bounding"


@check
def default_budget_maps_a_normal_project_fully():
    import tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    _make_tree(root, 20)
    out = sc.repo_map(str(root))          # default max_dirs — far above a normal project
    assert sum(1 for ln in out.splitlines() if "mod.py" in ln) == 20, "normal projects must be unaffected"


@check
def os_account_home_never_counts_as_a_project_even_with_home_overridden():
    import tempfile
    fake_os_home = pathlib.Path(tempfile.mkdtemp())          # stands in for pw_dir
    (fake_os_home / "package.json").write_text("{}")          # the stray marker that caused the live bug
    orig = sc._os_home
    sc._os_home = lambda: fake_os_home.resolve()
    try:
        # $HOME points elsewhere (the live-repro condition), yet the OS home must still be skipped
        assert sc._marker_root(fake_os_home) is None, \
            "a marker in the OS-account home must not make the home a project"
        # ...while a real project UNDER the home is still detected
        proj = fake_os_home / "code" / "myproj"
        proj.mkdir(parents=True)
        (proj / "pyproject.toml").write_text("[project]\nname='x'\n")
        got = sc._marker_root(proj)
        assert got is not None and got.name == "myproj", f"real subproject must still resolve, got {got}"
    finally:
        sc._os_home = orig


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
