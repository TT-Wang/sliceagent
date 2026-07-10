"""Dependency-scoped workspace revision contracts. No model, no pytest."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.workspace_revision import WorkspaceRevision  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


@check
def only_dependency_changes_stale_observation():
    root = tempfile.mkdtemp(prefix="revision-")
    _write(os.path.join(root, "a.py"), "a = 1\n")
    _write(os.path.join(root, "b.py"), "b = 1\n")
    revision = WorkspaceRevision.capture(root, ["a.py"])
    _write(os.path.join(root, "b.py"), "b = 2\n")
    assert revision.is_current(), "an unrelated edit must not globally stale every claim"
    _write(os.path.join(root, "a.py"), "a = 2\n")
    assert [d.path for d in revision.drifted()] == ["a.py"]


@check
def dirty_and_untracked_bytes_are_observed_without_git():
    root = tempfile.mkdtemp(prefix="revision-")
    _write(os.path.join(root, "generated.txt"), "one")
    revision = WorkspaceRevision.capture(root, ["generated.txt"])
    _write(os.path.join(root, "generated.txt"), "two")
    assert not revision.is_current()


@check
def missing_to_created_is_drift():
    root = tempfile.mkdtemp(prefix="revision-")
    revision = WorkspaceRevision.capture(root, ["later.txt"])
    assert revision.is_current()
    _write(os.path.join(root, "later.txt"), "now here")
    assert not revision.is_current()


@check
def revision_round_trips_as_structured_data():
    root = tempfile.mkdtemp(prefix="revision-")
    _write(os.path.join(root, "a"), "x")
    revision = WorkspaceRevision.capture(root, ["a"])
    assert WorkspaceRevision.from_dict(revision.as_dict()) == revision


@check
def dependency_cannot_escape_workspace():
    root = tempfile.mkdtemp(prefix="revision-")
    try:
        WorkspaceRevision.capture(root, ["../secret"])
        assert False, "out-of-root dependency must be rejected"
    except ValueError as exc:
        assert "escapes" in str(exc)


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
