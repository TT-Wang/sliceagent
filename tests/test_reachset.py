"""Focused checks for the grounded ReachSet abstraction."""
import os
import tempfile

from sliceagent.reach import ReachSet, is_sensitive_path


def test_primary_and_focus_are_separate():
    with tempfile.TemporaryDirectory() as primary, tempfile.TemporaryDirectory() as focus:
        reach = ReachSet(primary)
        assert reach.roots == (os.path.realpath(primary),)
        assert reach.add(focus, source="user") == os.path.realpath(focus)
        reach.active_focus = focus
        assert reach.active_focus == os.path.realpath(focus)
        assert reach.roots == (os.path.realpath(primary), os.path.realpath(focus))


def test_explicit_observation_auto_admits_narrow_home_directory():
    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory() as primary, tempfile.TemporaryDirectory(dir=home) as focus:
        path = os.path.join(focus, "fact.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("fact")
        reach = ReachSet(primary)
        assert reach.observation_root(path) == os.path.realpath(focus)
        assert reach.contains(path)


def test_blanket_and_sensitive_roots_are_not_implicitly_admitted():
    with tempfile.TemporaryDirectory() as primary:
        reach = ReachSet(primary)
        assert reach.add("/") is None
        assert reach.add("~") is None
        assert is_sensitive_path(os.path.join("~", ".ssh", "id_ed25519"))
        assert reach.observation_root(os.path.join("~", ".ssh")) is None
