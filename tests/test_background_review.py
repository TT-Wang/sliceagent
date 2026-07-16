"""Item 16 — async background-review fork (OPT-IN, OFF by default). No model, no pytest.
Run: python tests/test_background_review.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.background_review import (  # noqa: E402
    BackgroundReviewer, background_review_enabled, make_background_reviewer,
)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Mem:
    is_durable = True
    def __init__(self, records):
        self._records = records
        self.saved = []
    def read_episodes(self, session_id, *, limit=None):
        return self._records
    def remember(self, content, *, title="", scope="default", tags="", paths=None):
        self.saved.append((title, content, paths))


class _ProjectBoundMem(_Mem):
    def remember_for_project(
        self, content, *, project_id, title="", tags="", paths=None,
    ):
        self.saved.append((project_id, title, content, paths))


def _corrective_records():
    # a corrective-and-cleared episode: an early turn hit an error (meta.failing=True), a later
    # turn edited the file and ended CLEAN (meta.failing=False, stop_reason=end_turn). That last-
    # clean shape is what promote_episodes requires to mine a lesson.
    return [
        {"session_id": "s", "task_id": "t1", "turn": 1,
         "record": {"title": "fix the parser", "note": "",
                    "steps": [{"slice": "", "action": [{"name": "run_command",
                               "args": {"command": "pytest"}, "failing": True}],
                               "observation": ["Error: index out of range in parser.py"]}],
                    "meta": {"failing": True, "files": ["parser.py"], "stop_reason": "tool_use"}}},
        {"session_id": "s", "task_id": "t1", "turn": 2,
         "record": {"title": "fix the parser", "note": "the tokenizer dropped the last token",
                    "steps": [{"slice": "", "action": [{"name": "edit_file",
                               "args": {"path": "parser.py"}, "failing": False}],
                               "observation": ["edited parser.py"]}],
                    "meta": {"failing": False, "files": ["parser.py"], "stop_reason": "end_turn"}}},
    ]


@check
def disabled_by_default():
    os.environ.pop("AGENT_BACKGROUND_REVIEW", None)
    assert background_review_enabled() is False
    # factory returns None → host wires nothing
    assert make_background_reviewer(_Mem([]), scope="t") is None


@check
def review_is_noop_when_disabled():
    os.environ.pop("AGENT_BACKGROUND_REVIEW", None)
    mem = _Mem(_corrective_records())
    # construct directly to bypass the factory's None gate, prove review() self-gates too
    r = BackgroundReviewer(mem, scope="t")
    r.review("s")
    r.join(2.0)
    assert mem.saved == []   # disabled → nothing written


@check
def enabled_writes_lesson_off_thread():
    os.environ["AGENT_BACKGROUND_REVIEW"] = "1"
    try:
        assert background_review_enabled() is True
        mem = _Mem(_corrective_records())
        r = make_background_reviewer(mem, scope="proj")
        assert r is not None
        r.review("s")
        r.join(5.0)
        assert len(mem.saved) == 1
        title, content, paths = mem.saved[0]
        assert "parser.py" in content
        assert paths and "parser.py" in " ".join(paths)   # fix #1: file-context tag now flows through
    finally:
        os.environ.pop("AGENT_BACKGROUND_REVIEW", None)


@check
def review_uses_the_workspace_project_binding_not_mutable_foreground_scope():
    os.environ["AGENT_BACKGROUND_REVIEW"] = "1"
    try:
        mem = _ProjectBoundMem(_corrective_records())
        reviewer = make_background_reviewer(
            mem, scope="human-label", project_id="stable-project-a",
        )
        assert reviewer is not None
        reviewer.review("s")
        reviewer.join(5.0)
        assert len(mem.saved) == 1
        assert mem.saved[0][0] == "stable-project-a"
    finally:
        os.environ.pop("AGENT_BACKGROUND_REVIEW", None)


@check
def not_durable_memory_is_skipped():
    os.environ["AGENT_BACKGROUND_REVIEW"] = "1"
    try:
        class _Null:
            is_durable = False
        assert make_background_reviewer(_Null(), scope="t") is None
    finally:
        os.environ.pop("AGENT_BACKGROUND_REVIEW", None)


@check
def empty_cache_writes_nothing():
    os.environ["AGENT_BACKGROUND_REVIEW"] = "1"
    try:
        mem = _Mem([])
        r = make_background_reviewer(mem, scope="t")
        r.review("s")
        r.join(5.0)
        assert mem.saved == []
    finally:
        os.environ.pop("AGENT_BACKGROUND_REVIEW", None)


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
