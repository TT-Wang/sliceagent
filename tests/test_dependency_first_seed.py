from __future__ import annotations

from sliceagent.active_work import ResourceRef, WorkDelta, WorkItem
from sliceagent.memory import NullMemory
from sliceagent.pfc import Slice, record_user
from sliceagent.seed import make_build_slice
from sliceagent.tools import LocalToolHost


class Retriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=6):
        self.queries.append((query, k))
        return []


class Memory(NullMemory):
    def __init__(self):
        self.recalls = []

    def recall(self, query, k=6, paths=None):
        self.recalls.append((query, k, paths))
        return []


class Ledger:
    def user_sources(self):
        return {"event": "inspect this project"}


def state_with_graph():
    state = Slice(); state.reset("inspect this project")
    record_user(
        state, "inspect this project", source_artifact="local",
        source_event_id="event", logical_id="logical", workspace_epoch=0,
    )
    return state


def test_active_work_without_dependencies_does_not_eagerly_fetch_global_context(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0'\n")
    state = state_with_graph()
    retriever, memory = Retriever(), Memory()
    seed = make_build_slice(
        state, LocalToolHost(str(tmp_path)), retriever, memory, state.goal,
        event_ledger=Ledger(), model_id="test-model",
    )()
    system, user = seed[0]["content"], seed[1]["content"]
    assert retriever.queries == [] and memory.recalls == []
    assert "# REPO MAP" not in system
    assert "# RELATED CODE" not in user and "# RELEVANT MEMORY" not in user
    assert user.count("inspect this project") == 1


def test_typed_file_dependency_faults_in_only_that_live_resource(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0'\n")
    (tmp_path / "target.py").write_text("VALUE = 7\n")
    (tmp_path / "noise.py").write_text("NOISE = 99\n")
    state = state_with_graph()
    root = state.active_work.request_roots[0]
    child = WorkItem(
        id="inspect-target", root_id=root.id, source_refs=root.source_refs,
        description="Inspect target.py", status="in_progress",
        resource_refs=(ResourceRef("workspace_file", "target.py", workspace_epoch=0),),
    )
    state.active_work = state.active_work.apply(WorkDelta(expected_revision=1, creates=(child,)))
    state.active_files = ["noise.py", "target.py"]
    retriever, memory = Retriever(), Memory()
    user = make_build_slice(
        state, LocalToolHost(str(tmp_path)), retriever, memory, state.goal,
        event_ledger=Ledger(), model_id="test-model",
    )()[1]["content"]
    assert "VALUE = 7" in user and "NOISE = 99" not in user
    assert retriever.queries, "a typed live-file dependency may invoke focused code discovery"
    assert memory.recalls == [], "file work must not pull unrelated cross-session lessons"
