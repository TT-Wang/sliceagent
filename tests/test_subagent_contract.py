"""Focused offline tests for the typed subagent brief/artifact contract.

Run: PYTHONPATH=src python tests/test_subagent_contract.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.hippocampus import HippocampusMixin  # noqa: E402
from sliceagent.intent import IntentEntry  # noqa: E402
from sliceagent.memory import NullMemory  # noqa: E402
from sliceagent.retriever import NullRetriever  # noqa: E402
from sliceagent.subagent import SubagentHost, run_subagent  # noqa: E402
from sliceagent.subagent_contract import (  # noqa: E402
    SubagentArtifact,
    SubagentBrief,
)


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


class _Archive(HippocampusMixin, NullMemory):
    # The archive methods are real even though optional semantic memory is intentionally off.
    is_durable = False

    def __init__(self, root):
        self._vault = root
        self._idx_lock = threading.Lock()


class _DurableFailure(NullMemory):
    is_durable = True

    def append_subagent_artifact(self, session_id, artifact):
        return ""


class _DurableRaise(_DurableFailure):
    def append_subagent_artifact(self, session_id, artifact):
        raise OSError("disk full")


class _Response:
    def __init__(self, text):
        self.content = text
        self.tool_calls = []
        self.finish_reason = "stop"
        self.usage = {"prompt_tokens": 10, "completion_tokens": 2}


class _LLM:
    def __init__(self, text="child report"):
        self.text = text
        self.reasoning = "fast"
        self.calls = [0]
        self.last_prompt = [""]

    def complete(self, messages, schemas):
        self.calls[0] += 1
        self.last_prompt[0] = "\n".join(str(message.get("content", "")) for message in messages)
        return _Response(self.text)


class _Tools:
    def __init__(self, root):
        self._root = root

    def root(self):
        return self._root

    def schemas(self):
        return []

    def accesses(self, name, args):
        return []

    def run(self, name, args):
        return ""

    def read_text(self, path):
        return ""


def _legacy_artifact(report, *, name=""):
    return {"kind": "explorer", "name": name, "task": "old", "brief": {"task": "old"},
            "status": "ok", "steps": 1, "report": report, "findings": [], "change_set": [],
            "files": [], "coverage": "old coverage", "refs": []}


@check
def brief_preserves_exact_clauses_sources_and_only_explicit_context():
    clause = IntentEntry(id="intent-7", verbatim_clause="Preserve APIName exactly — including case.",
                         source_artifact="history/turn-2.md", source_range=(14, 54), authority="user")
    brief = SubagentBrief.create(
        "Audit only the parser.", intent_entries=[clause], scope=["src/parser.py"],
        exclusions=["Do not edit files"], report_shape="Return findings with line evidence.",
        canonical_refs=["subagents/sub-3.md"], drift_policy="fail")
    rendered = brief.render()
    assert clause.verbatim_clause in rendered and "history/turn-2.md" in rendered and "14:54" in rendered
    assert "Audit only the parser." in rendered and "Do not edit files" in rendered
    assert 'read_file("subagents/sub-3.md")' in rendered
    assert "parent assistant transcript that was never selected" not in rendered
    assert SubagentBrief.from_dict(brief.to_dict()) == brief


@check
def brief_keeps_corrections_distinct_from_binding_constraints():
    correction = IntentEntry(
        id="intent-correction", verbatim_clause="Actually, Python 3.12 is installed.",
        source_artifact="history/turn-3.md", authority="user", kind="correction",
    )
    brief = SubagentBrief.create("Inspect runtime.", intent_entries=[correction])
    rendered = brief.render()
    assert "USER CORRECTIONS / CLARIFICATIONS" in rendered
    assert "Python 3.12 is installed" in rendered
    assert "BINDING USER CONSTRAINTS" not in rendered


@check
def current_correction_is_not_duplicated_as_a_binding_child_constraint():
    from sliceagent.intent import IntentState

    request = "Actually, there are 2 failing tests."
    intent = IntentState(current_request=request, current_source="turn-current")
    intent.add_exact(
        request, source_artifact="turn-current", source_range=(0, len(request)),
        authority="user", kind="correction",
    )
    rendered = SubagentBrief.create("Inspect tests.", intent_entries=intent).render()
    assert rendered.count(request) == 1
    assert request in rendered.split("USER CORRECTIONS / CLARIFICATIONS", 1)[1]
    assert "BINDING USER CONSTRAINTS" not in rendered


@check
def legacy_mutable_grant_record_stays_readable_but_is_not_upgraded_to_authority():
    brief = SubagentBrief.from_dict({"task": "old job", "grants": ["subagents/auth-explorer.md"]})
    assert brief.objective == "old job" and brief.canonical_refs == ()


@check
def host_threads_selected_intent_and_typed_identity_into_seal():
    with tempfile.TemporaryDirectory() as root:
        memory = _Archive(os.path.join(root, "vault"))
        tools = _Tools(root)
        llm = _LLM("found parser issue")
        clause = IntentEntry(id="intent-1", verbatim_clause="Keep parse_v1 public.",
                             source_artifact="history/turn-1.md", source_range=(0, 21), authority="user")
        host = SubagentHost(
            tools, llm=llm, retriever=NullRetriever(), memory=memory, policy=None,
            max_depth=1, session_id="session-9", workspace_id="workspace-9",
            intent_provider=lambda objective: [clause], task_id_fn=lambda: "task-4",
            parent_id_fn=lambda: "turn-parent-8")
        out = host.run("spawn_agent", {
            "agent": "explorer", "task": "audit parser", "scope": ["src/parser.py"],
            "exclusions": ["no edits"], "report_shape": "status + evidenced findings",
            "drift_policy": "report",
        })
        assert out.startswith("[explore ok") and 'read_file("subagents/sub-1.md")' in out
        record = memory.read_subagent_artifacts("session-9")[-1]["artifact"]
        assert record["contract_v"] == 1
        assert record["workspace_id"] == "workspace-9" and record["session_id"] == "session-9"
        assert record["task_id"] == "task-4" and record["parent_id"] == "turn-parent-8"
        assert record["brief"]["objective"] == "audit parser"
        carried = record["brief"]["intent_clauses"][0]
        assert carried["id"] == clause.id and carried["verbatim_clause"] == clause.verbatim_clause
        assert carried["source_artifact"] == clause.source_artifact and carried["source_range"] == [0, 21]
        assert record["brief"]["scope"] == ["src/parser.py"]
        assert record["evidence_refs"] == ["history/turn-1.md"]
        for required in ("status", "coverage", "gaps", "uncertainty", "conflicts", "error",
                         "evidence_refs", "workspace_revision"):
            assert required in record, required
        assert clause.verbatim_clause in llm.last_prompt[0]
        assert "history/turn-1.md" in llm.last_prompt[0]


@check
def null_semantic_memory_still_seals_to_canonical_local_artifact():
    from sliceagent.persistence import ArtifactStore, PendingTurnJournal

    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))
        refs = []
        llm = _LLM("local durable report")
        host = SubagentHost(
            _Tools(root), llm=llm, retriever=NullRetriever(), memory=NullMemory(), policy=None,
            max_depth=1, session_id="session-1", workspace_id="workspace-1",
            task_id_fn=lambda: "task-1", parent_id_fn=lambda: "turn-parent",
            artifact_store=store, artifact_ref_sink=refs.append,
        )
        out = host.run("spawn_agent", {"agent": "explorer", "task": "inspect parser"})
        artifacts = store.list_all()
        assert len(artifacts) == 1 and artifacts[0].kind == "subagent"
        assert refs == [artifacts[0].id]
        assert f'artifacts/{artifacts[0].id}.md' in out
        usage_effects = [effect for effect in out.effects if effect.kind == "model_usage"]
        assert len(usage_effects) == 1
        assert usage_effects[0].payload["prompt_tokens"] == 10
        assert usage_effects[0].payload["completion_tokens"] == 2
        assert PendingTurnJournal.pending(store.root) == [], \
            "a successfully sealed child must not leave an in-flight journal"


@check
def canonical_child_failure_cannot_be_masked_by_semantic_mirror():
    from sliceagent.persistence import ArtifactStore, PendingTurnJournal

    class FailingStore(ArtifactStore):
        def put(self, _artifact):
            raise OSError("core disk full")

    class Mirror(NullMemory):
        is_durable = True

        def __init__(self):
            self.calls = 0

        def append_subagent_artifact(self, _session_id, _artifact):
            self.calls += 1
            return "sub-1"

    with tempfile.TemporaryDirectory() as root:
        store, mirror = FailingStore(os.path.join(root, "core")), Mirror()
        host = SubagentHost(
            _Tools(root), llm=_LLM("report"), retriever=NullRetriever(), memory=mirror, policy=None,
            max_depth=1, session_id="session-1", workspace_id="workspace-1",
            task_id_fn=lambda: "task-1", parent_id_fn=lambda: "turn-parent",
            artifact_store=store,
        )
        out = host.run("spawn_agent", {"agent": "explorer", "task": "inspect"})
        assert out.startswith("Error: subagent result is indeterminate")
        assert mirror.calls == 0, "derived memory must not create a competing successful truth"
        assert len(PendingTurnJournal.pending(store.root)) == 1


@check
def canonical_child_archive_and_pending_header_are_redacted():
    from sliceagent.persistence import ArtifactStore, PendingTurnJournal

    secret = "sk-test-secret"
    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))
        seen_headers = []

        class InspectLLM(_LLM):
            def complete(self, messages, schemas):
                pending = PendingTurnJournal.pending(store.root)
                seen_headers.append(dict(pending[0].snapshot().header))
                return super().complete(messages, schemas)

        host = SubagentHost(
            _Tools(root), llm=InspectLLM(f"report {secret}"), retriever=NullRetriever(),
            memory=NullMemory(), policy=None, max_depth=1, session_id="session-1",
            workspace_id="workspace-1", task_id_fn=lambda: "task-1",
            parent_id_fn=lambda: "turn-parent", artifact_store=store,
        )
        out = host.run("spawn_agent", {"agent": "explorer", "task": f"inspect {secret}"})
        artifact = store.list_all()[0]
        assert secret not in str(artifact.to_dict())
        assert secret not in str(seen_headers) and secret not in str(out)


@check
def failed_parent_ref_handoff_keeps_child_unaccepted_and_recoverable():
    from sliceagent.persistence import ArtifactStore, PendingTurnJournal

    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))

        def fail_ref(_artifact_id):
            raise OSError("parent journal unavailable")

        host = SubagentHost(
            _Tools(root), llm=_LLM("report"), retriever=NullRetriever(), memory=NullMemory(),
            policy=None, max_depth=1, session_id="session-1", workspace_id="workspace-1",
            task_id_fn=lambda: "task-1", parent_id_fn=lambda: "turn-parent",
            artifact_store=store, artifact_ref_sink=fail_ref,
        )
        out = host.run("spawn_agent", {"agent": "explorer", "task": "inspect"})
        assert out.startswith("Error: subagent result is indeterminate")
        assert len(store.list_all()) == 1 and len(PendingTurnJournal.pending(store.root)) == 1


@check
def mutable_name_grant_resolves_once_to_canonical_job_handle():
    with tempfile.TemporaryDirectory() as root:
        memory = _Archive(os.path.join(root, "vault"))
        first = memory.append_subagent_artifact("s1", _legacy_artifact("first auth survey", name="auth"))
        assert first == "sub-1"
        host = SubagentHost(_Tools(root), llm=_LLM("synthesis"), retriever=NullRetriever(),
                            memory=memory, policy=None, max_depth=1, session_id="s1")
        out = host.run("spawn_agent", {"agent": "synthesiser", "task": "use auth survey",
                                       "grants": ["subagents/auth.md"]})
        assert 'read_file("subagents/sub-2.md")' in out
        synthesis = memory.read_subagent_artifacts("s1")[-1]["artifact"]
        assert synthesis["brief"]["grants"] == ["subagents/sub-1.md"]
        assert synthesis["refs"] == ["subagents/sub-1.md"]
        assert 'read_file("subagents/sub-1.md")' in host.llm.last_prompt[0]
        assert "subagents/auth.md" not in host.llm.last_prompt[0]

        # Retargeting the convenience alias later cannot mutate the sealed dependency.
        memory.append_subagent_artifact("s1", _legacy_artifact("second auth survey", name="auth"))
        assert synthesis["refs"] == ["subagents/sub-1.md"]


@check
def artifact_fingerprints_only_its_dependency_paths_and_detects_drift():
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "a.py"), "w", encoding="utf-8") as stream:
            stream.write("VALUE = 1\n")
        with open(os.path.join(root, "unrelated.py"), "w", encoding="utf-8") as stream:
            stream.write("OTHER = 1\n")
        brief = SubagentBrief.create("inspect a.py")
        artifact = SubagentArtifact.create(
            kind="explorer", name="", workspace_id="w", session_id="s", task_id="t",
            parent_id="turn-1", brief=brief, status="ok", coverage="a.py inspected",
            report="VALUE is one", files=("a.py",), workspace_root=root,
            evidence_refs=("file:a.py",), gaps=(), uncertainty=())
        record = artifact.to_record()
        assert [row["path"] for row in record["workspace_revision"]["dependencies"]] == ["a.py"]
        restored = SubagentArtifact.from_record(record)
        assert restored.workspace_revision.is_current()

        with open(os.path.join(root, "unrelated.py"), "w", encoding="utf-8") as stream:
            stream.write("OTHER = 2\n")
        assert restored.workspace_revision.is_current(), "unrelated edits must not stale the child evidence"
        with open(os.path.join(root, "a.py"), "w", encoding="utf-8") as stream:
            stream.write("VALUE = 2\n")
        drift = restored.workspace_revision.drifted()
        assert len(drift) == 1 and drift[0].path == "a.py"


@check
def unscoped_external_dependency_becomes_an_explicit_gap():
    with tempfile.TemporaryDirectory() as root:
        outside = tempfile.NamedTemporaryFile(delete=False)
        outside.close()
        try:
            artifact = SubagentArtifact.create(
                kind="explorer", name="", workspace_id="w", session_id="s", task_id="t",
                parent_id="", brief=SubagentBrief.create("inspect"), status="ok",
                coverage="attempted external file", report="", files=(outside.name,), workspace_root=root)
            assert artifact.gaps and "could not fingerprint" in artifact.gaps[0]
            assert artifact.workspace_revision.dependencies == ()
        finally:
            os.unlink(outside.name)


@check
def durable_archive_failure_is_indeterminate_and_never_accepted_inline():
    with tempfile.TemporaryDirectory() as root:
        for memory in (_DurableFailure(), _DurableRaise()):
            result = run_subagent(
                "inspect", tools=_Tools(root), llm=_LLM("authoritative-looking child claim"),
                retriever=NullRetriever(), memory=memory, policy=None, max_steps=2,
                read_only=True, session_id="s1")
            assert result.startswith("Error: subagent result is indeterminate"), result
            assert "durable report could not be sealed" in result
            assert "authoritative-looking child claim" not in result


@check
def intentional_non_durable_mode_keeps_inline_compatibility_fallback():
    with tempfile.TemporaryDirectory() as root:
        result = run_subagent(
            "inspect", tools=_Tools(root), llm=_LLM("ephemeral result"), retriever=NullRetriever(),
            memory=NullMemory(), policy=None, max_steps=2, read_only=True, session_id="")
        assert result.startswith("[explore ok") and "ephemeral result" in result
        assert "subagents/sub-" not in result and "indeterminate" not in result


@check
def configured_intent_seam_failure_blocks_child_before_model_call():
    with tempfile.TemporaryDirectory() as root:
        llm = _LLM()
        def broken_provider(objective):
            raise RuntimeError("intent store unavailable")
        host = SubagentHost(_Tools(root), llm=llm, retriever=NullRetriever(), memory=NullMemory(),
                            policy=None, max_depth=1, intent_provider=broken_provider)
        result = host.run("spawn_agent", {"agent": "explorer", "task": "inspect"})
        assert result.startswith("Error: invalid subagent brief") and "intent store unavailable" in result
        assert llm.calls[0] == 0


@check
def core_mode_rejects_legacy_and_writable_delegation_at_runtime():
    with tempfile.TemporaryDirectory() as root:
        llm = _LLM()
        host = SubagentHost(
            _Tools(root), llm=llm, retriever=NullRetriever(), memory=NullMemory(),
            policy=None, max_depth=1, core_mode=True,
        )
        legacy = host.run("spawn_subagent", {"task": "edit the project"})
        writable = host.run("spawn_agent", {"agent": "general", "task": "edit the project"})
        assert legacy.startswith("Error: core delegation")
        assert writable.startswith("Error: core delegation")
        assert llm.calls[0] == 0


@check
def child_budget_is_enforced_during_the_child_loop():
    class LoopLLM:
        reasoning = "fast"

        def __init__(self):
            self.calls = [0]

        def complete(self, _messages, _schemas):
            self.calls[0] += 1
            index = self.calls[0]
            return NS(
                content="", finish_reason="tool_calls",
                tool_calls=[NS(name="read_file", args={"path": "a.py"}, id=f"read-{index}")],
                usage={"prompt_tokens": 3, "completion_tokens": 3},
            )

    with tempfile.TemporaryDirectory() as root:
        llm = LoopLLM()
        result = run_subagent(
            "inspect", tools=_Tools(root), llm=llm, retriever=NullRetriever(),
            memory=NullMemory(), policy=None, max_steps=8, read_only=True,
            token_budget=10,
        )
        assert llm.calls[0] == 2, "the child must stop as soon as its own usage crosses the reservation"
        assert result.startswith("Error: subagent did not finish cleanly")
        usage_effect = next(effect for effect in result.effects if effect.kind == "model_usage")
        assert usage_effect.payload["prompt_tokens"] == 6
        assert usage_effect.payload["completion_tokens"] == 6


@check
def indeterminate_child_propagates_typed_uncertainty_to_the_parent():
    from sliceagent.execution import ToolStatus
    from sliceagent.registry import ToolText

    class LLM:
        reasoning = "fast"

        def __init__(self):
            self.calls = [0]

        def complete(self, _messages, _schemas):
            self.calls[0] += 1
            return NS(
                content="", finish_reason="tool_calls",
                tool_calls=[NS(name="run_command", args={"command": "slow"}, id="slow")],
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

    class Tools(_Tools):
        def run(self, _name, _args):
            return ToolText("operation may still be running", status="indeterminate")

    with tempfile.TemporaryDirectory() as root:
        result = run_subagent(
            "run safely", tools=Tools(root), llm=LLM(), retriever=NullRetriever(),
            memory=NullMemory(), policy=None, max_steps=3,
        )
        assert result.status is ToolStatus.INDETERMINATE


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
