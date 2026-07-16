"""Focused offline tests for the typed subagent brief/artifact contract.

Run: PYTHONPATH=src python tests/test_subagent_contract.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.hippocampus import HippocampusMixin  # noqa: E402
from sliceagent.events import ToolResult  # noqa: E402
from sliceagent.execution import ToolStatus  # noqa: E402
from sliceagent.intent import IntentEntry  # noqa: E402
from sliceagent.memory import NullMemory  # noqa: E402
from sliceagent.retriever import NullRetriever  # noqa: E402
from sliceagent.subagent import (SubagentHost, _ObservationSink,
                                 _canonical_artifact_for_seal,
                                 _explorer_navigation_steps,
                                 _task_mentions_exact_target,
                                 run_subagent)  # noqa: E402
from sliceagent.subagent_contract import (  # noqa: E402
    ExplorerEvidenceAccount,
    SubagentArtifact,
    SubagentBrief,
    SubagentClaim,
    SubagentObservation,
)


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _child_outcome(result):
    effects = [effect for effect in result.effects if effect.kind == "child_outcome"]
    assert len(effects) == 1, "each terminal child report must carry one child_outcome effect"
    return effects[0].payload


def _assert_child_envelope(result, status: str, report: str = ""):
    assert str(result).startswith("[child"), result
    assert f"· {status}" in str(result).splitlines()[0], result
    assert "BEGIN CHILD REPORT" in result and "END CHILD REPORT" in result
    outcome = _child_outcome(result)
    assert outcome["status"] == status
    if report:
        assert report in result, "the accepted canonical report must return directly and in full"
        assert outcome["report_bytes"] == len(report.encode("utf-8"))
        assert outcome["report_sha256"] == hashlib.sha256(report.encode("utf-8")).hexdigest()
    return outcome


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


class _GroundedTools(_Tools):
    """Minimal real workspace observation for tests whose subject is persistence, not epistemics."""

    def schemas(self):
        return [{"type": "function", "function": {
            "name": "read_file", "parameters": {"type": "object", "properties": {}},
        }}]

    def run(self, name, args):
        assert name == "read_file" and args.get("path") == "fixture.py"
        return "     1\tFIXTURE = True"


class _GroundedLLM(_LLM):
    def complete(self, messages, schemas):
        if self.calls[0] == 0:
            self.calls[0] += 1
            self.last_prompt[0] = "\n".join(
                str(message.get("content", "")) for message in messages
            )
            return NS(
                content="inspect fixture", finish_reason="tool_calls",
                tool_calls=[NS(name="read_file", args={"path": "fixture.py"}, id="fixture-read")],
                usage={"prompt_tokens": 2, "completion_tokens": 1},
            )
        return super().complete(messages, schemas)


def _legacy_artifact(report, *, name=""):
    return {"kind": "explorer", "name": name, "task": "old", "brief": {"task": "old"},
            "status": "ok", "steps": 1, "report": report, "findings": [], "change_set": [],
            "files": [], "coverage": "old coverage", "refs": []}


@check
def observation_archive_is_complete_while_the_inline_projection_stays_bounded():
    sink = _ObservationSink()
    auth_view = (
        "     1\tdef login(username, password):\n"
        "     2\t    stored = get_password(username)\n"
        "     3\t    return password == stored"
    )
    sink(ToolResult(
        "read_file", {"path": "auth.py", "offset": 1, "note": "not retained"},
        auth_view, False, status="succeeded",
    ))
    sink(ToolResult(
        "read_file", {"path": "history/turn-1.md"}, "virtual bytes", False, status="succeeded",
    ))
    sink(ToolResult(
        "read_file", {"path": "failed.py"}, "denied", True, status="failed",
    ))
    sink(ToolResult(
        "grep", {"pattern": "needle", "path": "src"}, "x" * 20_000, False, status="succeeded",
    ))

    assert len(sink.observations) == 3
    auth, failed, large = sink.observations
    assert auth.tool == "read_file" and dict(auth.args) == {"path": "auth.py", "offset": 1}
    assert auth.view == auth_view and not auth.redacted and not auth.truncated
    assert auth.raw_sha256 == hashlib.sha256(auth_view.encode()).hexdigest()
    assert auth.view_sha256 == auth.raw_sha256
    assert failed.status == "failed" and dict(failed.args) == {"path": "failed.py"}
    assert any("failed.py" in gap and "denied" in gap for gap in sink.gaps)
    assert sink.successful_observations == (auth, large)
    assert large.tool == "grep" and not large.truncated and large.view == "x" * 20_000
    assert "sealed observation view truncated" not in large.view
    inline = sink.inline_observations
    inline_large = next(item for item in inline if item.tool == "grep")
    assert inline_large.truncated and inline_large.view_bytes <= 8 * 1024
    assert "sealed observation view truncated" in inline_large.view
    assert sum(item.view_bytes for item in inline if item.status == "succeeded") <= 16 * 1024


@check
def failed_observation_burst_cannot_crowd_out_later_success_evidence():
    sink = _ObservationSink()
    for index in range(8):
        sink(ToolResult(
            "read_file", {"path": f"missing-{index}.py"}, "missing " + ("x" * 5000),
            True, status="failed",
        ))
    source = "s" * (8 * 1024)
    sink(ToolResult(
        "read_file", {"path": "src/real.py"}, source, False, status="succeeded",
    ))

    successful = sink.successful_observations
    assert len(successful) == 1 and successful[0].view == source
    assert successful[0].view_bytes == 8 * 1024 and not successful[0].truncated
    assert len([item for item in sink.observations if item.status == "failed"]) == 8
    assert len([item for item in sink.inline_observations if item.status == "failed"]) <= 4
    assert all(f"missing-{index}.py" in "\n".join(sink.gaps) for index in range(8))


@check
def scope_diverse_content_evidence_displaces_early_navigation_under_capsule_pressure():
    scope = tuple(f"src/scoped-{index}.py" for index in range(10))
    sink = _ObservationSink(scope=scope)
    for path in scope:
        sink(ToolResult(
            "glob", {"path": ".", "pattern": path},
            f"navigation-only match for {path}\n" + ("n" * 5000), False, status="succeeded",
        ))

    navigation_account = sink.evidence_account()
    assert navigation_account.status == "navigation_only"
    assert navigation_account.navigation_success_count == 10
    assert navigation_account.content_success_count == 0

    for path in scope:
        sink(ToolResult(
            "read_file", {"path": path},
            f"CONTENT FOR {path}\n" + (path * 400), False, status="succeeded",
        ))

    retained = sink.successful_observations
    content = tuple(item for item in retained if item.tool == "read_file")
    inline = sink.inline_observations
    inline_content = tuple(item for item in inline if item.tool == "read_file")
    account = sink.evidence_account()
    assert len(content) == 10 and len(retained) == 20
    assert {item.args["path"] for item in content} == set(scope)
    assert all(item.view and not item.truncated for item in content)
    assert len(inline_content) == 10 and all(item.truncated for item in inline_content)
    assert not [item for item in inline if item.tool in {"glob", "list_files"}]
    assert sum(item.view_bytes for item in inline) <= 16 * 1024
    assert account.status == "content_retained"
    assert account.scope_path_count == 10 and account.scope_paths == scope
    assert account.navigation_success_count == 10 and len(account.navigation_paths) == 10
    assert account.content_success_count == 10 and account.content_paths == scope
    assert account.retained_content_view_count == 10 and account.omitted_content_view_count == 0
    assert account.retained_navigation_view_count == 10 and account.omitted_navigation_view_count == 0
    assert account.truncated_content_view_count == 0


@check
def directory_scope_prioritizes_nested_content_when_view_count_is_contested():
    sink = _ObservationSink(scope=("src/auth/",))
    for index in range(16):
        sink(ToolResult(
            "read_file", {"path": f"other/file-{index}.py"}, f"other {index}",
            False, status="succeeded",
        ))
    sink(ToolResult(
        "read_file", {"path": "src/auth/handler.py"}, "def authenticate(): return True",
        False, status="succeeded",
    ))

    retained_paths = {item.args["path"] for item in sink.successful_content_observations}
    inline_paths = {
        item.args["path"] for item in sink.inline_observations
        if item.status == "succeeded" and item.tool == "read_file"
    }
    account = sink.evidence_account()
    assert "src/auth/handler.py" in retained_paths
    assert len(retained_paths) == 17
    assert "src/auth/handler.py" in inline_paths and len(inline_paths) == 16
    assert account.content_success_count == 17
    assert account.retained_content_view_count == 17 and account.omitted_content_view_count == 0
    assert account.status == "content_retained"


@check
def every_successful_observation_survives_old_candidate_and_byte_caps():
    sink = _ObservationSink()
    sentinels = []
    for index in range(70):
        sentinel = f"FULL-EVIDENCE-{index:02d}"
        sentinels.append(sentinel)
        body = f"{index}:start\n" + ("x" * (80_000 if index == 69 else 700)) + f"\n{sentinel}"
        sink(ToolResult(
            "read_file", {"path": f"src/file-{index}.py"}, body, False, status="succeeded",
        ))

    archived = sink.observations
    assert len(archived) == 70
    assert [item.args["path"] for item in archived] == [f"src/file-{index}.py" for index in range(70)]
    assert all(sentinel in item.view for sentinel, item in zip(sentinels, archived))
    assert not any("sealed observation view truncated" in item.view for item in archived)
    assert all(item.view_sha256 == hashlib.sha256(item.view.encode()).hexdigest() for item in archived)
    assert len(sink.inline_observations) <= 16
    assert sum(item.view_bytes for item in sink.inline_observations) <= 16 * 1024
    account = sink.evidence_account()
    assert account.status == "content_retained"
    assert account.content_success_count == account.retained_content_view_count == 70
    assert account.omitted_content_view_count == account.truncated_content_view_count == 0


@check
def source_paging_is_partial_even_when_the_complete_returned_page_is_sealed():
    sink = _ObservationSink()
    returned_page = (
        "     1\tfirst\n     2\tsecond\n"
        "<system>read_file big.py: lines 1-2 of 4 · +2 more — "
        "read_file(path, offset=3) to continue</system>"
    )
    sink(ToolResult(
        "read_file", {"path": "big.py"}, returned_page, False, status="succeeded",
    ))
    observation = sink.observations[0]
    assert observation.view == returned_page and observation.truncated
    assert observation.raw_bytes == observation.view_bytes == len(returned_page.encode())
    account = sink.evidence_account()
    assert account.status == "content_partial"
    assert account.truncated_content_view_count == 1


@check
def typed_observation_roundtrips_and_legacy_artifacts_default_to_none():
    view = "     1\treturn password == stored"
    body = view.encode("utf-8")
    observation = SubagentObservation(
        tool="read_file", args={"path": "auth.py"}, status="succeeded", view=view,
        raw_sha256=hashlib.sha256(body).hexdigest(), view_sha256=hashlib.sha256(body).hexdigest(),
        raw_bytes=len(body), view_bytes=len(body), redacted=False, truncated=False,
    )
    artifact = SubagentArtifact.create(
        kind="explorer", name="", workspace_id="w", session_id="s", task_id="t",
        parent_id="turn-1", brief=SubagentBrief.create("inspect auth.py"), status="ok",
        coverage="auth.py inspected", report="claim", observations=(observation,),
    )
    restored = SubagentArtifact.from_record(artifact.to_record())
    assert restored.observations == (observation,)
    assert restored.to_record()["observations"] == [observation.to_dict()]
    assert SubagentArtifact.from_record(_legacy_artifact("legacy")).observations == ()


@check
def explorer_evidence_account_roundtrips_and_legacy_artifacts_remain_unassessed():
    account = ExplorerEvidenceAccount(
        status="content_retained", scope_path_count=1, content_success_count=1,
        retained_content_view_count=1, scope_paths=("auth.py",), content_paths=("auth.py",),
    )
    artifact = SubagentArtifact.create(
        kind="explorer", name="", workspace_id="w", session_id="s", task_id="t",
        parent_id="turn-1", brief=SubagentBrief.create("inspect", scope=("auth.py",)), status="ok",
        coverage="typed host account", report="claim", explorer_evidence=account,
    )
    restored = SubagentArtifact.from_record(artifact.to_record())
    assert restored.explorer_evidence == account
    assert restored.to_record()["explorer_evidence"] == account.to_dict()
    assert SubagentArtifact.from_record(_legacy_artifact("legacy")).explorer_evidence.status == "not_assessed"


@check
def typed_claims_are_exact_bounded_and_closed_over_their_artifact():
    view = "     1\treturn query"
    body = view.encode("utf-8")
    observation = SubagentObservation(
        tool="read_file", args={"path": "app.py"}, status="succeeded", view=view,
        raw_sha256=hashlib.sha256(body).hexdigest(), view_sha256=hashlib.sha256(body).hexdigest(),
        raw_bytes=len(body), view_bytes=len(body), redacted=False, truncated=False,
    )
    report = "Analysis\nTOP CLAIM: Query construction is risky if a downstream caller executes it."
    claim = SubagentClaim(
        text="TOP CLAIM: Query construction is risky if a downstream caller executes it.",
        report_exact="TOP CLAIM: Query construction is risky if a downstream caller executes it.",
        modality="conditional", observation_refs=(observation.view_sha256,),
    )
    artifact = SubagentArtifact.create(
        kind="explorer", name="", workspace_id="w", session_id="s", task_id="t",
        parent_id="turn-1", brief=SubagentBrief.create("inspect app.py"), status="ok",
        coverage="app.py inspected", report=report, observations=(observation,), claims=(claim,),
    )
    restored = SubagentArtifact.from_record(artifact.to_record())
    assert restored.claims == (claim,) and restored.claims[0].to_dict()["v"] == 1
    assert SubagentArtifact.from_record(_legacy_artifact("legacy")).claims == ()
    malformed = artifact.to_record()
    malformed["claims"] = "not-an-array"
    salvaged = SubagentArtifact.from_record(malformed)
    assert salvaged.report == report and salvaged.observations == (observation,)
    assert not salvaged.claims and any("malformed legacy claims" in row for row in salvaged.projection_gaps)

    for bad in (
        SubagentClaim(text="x", report_exact="not in report"),
        SubagentClaim(text="x", report_exact="Analysis", observation_refs=("0" * 64,)),
    ):
        salvaged = SubagentArtifact.create(
            kind="explorer", name="", workspace_id="w", session_id="s", task_id="t",
            parent_id="turn-1", brief=SubagentBrief.create("inspect app.py"), status="ok",
            coverage="app.py inspected", report=report, observations=(observation,), claims=(bad,),
        )
        assert not salvaged.claims and salvaged.projection_gaps


@check
def canonical_seal_preserves_evidence_after_non_idempotent_redaction_without_guessing_claim_links():
    from sliceagent.safety import redact_text

    raw_view = 'CHILD_TOKEN_BUDGET_ARG = "__sliceagent_token_budget"'
    child_view = redact_text(raw_view)
    stored_view = redact_text(child_view)
    assert child_view != stored_view, "fixture must exercise the live double-redaction hash drift"
    body = child_view.encode("utf-8")
    observation = SubagentObservation(
        tool="read_file", args={"path": "src/sliceagent/execution.py"}, status="succeeded",
        view=child_view, raw_sha256=hashlib.sha256(raw_view.encode("utf-8")).hexdigest(),
        view_sha256=hashlib.sha256(body).hexdigest(), raw_bytes=len(raw_view.encode("utf-8")),
        view_bytes=len(body), redacted=True, truncated=False,
    )
    report = "TOP CLAIM: execution.py contains a scheduler token-budget control."
    stale_claim = SubagentClaim(
        text=report, report_exact=report, modality="inference",
        observation_refs=(observation.view_sha256,),
    )
    artifact = SubagentArtifact.create(
        kind="explorer", name="", workspace_id="w", session_id="s", task_id="t",
        parent_id="turn-1",
        brief=SubagentBrief.create("inspect execution.py", scope=("src/sliceagent/execution.py",)),
        status="ok", coverage="execution.py inspected", report=report,
        observations=(observation,), claims=(stale_claim,),
    )

    sealed = _canonical_artifact_for_seal(artifact)
    assert sealed.observations[0].view == child_view
    assert redact_text(sealed.observations[0].view) == stored_view, \
        "a second generic pass would change the exact child-visible evidence"
    assert sealed.claims == (), "the host must not bind a report line to an arbitrary primary read"


@check
def live_child_with_non_idempotent_source_redaction_still_seals_successfully():
    from sliceagent.persistence import ArtifactStore

    raw_view = 'CHILD_TOKEN_BUDGET_ARG = "__sliceagent_token_budget"'

    class TokenTools(_GroundedTools):
        def run(self, name, args):
            assert name == "read_file" and args.get("path") == "fixture.py"
            return raw_view

    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))
        result = run_subagent(
            "inspect fixture.py", tools=TokenTools(root),
            llm=_GroundedLLM("TOP CLAIM: fixture.py defines the child token-budget argument."),
            retriever=NullRetriever(), memory=NullMemory(), max_steps=3, read_only=True,
            workspace_id="workspace-1", session_id="session-1", task_id="task-1",
            parent_id="turn-1", artifact_store=store, artifact_id="subagent-redaction-seal",
        )
        artifact = store.get("subagent-redaction-seal")
        observation = artifact.structured_body["observations"][0]
        assert result.status is ToolStatus.SUCCEEDED, result
        assert observation["view"] == 'CHILD_TOKEN_BUDGET_ARG="__slic...dget"'
        assert observation["view_sha256"] == hashlib.sha256(
            observation["view"].encode("utf-8")
        ).hexdigest()
        assert not artifact.structured_body["claims"]
        effect = next(item for item in result.effects if item.kind == "child_artifact")
        assert effect.payload["claims"] == []


@check
def deterministic_optional_projection_failure_salvages_the_report_envelope():
    import sliceagent.subagent as subagent_module
    from sliceagent.persistence import ArtifactStore

    original = subagent_module._canonical_artifact_for_seal

    def fail_projection(_artifact):
        raise ValueError("injected optional projection failure")

    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))
        subagent_module._canonical_artifact_for_seal = fail_projection
        try:
            result = run_subagent(
                "inspect fixture.py", tools=_GroundedTools(root),
                llm=_GroundedLLM("TOP CLAIM: fixture.py was inspected."),
                retriever=NullRetriever(), memory=NullMemory(), max_steps=3, read_only=True,
                workspace_id="workspace-1", session_id="session-1", task_id="task-1",
                parent_id="turn-1", artifact_store=store, artifact_id="subagent-envelope-salvage",
            )
        finally:
            subagent_module._canonical_artifact_for_seal = original
        assert result.status is ToolStatus.SUCCEEDED, result
        artifact = store.get("subagent-envelope-salvage")
        body = artifact.structured_body
        report = body["report"]
        assert report == "TOP CLAIM: fixture.py was inspected."
        assert body["report_bytes"] == len(report.encode())
        assert body["report_sha256"] == hashlib.sha256(report.encode()).hexdigest()
        assert body["report_completion"] == "complete" and body["report_stop_reason"] == "end_turn"
        assert body["observations"] and not body["claims"]
        assert any("injected optional projection failure" in row for row in body["projection_gaps"])
        assert "Projection warning:" in result


@check
def observation_contract_rejects_unknown_status_and_claim_refs_to_failed_rows():
    body = b"missing"
    digest = hashlib.sha256(body).hexdigest()
    try:
        SubagentObservation(
            tool="read_file", args={"path": "missing.py"}, status="unknown", view="missing",
            raw_sha256=digest, view_sha256=digest, raw_bytes=7, view_bytes=7,
        )
        assert False, "observation status vocabulary must be closed"
    except ValueError as error:
        assert "status" in str(error)

    failed = SubagentObservation(
        tool="read_file", args={"path": "missing.py"}, status="failed", view="missing",
        raw_sha256=digest, view_sha256=digest, raw_bytes=7, view_bytes=7,
    )
    claim = SubagentClaim(
        text="TOP CLAIM: missing.py is insecure.",
        report_exact="TOP CLAIM: missing.py is insecure.",
        modality="inference", observation_refs=(digest,),
    )
    salvaged = SubagentArtifact.create(
        kind="explorer", name="", workspace_id="w", session_id="s", task_id="t",
        parent_id="turn-1", brief=SubagentBrief.create("inspect missing.py"), status="ok",
        coverage="attempted", report=claim.report_exact,
        observations=(failed,), claims=(claim,),
    )
    assert not salvaged.claims
    assert any("observation_refs" in row for row in salvaged.projection_gaps)


@check
def target_binding_accepts_sentence_punctuation_without_basename_collisions():
    assert _task_mentions_exact_target("Review app.py.", "app.py")
    assert _task_mentions_exact_target("Review /workspace/app.py, then summarize it.", "app.py")
    assert _task_mentions_exact_target("Review app.py (one file).", "app.py")

    assert not _task_mentions_exact_target("Review data.py.", "a.py")
    assert not _task_mentions_exact_target("Review /workspace/data.py.", "a.py")
    assert not _task_mentions_exact_target("Review a.py.bak.", "a.py")
    assert not _task_mentions_exact_target("Review a.py/child.", "a.py")


@check
def successful_child_read_is_sealed_into_the_canonical_artifact():
    from sliceagent.persistence import ArtifactStore
    from sliceagent.intent import IntentState, analyze_turn
    from sliceagent.runtime_persistence import CoreArtifactFS

    auth_view = (
        "     1\tdef login(username, password):\n"
        "     2\t    stored = get_password(username)\n"
        "     3\t    return password == stored"
    )

    class ReadThenReportLLM:
        reasoning = "fast"

        def __init__(self):
            self.calls = 0

        def complete(self, _messages, _schemas):
            self.calls += 1
            if self.calls == 1:
                return NS(
                    content="", finish_reason="tool_calls",
                    tool_calls=[NS(name="read_file", args={"path": "auth.py"}, id="read-auth")],
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            return _Response(
                "TOP CLAIM: password equality may be risky if the unseen storage and caller assumptions hold."
            )

    class AuthTools(_Tools):
        def run(self, name, args):
            return auth_view if name == "read_file" and args.get("path") == "auth.py" else ""

    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))
        request = (
            "spawn exactly 3 parallel explorer subagents — one each for app.py, auth.py and util.py — "
            "then give me a combined 3-line summary"
        )
        intent = IntentState()
        intent.begin_turn(request, source_artifact="turn-parent", contract=analyze_turn(request))
        host = SubagentHost(
            AuthTools(root), llm=ReadThenReportLLM(), retriever=NullRetriever(),
            memory=NullMemory(), max_depth=1, session_id="session-1",
            workspace_id="workspace-1", task_id_fn=lambda: "task-1",
            parent_id_fn=lambda: "turn-parent", artifact_store=store,
            intent_provider=lambda _task: intent,
        )
        result = host.run("spawn_agent", {"agent": "explorer", "task": "inspect auth.py"})
        report = "TOP CLAIM: password equality may be risky if the unseen storage and caller assumptions hold."
        outcome = _assert_child_envelope(result, "succeeded", report)
        assert result.status is ToolStatus.SUCCEEDED
        assert outcome["report_completion"] == "complete"
        assert outcome["explorer_evidence_status"] == "content_retained"
        assert "primary observation" not in result, \
            "evidence bytes stay behind their locator instead of duplicating the child report"
        artifact = store.list_all()[0]
        observations = artifact.structured_body["observations"]
        assert len(observations) == 1
        assert observations[0]["tool"] == "read_file"
        assert observations[0]["args"] == {"path": "auth.py"}
        assert observations[0]["view"] == auth_view
        assert not observations[0]["redacted"] and not observations[0]["truncated"]
        assert observations[0]["view_sha256"] == hashlib.sha256(
            observations[0]["view"].encode("utf-8")
        ).hexdigest()
        claims = artifact.structured_body["claims"]
        assert not claims, "the parent, not the host, owns report synthesis and evidence verification"
        child_effect = next(effect for effect in result.effects if effect.kind == "child_artifact")
        assert child_effect.payload["scope"] == ["auth.py"]
        assert child_effect.payload["delegation_target"] == "auth.py"
        assert "integration_policy" not in child_effect.payload
        assert child_effect.payload["explorer_evidence_status"] == "content_retained"
        assert child_effect.payload["explorer_evidence"]["content_success_count"] == 1
        assert child_effect.payload["claims"] == []
        assert outcome["artifact_id"] == artifact.id
        assert outcome["report_handle"] == f"artifacts/{artifact.id}.md"
        assert outcome["evidence_index_handle"] == f"artifacts/{artifact.id}/evidence/index.md"
        virtual = CoreArtifactFS(store)
        report_page = virtual.read_file(f"artifacts/{artifact.id}.md")
        assert "TOP CLAIM:" in report_page
        assert "## Page-backed workspace evidence" in report_page
        assert "Legacy/explicit child claims" not in report_page
        assert auth_view not in report_page, "full tool output belongs on evidence pages, not the report page"
        evidence_index = virtual.read_file(f"artifacts/{artifact.id}/evidence/index.md")
        assert "Observation 1 · read_file · succeeded" in evidence_index
        assert f"artifacts/{artifact.id}/evidence/obs-001-page-001.md" in evidence_index
        evidence_page = virtual.read_file(
            f"artifacts/{artifact.id}/evidence/obs-001-page-001.md"
        )
        assert auth_view in evidence_page


@check
def brief_preserves_exact_clauses_sources_and_only_explicit_context():
    clause = IntentEntry(id="intent-7", verbatim_clause="Preserve APIName exactly — including case.",
                         source_artifact="history/turn-2.md", source_range=(14, 54), authority="user")
    brief = SubagentBrief.create(
        "Audit only the parser.", intent_entries=[clause], scope=["src/parser.py"],
        exclusions=["Do not edit files"], report_shape="Return findings with line evidence.",
        canonical_refs=["subagents/sub-3.md"], drift_policy="fail",
        integration_policy="report_required")
    rendered = brief.render()
    assert clause.verbatim_clause in rendered and "history/turn-2.md" in rendered and "14:54" in rendered
    assert "Audit only the parser." in rendered and "Do not edit files" in rendered
    assert 'read_file("subagents/sub-3.md")' in rendered
    assert "parent assistant transcript that was never selected" not in rendered
    assert "EVIDENCE STANDARD (binding)" in rendered
    assert "CONDITIONAL CONSEQUENCE" in rendered
    assert "Constructing a command/query is not executing it" in rendered
    assert "PARENT INTEGRATION POLICY (legacy): report_required" in rendered
    assert SubagentBrief.from_dict(brief.to_dict()) == brief


@check
def spawn_schema_retires_integration_policy_but_reads_legacy_briefs():
    host = SubagentHost(
        _Tools("."), llm=_LLM(), retriever=NullRetriever(), memory=NullMemory(), max_depth=1,
    )
    schema = next(row for row in host.schemas() if row["function"]["name"] == "spawn_agent")
    parameters = schema["function"]["parameters"]
    assert "integration_policy" not in parameters["properties"]
    assert SubagentBrief.from_dict({"task": "legacy"}).integration_policy == "digest_ok"


@check
def canonical_local_artifact_handles_can_be_granted_without_legacy_aliases():
    from sliceagent.persistence import Artifact, ArtifactStore

    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))
        artifact = Artifact(
            id="subagent-canonical-123", kind="subagent", workspace_id="workspace-1",
            session_id="session-1", task_id="task-1", status="ok",
            summary="sealed child report", structured_body={"report": "sealed child report"},
        )
        store.put(artifact)
        host = SubagentHost(
            _Tools(root), llm=_LLM(), retriever=NullRetriever(), memory=NullMemory(),
            max_depth=1, artifact_store=store,
        )
        handle = "artifacts/subagent-canonical-123.md"
        error, grants = host._validate_grants([handle])
        assert error == "" and grants == frozenset({handle})
        brief = SubagentBrief.create("merge reports", canonical_refs=tuple(sorted(grants)))
        assert SubagentBrief.from_dict(brief.to_dict()).canonical_refs == (handle,)

        wrong_kind = Artifact(
            id="turn-not-child", kind="turn", workspace_id="workspace-1",
            session_id="session-1", task_id="task-1",
        )
        store.put(wrong_kind)
        error, grants = host._validate_grants(["artifacts/turn-not-child.md"])
        assert error.startswith("Error: cannot grant") and not grants


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
        tools = _GroundedTools(root)
        llm = _GroundedLLM("found parser issue")
        clause = IntentEntry(id="intent-1", verbatim_clause="Keep parse_v1 public.",
                             source_artifact="history/turn-1.md", source_range=(0, 21), authority="user")
        host = SubagentHost(
            tools, llm=llm, retriever=NullRetriever(), memory=memory,
            max_depth=1, session_id="session-9", workspace_id="workspace-9",
            intent_provider=lambda objective: [clause], task_id_fn=lambda: "task-4",
            parent_id_fn=lambda: "turn-parent-8")
        out = host.run("spawn_agent", {
            "agent": "explorer", "task": "audit parser", "scope": ["src/parser.py"],
            "exclusions": ["no edits"], "report_shape": "status + evidenced findings",
            "drift_policy": "report",
        })
        outcome = _assert_child_envelope(out, "succeeded", "found parser issue")
        assert "Archive: subagents/sub-1.md" in out
        assert outcome["report_handle"] == "subagents/sub-1.md"
        record = memory.read_subagent_artifacts("session-9")[-1]["artifact"]
        assert record["contract_v"] == 1
        assert record["workspace_id"] == "workspace-9" and record["session_id"] == "session-9"
        assert record["task_id"] == "task-4" and record["parent_id"] == "turn-parent-8"
        assert record["brief"]["objective"] == "audit parser"
        carried = record["brief"]["intent_clauses"][0]
        assert carried["id"] == clause.id and carried["verbatim_clause"] == clause.verbatim_clause
        assert carried["source_artifact"] == clause.source_artifact and carried["source_range"] == [0, 21]
        assert record["brief"]["scope"] == ["src/parser.py"]
        assert "integration_policy" not in record["brief"]
        assert record["evidence_refs"] == ["history/turn-1.md"]
        for required in ("status", "coverage", "gaps", "uncertainty", "conflicts", "error",
                         "evidence_refs", "observations", "workspace_revision"):
            assert required in record, required
        assert clause.verbatim_clause in llm.last_prompt[0]
        assert "history/turn-1.md" in llm.last_prompt[0]


@check
def parent_delegation_mechanism_does_not_replicate_into_each_child_brief():
    from sliceagent.intent import IntentState, analyze_turn

    request = (
        "spawn exactly 3 parallel explorer subagents — one each for app.py, auth.py and util.py — "
        "then give me a combined 3-line summary"
    )
    intent = IntentState()
    intent.begin_turn(request, source_artifact="turn-parent", contract=analyze_turn(request))
    intent.add_exact("Keep public APIs stable.", source_artifact="turn-standing", authority="user")
    host = SubagentHost(
        _Tools("."), llm=_LLM(), retriever=NullRetriever(), memory=NullMemory(),
        max_depth=1, intent_provider=lambda _task: intent,
    )
    brief = host._brief("Review only app.py and report its top bug.", {}, frozenset())
    rendered = brief.render()
    assert request not in rendered, "parent fan-out and reduce mechanics must stay at the parent boundary"
    assert "Keep public APIs stable." in rendered, "independent standing constraints still bind the child"
    assert brief.scope == ("app.py",)
    assert brief.delegation_target == "app.py"
    assert "PRIMARY DELEGATION TARGET (host-bound)\napp.py" in rendered
    assert SubagentBrief.from_dict(brief.to_dict()) == brief

    collision = host._brief("Review data.py; do not confuse it with a.py.", {}, frozenset())
    assert collision.delegation_target == "" and collision.scope == (), \
        "ambiguous/multiple exact targets must not be host-bound"


@check
def null_semantic_memory_still_seals_to_canonical_local_artifact():
    from sliceagent.persistence import ArtifactStore, PendingTurnJournal

    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))
        refs = []
        llm = _LLM("local durable report")
        host = SubagentHost(
            _Tools(root), llm=llm, retriever=NullRetriever(), memory=NullMemory(),
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
        assert llm.calls[0] == 1, "evidence-free navigation must not mint a synthesis call"
        assert usage_effects[0].payload["prompt_tokens"] == 10
        assert usage_effects[0].payload["completion_tokens"] == 2
        child_effects = [effect for effect in out.effects if effect.kind == "child_artifact"]
        assert len(child_effects) == 1
        assert child_effects[0].payload["artifact_id"] == artifacts[0].id
        assert child_effects[0].payload["kind"] == "explorer"
        assert PendingTurnJournal.pending(store.root) == [], \
            "a successfully sealed child must not leave an in-flight journal"


@check
def canonical_artifact_store_failure_returns_the_full_grounded_report_fail_soft():
    from sliceagent.persistence import ArtifactStore, PendingTurnJournal

    class FailingStore(ArtifactStore):
        def put(self, _artifact):
            raise OSError("core disk full")

    with tempfile.TemporaryDirectory() as root:
        store = FailingStore(os.path.join(root, "core"))
        report = "LONG-REPORT-BEGIN\n" + ("full canonical evidence-backed analysis\n" * 40) \
            + "LONG-REPORT-END"
        assert len(report.encode()) > 800
        host = SubagentHost(
            _GroundedTools(root), llm=_GroundedLLM(report),
            retriever=NullRetriever(), memory=NullMemory(),
            max_depth=1, session_id="session-1", workspace_id="workspace-1",
            task_id_fn=lambda: "task-1", parent_id_fn=lambda: "turn-parent",
            artifact_store=store,
        )
        out = host.run("spawn_agent", {"agent": "explorer", "task": "inspect fixture.py"})
        outcome = _assert_child_envelope(out, "succeeded", report)
        assert out.status is ToolStatus.SUCCEEDED
        assert "Persistence warning: canonical artifact was not stored (OSError: core disk full)" in out
        assert outcome["artifact_id"] == "" and outcome["report_handle"] == ""
        assert outcome["persistence_warnings"] == [
            "canonical artifact was not stored (OSError: core disk full)"
        ]
        assert not [effect for effect in out.effects if effect.kind == "child_artifact"], \
            "optional persistence failure cannot mint an artifact effect"
        assert len(PendingTurnJournal.pending(store.root)) == 1


@check
def canonical_child_archive_and_pending_header_are_redacted():
    from sliceagent.persistence import ArtifactStore, PendingTurnJournal

    secret = "sk-test-secret"
    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))
        seen_headers = []

        class InspectLLM(_GroundedLLM):
            def complete(self, messages, schemas):
                pending = PendingTurnJournal.pending(store.root)
                seen_headers.append(dict(pending[0].snapshot().header))
                return super().complete(messages, schemas)

        host = SubagentHost(
            _GroundedTools(root), llm=InspectLLM(f"TOP CLAIM: report {secret}"), retriever=NullRetriever(),
            memory=NullMemory(), max_depth=1, session_id="session-1",
            workspace_id="workspace-1", task_id_fn=lambda: "task-1",
            parent_id_fn=lambda: "turn-parent", artifact_store=store,
        )
        out = host.run("spawn_agent", {"agent": "explorer", "task": f"inspect {secret}"})
        artifact = store.list_all()[0]
        assert secret not in str(artifact.to_dict())
        assert secret not in str(seen_headers) and secret not in str(out)
        assert not artifact.structured_body["claims"]
        effect = next(item for item in out.effects if item.kind == "child_artifact")
        assert effect.payload["claims"] == []
        assert secret not in str(effect.payload)


@check
def failed_parent_ref_handoff_is_fail_soft_and_never_claims_a_committed_locator():
    from sliceagent.persistence import ArtifactStore, PendingTurnJournal

    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))

        def fail_ref(_artifact_id):
            raise OSError("parent journal unavailable")

        host = SubagentHost(
            _GroundedTools(root), llm=_GroundedLLM("reference-safe report"),
            retriever=NullRetriever(), memory=NullMemory(),
            max_depth=1, session_id="session-1", workspace_id="workspace-1",
            task_id_fn=lambda: "task-1", parent_id_fn=lambda: "turn-parent",
            artifact_store=store, artifact_ref_sink=fail_ref,
        )
        out = host.run("spawn_agent", {"agent": "explorer", "task": "inspect fixture.py"})
        outcome = _assert_child_envelope(out, "succeeded", "reference-safe report")
        assert out.status is ToolStatus.SUCCEEDED
        assert "Persistence warning: canonical artifact was not stored " \
               "(OSError: parent journal unavailable)" in out
        assert outcome["artifact_id"] == "" and outcome["report_handle"] == ""
        assert not [effect for effect in out.effects if effect.kind == "child_artifact"]
        assert len(store.list_all()) == 1 and len(PendingTurnJournal.pending(store.root)) == 1


@check
def launch_ref_binding_failure_runs_and_returns_full_report_without_artifact():
    from sliceagent.persistence import ArtifactStore

    class SinkOwner:
        def record(self, _artifact_id):
            return None

        def bind_artifact_ref_sink(self, **_kwargs):
            raise OSError("launch journal unavailable")

    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(os.path.join(root, "core"))
        owner = SinkOwner()
        full_report = "HEADLESS DELIVERY\n" + ("verified detail\n" * 100)
        host = SubagentHost(
            _GroundedTools(root), llm=_GroundedLLM(full_report), retriever=NullRetriever(),
            memory=NullMemory(), max_depth=1, session_id="session-1",
            workspace_id="workspace-1", task_id_fn=lambda: "task-1",
            parent_id_fn=lambda: "turn-parent", artifact_store=store,
            artifact_ref_sink=owner.record,
        )

        out = host.run("spawn_agent", {"agent": "explorer", "task": "inspect fixture.py"})
        outcome = _assert_child_envelope(out, "succeeded", full_report)
        assert out.status is ToolStatus.SUCCEEDED
        assert full_report in out
        assert "Persistence warning: canonical artifact was not stored " \
               "(launch-turn artifact reference allocation failed: OSError: " \
               "launch journal unavailable)" in out
        assert outcome["artifact_id"] == "" and outcome["report_handle"] == ""
        assert not any(effect.kind == "child_artifact" for effect in out.effects)
        assert not store.list_all()


@check
def mutable_name_grant_resolves_once_to_canonical_job_handle():
    with tempfile.TemporaryDirectory() as root:
        memory = _Archive(os.path.join(root, "vault"))
        first = memory.append_subagent_artifact("s1", _legacy_artifact("first auth survey", name="auth"))
        assert first == "sub-1"
        host = SubagentHost(_Tools(root), llm=_LLM("synthesis"), retriever=NullRetriever(),
                            memory=memory, max_depth=1, session_id="s1")
        out = host.run("spawn_agent", {"agent": "synthesiser", "task": "use auth survey",
                                       "grants": ["subagents/auth.md"]})
        outcome = _assert_child_envelope(out, "succeeded", "synthesis")
        assert "Archive: subagents/sub-2.md" in out
        assert outcome["report_handle"] == "subagents/sub-2.md"
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
def durable_mirror_failure_is_fail_soft_and_keeps_the_full_report_inline():
    with tempfile.TemporaryDirectory() as root:
        for memory in (_DurableFailure(), _DurableRaise()):
            report = f"authoritative grounded child report from {type(memory).__name__}"
            result = run_subagent(
                "inspect fixture.py", tools=_GroundedTools(root), llm=_GroundedLLM(report),
                retriever=NullRetriever(), memory=memory, max_steps=3,
                read_only=True, session_id="s1")
            outcome = _assert_child_envelope(result, "succeeded", report)
            assert result.status is ToolStatus.SUCCEEDED
            assert outcome["artifact_id"] == ""
            assert outcome["persistence_warnings"]
            assert "Persistence warning: memory mirror was not stored" in result
            assert not [effect for effect in result.effects if effect.kind == "child_artifact"]


@check
def intentional_non_durable_mode_keeps_inline_compatibility_fallback():
    with tempfile.TemporaryDirectory() as root:
        result = run_subagent(
            "inspect", tools=_GroundedTools(root), llm=_GroundedLLM("ephemeral result"), retriever=NullRetriever(),
            memory=NullMemory(), max_steps=2, read_only=True, session_id="")
        outcome = _assert_child_envelope(result, "succeeded", "ephemeral result")
        assert result.status is ToolStatus.SUCCEEDED
        assert outcome["artifact_id"] == "" and outcome["persistence_warnings"] == []
        assert "subagents/sub-" not in result and "indeterminate" not in result


@check
def nonpersistent_nested_progress_ids_remain_hierarchical_and_unique():
    import sliceagent.subagent as subagent_module
    from sliceagent.registry import ToolText

    captured = []
    original = subagent_module.run_subagent

    def fake_run(_task, *, tools, parent_id="", artifact_id="", presentation_turn_id="", **_kwargs):
        captured.append((tools, parent_id, artifact_id, presentation_turn_id))
        return ToolText("ephemeral report", ok=True)

    with tempfile.TemporaryDirectory() as root:
        try:
            subagent_module.run_subagent = fake_run
            parent = SubagentHost(
                _Tools(root), llm=_LLM(), retriever=NullRetriever(), memory=NullMemory(),
                max_depth=2, task_id_fn=lambda: "task", parent_id_fn=lambda: "turn-root",
            )
            assert not parent.run("spawn_agent", {"agent": "general", "task": "top"}).startswith("Error:")
            child_host = captured[0][0]
            assert child_host.parent_id_fn() == "turn-root:agent:1"
            assert not child_host.run(
                "spawn_agent", {"agent": "explorer", "task": "nested"},
            ).startswith("Error:")
            nested_host = captured[1][0]
            assert captured[1][1] == "turn-root:agent:1"
            assert nested_host.parent_id_fn() == "turn-root:agent:1:agent:1"
            assert captured[1][3] == "turn-root", "nested UI ownership stays on the physical root turn"
        finally:
            subagent_module.run_subagent = original


@check
def configured_intent_seam_failure_blocks_child_before_model_call():
    with tempfile.TemporaryDirectory() as root:
        llm = _LLM()
        def broken_provider(objective):
            raise RuntimeError("intent store unavailable")
        host = SubagentHost(_Tools(root), llm=llm, retriever=NullRetriever(), memory=NullMemory(),
                            max_depth=1, intent_provider=broken_provider)
        result = host.run("spawn_agent", {"agent": "explorer", "task": "inspect"})
        assert result.startswith("Error: invalid subagent brief") and "intent store unavailable" in result
        assert llm.calls[0] == 0


@check
def core_mode_rejects_legacy_and_writable_delegation_at_runtime():
    with tempfile.TemporaryDirectory() as root:
        llm = _LLM()
        host = SubagentHost(
            _Tools(root), llm=llm, retriever=NullRetriever(), memory=NullMemory(),
            max_depth=1, core_mode=True,
        )
        legacy = host.run("spawn_subagent", {"task": "edit the project"})
        writable = host.run("spawn_agent", {"agent": "general", "task": "edit the project"})
        assert legacy.status is ToolStatus.STEERED and "core delegation" in legacy
        assert writable.status is ToolStatus.STEERED and "core delegation" in writable
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
            memory=NullMemory(), max_steps=8, read_only=True,
            token_budget=10,
        )
        assert llm.calls[0] == 2, "the child must stop as soon as its own usage crosses the reservation"
        outcome = _assert_child_envelope(result, "failed")
        assert result.status is ToolStatus.FAILED
        assert outcome["report_completion"] == "absent"
        assert outcome["stop_reason"] == "token_budget"
        usage_effect = next(effect for effect in result.effects if effect.kind == "model_usage")
        assert usage_effect.payload["prompt_tokens"] == 6
        assert usage_effect.payload["completion_tokens"] == 6


@check
def staged_navigation_ceiling_is_env_backed_and_reserves_synthesis():
    original = os.environ.get("AGENT_EXPLORER_NAV_STEPS")
    try:
        os.environ.pop("AGENT_EXPLORER_NAV_STEPS", None)
        assert _explorer_navigation_steps(20) == 6
        assert _explorer_navigation_steps(4) == 3
        for raw, expected in (("0", 1), ("-9", 1), ("999", 7), ("invalid", 6)):
            os.environ["AGENT_EXPLORER_NAV_STEPS"] = raw
            assert _explorer_navigation_steps(8) == expected, (raw, expected)
    finally:
        if original is None:
            os.environ.pop("AGENT_EXPLORER_NAV_STEPS", None)
        else:
            os.environ["AGENT_EXPLORER_NAV_STEPS"] = original


@check
def evidence_at_planned_navigation_ceiling_enters_exactly_one_full_synthesis():
    import sliceagent.subagent as subagent_module
    from sliceagent.persistence import ArtifactStore

    class ReadTools(_Tools):
        def schemas(self):
            return [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object", "properties": {}},
            }}]

        def run(self, name, args):
            assert name == "read_file"
            return f"{args['path']}: observed implementation line"

    class BudgetBoundLLM:
        reasoning = "full"

        def __init__(self):
            self.shared = {"calls": 0, "profiles": [], "schemas": [], "synthesis": ""}

        def complete(self, messages, schemas):
            shared = self.shared
            shared["calls"] += 1
            shared["profiles"].append(self.reasoning)
            shared["schemas"].append(tuple(
                row.get("function", {}).get("name", "") for row in schemas
            ))
            call = shared["calls"]
            if call <= 2:
                return NS(
                    content=f"navigation step {call}", finish_reason="tool_calls",
                    tool_calls=[NS(
                        name="read_file", args={"path": f"src/area_{call}.py"}, id=f"read-{call}",
                    )],
                    usage={"prompt_tokens": 3, "completion_tokens": 2},
                )
            assert call == 3, "no generic fast closeout or wrapper recovery call may exist"
            assert schemas == [], "the reserved synthesis is tool-free"
            shared["synthesis"] = "\n".join(str(row.get("content", "")) for row in messages)
            return NS(
                content="FINAL grounded report\nCoverage gaps: remaining callers were not inspected.",
                finish_reason="stop", tool_calls=[],
                usage={"prompt_tokens": 5, "completion_tokens": 4},
            )

    original_profile = subagent_module.EXPLORER_REASONING
    original_steps = os.environ.get("AGENT_EXPLORER_NAV_STEPS")
    subagent_module.EXPLORER_REASONING = "staged"
    os.environ["AGENT_EXPLORER_NAV_STEPS"] = "2"
    try:
        with tempfile.TemporaryDirectory() as root:
            llm = BudgetBoundLLM()
            store = ArtifactStore(os.path.join(root, "core"))
            result = run_subagent(
                "inspect two areas", tools=ReadTools(root), llm=llm,
                retriever=NullRetriever(), memory=NullMemory(), max_steps=20, read_only=True,
                workspace_id="workspace-1", session_id="session-1", task_id="task-1",
                parent_id="turn-1", artifact_store=store,
                artifact_id="subagent-budget-bound-synthesis",
            )
            artifact = store.get("subagent-budget-bound-synthesis")
            child = next(effect for effect in result.effects if effect.kind == "child_artifact").payload

            assert result.status is ToolStatus.SUCCEEDED, result
            assert llm.shared["calls"] == 3
            assert llm.shared["profiles"] == ["fast", "fast", "full"]
            assert llm.shared["schemas"][0] == llm.shared["schemas"][1] == ("read_file",)
            assert llm.shared["schemas"][2] == ()
            assert "planned fast-navigation budget ended after 2 model step(s)" in llm.shared["synthesis"]
            assert "NOT evidence of complete coverage" in llm.shared["synthesis"]
            assert "explicitly report files, paths, callers" in llm.shared["synthesis"]
            assert "observed implementation line" in llm.shared["synthesis"]
            assert artifact.status == "ok" and artifact.error == ""
            assert artifact.structured_body["steps"] == 3
            assert len(artifact.structured_body["observations"]) == 2
            assert any("planned fast navigation reached its 2-step ceiling" in note
                       for note in artifact.uncertainty)
            assert child["status"] == "ok" and child["stop_reason"] == "end_turn"
            assert child["partial"] is False
    finally:
        subagent_module.EXPLORER_REASONING = original_profile
        if original_steps is None:
            os.environ.pop("AGENT_EXPLORER_NAV_STEPS", None)
        else:
            os.environ["AGENT_EXPLORER_NAV_STEPS"] = original_steps


@check
def code_review_is_typed_navigation_evidence_and_enters_full_synthesis():
    import sliceagent.subagent as subagent_module
    from sliceagent.persistence import ArtifactStore

    class ReviewTools(_Tools):
        def schemas(self):
            return [{"type": "function", "function": {
                "name": "code_review", "parameters": {"type": "object", "properties": {}},
            }}]

        def run(self, name, args):
            assert name == "code_review" and args == {"ref": "HEAD"}
            return "diff --git a/src/a.py b/src/a.py\n+return safe_value"

    class ReviewLLM:
        reasoning = "full"

        def __init__(self):
            self.shared = {"calls": 0, "profiles": [], "synthesis": ""}

        def complete(self, messages, schemas):
            self.shared["calls"] += 1
            self.shared["profiles"].append(self.reasoning)
            call = self.shared["calls"]
            if call == 1:
                return NS(
                    content="inspect the current diff", finish_reason="tool_calls",
                    tool_calls=[NS(name="code_review", args={"ref": "HEAD"}, id="review-1")],
                    usage={"prompt_tokens": 2, "completion_tokens": 1},
                )
            if call == 2:
                return NS(
                    content="handoff: the retained diff is the only source", finish_reason="stop",
                    tool_calls=[], usage={"prompt_tokens": 2, "completion_tokens": 1},
                )
            assert call == 3 and schemas == []
            self.shared["synthesis"] = "\n".join(
                str(message.get("content", "")) for message in messages
            )
            return NS(
                content="FINAL: the observed diff returns safe_value.", finish_reason="stop",
                tool_calls=[], usage={"prompt_tokens": 3, "completion_tokens": 2},
            )

    original_profile = subagent_module.EXPLORER_REASONING
    subagent_module.EXPLORER_REASONING = "staged"
    try:
        with tempfile.TemporaryDirectory() as root:
            llm = ReviewLLM()
            store = ArtifactStore(os.path.join(root, "core"))
            result = run_subagent(
                "review the current diff", tools=ReviewTools(root), llm=llm,
                retriever=NullRetriever(), memory=NullMemory(), max_steps=6, read_only=True,
                workspace_id="workspace-1", session_id="session-1", task_id="task-1",
                parent_id="turn-1", artifact_store=store,
                artifact_id="subagent-code-review-staging",
            )
            artifact = store.get("subagent-code-review-staging")
            assert result.status is ToolStatus.SUCCEEDED
            assert llm.shared["calls"] == 3
            assert llm.shared["profiles"] == ["fast", "fast", "full"]
            assert "tool=code_review" in llm.shared["synthesis"]
            assert "return safe_value" in llm.shared["synthesis"]
            assert artifact.structured_body["observations"][0]["tool"] == "code_review"
    finally:
        subagent_module.EXPLORER_REASONING = original_profile


@check
def failed_navigation_scope_is_visible_to_synthesis_and_cannot_support_claims():
    import sliceagent.subagent as subagent_module
    from sliceagent.persistence import ArtifactStore
    from sliceagent.registry import ToolText

    class MixedTools(_Tools):
        def schemas(self):
            return [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object", "properties": {}},
            }}]

        def run(self, name, args):
            assert name == "read_file"
            if args["path"] == "src/missing.py":
                return ToolText("file does not exist", status=ToolStatus.FAILED)
            if args["path"] == "src/outside.py":
                return ToolText("outside authorized reach", status=ToolStatus.STEERED)
            return "     1\tdef observed(): return True"

    class MixedLLM:
        reasoning = "full"

        def __init__(self):
            self.shared = {"calls": 0, "synthesis": ""}

        def complete(self, messages, schemas):
            self.shared["calls"] += 1
            call = self.shared["calls"]
            if call <= 3:
                path = {
                    1: "src/seen.py",
                    2: "src/missing.py",
                    3: "src/outside.py",
                }[call]
                return NS(
                    content=f"inspect {path}", finish_reason="tool_calls",
                    tool_calls=[NS(name="read_file", args={"path": path}, id=f"read-{call}")],
                    usage={"prompt_tokens": 2, "completion_tokens": 1},
                )
            assert call == 4 and schemas == []
            self.shared["synthesis"] = "\n".join(
                str(message.get("content", "")) for message in messages
            )
            return NS(
                content=(
                    "TOP CLAIM: src/seen.py defines observed().\n"
                    "Coverage gaps: src/missing.py could not be read and src/outside.py was outside reach."
                ),
                finish_reason="stop", tool_calls=[],
                usage={"prompt_tokens": 3, "completion_tokens": 2},
            )

    original_profile = subagent_module.EXPLORER_REASONING
    original_steps = os.environ.get("AGENT_EXPLORER_NAV_STEPS")
    subagent_module.EXPLORER_REASONING = "staged"
    os.environ["AGENT_EXPLORER_NAV_STEPS"] = "3"
    try:
        with tempfile.TemporaryDirectory() as root:
            llm = MixedLLM()
            store = ArtifactStore(os.path.join(root, "core"))
            result = run_subagent(
                "inspect both files", tools=MixedTools(root), llm=llm,
                retriever=NullRetriever(), memory=NullMemory(), max_steps=8, read_only=True,
                workspace_id="workspace-1", session_id="session-1", task_id="task-1",
                parent_id="turn-1", artifact_store=store,
                artifact_id="subagent-failed-scope",
            )
            artifact = store.get("subagent-failed-scope")
            rows = artifact.structured_body["observations"]
            assert result.status is ToolStatus.SUCCEEDED and llm.shared["calls"] == 4
            assert [row["status"] for row in rows] == ["succeeded", "failed", "steered"]
            assert "status=failed" in llm.shared["synthesis"]
            assert "status=steered" in llm.shared["synthesis"]
            assert "src/missing.py" in llm.shared["synthesis"]
            assert "src/outside.py" in llm.shared["synthesis"]
            assert "Any non-success observation status proves attempted scope" in llm.shared["synthesis"]
            assert any("src/missing.py" in gap for gap in artifact.structured_body["gaps"])
            assert any("src/outside.py" in gap for gap in artifact.structured_body["gaps"])
            assert not artifact.structured_body["claims"]
    finally:
        subagent_module.EXPLORER_REASONING = original_profile
        if original_steps is None:
            os.environ.pop("AGENT_EXPLORER_NAV_STEPS", None)
        else:
            os.environ["AGENT_EXPLORER_NAV_STEPS"] = original_steps


@check
def staged_synthesis_owns_the_report_and_host_empty_fallback_is_never_accepted():
    import sliceagent.subagent as subagent_module
    from sliceagent.persistence import ArtifactStore

    class ReadTools(_Tools):
        def schemas(self):
            return [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object", "properties": {}},
            }}]

        def run(self, _name, _args):
            return "     1\tdef observed(): return True"

    class FinalLLM:
        reasoning = "full"

        def __init__(self, final):
            self.final = final
            self.calls = [0]

        def complete(self, _messages, schemas):
            self.calls[0] += 1
            if self.calls[0] == 1:
                return NS(
                    content="TOP CLAIM: navigator speculation must never become the report",
                    finish_reason="tool_calls",
                    tool_calls=[NS(name="read_file", args={"path": "src/seen.py"}, id="read-1")],
                    usage={"prompt_tokens": 2, "completion_tokens": 1},
                )
            assert self.calls[0] == 2 and schemas == []
            if self.final == "raise":
                raise RuntimeError("provider stopped before final report text")
            return NS(
                content="", finish_reason="stop", tool_calls=[],
                usage={"prompt_tokens": 3, "completion_tokens": 0},
            )

    original_profile = subagent_module.EXPLORER_REASONING
    original_steps = os.environ.get("AGENT_EXPLORER_NAV_STEPS")
    subagent_module.EXPLORER_REASONING = "staged"
    os.environ["AGENT_EXPLORER_NAV_STEPS"] = "1"
    try:
        for final in ("empty", "raise"):
            with tempfile.TemporaryDirectory() as root:
                llm = FinalLLM(final)
                store = ArtifactStore(os.path.join(root, "core"))
                artifact_id = f"subagent-final-{final}"
                result = run_subagent(
                    "inspect seen", tools=ReadTools(root), llm=llm,
                    retriever=NullRetriever(), memory=NullMemory(), max_steps=6, read_only=True,
                    workspace_id="workspace-1", session_id="session-1", task_id="task-1",
                    parent_id="turn-1", artifact_store=store, artifact_id=artifact_id,
                )
                artifact = store.get(artifact_id)
                child = next(
                    effect for effect in result.effects if effect.kind == "child_artifact"
                ).payload
                assert llm.calls[0] == 2
                assert result.status is ToolStatus.FAILED
                assert "navigator speculation" not in result
                assert "Done — no summary to add" not in result
                assert artifact.structured_body["report"] == ""
                assert not artifact.structured_body["claims"]
                assert len(artifact.structured_body["observations"]) == 1
                if final == "empty":
                    assert child["stop_cause"] == "empty_synthesis"
                    assert "final synthesis returned no model-authored report text" in result
                    assert "max_steps" not in artifact.error
                    assert "final synthesis returned no model-authored report text" in artifact.error
    finally:
        subagent_module.EXPLORER_REASONING = original_profile
        if original_steps is None:
            os.environ.pop("AGENT_EXPLORER_NAV_STEPS", None)
        else:
            os.environ["AGENT_EXPLORER_NAV_STEPS"] = original_steps


@check
def nonstaged_tool_preamble_never_becomes_report_after_empty_or_failed_followup():
    import sliceagent.subagent as subagent_module
    from sliceagent.persistence import ArtifactStore

    class ReadTools(_Tools):
        def schemas(self):
            return [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object", "properties": {}},
            }}]

        def run(self, _name, _args):
            return "     1\tdef observed(): return True"

    class SpyMemory(NullMemory):
        def __init__(self):
            self.index_jobs = []

        def append_subagent_artifact(self, _session_id, _artifact):
            return "sub-diagnostic"

        def index_subagent_artifact(self, *args):
            self.index_jobs.append(args)

    class FollowupLLM:
        reasoning = "full"

        def __init__(self, followup):
            self.followup = followup
            self.calls = [0]

        def complete(self, _messages, _schemas):
            self.calls[0] += 1
            if self.calls[0] == 1:
                return NS(
                    content="TOP CLAIM: pre-tool speculation must never become the report",
                    finish_reason="tool_calls",
                    tool_calls=[NS(name="read_file", args={"path": "src/seen.py"}, id="read-1")],
                    usage={"prompt_tokens": 2, "completion_tokens": 1},
                )
            assert self.calls[0] == 2
            if self.followup == "timeout":
                raise TimeoutError("provider stopped before report text")
            return NS(
                content="", finish_reason="stop", tool_calls=[],
                usage={"prompt_tokens": 3, "completion_tokens": 0},
            )

    original_profile = subagent_module.EXPLORER_REASONING
    try:
        for profile in ("fast", "full"):
            for followup in ("empty", "timeout"):
                subagent_module.EXPLORER_REASONING = profile
                with tempfile.TemporaryDirectory() as root:
                    llm, memory = FollowupLLM(followup), SpyMemory()
                    store = ArtifactStore(os.path.join(root, "core"))
                    artifact_id = f"subagent-{profile}-{followup}"
                    result = run_subagent(
                        "inspect seen", tools=ReadTools(root), llm=llm,
                        retriever=NullRetriever(), memory=memory, max_steps=4, read_only=True,
                        workspace_id="workspace-1", session_id="session-1", task_id="task-1",
                        parent_id="turn-1", artifact_store=store, artifact_id=artifact_id,
                    )
                    artifact = store.get(artifact_id)
                    assert llm.calls[0] == 2
                    assert result.status is ToolStatus.FAILED
                    assert "pre-tool speculation" not in result
                    assert artifact.structured_body["report"] == ""
                    assert not artifact.structured_body["claims"]
                    assert len(artifact.structured_body["observations"]) == 1
                    assert memory.index_jobs == []
    finally:
        subagent_module.EXPLORER_REASONING = original_profile


@check
def unexpected_tool_call_in_final_synthesis_never_mints_a_hidden_closeout():
    import sliceagent.subagent as subagent_module

    class ReadTools(_Tools):
        def schemas(self):
            return [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object", "properties": {}},
            }}]

        def run(self, _name, _args):
            return "observed source"

    class UnexpectedToolLLM:
        reasoning = "full"

        def __init__(self):
            self.shared = {"calls": 0, "profiles": []}

        def complete(self, _messages, schemas):
            self.shared["calls"] += 1
            self.shared["profiles"].append(self.reasoning)
            if self.shared["calls"] == 1:
                return NS(
                    content="read", finish_reason="tool_calls",
                    tool_calls=[NS(name="read_file", args={"path": "src/a.py"}, id="read-1")],
                    usage={"prompt_tokens": 2, "completion_tokens": 1},
                )
            assert self.shared["calls"] == 2 and schemas == []
            return NS(
                content="unexpected", finish_reason="tool_calls",
                tool_calls=[NS(name="read_file", args={"path": "src/b.py"}, id="bad-1")],
                usage={"prompt_tokens": 2, "completion_tokens": 1},
            )

    original_profile = subagent_module.EXPLORER_REASONING
    original_steps = os.environ.get("AGENT_EXPLORER_NAV_STEPS")
    subagent_module.EXPLORER_REASONING = "staged"
    os.environ["AGENT_EXPLORER_NAV_STEPS"] = "1"
    try:
        with tempfile.TemporaryDirectory() as root:
            llm = UnexpectedToolLLM()
            result = run_subagent(
                "inspect", tools=ReadTools(root), llm=llm, retriever=NullRetriever(),
                memory=NullMemory(), max_steps=5, read_only=True,
            )
            assert llm.shared == {"calls": 2, "profiles": ["fast", "full"]}
            assert result.status is ToolStatus.FAILED
            outcome = _assert_child_envelope(result, "failed")
            assert outcome["report_completion"] == "absent"
            assert outcome["explorer_evidence_status"] == "content_retained"
    finally:
        subagent_module.EXPLORER_REASONING = original_profile
        if original_steps is None:
            os.environ.pop("AGENT_EXPLORER_NAV_STEPS", None)
        else:
            os.environ["AGENT_EXPLORER_NAV_STEPS"] = original_steps


@check
def failed_reserved_synthesis_keeps_budget_bound_navigation_partial_and_typed():
    import sliceagent.subagent as subagent_module
    from sliceagent.persistence import ArtifactStore

    class ReadTools(_Tools):
        def schemas(self):
            return [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object", "properties": {}},
            }}]

        def run(self, _name, _args):
            return "observed source line"

    class TruncatedSynthesisLLM:
        reasoning = "full"

        def __init__(self):
            self.shared = {"calls": 0, "profiles": []}

        def complete(self, _messages, schemas):
            self.shared["calls"] += 1
            self.shared["profiles"].append(self.reasoning)
            if self.shared["calls"] == 1:
                return NS(
                    content="navigation evidence", finish_reason="tool_calls",
                    tool_calls=[NS(name="read_file", args={"path": "src/a.py"}, id="read-a")],
                    usage={"prompt_tokens": 2, "completion_tokens": 1},
                )
            assert self.shared["calls"] == 2 and schemas == []
            return NS(
                content="partial final synthesis", finish_reason="length", tool_calls=[],
                usage={"prompt_tokens": 3, "completion_tokens": 4},
            )

    original_profile = subagent_module.EXPLORER_REASONING
    original_steps = os.environ.get("AGENT_EXPLORER_NAV_STEPS")
    subagent_module.EXPLORER_REASONING = "staged"
    os.environ["AGENT_EXPLORER_NAV_STEPS"] = "1"
    try:
        with tempfile.TemporaryDirectory() as root:
            llm = TruncatedSynthesisLLM()
            store = ArtifactStore(os.path.join(root, "core"))
            result = run_subagent(
                "inspect a", tools=ReadTools(root), llm=llm,
                retriever=NullRetriever(), memory=NullMemory(), max_steps=20, read_only=True,
                workspace_id="workspace-1", session_id="session-1", task_id="task-1",
                parent_id="turn-1", artifact_store=store,
                artifact_id="subagent-truncated-reserved-synthesis",
            )
            artifact = store.get("subagent-truncated-reserved-synthesis")
            child = next(effect for effect in result.effects if effect.kind == "child_artifact").payload
            assert llm.shared == {"calls": 2, "profiles": ["fast", "full"]}
            assert result.status is ToolStatus.FAILED
            assert child["stop_reason"] == "max_tokens"
            assert child["stop_cause"] == "output_truncated" and child["partial"] is True
            assert "partial final synthesis" in artifact.structured_body["report"]
            assert any("planned full synthesis stopped as max_tokens" in note
                       for note in artifact.uncertainty)
    finally:
        subagent_module.EXPLORER_REASONING = original_profile
        if original_steps is None:
            os.environ.pop("AGENT_EXPLORER_NAV_STEPS", None)
        else:
            os.environ["AGENT_EXPLORER_NAV_STEPS"] = original_steps


@check
def evidence_free_navigation_ceiling_never_starts_synthesis():
    import sliceagent.subagent as subagent_module
    from sliceagent.persistence import ArtifactStore

    class SkillTools(_Tools):
        def schemas(self):
            return [{"type": "function", "function": {
                "name": "skill", "parameters": {"type": "object", "properties": {}},
            }}]

        def run(self, name, _args):
            assert name == "skill"
            return "general workflow guidance, not a workspace observation"

    class NoEvidenceLLM:
        reasoning = "full"

        def __init__(self):
            self.shared = {"calls": 0, "profiles": []}

        def complete(self, _messages, _schemas):
            self.shared["calls"] += 1
            self.shared["profiles"].append(self.reasoning)
            call = self.shared["calls"]
            assert call <= 2, "no typed workspace evidence means no synthesis call"
            return NS(
                content=f"consulting guidance {call}", finish_reason="tool_calls",
                tool_calls=[NS(name="skill", args={"name": "review"}, id=f"skill-{call}")],
                usage={"prompt_tokens": 2, "completion_tokens": 1},
            )

    original_profile = subagent_module.EXPLORER_REASONING
    original_steps = os.environ.get("AGENT_EXPLORER_NAV_STEPS")
    subagent_module.EXPLORER_REASONING = "staged"
    os.environ["AGENT_EXPLORER_NAV_STEPS"] = "2"
    try:
        with tempfile.TemporaryDirectory() as root:
            llm = NoEvidenceLLM()
            store = ArtifactStore(os.path.join(root, "core"))
            result = run_subagent(
                "inspect the workspace", tools=SkillTools(root), llm=llm,
                retriever=NullRetriever(), memory=NullMemory(), max_steps=20, read_only=True,
                workspace_id="workspace-1", session_id="session-1", task_id="task-1",
                parent_id="turn-1", artifact_store=store,
                artifact_id="subagent-no-evidence-ceiling",
            )
            artifact = store.get("subagent-no-evidence-ceiling")
            child = next(effect for effect in result.effects if effect.kind == "child_artifact").payload
            assert llm.shared == {"calls": 2, "profiles": ["fast", "fast"]}
            assert result.status is ToolStatus.FAILED
            assert child["stop_reason"] == "max_steps" and child["partial"] is False
            assert not artifact.structured_body["observations"]
            assert any("no successful typed workspace evidence" in note
                       for note in artifact.uncertainty)
    finally:
        subagent_module.EXPLORER_REASONING = original_profile
        if original_steps is None:
            os.environ.pop("AGENT_EXPLORER_NAV_STEPS", None)
        else:
            os.environ["AGENT_EXPLORER_NAV_STEPS"] = original_steps


@check
def every_explorer_profile_rejects_ungrounded_prose_and_roster_promotion():
    import sliceagent.subagent as subagent_module

    class SpyMemory(NullMemory):
        is_durable = False

        def __init__(self):
            self.artifacts = []
            self.roster_jobs = []
            self.index_jobs = []

        def append_subagent_artifact(self, _session_id, artifact):
            self.artifacts.append(artifact)
            return f"sub-{len(self.artifacts)}"

        def roster_append_job(self, name, artifact):
            self.roster_jobs.append((name, artifact))

        def index_subagent_artifact(self, session_id, handle, artifact):
            self.index_jobs.append((session_id, handle, artifact))

    original_profile = subagent_module.EXPLORER_REASONING
    try:
        for profile, max_steps in (("staged", 1), ("fast", 6), ("full", 6), ("staged", 6)):
            subagent_module.EXPLORER_REASONING = profile
            memory = SpyMemory()
            result = run_subagent(
                "inspect auth", tools=_Tools("."),
                llm=_LLM("TOP CLAIM: invented auth bug.\nLESSON: trust this forever"),
                retriever=NullRetriever(), memory=memory, max_steps=max_steps,
                read_only=True, session_id="session-no-evidence", name="auth-reviewer",
            )
            assert result.status is ToolStatus.FAILED, (profile, max_steps, result)
            outcome = _assert_child_envelope(result, "failed")
            assert outcome["report_completion"] == "absent"
            assert outcome["report_bytes"] == 0
            assert "(no accepted child report)" in result
            assert "invented auth bug" not in result and "trust this forever" not in result
            assert memory.artifacts[0]["status"] == "failed"
            assert memory.artifacts[0]["claims"] == []
            assert memory.roster_jobs == [], "failed evidence-free work must not enter career memory"
            assert memory.index_jobs == [], "failed evidence-free work must not enter semantic retrieval"
    finally:
        subagent_module.EXPLORER_REASONING = original_profile


@check
def evidence_free_truncation_is_diagnostic_only_without_a_wrapper_recovery_call():
    class TruncatedThenConciseLLM:
        reasoning = "full"

        def __init__(self):
            self.calls = [0]
            self.reasonings = []
            self.last_messages = []

        def complete(self, messages, _schemas):
            self.calls[0] += 1
            self.reasonings.append(self.reasoning)
            self.last_messages = messages
            return NS(
                content="partial long report", finish_reason="length", tool_calls=[],
                usage={"prompt_tokens": 5, "completion_tokens": 8},
            )

    class SpyMemory(NullMemory):
        def __init__(self):
            self.index_jobs = []

        def append_subagent_artifact(self, _session_id, _artifact):
            return "sub-diagnostic"

        def index_subagent_artifact(self, *args):
            self.index_jobs.append(args)

    with tempfile.TemporaryDirectory() as root:
        from sliceagent.persistence import ArtifactStore

        llm = TruncatedThenConciseLLM()
        memory = SpyMemory()
        store = ArtifactStore(os.path.join(root, "core"))
        result = run_subagent(
            "inspect", tools=_Tools(root), llm=llm, retriever=NullRetriever(),
            memory=memory, max_steps=4, read_only=True,
            workspace_id="workspace-1", session_id="session-1", task_id="task-1",
            parent_id="turn-1", artifact_store=store,
            artifact_id="subagent-truncation-recovery",
        )
        assert llm.calls[0] == 1, "truncation must not mint a hidden wrapper request"
        assert llm.reasonings == ["fast"], llm.reasonings
        outcome = _assert_child_envelope(result, "failed")
        assert result.status is ToolStatus.FAILED
        assert "partial long report" not in result
        assert "(no accepted child report)" in result
        usage = next(effect for effect in result.effects if effect.kind == "model_usage").payload
        assert usage["prompt_tokens"] == 5 and usage["completion_tokens"] == 8
        assert outcome["stop_reason"] == "max_tokens" and outcome["stop_cause"] == "output_truncated"
        assert outcome["partial"] is False, \
            "unobserved evidence-free prose is not an accepted partial report"
        child = next(effect for effect in result.effects if effect.kind == "child_artifact").payload
        assert child["recovered_from"] == []
        artifact = store.get("subagent-truncation-recovery")
        assert artifact.structured_body["report"] == "partial long report"
        assert not artifact.structured_body["claims"]
        assert memory.index_jobs == []


@check
def truncated_child_does_not_replay_or_mint_a_fresh_budget():
    class AlwaysTruncatedLLM:
        reasoning = "full"

        def __init__(self):
            self.calls = [0]

        def complete(self, _messages, _schemas):
            self.calls[0] += 1
            return NS(
                content=f"partial {self.calls[0]}", finish_reason="length", tool_calls=[],
                usage={"prompt_tokens": 3, "completion_tokens": 3},
            )

    with tempfile.TemporaryDirectory() as root:
        llm = AlwaysTruncatedLLM()
        result = run_subagent(
            "inspect", tools=_Tools(root), llm=llm, retriever=NullRetriever(),
            memory=NullMemory(), max_steps=6, read_only=True, token_budget=10,
        )
        assert llm.calls[0] == 1, "a partial result must not be replayed by a wrapper recovery call"
        outcome = _assert_child_envelope(result, "failed")
        assert "partial 1" not in result
        assert "(no accepted child report)" in result
        assert outcome["report_completion"] == "absent" and outcome["report_bytes"] == 0
        assert outcome["stop_reason"] == "max_tokens"
        usage = next(effect for effect in result.effects if effect.kind == "model_usage").payload
        assert usage["prompt_tokens"] == 3 and usage["completion_tokens"] == 3


@check
def provider_timeout_is_not_replayed_by_the_child_wrapper():
    class TimeoutThenReportLLM:
        reasoning = "fast"

        def __init__(self):
            self.calls = [0]

        def is_retryable(self, _error):
            return False  # exercise the child-level continuation without sleeping through SDK retries

        def complete(self, messages, _schemas):
            self.calls[0] += 1
            raise TimeoutError("Request timed out")

    with tempfile.TemporaryDirectory() as root:
        from sliceagent.persistence import ArtifactStore

        llm = TimeoutThenReportLLM()
        store = ArtifactStore(os.path.join(root, "core"))
        result = run_subagent(
            "inspect", tools=_Tools(root), llm=llm, retriever=NullRetriever(),
            memory=NullMemory(), max_steps=4, read_only=True,
            workspace_id="workspace-1", session_id="session-1", task_id="task-1",
            parent_id="turn-1", artifact_store=store,
            artifact_id="subagent-timeout-recovery",
        )
        assert llm.calls[0] == 1, "provider retry policy has one owner; the child wrapper cannot replay it"
        outcome = _assert_child_envelope(result, "failed")
        usage = next(effect for effect in result.effects if effect.kind == "model_usage").payload
        assert usage["prompt_tokens"] == 0 and usage["completion_tokens"] == 0
        assert outcome["stop_reason"] == "error" and outcome["partial"] is False
        child = next(effect for effect in result.effects if effect.kind == "child_artifact").payload
        assert child["recovered_from"] == []


@check
def watchdog_abandonment_never_launches_overlapping_child_recovery():
    from sliceagent.errors import IndeterminateModelCallError
    from sliceagent.execution import ToolStatus

    class AbandonedCallLLM:
        reasoning = "fast"

        def __init__(self):
            self.calls = [0]

        def is_retryable(self, _error):
            return False

        def complete(self, _messages, _schemas):
            self.calls[0] += 1
            raise IndeterminateModelCallError(
                "provider request exceeded its watchdog deadline and may still be in flight"
            )

    with tempfile.TemporaryDirectory() as root:
        from sliceagent.persistence import ArtifactStore

        llm = AbandonedCallLLM()
        store = ArtifactStore(os.path.join(root, "core"))
        result = run_subagent(
            "inspect", tools=_Tools(root), llm=llm, retriever=NullRetriever(),
            memory=NullMemory(), max_steps=4, read_only=True,
            workspace_id="workspace-1", session_id="session-1", task_id="task-1",
            parent_id="turn-1", artifact_store=store,
            artifact_id="subagent-indeterminate-model-call",
        )
        assert llm.calls[0] == 1, (
            f"a still-live provider socket must not overlap a recovery model call (got {llm.calls[0]})"
        )
        outcome = _assert_child_envelope(result, "indeterminate")
        assert result.status is ToolStatus.INDETERMINATE, (
            "a watchdog-abandoned provider request must remain uncertain at the parent boundary"
        )
        assert outcome["stop_cause"] == "indeterminate_model_call"
        child = next(effect for effect in result.effects if effect.kind == "child_artifact").payload
        assert child["recovered_from"] == []


@check
def timeout_after_evidence_seals_only_observed_partial_without_a_fallback_namespace():
    class ToolThenTimeoutThenToolLLM:
        reasoning = "fast"

        def __init__(self):
            self.calls = [0]
            self.schemas = []

        def is_retryable(self, _error):
            return False

        def complete(self, _messages, schemas):
            self.calls[0] += 1
            self.schemas.append(tuple(schemas))
            if self.calls[0] == 1:
                return NS(
                    content="", finish_reason="tool_calls",
                    tool_calls=[NS(name="read_file", args={"path": "a.py"}, id=None)],
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            if self.calls[0] == 2:
                raise TimeoutError("Request timed out")
            raise AssertionError("the child wrapper must not launch a recovery namespace")

    class ReadTools(_Tools):
        def __init__(self, root):
            super().__init__(root)
            self.ran = []

        def schemas(self):
            return [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object", "properties": {}},
            }}]

        def run(self, _name, args):
            self.ran.append(args.get("path"))
            return f"contents of {args.get('path')}"

    with tempfile.TemporaryDirectory() as root:
        from sliceagent.persistence import ArtifactStore

        llm = ToolThenTimeoutThenToolLLM()
        store = ArtifactStore(os.path.join(root, "core"))
        tools = ReadTools(root)
        result = run_subagent(
            "inspect", tools=tools, llm=llm, retriever=NullRetriever(),
            memory=NullMemory(), max_steps=6, read_only=True,
            workspace_id="workspace-1", session_id="session-1", task_id="task-1",
            parent_id="turn-1", artifact_store=store,
            artifact_id="subagent-timeout-tool-namespace",
        )
        assert llm.calls[0] == 2
        outcome = _assert_child_envelope(result, "failed")
        artifact = store.get("subagent-timeout-tool-namespace")
        assert "read_file a.py" in artifact.structured_body["trace"]
        assert "read_file b.py" not in artifact.structured_body["trace"]
        assert tools.ran == ["a.py"], "an unoffered recovery call must not reach the real tool host"
        assert outcome["partial"] is True and outcome["report_completion"] == "absent"
        child = next(effect for effect in result.effects if effect.kind == "child_artifact").payload
        assert child["recovered_from"] == []


@check
def cancelled_child_never_publishes_a_late_report_or_reference():
    entered = threading.Event()
    release = threading.Event()
    signal = threading.Event()

    class BlockingLLM(_LLM):
        def complete(self, _messages, _schemas):
            self.calls[0] += 1
            entered.set()
            assert release.wait(2)
            return _Response("late report that must not be accepted")

    with tempfile.TemporaryDirectory() as root:
        from sliceagent.persistence import ArtifactStore, PendingTurnJournal

        store = ArtifactStore(os.path.join(root, "core"))
        refs = []
        box = {}
        llm = BlockingLLM()
        thread = threading.Thread(target=lambda: box.setdefault("result", run_subagent(
            "inspect", tools=_Tools(root), llm=llm, retriever=NullRetriever(),
            memory=NullMemory(), max_steps=4, read_only=True, signal=signal,
            workspace_id="workspace-1", session_id="session-1", task_id="task-A",
            parent_id="turn-A", artifact_store=store,
            artifact_id="subagent-cancelled-late", artifact_ref_sink=refs.append,
        )), daemon=True)
        thread.start()
        assert entered.wait(1)
        signal.set()
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        result = box["result"]
        assert result.status.value == "cancelled", result
        assert llm.calls[0] == 1, "cancellation must not enter planned synthesis"
        assert refs == []
        assert not store.exists("subagent-cancelled-late")
        assert PendingTurnJournal.pending(store.root) == []


@check
def cancellation_immediately_after_generic_put_finishes_the_required_reference():
    signal = threading.Event()

    with tempfile.TemporaryDirectory() as root:
        from sliceagent.persistence import ArtifactStore, PendingTurnJournal

        canonical = ArtifactStore(os.path.join(root, "core"))
        refs = []

        class CancelAfterPut:
            root = canonical.root

            def put(self, artifact):
                result = canonical.put(artifact)
                signal.set()  # exact old split: immutable child exists, parent ref is not written yet
                return result

        result = run_subagent(
            "inspect", tools=_GroundedTools(root), llm=_GroundedLLM("committed report"),
            retriever=NullRetriever(), memory=NullMemory(), max_steps=4, read_only=True,
            signal=signal, workspace_id="workspace-1", session_id="session-1",
            task_id="task-A", parent_id="turn-A", artifact_store=CancelAfterPut(),
            artifact_id="subagent-cancel-after-put", artifact_ref_sink=refs.append,
        )

        assert signal.is_set()
        assert result.status is ToolStatus.SUCCEEDED, result
        assert canonical.exists("subagent-cancel-after-put")
        assert refs == ["subagent-cancel-after-put"]
        child = next(effect for effect in result.effects if effect.kind == "child_artifact")
        assert child.payload["artifact_id"] == "subagent-cancel-after-put"
        assert PendingTurnJournal.pending(canonical.root) == []


@check
def cancellation_during_atomic_publication_returns_committed_child_and_seals_its_ref():
    signal = threading.Event()

    with tempfile.TemporaryDirectory() as root:
        from sliceagent.runtime_persistence import LocalTurnStore

        turn_store = LocalTurnStore(
            root, "session-atomic-child", store_root=os.path.join(root, "core"),
        )
        active = turn_store.begin(
            task_id="task-A", logical_id="turn-A", user_request="inspect A",
        )
        canonical = turn_store.coordinator.artifacts

        class CancelAfterPut:
            root = canonical.root

            def put(self, artifact):
                result = canonical.put(artifact)
                signal.set()  # publication owns the turn lock; cancellation cannot split its ref
                return result

        publish = turn_store.bind_artifact_ref_sink(
            task_id=active.task_id, parent_id=active.artifact_id,
        )
        result = run_subagent(
            "inspect", tools=_GroundedTools(root), llm=_GroundedLLM("committed report"),
            retriever=NullRetriever(), memory=NullMemory(), max_steps=4, read_only=True,
            signal=signal, workspace_id=turn_store.workspace_id,
            session_id=turn_store.session_id, task_id=active.task_id,
            parent_id=active.artifact_id, artifact_store=CancelAfterPut(),
            artifact_id="subagent-atomic-cancel-after-put", artifact_ref_sink=publish,
        )

        assert signal.is_set()
        assert result.status is ToolStatus.SUCCEEDED, result
        assert canonical.exists("subagent-atomic-cancel-after-put")
        assert "subagent-atomic-cancel-after-put" in active.journal.snapshot().artifact_refs
        turn_store.seal(state={}, record={}, status="end_turn")
        parent = canonical.get(active.artifact_id)
        checkpoint = turn_store.coordinator.checkpoints.load(
            turn_store.workspace_id, active.task_id,
        )
        assert "subagent-atomic-cancel-after-put" in parent.refs
        assert "subagent-atomic-cancel-after-put" in checkpoint.artifact_refs
        turn_store.close()


@check
def cancellation_after_parent_ref_publication_preserves_committed_child_truth():
    signal = threading.Event()

    class OptionalMirror(NullMemory):
        is_durable = True

        def __init__(self):
            self.append_calls = 0
            self.index_calls = 0

        def append_subagent_artifact(self, _session_id, _artifact):
            self.append_calls += 1
            return "legacy-sub-1"

        def index_subagent_artifact(self, _session_id, _handle, _artifact):
            self.index_calls += 1

    with tempfile.TemporaryDirectory() as root:
        from sliceagent.persistence import ArtifactStore, PendingTurnJournal

        store = ArtifactStore(os.path.join(root, "core"))
        refs = []
        mirror = OptionalMirror()

        def publish_ref(artifact_id):
            refs.append(artifact_id)
            signal.set()  # exact race: cancellation arrives after the parent dependency is durable

        result = run_subagent(
            "inspect", tools=_GroundedTools(root), llm=_GroundedLLM("committed report"),
            retriever=NullRetriever(), memory=mirror, max_steps=4, read_only=True,
            signal=signal, workspace_id="workspace-1", session_id="session-1",
            task_id="task-A", parent_id="turn-A", artifact_store=store,
            artifact_id="subagent-committed-race", artifact_ref_sink=publish_ref,
        )

        assert result.status is ToolStatus.SUCCEEDED, result
        assert refs == ["subagent-committed-race"]
        assert store.exists("subagent-committed-race")
        outcome = _assert_child_envelope(result, "succeeded", "committed report")
        assert outcome["artifact_id"] == "subagent-committed-race"
        child = next(effect for effect in result.effects if effect.kind == "child_artifact")
        assert child.payload["artifact_id"] == "subagent-committed-race"
        assert child.payload["status"] == "ok"
        assert child.payload["operational_status"] == "succeeded"
        assert mirror.append_calls == mirror.index_calls == 0, \
            "post-commit cancellation may skip optional mirrors without rewriting canonical truth"
        assert PendingTurnJournal.pending(store.root) == []


@check
def host_binds_child_reference_to_the_launch_turn_before_the_child_runs():
    entered = threading.Event()
    release = threading.Event()

    class BlockingLLM(_LLM):
        def complete(self, _messages, _schemas):
            self.calls[0] += 1
            if self.calls[0] == 1:
                return NS(
                    content="", finish_reason="tool_calls",
                    tool_calls=[NS(name="read_file", args={"path": "fixture.py"}, id="read")],
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            entered.set()
            assert release.wait(2)
            return _Response("late but otherwise complete report")

    with tempfile.TemporaryDirectory() as root:
        from sliceagent.runtime_persistence import LocalTurnStore

        turn_store = LocalTurnStore(
            root, "session-bound-child", store_root=os.path.join(root, "core"),
        )
        active_a = turn_store.begin(
            task_id="task-A", logical_id="turn-A", user_request="inspect A",
        )
        host = SubagentHost(
            _GroundedTools(root), llm=BlockingLLM(),
            retriever=NullRetriever(), memory=NullMemory(),
            max_depth=1, session_id="session-bound-child", workspace_id=turn_store.workspace_id,
            task_id_fn=lambda: "task-A", parent_id_fn=lambda: active_a.artifact_id,
            artifact_store=turn_store.coordinator.artifacts,
            artifact_ref_sink=turn_store.record_artifact_ref,
        )
        box = {}
        thread = threading.Thread(target=lambda: box.setdefault(
            "result", host.run("spawn_agent", {"agent": "explorer", "task": "inspect A"}),
        ), daemon=True)
        thread.start()
        assert entered.wait(1)
        turn_store.seal(state={}, record={}, status="aborted")
        active_b = turn_store.begin(
            task_id="task-B", logical_id="turn-B", user_request="inspect B",
        )
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        result = box["result"]
        outcome = _assert_child_envelope(
            result, "succeeded", "late but otherwise complete report",
        )
        assert result.status is ToolStatus.SUCCEEDED
        assert outcome["artifact_id"] == "" and outcome["report_handle"] == ""
        assert "Persistence warning: canonical artifact was not stored " \
               "(RuntimeError: child launch turn is no longer active)" in result
        assert not [effect for effect in result.effects if effect.kind == "child_artifact"]
        assert active_b.journal.snapshot().artifact_refs == ()
        turn_store.seal(state={}, record={}, status="end_turn")
        assert "subagent" not in " ".join(
            turn_store.coordinator.artifacts.get(active_b.artifact_id).refs
        )
        turn_store.close()


@check
def setup_timeout_is_not_misclassified_as_a_recoverable_model_timeout():
    class NeverCalledLLM(_LLM):
        def complete(self, messages, schemas):
            raise AssertionError("model must not run when schema preparation failed")

    class SetupTimeoutTools(_Tools):
        def schemas(self):
            raise TimeoutError("tool host setup timed out")

    with tempfile.TemporaryDirectory() as root:
        llm = NeverCalledLLM()
        result = run_subagent(
            "inspect", tools=SetupTimeoutTools(root), llm=llm, retriever=NullRetriever(),
            memory=NullMemory(), max_steps=4, read_only=True,
        )
        assert llm.calls[0] == 0
        outcome = _assert_child_envelope(result, "failed")
        assert result.status is ToolStatus.FAILED
        assert outcome["stop_reason"] == "error"
        assert outcome["report_completion"] == "absent"


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
            memory=NullMemory(), max_steps=3,
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
