"""Projection truth: virtual resources, live capability guidance, and prompt A/B seam."""
import hashlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import ToolResult  # noqa: E402
from sliceagent.code_grep import make_grep_tool  # noqa: E402
from sliceagent.agents import BUILTIN_AGENTS  # noqa: E402
from sliceagent.discourse import (_deterministic_response_constraint_mismatches,  # noqa: E402
                                  _subagent_grounding_envelope,
                                  interpret_turn, make_evidence_snapshot)
from sliceagent.execution import ToolInvocation  # noqa: E402
from sliceagent.pfc import Slice, slice_sink  # noqa: E402
from sliceagent.prompt import (MEMORY_ACCUMULATE, memory_model_for_eval,
                               render_delegation_guidance)  # noqa: E402
from sliceagent.regions import build_context_blocks  # noqa: E402
from sliceagent.runtime_persistence import CoreArtifactFS  # noqa: E402
from sliceagent.seed import (_slice_context, build_artifacts,
                             physical_active_files)  # noqa: E402
from sliceagent.subagent import SubagentHost, _ObservationSink  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def deterministic_quality_checks_cover_only_measurable_explicit_constraints():
    long = " ".join(["word"] * 81)
    brief = _deterministic_response_constraint_mismatches(
        "What else can you help with, briefly?", long,
    )
    assert len(brief) == 1 and brief[0]["constraint"] == "brief_response"
    assert brief[0]["produced_exact"] in long and "81 words" in brief[0]["explanation"]
    assert brief[0]["produced_exact_is_bounded_prefix"]
    assert not brief[0]["produced_exact"].endswith("wor"), "the exact prefix must end at a word boundary"
    assert not _deterministic_response_constraint_mismatches(
        "What else can you help with?", long,
    ), "brevity is not inferred when the user did not request it"
    assert not _deterministic_response_constraint_mismatches(
        "Do not be brief; explain fully.", long,
    )

    lines = _deterministic_response_constraint_mismatches(
        "Return exactly 3 lines.", "one\ntwo",
    )
    assert len(lines) == 1 and lines[0]["measurements"] == {
        "expected_nonempty_lines": 3, "actual_nonempty_lines": 2,
    }
    assert not _deterministic_response_constraint_mismatches(
        "Return exactly 3 lines.", "one\ntwo\nthree",
    )
    assert _deterministic_response_constraint_mismatches(
        "Return exactly JSON.", "plain text",
    )[0]["constraint"] == "valid_json"
    assert not _deterministic_response_constraint_mismatches(
        "Return exactly JSON.", '{"ok": true}',
    )


class _ArtifactView:
    def read_file(self, path):
        return f"sealed virtual artifact: {path}"


class _ArtifactStore:
    def __init__(self):
        self.artifact = SimpleNamespace(
            id="turn-absolute", kind="turn", title="absolute route", task_id="task-1",
            status="completed", timestamp="2026-07-11T00:00:00Z",
            brief={"request": "find the canonical needle"}, summary="canonical virtual needle",
            structured_body={"assistant": "canonical virtual needle"}, refs=(),
        )

    def list_all(self):
        return [self.artifact]

    def get(self, artifact_id):
        if artifact_id != self.artifact.id:
            raise KeyError(artifact_id)
        return self.artifact


def _invoke_read(host, path, invocation_id="read-1"):
    invocation = ToolInvocation(invocation_id, "read_file", {"path": path}, 0)
    outcome = host.registry.invoke(invocation)
    return outcome


@check
def virtual_artifact_read_stays_typed_and_never_enters_open_files():
    host = LocalToolHost(tempfile.mkdtemp())
    host._artifacts = _ArtifactView()
    outcome = _invoke_read(host, "artifacts/turn-1.md")
    resource = next(effect for effect in outcome.effects if effect.kind == "resource_observed")
    assert resource.payload["resource_kind"] == "artifact"
    assert resource.payload["handle"] == "artifacts/turn-1.md"
    assert resource.payload["artifact_id"] == "turn-1"
    assert resource.payload["read_coverage"] == "complete"
    assert len(resource.payload["content_sha256"]) == 64
    assert resource.payload["content_bytes"] == len(outcome.text.encode("utf-8"))

    state = Slice(); state.reset("recall")
    slice_sink(state)(ToolResult(
        "read_file", {"path": "artifacts/turn-1.md"}, outcome.text, outcome.failing,
        status=outcome.status.value, invocation_id=outcome.invocation.id, outcome=outcome,
    ))
    assert state.active_files == []

    # A poisoned legacy checkpoint is filtered defensively too.
    state.active_files = ["artifacts/turn-1.md"]
    rendered = build_artifacts(state, host)
    assert rendered == "(no workspace files opened yet)" and "not created yet" not in rendered


@check
def child_evidence_page_read_keeps_root_artifact_provenance_but_is_not_a_report_read():
    from sliceagent.persistence import Artifact, ArtifactStore

    store = ArtifactStore(tempfile.mkdtemp(prefix="projection-child-evidence-"))
    view = "     1\treturn verified_value"
    encoded = view.encode()
    store.put(Artifact(
        id="subagent-evidence-root", kind="subagent", workspace_id="workspace",
        session_id="session", task_id="task", status="ok",
        structured_body={
            "report": "verified_value is returned",
            "observations": [{
                "v": 1, "tool": "read_file", "args": {"path": "value.py"},
                "status": "succeeded", "view": view,
                "raw_sha256": hashlib.sha256(encoded).hexdigest(),
                "view_sha256": hashlib.sha256(encoded).hexdigest(),
                "raw_bytes": len(encoded), "view_bytes": len(encoded),
                "redacted": False, "truncated": False,
            }],
        },
    ))
    host = LocalToolHost(tempfile.mkdtemp(prefix="projection-child-workspace-"))
    host._artifacts = CoreArtifactFS(store)
    path = "artifacts/subagent-evidence-root/evidence/obs-001-page-001.md"
    outcome = _invoke_read(host, path, "read-child-evidence")
    resource = next(effect for effect in outcome.effects if effect.kind == "resource_observed")
    assert resource.payload["artifact_id"] == "subagent-evidence-root"
    assert resource.payload["artifact_view"] == "evidence"
    assert resource.payload["read_coverage"] == "complete"

    state = Slice(); state.reset("verify child evidence")
    slice_sink(state)(ToolResult(
        "read_file", {"path": path}, outcome.text, outcome.failing,
        status=outcome.status.value, invocation_id=outcome.invocation.id, outcome=outcome,
    ))
    assert state.active_files == []
    assert state.runtime.recent_calls[-1]["observed_artifact_id"] == "subagent-evidence-root"
    assert state.runtime.recent_calls[-1]["observed_artifact_view"] == "evidence"


@check
def quality_grounding_uses_only_the_bounded_preview_not_the_full_evidence_archive():
    full = ("FULL-ARCHIVE-ONLY\n" * 10_000) + "ARCHIVE-TAIL-SENTINEL"
    preview = "bounded preview bytes"

    def row(text, *, truncated):
        encoded = text.encode()
        digest = hashlib.sha256(encoded).hexdigest()
        return {
            "v": 1, "tool": "read_file", "args": {"path": "large.py"},
            "status": "succeeded", "view": text,
            "raw_sha256": digest, "view_sha256": digest,
            "raw_bytes": len(encoded), "view_bytes": len(encoded),
            "redacted": False, "truncated": truncated,
        }

    envelope = _subagent_grounding_envelope({
        "status": "ok",
        "structured_body": {
            "brief": {"objective": "inspect large.py", "scope": ["large.py"]},
            "report": "child conclusion", "claims": [],
            "observations": [row(full, truncated=False)],
            "observation_preview": [row(preview, truncated=True)],
        },
    })
    assert envelope["observations"][0]["view"] == preview
    assert "ARCHIVE-TAIL-SENTINEL" not in json.dumps(envelope)


@check
def absolute_artifact_paths_share_the_canonical_virtual_handle_for_read_list_and_grep():
    root = tempfile.mkdtemp()
    host = LocalToolHost(root)
    host._artifacts = CoreArtifactFS(_ArtifactStore())
    host.registry.register(make_grep_tool(host))
    absolute_mount = os.path.join(root, "artifacts")
    absolute_file = os.path.join(absolute_mount, "turn-absolute.md")

    outcome = _invoke_read(host, absolute_file, "read-absolute")
    assert "canonical virtual needle" in outcome.text
    resource = next(effect for effect in outcome.effects if effect.kind == "resource_observed")
    assert resource.payload["resource_kind"] == "artifact"
    assert resource.payload["handle"] == "artifacts/turn-absolute.md"
    assert resource.payload["artifact_id"] == "turn-absolute"
    assert resource.payload["read_coverage"] == "complete"
    assert "turn-absolute.md" in host.run("list_files", {"path": absolute_mount})
    grep = host.run("grep", {"pattern": "canonical virtual needle", "path": absolute_file})
    assert "artifacts/turn-absolute.md:" in grep

    blocked = host.run("edit_file", {"path": absolute_file, "content": "shadow"})
    assert "read-only authoritative local artifact archive" in blocked
    assert not os.path.exists(absolute_file)


@check
def absolute_physical_artifact_file_still_shadows_the_virtual_handle():
    root = tempfile.mkdtemp()
    physical = os.path.join(root, "artifacts", "turn-absolute.md")
    os.makedirs(os.path.dirname(physical))
    with open(physical, "w", encoding="utf-8") as stream:
        stream.write("physical workspace bytes")
    host = LocalToolHost(root)
    host._artifacts = CoreArtifactFS(_ArtifactStore())

    outcome = _invoke_read(host, physical, "read-absolute-shadow")
    assert "physical workspace bytes" in outcome.text
    assert "canonical virtual needle" not in outcome.text
    resource = next(effect for effect in outcome.effects if effect.kind == "resource_observed")
    assert resource.payload == {"resource_kind": "workspace_file", "handle": physical}


@check
def child_isolation_and_observation_capsules_follow_canonical_host_routing():
    root = tempfile.mkdtemp()
    host = LocalToolHost(root)
    host._artifacts = CoreArtifactFS(_ArtifactStore())
    absolute_mount = os.path.join(root, "artifacts")
    absolute_file = os.path.join(absolute_mount, "turn-absolute.md")
    child = SubagentHost(
        host, llm=None, retriever=None, memory=None,
        max_depth=1, depth=1, spec=BUILTIN_AGENTS["explorer"],
    )

    # Absolute and relative spellings route to the same virtual archive and are both parent-private.
    for path in ("artifacts/turn-absolute.md", absolute_file):
        blocked = child.run("read_file", {"path": path})
        assert "private namespace" in blocked, (path, blocked)
    blocked_list = child.run("list_files", {"path": absolute_mount})
    assert "private namespace" in blocked_list, blocked_list

    # A virtual read result is archive testimony, not a workspace observation capsule.
    virtual = _invoke_read(host, absolute_file, "virtual-child-read")
    sink = _ObservationSink(host.resource_ref, host._archive_handle)
    sink(ToolResult(
        "read_file", {"path": absolute_file}, virtual.text, virtual.failing,
        status=virtual.status.value, invocation_id=virtual.invocation.id, outcome=virtual,
    ))
    assert sink.observations == ()

    # Once real project bytes shadow the mount, canonical routing classifies them as physical. Child reads and
    # sealed observations must both remain available; a lexical `artifacts/` deny would get this wrong.
    os.makedirs(absolute_mount, exist_ok=True)
    with open(absolute_file, "w", encoding="utf-8") as stream:
        stream.write("physical child-visible bytes")
    assert "physical child-visible bytes" in child.run("read_file", {"path": absolute_file})
    assert "turn-absolute.md" in child.run("list_files", {"path": absolute_mount})
    physical = _invoke_read(host, absolute_file, "physical-child-read")
    sink(ToolResult(
        "read_file", {"path": absolute_file}, physical.text, physical.failing,
        status=physical.status.value, invocation_id=physical.invocation.id, outcome=physical,
    ))
    assert len(sink.observations) == 1
    assert sink.observations[0].args["path"] == absolute_file
    assert "physical child-visible bytes" in sink.observations[0].view

    # The host-private physical store is not a project shadow and stays isolated under its absolute spelling.
    private_file = os.path.join(root, ".sliceagent", "blobs", "parent.txt")
    os.makedirs(os.path.dirname(private_file), exist_ok=True)
    with open(private_file, "w", encoding="utf-8") as stream:
        stream.write("parent-private")
    assert "private namespace" in child.run("read_file", {"path": private_file})


@check
def real_workspace_file_shadowing_reserved_mount_remains_physical():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "artifacts"))
    with open(os.path.join(root, "artifacts", "real.md"), "w", encoding="utf-8") as stream:
        stream.write("real workspace bytes\n" + ("payload\n" * 80))
    host = LocalToolHost(root); host._artifacts = _ArtifactView()
    outcome = _invoke_read(host, "artifacts/real.md", "read-real")
    resource = next(effect for effect in outcome.effects if effect.kind == "resource_observed")
    assert resource.payload["resource_kind"] == "workspace_file"
    state = Slice(); state.reset("read")
    slice_sink(state)(ToolResult(
        "read_file", {"path": "artifacts/real.md"}, outcome.text, outcome.failing,
        status=outcome.status.value, invocation_id=outcome.invocation.id, outcome=outcome,
    ))
    assert state.active_files == ["artifacts/real.md"]
    assert "real workspace bytes" in build_artifacts(state, host)
    paths = physical_active_files(state, host)
    blocks = build_context_blocks(_slice_context(
        state, build_artifacts(state, host), open_file_paths=paths,
    ))
    locator = next(block for block in blocks
                   if block.item_id == "region:open_files" and block.fidelity.value == "locator")
    assert 'read_file("artifacts/real.md")' in locator.content
    assert locator.resource_refs[0].kind.value == "workspace_file"


class _Inner:
    def schemas(self):
        return []

    def accesses(self, _name, _args):
        return []

    def run(self, _name, _args):
        return ""


@check
def delegation_guidance_is_compiled_from_core_and_advanced_schemas():
    core = SubagentHost(_Inner(), llm=None, retriever=None, memory=None,
                        max_depth=1, core_mode=True)
    core_schema = next(s for s in core.schemas() if s["function"]["name"] == "spawn_agent")
    props = core_schema["function"]["parameters"]["properties"]
    assert props["agent"]["enum"] == ["explorer"]
    assert "name" not in props and "grants" not in props
    core_text = render_delegation_guidance(core.schemas())
    assert "Available agent kinds: explorer" in core_text
    assert "standing specialist" not in core_text and "grants field" not in core_text
    assert "complete normalized report directly as this tool result" in core_text
    assert "archive and evidence locators" in core_text and "not required for delivery" in core_text
    assert "ignore-aware source map" in core_text
    assert "20-30k source tokens" in core_text and "80-120 KB" in core_text
    assert "typed scope field" in core_text
    assert "scheduler owns those physical waves" in core_text
    assert "user explicitly requests a child count" in core_text
    assert "blindly reading every file in full" in core_text
    assert "coverage gaps" in core_text and "cite the sources" in core_text
    assert "work_item_id" not in core_text and "DELEGATION FAN-IN" not in core_text

    advanced = SubagentHost(_Inner(), llm=None, retriever=None, memory=None,
                            max_depth=1, core_mode=False)
    advanced_text = render_delegation_guidance(advanced.schemas())
    assert "general" in advanced_text and "standing specialist" in advanced_text
    assert "grants field" in advanced_text


@check
def memory_model_file_replaces_only_the_contract_and_allows_empty_arm():
    prior = os.environ.get("SLICEAGENT_MEMORY_MODEL_FILE")
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as stream:
            stream.write("# TEST OPERATING CONTRACT\nexact arm")
            path = stream.name
        os.environ["SLICEAGENT_MEMORY_MODEL_FILE"] = path
        assert memory_model_for_eval(MEMORY_ACCUMULATE) == "# TEST OPERATING CONTRACT\nexact arm"
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("")
        assert memory_model_for_eval(MEMORY_ACCUMULATE) == ""
    finally:
        if prior is None:
            os.environ.pop("SLICEAGENT_MEMORY_MODEL_FILE", None)
        else:
            os.environ["SLICEAGENT_MEMORY_MODEL_FILE"] = prior
        try:
            os.unlink(path)
        except (NameError, OSError):
            pass

    assert "directly obeying requested delegation/scope" in MEMORY_ACCUMULATE
    assert "No supported response-quality issue is evidenced" in MEMORY_ACCUMULATE


@check
def sealed_evidence_queries_preselect_relevant_sources_before_elasticity():
    from types import SimpleNamespace

    operation = {
        "invocation_id": "spawn-1", "name": "spawn_agent", "args": {"agent": "explorer"},
        "requested": True, "rejected_before_execution": False, "execution_started": True,
        "settled": True, "disposition": "succeeded",
    }
    artifact = SimpleNamespace(
        id="turn-review", kind="turn", timestamp="2026-07-11T00:00:00Z", task_id="task", summary="",
        structured_body={"turn_receipt": {
            "turn_id": "turn-review", "disposition": "completed", "warnings": [],
            "operations": [operation],
        }},
    )
    request = "Own up to your failures: which explorer failed and why?"
    preview = interpret_turn(request, (artifact,), task_id="task")
    state = Slice(); state.reset("Review the project")
    state.intent.current_request = request
    state.intent.turn_admission = preview.admission
    state.findings = ["unrelated assistant diagnostic note"]
    state.finding_source = {"unrelated assistant diagnostic note": "claim"}
    state.plan = [{"step": "unrelated plan", "status": "pending"}]
    state.world = {"unrelated": "state"}
    blocks = build_context_blocks(_slice_context(
        state,
        "# unrelated.py\n1: bytes that should not enter a sealed evidence query",
        discovery="unrelated related-code candidate", memory="unrelated retrieved memory",
        worktree="branch main", cache_manifest='turn 1 → read_file("history/turn-1.md")',
    ))
    names = {block.item_id for block in blocks}
    assert "region:evidence_result" in names and "region:turn_contract" in names
    assert "region:cache_manifest" in names
    assert not {
        "region:open_files", "region:related_code", "region:memory", "region:findings",
        "region:plan", "region:world", "region:worktree", "region:action_header",
    }.intersection(names)

    live = interpret_turn("Is that failure still present now?", (artifact,), task_id="task")
    state.intent.current_request = "Is that failure still present now?"
    state.intent.turn_admission = live.admission
    live_names = {block.item_id for block in build_context_blocks(_slice_context(
        state, "# unrelated.py\n1: live bytes", discovery="live candidate", worktree="branch main",
    ))}
    assert "region:open_files" in live_names and "region:worktree" in live_names, \
        "a mixed live query retains current-world sources"

    recall = interpret_turn("What did you say in your previous response?", (), task_id="task")
    assert recall.admission.evidence_query is None
    assert recall.admission.source_needs == ("prior_assistant_utterance",)
    state.intent.current_request = "What did you say in your previous response?"
    state.intent.turn_admission = recall.admission
    recall_names = {block.item_id for block in build_context_blocks(_slice_context(
        state, "# unrelated.py\n1: live bytes", discovery="unrelated code",
        memory="unrelated memory", worktree="branch main",
        cache_manifest='turn 1 → read_file("history/turn-1.md")',
    ))}
    assert "region:cache_manifest" in recall_names and "region:turn_contract" in recall_names
    assert not {
        "region:open_files", "region:related_code", "region:memory", "region:plan",
        "region:world", "region:worktree", "region:action_header",
    }.intersection(recall_names), "utterance recall should not receive unrelated roomy task furniture"


@check
def self_audit_projects_exact_paged_exchange_pairs_and_a_mandatory_quality_gate():
    from types import SimpleNamespace

    artifacts = (
        SimpleNamespace(
            id="turn-origin", kind="turn", timestamp="2026-07-11T00:00:00Z",
            task_id="task", session_id="session", status="end_turn", brief={}, summary="lossy preview",
            structured_body={
                "request": "Spawn exactly three explorers and give a three-line summary.",
                "assistant": "1. app issue\n2. auth issue\n3. util issue",
                "turn_receipt": {"turn_id": "turn-origin", "disposition": "completed", "operations": []},
            },
        ),
        SimpleNamespace(
            id="turn-filler", kind="turn", timestamp="2026-07-11T00:01:00Z",
            task_id="task", session_id="session", status="end_turn", brief={}, summary="different preview",
            structured_body={
                "request": "What model are you?", "assistant": "deepseek-chat",
                "turn_receipt": {"turn_id": "turn-filler", "disposition": "completed", "operations": []},
            },
        ),
    )
    request = "Reflect on your performance this session: what went wrong?"
    preview = interpret_turn(request, artifacts, task_id="task", session_id="session")
    assert preview.admission.quality_evidence_query is not None
    assert preview.admission.quality_evidence_query.prospective_requested is False
    coverage = next(item for item in preview.projections
                    if item.get("kind") == "quality_exchange_coverage")
    assert coverage["coverage"] == "complete" and coverage["complete_exchange_pairs"] == 2
    pairs = [item for item in preview.projections if item.get("kind") == "quality_exchange"]
    assert pairs[0]["request"] == "Spawn exactly three explorers and give a three-line summary."
    assert pairs[0]["assistant"] == "1. app issue\n2. auth issue\n3. util issue"
    assert all("lossy preview" not in str(item) for item in preview.projections)

    state = Slice(); state.reset("task")
    state.intent.current_request = request
    state.intent.turn_admission = preview.admission
    state.runtime.source_projections = preview.projections
    state.conversation = [{
        "user": "What model are you?", "assistant": "deepseek-chat", "artifact_id": "turn-filler",
    }, {"user": request, "assistant": "", "artifact_id": "turn-active"}]
    blocks = build_context_blocks(_slice_context(state, "(no files opened yet)"))
    by_id = {}
    for block in blocks:
        by_id.setdefault(block.item_id, []).append(block)
    assert "region:quality_evidence_result" in by_id
    assert "region:conversation" not in by_id, \
        "exact quality pairs replace, rather than duplicate, recent exchange bytes during the audit"
    assert all(block.mandatory for block in by_id["region:quality_evidence_result"])
    full = next(block for block in by_id["region:quality_evidence_detail"]
                if block.fidelity.value == "full")
    locator = next(block for block in by_id["region:quality_evidence_detail"]
                   if block.fidelity.value == "locator")
    assert "three-line summary" in full.content and "1. app issue" in full.content
    assert "four fields" in by_id["region:quality_evidence_result"][0].content
    assert "NOT proof that every response was correct" in by_id["region:quality_evidence_result"][0].content
    quality_protocol = by_id["region:quality_evidence_result"][0].content
    assert "source-complete audit: examine every exact request/response pair" in quality_protocol
    assert "private coverage certificate" in quality_protocol
    for stale_host_promise in (
        "host checks", "host strips", "host replaces", "before publication", "before publishing",
    ):
        assert stale_host_promise not in quality_protocol
    assert "Grounding exact: <JSON string copied verbatim" \
        in by_id["region:quality_evidence_result"][0].content
    assert "report prose alone proves only that the child said it" \
        in by_id["region:quality_evidence_result"][0].content
    assert "legacy/explicit `claims`" in by_id["region:quality_evidence_result"][0].content
    assert "redacted or truncated" in by_id["region:quality_evidence_result"][0].content
    assert "open the exact immutable turn" in locator.content


@check
def quality_projection_carries_receipt_grounding_and_freezes_its_exact_bytes():
    from types import SimpleNamespace

    auth_view = (
        "     1\tdef login(username, password):\n"
        "     2\t    stored = get_password(username)\n"
        "     3\t    return password == stored"
    )
    auth_bytes = auth_view.encode("utf-8")
    child = SimpleNamespace(
        id="subagent-grounding", kind="subagent", schema_version=1,
        workspace_id="workspace", timestamp="2026-07-11T00:00:00Z",
        task_id="task", session_id="session", parent_id="turn-grounded",
        status="ok", title="auth explorer",
        brief={"objective": "Inspect auth.py.", "scope": ["auth.py"]},
        summary="password finding", files=("auth.py",), refs=(), uncertainty=(), error="",
        structured_body={
            "brief": {"objective": "Inspect auth.py.", "scope": ["auth.py"],
                      "report_shape": "Report the top bug."},
            # Deliberately stronger than the observed bytes: the capsule must preserve this distinction.
            "report": "auth.py stores passwords in plaintext before comparing them.",
            "claims": [{
                "v": 1,
                "text": "auth.py stores passwords in plaintext before comparing them.",
                "report_exact": "auth.py stores passwords in plaintext before comparing them.",
                "modality": "inference",
                "observation_refs": [hashlib.sha256(auth_bytes).hexdigest()],
                "prerequisites": [],
            }],
            "findings": ["Passwords are stored in plaintext."],
            "coverage": "auth.py inspected", "files": ["auth.py"],
            "gaps": [], "uncertainty": [], "conflicts": [],
            "observations": [{
                "v": 1, "tool": "read_file", "args": {"path": "auth.py"},
                "status": "succeeded", "view": auth_view,
                "raw_sha256": hashlib.sha256(auth_bytes).hexdigest(),
                "view_sha256": hashlib.sha256(auth_bytes).hexdigest(),
                "raw_bytes": len(auth_bytes), "view_bytes": len(auth_bytes),
                "redacted": False, "truncated": False,
            }],
        },
    )
    operation = {
        "invocation_id": "spawn-1", "name": "spawn_agent", "args": {"agent": "explorer"},
        "requested": True, "rejected_before_execution": False, "execution_started": True,
        "settled": True, "disposition": "succeeded", "artifact_refs": [child.id],
    }
    grounded_turn = SimpleNamespace(
        id="turn-grounded", kind="turn", schema_version=1,
        workspace_id="workspace", timestamp="2026-07-11T00:01:00Z",
        task_id="task", session_id="session", parent_id="", status="end_turn", title="",
        brief={}, summary="", files=(), refs=(child.id,), uncertainty=(), error="",
        structured_body={
            "request": "Ask an explorer for the top bug in auth.py.",
            "assistant": "The explorer found that auth.py stores passwords in plaintext.",
            "turn_receipt": {
                "turn_id": "turn-grounded", "disposition": "completed", "warnings": [],
                "artifact_refs": [child.id], "operations": [operation],
            },
        },
    )
    audit_request = "Reflect on your performance this session: what went wrong?"
    leading = interpret_turn(
        audit_request, (child, grounded_turn), task_id="task", session_id="session",
    )
    coverage = next(item for item in leading.projections
                    if item.get("kind") == "quality_exchange_coverage")
    row = next(item for item in leading.projections if item.get("kind") == "quality_exchange")
    grounding = row["grounding_artifacts"][0]
    assert coverage["coverage"] == "complete"
    assert coverage["grounding_artifact_count"] == 1
    assert coverage["missing_grounding_artifact_count"] == 0
    assert row["grounding_artifact_ids"] == [child.id]
    assert grounding["artifact_id"] == child.id and grounding["artifact_kind"] == "subagent"
    assert grounding["source_text_kind"] == "subagent_grounding_v1"
    envelope = json.loads(grounding["source_text"])
    assert envelope["brief"]["scope"] == ["auth.py"]
    assert envelope["report"] == child.structured_body["report"]
    assert envelope["claims"] == child.structured_body["claims"]
    assert envelope["claims"][0]["report_exact"] in envelope["report"]
    assert envelope["observations"][0]["view"] == auth_view
    assert "plaintext" not in envelope["observations"][0]["view"].casefold(), \
        "the child report's plaintext-storage assertion is not supported by the returned auth.py bytes"
    assert grounding["observation_count"] == 1
    assert grounding["complete_observation_count"] == 1
    assert len(grounding["record_sha256"]) == 64
    assert "record" not in grounding, "the full sealed record must not be duplicated into the model slice"
    assert leading.snapshot_basis["quality_grounding_artifact_ids"] == [child.id]

    state = Slice(); state.reset("task")
    state.intent.current_request = audit_request
    state.intent.turn_admission = leading.admission
    state.runtime.source_projections = leading.projections
    blocks = build_context_blocks(_slice_context(state, "(no files opened yet)"))
    full = next(block for block in blocks
                if block.item_id == "region:quality_evidence_detail"
                and block.fidelity.value == "full")
    locator = next(block for block in blocks
                   if block.item_id == "region:quality_evidence_detail"
                   and block.fidelity.value == "locator")
    assert "Grounding source: artifacts/subagent-grounding.md" in full.content
    assert "Grounding exact source text (verbatim)" in full.content
    assert child.structured_body["report"] in full.content
    assert "stored = get_password(username)" in full.content
    assert "return password == stored" in full.content
    assert 'read_file("artifacts/subagent-grounding.md")' in locator.content

    assessment = SimpleNamespace(
        id="turn-assessment", kind="turn", schema_version=1,
        workspace_id="workspace", timestamp="2026-07-11T00:02:00Z",
        task_id="task", session_id="session", parent_id="", status="end_turn", title="",
        brief={}, summary="", files=(), refs=(), uncertainty=(), error="",
        structured_body={
            "request": audit_request,
            "assistant": "No supported response-quality issue is evidenced.",
            "turn_receipt": {
                "turn_id": "turn-assessment", "disposition": "completed", "warnings": [],
                "artifact_refs": [], "operations": [],
            },
        },
    )
    snapshot = make_evidence_snapshot(
        leading.admission, leading.projections, assessment.id,
        snapshot_basis=leading.snapshot_basis, source_generation=2,
    )
    assert snapshot["basis"]["quality_grounding_artifact_ids"] == [child.id]
    challenge = interpret_turn(
        "Is that assessment accurate? Verify it against your records.",
        (child, grounded_turn, assessment), task_id="task", session_id="session",
        previous_evidence_snapshot=snapshot, current_generation=2,
    )
    assert challenge.projections == leading.projections, \
        "adjacent verification must rebuild the identical pair and grounding projection"
    assert any(isinstance(ref, dict) and ref.get("kind") == "evidence_snapshot"
               and ref.get("status") == "frozen" for ref in challenge.admission.referents)

    tampered_child = SimpleNamespace(**{
        **vars(child),
        "structured_body": {
            **child.structured_body,
            "report": "A different report was substituted under the same artifact id.",
        },
    })
    rejected = interpret_turn(
        "Is that assessment accurate? Verify it against your records.",
        (tampered_child, grounded_turn, assessment), task_id="task", session_id="session",
        previous_evidence_snapshot=snapshot, current_generation=2,
    )
    rejected_coverage = next(item for item in rejected.projections
                             if item.get("kind") == "quality_exchange_coverage")
    assert rejected_coverage["coverage"] == "unavailable"
    assert any(isinstance(ref, dict) and ref.get("kind") == "evidence_snapshot"
               and ref.get("status") == "unavailable" for ref in rejected.admission.referents)


@check
def quality_grounding_uses_operation_provenance_not_inherited_turn_dependencies():
    from types import SimpleNamespace

    child = SimpleNamespace(
        id="subagent-source", kind="subagent", schema_version=1,
        workspace_id="workspace", timestamp="2026-07-11T00:00:00Z",
        task_id="task", session_id="session", parent_id="turn-review",
        status="ok", title="reviewer", brief={"objective": "Review app.py."},
        summary="finding", files=("app.py",), refs=(), uncertainty=(), error="",
        structured_body={"report": "app.py line 2 concatenates input into SQL."},
    )
    review = SimpleNamespace(
        id="turn-review", kind="turn", schema_version=1,
        workspace_id="workspace", timestamp="2026-07-11T00:01:00Z",
        task_id="task", session_id="session", parent_id="", status="end_turn", title="",
        brief={}, summary="", files=(), refs=(child.id,), uncertainty=(), error="",
        structured_body={
            "request": "Review app.py with one explorer.",
            "assistant": "The explorer found SQL concatenation on line 2.",
            "turn_receipt": {
                "turn_id": "turn-review", "disposition": "completed", "warnings": [],
                "artifact_refs": [child.id],
                "operations": [{
                    "invocation_id": "spawn-1", "name": "spawn_agent",
                    "args": {"agent": "explorer"}, "requested": True,
                    "rejected_before_execution": False, "execution_started": True,
                    "settled": True, "disposition": "succeeded", "artifact_refs": [child.id],
                }],
            },
        },
    )
    filler = SimpleNamespace(
        id="turn-filler", kind="turn", schema_version=1,
        workspace_id="workspace", timestamp="2026-07-11T00:02:00Z",
        task_id="task", session_id="session", parent_id="", status="end_turn", title="",
        brief={}, summary="", files=(), refs=(review.id,), uncertainty=(), error="",
        structured_body={
            "request": "What else can you help with?", "assistant": "I can help with tests.",
            "turn_receipt": {
                "turn_id": "turn-filler", "disposition": "completed", "warnings": [],
                # This is a continuity/checkpoint dependency, not evidence for the filler response.
                "artifact_refs": [review.id], "operations": [],
            },
        },
    )

    result = interpret_turn(
        "Reflect on your performance this session: what went wrong?",
        (child, review, filler), task_id="task", session_id="session",
    )
    rows = {
        item["artifact_id"]: item for item in result.projections
        if item.get("kind") == "quality_exchange"
    }
    coverage = next(item for item in result.projections
                    if item.get("kind") == "quality_exchange_coverage")
    assert rows[review.id]["grounding_artifact_ids"] == [child.id]
    assert [item["artifact_id"] for item in rows[review.id]["grounding_artifacts"]] == [child.id]
    assert rows[filler.id]["grounding_artifact_ids"] == []
    assert rows[filler.id]["grounding_artifacts"] == []
    assert coverage["grounding_artifact_count"] == 1
    assert result.snapshot_basis["quality_grounding_artifact_ids"] == [child.id]


@check
def quality_projection_labels_interrupted_visible_text_as_partial_not_final():
    from types import SimpleNamespace

    artifact = SimpleNamespace(
        id="turn-interrupted", kind="turn", timestamp="2026-07-11T00:00:00Z",
        task_id="task", session_id="session", status="interrupted", brief={}, summary="",
        structured_body={
            "request": "Explain the issue.",
            "assistant": "I started explaining before interruption.",
            "assistant_provenance": "partial_or_note",
            "turn_receipt": {"turn_id": "turn-interrupted", "disposition": "interrupted",
                             "operations": []},
        },
    )
    preview = interpret_turn(
        "Audit your response quality.", (artifact,), task_id="task", session_id="session",
    )
    coverage = next(item for item in preview.projections
                    if item.get("kind") == "quality_exchange_coverage")
    row = next(item for item in preview.projections if item.get("kind") == "quality_exchange")
    assert coverage["coverage"] == "complete" and coverage["partial_response_pairs"] == 1
    assert row["assistant_provenance"] == "partial_or_note"

    state = Slice(); state.reset("task")
    state.intent.current_request = "Audit your response quality."
    state.intent.turn_admission = preview.admission
    state.runtime.source_projections = preview.projections
    rendered = "\n".join(block.content for block in build_context_blocks(
        _slice_context(state, "(no files opened yet)"),
    ))
    assert "assistant record provenance: partial_or_note" in rendered
    assert "never describe it as a final answer" in rendered


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as error:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
