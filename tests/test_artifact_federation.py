from __future__ import annotations

import tempfile

from sliceagent.persistence import Artifact, ArtifactStore
from sliceagent.memory import NullMemory
from sliceagent.retriever import NullRetriever
from sliceagent.runtime_persistence import CoreArtifactFS
from sliceagent.subagent import SubagentHost
from sliceagent.tools import LocalToolHost


def test_exact_artifact_handle_faults_across_workspaces_without_listing_the_archive():
    archive = tempfile.mkdtemp(prefix="artifact-federation-")
    source = ArtifactStore(f"{archive}/workspace-a")
    target = ArtifactStore(f"{archive}/workspace-b")
    source.put(Artifact(
        id="turn-source-receipt", kind="turn", workspace_id="workspace-a",
        session_id="session", task_id="task", title="Source receipt",
        brief={"request": "switch and inspect"},
        structured_body={"assistant": "source result", "markdown": "exact source evidence"},
    ))

    virtual = CoreArtifactFS(target, archive_root=archive)
    rendered = virtual.read_file("artifacts/turn-source-receipt.md")
    assert "exact source evidence" in rendered
    assert "turn-source-receipt.md" not in virtual.index()
    assert "turn-source-receipt.md" not in virtual.listing()


def test_federation_is_disabled_without_an_explicit_archive_root():
    archive = tempfile.mkdtemp(prefix="artifact-local-only-")
    source = ArtifactStore(f"{archive}/workspace-a")
    target = ArtifactStore(f"{archive}/workspace-b")
    source.put(Artifact(
        id="child-other-workspace", kind="subagent", workspace_id="workspace-a",
        session_id="session", task_id="task", structured_body={"markdown": "child proof"},
    ))
    assert "no such retained artifact" in CoreArtifactFS(target).read_file(
        "artifacts/child-other-workspace.md",
    )


def test_federated_subagent_report_is_grantable_by_its_exact_readable_handle():
    archive = tempfile.mkdtemp(prefix="artifact-federated-grant-")
    source = ArtifactStore(f"{archive}/workspace-a")
    target = ArtifactStore(f"{archive}/workspace-b")
    source.put(Artifact(
        id="subagent-before-switch", kind="subagent", workspace_id="workspace-a",
        session_id="session", task_id="task", structured_body={"report": "sealed report"},
    ))
    tools = LocalToolHost(tempfile.mkdtemp(prefix="workspace-b-files-"))
    tools._artifacts = CoreArtifactFS(target, archive_root=archive)
    host = SubagentHost(
        tools, llm=None, retriever=NullRetriever(), memory=NullMemory(),
        max_depth=1, artifact_store=target,
    )
    handle = "artifacts/subagent-before-switch.md"
    error, grants = host._validate_grants([handle])
    assert error == "" and grants == frozenset({handle})

    source.put(Artifact(
        id="turn-before-switch", kind="turn", workspace_id="workspace-a",
        session_id="session", task_id="task", structured_body={"assistant": "not a child"},
    ))
    wrong_error, wrong_grants = host._validate_grants(["artifacts/turn-before-switch.md"])
    assert "cannot grant" in wrong_error and wrong_grants == frozenset()
    tools.cleanup()


def test_core_artifact_renderer_never_doubles_an_already_rendered_reference_handle():
    artifact = Artifact(
        id="subagent-synthesis", kind="subagent", workspace_id="workspace",
        session_id="session", task_id="task", refs=("artifacts/subagent-source.md",),
        structured_body={"report": "synthesis"},
    )
    rendered = CoreArtifactFS._render(artifact)
    assert 'read_file("artifacts/subagent-source.md")' in rendered
    assert "artifacts/artifacts/" not in rendered and ".md.md" not in rendered
