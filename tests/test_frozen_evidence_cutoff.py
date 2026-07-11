"""A verification continuation cannot silently move its evidence cutoff."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.context import ResourceKind, ResourceRef  # noqa: E402
from sliceagent.hooks import FrozenEvidenceCutoffHook  # noqa: E402
from sliceagent.registry import ToolIntentEffect  # noqa: E402


def _state(*, status="frozen", continuation=True, grounding="sealed_past",
           needs=("sealed_exchange",), reconciliation=""):
    contract = NS(
        evidence_continuation=continuation,
        grounding=grounding,
        source_needs=needs,
        referents=(
            {"kind": "evidence_snapshot", "status": status, "source_turn_id": "turn-assessment"},
            {"kind": "execution_receipt", "artifact_id": "turn-execution"},
        ),
    )
    return NS(
        intent=NS(turn_contract=contract),
        reconciliation_required=reconciliation,
        runtime=NS(source_projections=(
            {
                "kind": "quality_exchange", "artifact_id": "turn-quality",
                "grounding_artifacts": [{"artifact_id": "child-grounding"}],
            },
        )),
    )


def _effect(name, _args):
    if name == "ask_user":
        return ToolIntentEffect.DIALOGUE
    if name in {"edit_file", "change_workspace"}:
        return ToolIntentEffect.TASK_STATE
    if name == "unknown_tool":
        return ToolIntentEffect.UNKNOWN
    return ToolIntentEffect.OBSERVE


def _artifact_ref(path):
    return ResourceRef(ResourceKind.ARTIFACT, path)


def _hook(state, resource_resolver=_artifact_ref):
    return FrozenEvidenceCutoffHook(lambda: state, _effect, resource_resolver)


def test_normal_and_mixed_live_turns_are_untouched():
    assert _hook(_state(continuation=False)).authorize_tool("run_command", {"command": "ls"}).allow
    assert _hook(_state(grounding="both")).authorize_tool("read_file", {"path": "live.py"}).allow
    assert _hook(_state(needs=("sealed_exchange", "current_world"))).authorize_tool(
        "web_search", {"query": "now"},
    ).allow


def test_pure_frozen_challenge_allows_only_exact_materialized_artifacts():
    hook = _hook(_state())
    for path in (
        "artifacts/turn-assessment.md",
        "./artifacts/turn-execution.md",
        "artifacts/turn-quality.md",
        "artifacts/child-grounding.md",
    ):
        assert hook.authorize_tool("read_file", {"path": path}).allow, path
    for name, args in (
        ("read_file", {"path": "artifacts/index.md"}),
        ("read_file", {"path": "artifacts/newer-turn.md"}),
        ("read_file", {"path": "src/live.py"}),
        ("list_files", {"path": "."}),
        ("grep", {"query": "claim"}),
        ("search_history", {"query": "claim"}),
        ("run_command", {"command": "git status"}),
        ("web_search", {"query": "claim"}),
        ("plugin_lookup", {"query": "claim"}),
    ):
        decision = hook.authorize_tool(name, args)
        assert not decision.allow and not decision.counts_as_stuck, (name, args, decision)
        assert "frozen_evidence_cutoff" in decision.reason


def test_dialogue_and_non_observation_effects_remain_owned_by_their_normal_gates():
    hook = _hook(_state())
    assert hook.authorize_tool("ask_user", {"question": "Which claim?"}).allow
    assert hook.authorize_tool("edit_file", {"path": "x.py"}).allow
    assert hook.authorize_tool("unknown_tool", {}).allow


def test_unavailable_snapshot_allows_no_observation_reads():
    hook = _hook(_state(status="unavailable"))
    denied = hook.authorize_tool("read_file", {"path": "artifacts/turn-assessment.md"})
    assert not denied.allow and "no artifact reads are available" in denied.reason
    assert hook.authorize_tool("ask_user", {"question": "Can you provide the record?"}).allow


def test_real_workspace_shadow_cannot_masquerade_as_frozen_artifact():
    def shadow(path):
        return ResourceRef(ResourceKind.WORKSPACE_FILE, path)

    decision = _hook(_state(), shadow).authorize_tool(
        "read_file", {"path": "artifacts/turn-quality.md"},
    )
    assert not decision.allow


def test_active_reconciliation_bypasses_cutoff_to_avoid_recovery_deadlock():
    hook = _hook(_state(reconciliation="late command may still be running"))
    assert hook.authorize_tool("read_file", {"path": "src/live.py"}).allow
    assert hook.authorize_tool("run_command", {"command": "git status"}).allow


def main():
    checks = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    failed = 0
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {check.__name__}: {error!r}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
