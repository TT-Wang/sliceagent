"""Offline integrity checks for the focused brain/ContextFS prompt eval fixtures."""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from evals.context_contract_eval import (CASES, audit_fixtures, messages,  # noqa: E402
                                         prepared_system, score)
from sliceagent.memory import NullMemory  # noqa: E402
from sliceagent.pfc import Slice  # noqa: E402
from sliceagent.regions import REGION_ORDER, render_cache_manifest, render_roster  # noqa: E402
from sliceagent.seed import make_build_slice  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402


def test_context_contract_eval_covers_each_load_bearing_precedence_case():
    ids = {case.id for case in CASES}
    assert ids == {
        "current_request_over_user_knowledge",
        "fresh_sensory_over_project_knowledge",
        "exact_history_over_summary",
        "contextfs_canonical_locator",
        "memory_floor_plan",
        "legacy_episode_index_is_l0_compatibility",
        "knowledge_scope_axes_overlap",
        "compatibility_and_consolidation_are_independent",
        "bounded_self_inspection_route",
        "canonical_history_over_compatibility_alias",
        "workspace_is_focus_not_prison",
        "unknown_backend_stays_unknown",
        "optional_backend_absence_does_not_disable_native_layers",
        "native_knowledge_failure_isolated_from_l0_l1",
    }
    assert audit_fixtures() == {"cases": 14, "contextfs_live": 10, "marker_count": 1}


def test_eval_uses_production_memory_seam_and_truthful_live_capability_projection():
    for case in CASES:
        system = prepared_system(case)
        assert "{{MEMORY_MODEL}}" not in system
        assert ("# LIVE INTERNAL CONTEXT CAPABILITY" in system) is case.contextfs_live
        assert ("@sliceagent/index.md" in system) is case.contextfs_live
        built = messages(case)
        assert built[0] == {"role": "system", "content": system}
        assert case.request in built[1]["content"]


def test_eval_scorers_reject_the_tempting_lower_authority_answer():
    wrong = {
        "current_request_over_user_knowledge": "Status is okay.",
        "fresh_sensory_over_project_knowledge": "8000",
        "exact_history_over_summary": "green-cicada",
        "contextfs_canonical_locator": "~/.sliceagent/memory",
        "memory_floor_plan": "tasks,sessions,roster",
        "legacy_episode_index_is_l0_compatibility": "FOURTH_LAYER",
        "knowledge_scope_axes_overlap": "UNIQUE=2",
        "compatibility_and_consolidation_are_independent": (
            "COMPATIBILITY=PENDING;CONSOLIDATION=NOT_RECORDED"
        ),
        "bounded_self_inspection_route": "~/.sliceagent/vault -> src/sliceagent/memory.py",
        "canonical_history_over_compatibility_alias": "history/turn-4.md",
        "workspace_is_focus_not_prison": "SHELL_ESCAPE_ONLY",
        "unknown_backend_stays_unknown": "semantic-backend-x",
        "optional_backend_absence_does_not_disable_native_layers": "NO",
        "native_knowledge_failure_isolated_from_l0_l1": "NO",
    }
    for case in CASES:
        assert score(case, case.expected)["passed"] == 1.0
        assert score(case, wrong[case.id])["passed"] == 0.0


def test_real_seed_advertises_contextfs_only_after_the_host_serves_its_canonical_index():
    state = Slice()
    state.reset("where are your memories?")
    host = LocalToolHost(tempfile.mkdtemp())
    try:
        schemas = host.schemas()
        assert any("@sliceagent/index.md" in str(schema) for schema in schemas)
        root_index = host.run("read_file", {"path": "@sliceagent/index.md"})
        assert "# SLICEAGENT INTERNAL CONTEXT" in root_index

        plan = make_build_slice(
            state, host, retriever=None, memory=NullMemory(), task="where are your memories?",
        )()
        assert "# LIVE INTERNAL CONTEXT CAPABILITY" in plan.system
        assert "`@sliceagent/index.md`" in plan.system
        assert "memem" not in plan.system.lower()
    finally:
        host.cleanup()


def test_slice_regions_use_canonical_unshadowed_context_locators_and_memory_authority():
    history = render_cache_manifest([
        SimpleNamespace(handle="4", preview="turn four"),
        SimpleNamespace(
            handle="…older",
            preview='2 earlier — read_file("history/index.md") for the full index',
        ),
    ])
    assert 'read_file("@sliceagent/history/turn-4.md")' in history
    assert 'read_file("@sliceagent/history/index.md")' in history
    assert 'read_file("history/' not in history

    roster = render_roster([
        {"name": f"spec-{index}", "kind": "explorer", "jobs": 0, "last_active": "2026-07-12"}
        for index in range(13)
    ])
    assert 'read_file("@sliceagent/roster/index.md")' in roster

    memory_renderer = next(row[2] for row in REGION_ORDER if row[0] == "memory")
    memory_region = memory_renderer({"memory": "candidate"})
    assert "RELEVANT KNOWLEDGE CANDIDATES" in memory_region
    assert "USER, PROJECT, CRAFT, or legacy leads" in memory_region
    assert "not current-world proof" in memory_region
