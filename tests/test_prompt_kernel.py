"""Stable operating-kernel contract: compact, source-linked, and evidence-typed."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.prompt import (MEMORY_ACCUMULATE, SYSTEM_PROMPT,  # noqa: E402
                               render_contextfs_guidance,
                               render_delegation_guidance)


def test_kernel_keeps_exact_request_above_derived_state():
    low = SYSTEM_PROMPT.lower()
    assert "current request is the user's exact text" in low
    assert "the exact request wins" in low
    assert "typed active-work delta" in low
    assert "working state, not a second user and not an autobiography" in low


def test_context_is_dependency_first_and_elastic_without_transcript_growth():
    contract = MEMORY_ACCUMULATE.lower()
    assert "context selection happens before elasticity" in contract
    assert "dependency closure" in contract
    assert "not an accumulating transcript" in contract
    assert "recent conversation" not in SYSTEM_PROMPT.lower()
    assert "last few user<->assistant exchanges" not in SYSTEM_PROMPT.lower()


def test_proof_families_are_not_interchangeable():
    contract = MEMORY_ACCUMULATE.lower()
    assert "fresh observations" in contract and "current world state" in contract
    assert "execution receipts prove only" in contract
    assert "response artifacts" in contract and "what text was delivered" in contract
    assert "child artifacts are attributed testimony" in contract
    assert "never use one proof family as another" in contract


def test_workspace_switch_continues_one_logical_request():
    low = SYSTEM_PROMPT.lower()
    assert "default focus for relative paths and project scope, not a prison" in low
    assert "explicit user targets and host focus roots" in low
    assert "workspace transition continues the same logical request" in low
    assert "do not demand a synthetic `go`" in low


def test_brain_regions_have_distinct_authority_and_memory_never_outranks_now():
    prompt = (SYSTEM_PROMPT + MEMORY_ACCUMULATE).lower()
    assert "sensory cortex" in prompt and "fresh derived view of the live world" in prompt
    assert "history / hippocampus" in prompt and "canonical evidence of what happened" in prompt
    assert "pfc / active work" in prompt and "open commitments" in prompt
    assert "knowledge" in prompt and "user, project, and craft leads" in prompt
    assert "current request and fresh world observations" in prompt
    assert "outrank every memory or knowledge record" in prompt


def _file_schema(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_contextfs_contract_is_compiled_only_from_a_live_advertised_schema():
    ordinary = [_file_schema("read_file", "Read a file in the current workspace.")]
    assert render_contextfs_guidance(ordinary) == ""
    assert "@sliceagent" not in render_delegation_guidance(ordinary)

    live = [_file_schema(
        "read_file",
        "Read a file. Internal context is rooted at @sliceagent/index.md.",
    )]
    guidance = render_contextfs_guidance(live)
    assert "LIVE INTERNAL CONTEXT CAPABILITY" in guidance
    assert "`@sliceagent/index.md`" in guidance
    assert "`@sliceagent/history/`" in guidance
    assert "`@sliceagent/work/`" in guidance
    assert "`@sliceagent/memory/`" in guidance
    assert "`@sliceagent/memory/status.md`" in guidance
    assert "mounted views, not physical workspace paths" in guidance
    assert "canonical self-description" in guidance
    assert "read only the relevant region or status page" in guidance
    assert "stop when the answer is grounded" in guidance
    assert "unless the exact current request explicitly asks to debug the implementation" in guidance
    assert "exactly three layers" in guidance
    assert "episode search indexes are l0 compatibility/discovery surfaces" in guidance.lower()
    assert "retrieval backends, roster, and skills are capabilities, not memory layers" in guidance
    assert "`artifacts/` or `roster/` are compatibility aliases" in guidance
    assert "bare `history/` locator is instead a legacy episodic mirror" in guidance
    assert "Prefer `@sliceagent/`" in guidance
    assert "retrieval backend" in guidance and "memem" not in guidance.lower()
    assert render_delegation_guidance(live) == guidance


def test_kernel_is_cacheable_and_materially_compact():
    assert SYSTEM_PROMPT.count("{{MEMORY_MODEL}}") == 1
    rendered = SYSTEM_PROMPT.replace("{{MEMORY_MODEL}}", MEMORY_ACCUMULATE)
    assert len(rendered) < 12_000, "stable operating kernel regrew into a policy encyclopedia"
