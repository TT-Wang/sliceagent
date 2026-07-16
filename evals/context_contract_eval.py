"""Focused prompt eval for SliceAgent's brain, ContextFS, and reach authority contract.

The fixtures deliberately present one authoritative source and one tempting conflicting source.  They
exercise the production ``{{MEMORY_MODEL}}`` seam and the live-schema ContextFS compiler without needing a
workspace or durable store.  ``--dry-run`` validates and prints the fixtures without making a model call.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sliceagent.prompt import (MEMORY_ACCUMULATE, SYSTEM_PROMPT,  # noqa: E402
                               render_contextfs_guidance)


@dataclass(frozen=True)
class ContractCase:
    id: str
    context: str
    request: str
    expected: str
    contextfs_live: bool = False


CASES = (
    ContractCase(
        id="current_request_over_user_knowledge",
        context=(
            "KNOWLEDGE / USER: standing preference says to answer status questions in prose.\n"
            "PFC / ACTIVE WORK: no output format has been selected."
        ),
        request='Return only this JSON object: {"status":"ok"}',
        expected='{"status":"ok"}',
    ),
    ContractCase(
        id="fresh_sensory_over_project_knowledge",
        context=(
            "KNOWLEDGE / PROJECT: an older sourced record says server_port=8000.\n"
            "SENSORY CORTEX / OPEN FILES (fresh): config.toml currently contains server_port=9100."
        ),
        request="Return only the current server port as digits.",
        expected="9100",
    ),
    ContractCase(
        id="exact_history_over_summary",
        context=(
            "KNOWLEDGE summary: the earlier response probably used token green-cicada.\n"
            "HISTORY / HIPPOCAMPUS sealed response artifact: exact response bytes are blue-cicada-47."
        ),
        request="Return only the exact token used in the earlier response.",
        expected="blue-cicada-47",
    ),
    ContractCase(
        id="contextfs_canonical_locator",
        context="The user asks where SliceAgent can inspect its internal context in this runtime.",
        request="Return only the canonical root index locator.",
        expected="@sliceagent/index.md",
        contextfs_live=True,
    ),
    ContractCase(
        id="memory_floor_plan",
        context="The user asks for SliceAgent's three model-facing memory layers, with no filesystem expedition.",
        request="Return exactly `L0=HISTORY;L1=WORK;L2=KNOWLEDGE` with no other text.",
        expected="L0=HISTORY;L1=WORK;L2=KNOWLEDGE",
        contextfs_live=True,
    ),
    ContractCase(
        id="legacy_episode_index_is_l0_compatibility",
        context=(
            "The legacy episode search index discovers episodic records. Native typed USER/PROJECT/CRAFT "
            "knowledge is exposed separately under the canonical memory region."
        ),
        request=(
            "Classify the legacy episode search index. Return exactly `L0_COMPATIBILITY`, `L2_KNOWLEDGE`, "
            "or `FOURTH_LAYER`."
        ),
        expected="L0_COMPATIBILITY",
        contextfs_live=True,
    ),
    ContractCase(
        id="knowledge_scope_axes_overlap",
        context=(
            "@sliceagent/memory/status.md reports one unique active L2 record. That same record carries both "
            "PROJECT and CRAFT scope, so the axis counts are PROJECT=1 and CRAFT=1."
        ),
        request="Return exactly `UNIQUE=1`; do not add the overlapping scope-axis counts.",
        expected="UNIQUE=1",
        contextfs_live=True,
    ),
    ContractCase(
        id="compatibility_and_consolidation_are_independent",
        context=(
            "Global compatibility-layout state is retained. Separately, current-project consolidation status "
            "is historical-output-present because one typed record cites legacy episodic input; exact old run "
            "lifecycle metadata is unavailable. Compatibility counts are not an L2 backlog."
        ),
        request=(
            "Return exactly `COMPATIBILITY=RETAINED;CONSOLIDATION=HISTORICAL_OUTPUT_PRESENT` with no other text."
        ),
        expected="COMPATIBILITY=RETAINED;CONSOLIDATION=HISTORICAL_OUTPUT_PRESENT",
        contextfs_live=True,
    ),
    ContractCase(
        id="bounded_self_inspection_route",
        context=(
            "The user asks for SliceAgent's current memory status. The canonical root index and memory status "
            "page are available. Implementation modules and the private ~/.sliceagent store also exist."
        ),
        request=(
            "Return only the bounded inspection route as two locators joined by ` -> `; do not return a raw "
            "physical path or implementation source path."
        ),
        expected="@sliceagent/index.md -> @sliceagent/memory/status.md",
        contextfs_live=True,
    ),
    ContractCase(
        id="canonical_history_over_compatibility_alias",
        context=(
            "A legacy episodic mirror emitted history/turn-4.md. Its ordinal is not proof that the canonical "
            "artifact history uses the same number; the live ContextFS history index is available."
        ),
        request="Return only the canonical entry point to locate and verify the corresponding history record.",
        expected="@sliceagent/history/index.md",
        contextfs_live=True,
    ),
    ContractCase(
        id="workspace_is_focus_not_prison",
        context=(
            "PRIMARY WORKSPACE: /projects/alpha.\n"
            "LIVE FOCUS ROOT: /projects/hunter.\n"
            "The offered read_file schema explicitly admits /projects/hunter/reference.toml."
        ),
        request=(
            "The user named /projects/hunter/reference.toml. Return only USE_FILE_TOOLS if the live file tools "
            "should inspect it; otherwise return SHELL_ESCAPE_ONLY."
        ),
        expected="USE_FILE_TOOLS",
    ),
    ContractCase(
        id="unknown_backend_stays_unknown",
        context=(
            "@sliceagent/index.md reports: evidence available; work available; knowledge available; "
            "optional semantic retrieval backend status not reported."
        ),
        request="Return only UNKNOWN if no optional semantic retrieval backend is established by this context.",
        expected="UNKNOWN",
        contextfs_live=True,
    ),
    ContractCase(
        id="optional_backend_absence_does_not_disable_native_layers",
        context=(
            "@sliceagent/index.md reports: evidence available; history available; work available; "
            "native knowledge index healthy; optional semantic retrieval backend disabled."
        ),
        request="Return only YES if exact history, Active Work, and native knowledge remain available.",
        expected="YES",
        contextfs_live=True,
    ),
    ContractCase(
        id="native_knowledge_failure_isolated_from_l0_l1",
        context=(
            "@sliceagent/index.md reports: evidence available; history available; work available; "
            "native knowledge degraded due to an observed database error."
        ),
        request="Return only YES if exact L0 history and L1 Active Work must remain usable.",
        expected="YES",
        contextfs_live=True,
    ),
)


def _contextfs_schema() -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file or internal context beginning at @sliceagent/index.md.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }]


def prepared_system(case: ContractCase) -> str:
    """Build the same static contract splice plus the relevant live-capability projection."""
    if SYSTEM_PROMPT.count("{{MEMORY_MODEL}}") != 1:
        raise ValueError("SYSTEM_PROMPT must contain exactly one {{MEMORY_MODEL}} splice")
    system = SYSTEM_PROMPT.replace("{{MEMORY_MODEL}}", MEMORY_ACCUMULATE)
    schemas = _contextfs_schema() if case.contextfs_live else ()
    return system + render_contextfs_guidance(schemas)


def messages(case: ContractCase) -> list[dict]:
    user = (
        "# COMPILED CONTEXT EVAL FIXTURE\n"
        f"{case.context}\n\n"
        "# CURRENT REQUEST (exact and authoritative)\n"
        f"{case.request}"
    )
    return [{"role": "system", "content": prepared_system(case)},
            {"role": "user", "content": user}]


def normalize_answer(answer: str) -> str:
    text = str(answer or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


def score(case: ContractCase, answer: str) -> dict:
    actual = normalize_answer(answer)
    return {"item": case.id, "passed": float(actual == case.expected),
            "expected": case.expected, "actual": actual}


def audit_fixtures() -> dict:
    ids = [case.id for case in CASES]
    if len(ids) != len(set(ids)):
        raise ValueError("context-contract eval case IDs must be unique")
    for case in CASES:
        system = prepared_system(case)
        advertised = "@sliceagent/index.md" in system
        if advertised != case.contextfs_live:
            raise ValueError(f"{case.id}: ContextFS capability projection is not truthful")
        if score(case, case.expected)["passed"] != 1.0:
            raise ValueError(f"{case.id}: expected answer does not satisfy its scorer")
    return {"cases": len(CASES), "contextfs_live": sum(case.contextfs_live for case in CASES),
            "marker_count": SYSTEM_PROMPT.count("{{MEMORY_MODEL}}")}


def run(model: str) -> dict:
    from sliceagent.config import Config
    from sliceagent.llm import OpenAILLM

    audit = audit_fixtures()
    # Match the real CLI's resolved provider rather than assuming credentials
    # were exported into the shell. ``sliceagent init`` normally stores them in
    # the private user config, and an eval that ignores that config gives a
    # misleading "missing credentials" result against a working installation.
    config = Config.load()
    llm = OpenAILLM(
        model=model, api_key=config.api_key or None,
        base_url=config.base_url or None, timeout=90.0,
    )
    rows = []
    for case in CASES:
        response = llm.complete(messages(case), [])
        rows.append(score(case, response.content or ""))
    return {"model": model, "audit": audit, "passed": sum(row["passed"] for row in rows),
            "total": len(rows), "items": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL", ""))
    parser.add_argument("--out", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        result = {"audit": audit_fixtures(), "items": [case.id for case in CASES]}
    else:
        if not args.model:
            parser.error("--model (or AGENT_MODEL) is required unless --dry-run is used")
        result = run(args.model)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
