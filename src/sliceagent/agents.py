"""Named-agent registry — file-defined subagent KINDS.

sliceagent's subagents were two HARDCODED kinds (read-only explorer + writable). The kinds are now a
pluggable REGISTRY: each agent is a {name, description, tools-allowlist, reasoning, system-prompt}
definition, discovered from `<root>/agents/*.md` (markdown + frontmatter — sliceagent's own SKILL.md idiom),
and the model spawns one BY NAME via the generic `spawn_agent` tool. Built-ins (explorer, general) ship
in-tree; user files add or override by name.

Periphery, NOT the moat: a spawned agent still runs the bounded slice loop and returns only a summary.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

# An EXPLORER's read-only surface — the single source of truth (subagent.py imports this).
# `grep` (find by CONTENT) + `glob` (find by NAME) are the two discovery tools; both read-only.
READ_ONLY_TOOLS = ("read_file", "list_files", "grep", "glob", "skill", "search_history", "code_review")
_READ_ONLY_SET = frozenset(READ_ONLY_TOOLS)   # mutability is decided against this KNOWN-safe set (pessimistic)

# Tools NO subagent may use, regardless of its allowlist. A
# subagent must not stop to ask the END-USER — ambiguity is the parent's job; a child that blocks on input
# is a stall (and racy/meaningless when several run in parallel). It returns its summary instead.
SUBAGENT_EXCLUDED_TOOLS = frozenset({"ask_user"})

# Mutating tools — an agent whose allowlist includes ANY of these is "writable" (globally serialized vs
# other writers); an allowlist with none of them is read-only (parallelizes as a swarm).
WRITE_TOOLS = frozenset({
    "edit_file", "append_to_file", "str_replace", "run_command", "execute_code",
    "world_set", "world_clear", "require", "drop_requirement", "requirement_done", "update_plan",
    "terminal_open", "terminal_send", "terminal_read", "terminal_wait", "terminal_close",
    "proc_start", "proc_poll", "proc_tail", "proc_wait", "proc_kill",
    "spawn_subagent", "spawn_explore", "spawn_agent",
})


@dataclass(frozen=True)
class AgentSpec:
    """One subagent KIND. `tools=None` → inherit the parent's FULL tool surface (a 'general' agent)."""
    name: str
    description: str = ""
    tools: tuple[str, ...] | None = None   # allowlist of tool names the child may use (None = all)
    reasoning: str | None = None           # "fast" | "full" | None (inherit the parent's)
    system_prompt: str = ""                # extra system-prompt layer prepended for the child
    summary_is_deliverable: bool = False   # the child's SUMMARY is the product (a trailing failing check is
    #                                        intentional, not a crash) — like a verifier that ends on a FAIL.

    @property
    def read_only(self) -> bool:
        """A child is read-only iff EVERY tool in its allowlist is a KNOWN read-only tool. Pessimistic by
        design: an unknown / plugin / MCP tool is NOT assumed safe (the old check only excluded the static
        WRITE_TOOLS names, so a side-effecting plugin tool was mis-classified read-only and could be
        scheduled as a parallel non-writer). None (full surface) is writable."""
        return self.tools is not None and set(self.tools).issubset(_READ_ONLY_SET)


BUILTIN_AGENTS: dict[str, AgentSpec] = {
    "explorer": AgentSpec(
        name="explorer",
        description="Read-only investigation — find files, trace usages, understand code; returns a summary. "
                    "Fan out several in one turn for breadth.",
        tools=READ_ONLY_TOOLS, reasoning="fast",
        system_prompt="You are a read-only EXPLORER subagent: investigate the task by reading/grepping and "
                      "return a concise summary of what you found (files, locations, conclusions). You cannot "
                      "modify anything — do not attempt edits or commands.",
    ),
    "general": AgentSpec(
        name="general",
        description="A full sub-agent for ONE self-contained sub-task (can read AND edit/run); returns a summary.",
        tools=None, reasoning=None,
        system_prompt="You are a SUBAGENT handling one self-contained sub-task in the shared workspace. Do the "
                      "work, then return a concise summary of what you changed and verified. Do NOT ask the "
                      "user; if the task is ambiguous, make the best reasonable choice and note it in the summary.",
    ),
    # The REDUCE side of fan-out — and deliberately NOT special machinery: a synthesiser is just a
    # read-only child whose brief GRANTS it the sibling reports to merge (spawn_agent with
    # grants=[all N handles]). It pages them ONE AT A TIME through its own bounded slice (peak O(1) in N)
    # and seals a synthesis whose refs are those handles — so the reduce is bounded AND lossless: any
    # detail the synthesis dropped stays one read_file away.
    "synthesiser": AgentSpec(
        name="synthesiser",
        description="REDUCE a fan-out: merges the sealed reports you grant it (grants=[...]) into ONE "
                    "synthesis with per-claim citations, surfacing conflicts and coverage gaps. Spawn after "
                    "several explorers finish, granting it their handles.",
        tools=READ_ONLY_TOOLS, reasoning="full",
        summary_is_deliverable=True,   # its summary IS the synthesis
        system_prompt=(
            "You are a SYNTHESISER subagent: your job is to REDUCE several sealed sibling reports into one "
            "coherent synthesis WITHOUT laundering away detail or disagreement.\n"
            "Method: read the granted INPUT REPORTS one at a time (each is a read_file call from your task); "
            "extract each report's claims/findings; then merge.\n"
            "Rules:\n"
            "- CITE every merged claim to its source handle, e.g. (subagents/sub-2.md) — a claim you cannot "
            "cite does not go in the synthesis.\n"
            "- CONFLICTS between reports are FINDINGS, not noise: surface them explicitly ('sub-1 says X; "
            "sub-3 says Y') rather than picking a side silently.\n"
            "- COVERAGE GAPS are part of the synthesis: state what none of the inputs examined.\n"
            "- Do NOT re-investigate the codebase yourself beyond spot-checking a citation; your input is "
            "the reports. If they are insufficient, say exactly what is missing.\n"
            "Deliver: a structured synthesis (merged findings with citations · conflicts · gaps · "
            "recommendation if asked). Do NOT ask the user."
        ),
    ),
    # An independent ADVERSARIAL verifier. Runs in a FRESH
    # slice and returns only a VERDICT + evidence — so it complements the parent's structural done-gates
    # (OracleHook/SelfCheckHook) with a second, skeptical opinion WITHOUT any context crossing the seal.
    # Read-only EXCEPT running checks: read/grep + run_command/execute_code (to build/test/probe), no edit
    # tools (the allowlist is enforced at runtime in subagent.py). It is "writable" by classification (shell
    # is in WRITE_TOOLS) so it serializes vs other writers — correct for a verifier that runs tests.
    "verification": AgentSpec(
        name="verification",
        description="Independent adversarial VERIFIER — given a change/claim, TRY TO BREAK IT (reproduce, run "
                    "build/tests, probe edges) and return VERDICT: PASS/FAIL/PARTIAL with command evidence. "
                    "Read-only except running checks. Spawn after a non-trivial change, before reporting done.",
        tools=READ_ONLY_TOOLS + ("run_command", "execute_code"),
        reasoning="full",
        summary_is_deliverable=True,   # a FAIL verdict normally ends on a failing check — that's the product,
        #                                not a crash; don't reclassify it as "did not finish cleanly".
        system_prompt=(
            "You are an independent VERIFICATION subagent. Your job is NOT to confirm the work is done — it is "
            "to TRY TO BREAK IT. You are given a task/claim and the change that was made; verify it "
            "INDEPENDENTLY and decide.\n"
            "Avoid two failure modes: (1) verification AVOIDANCE — reading code and narrating what you WOULD "
            "test, then writing PASS. Reading is NOT verification; RUN it. (2) being seduced by the first 80% "
            "— a passing test suite or the happy path is not proof; your value is the last 20%.\n"
            "DO NOT MODIFY THE PROJECT: no editing/creating/deleting project files, no installing deps, no git "
            "writes. You MAY write EPHEMERAL probe scripts to a temp dir WHERE THE SANDBOX ALLOWS (e.g. $TMPDIR "
            "or /tmp, via run_command/execute_code) and clean up after yourself.\n"
            "Method: REPRODUCE the original issue/scenario; run the cheapest sufficient build/test; then RUN at "
            "least ONE adversarial probe — a boundary/empty/large input, idempotency, the EXACT property the "
            "task names, or a related path that could regress. The implementer is also an LLM, so its tests may "
            "be happy-path — verify end-to-end yourself.\n"
            "Before PASS: you must have RUN at least one adversarial probe and observed its real output. Before "
            "FAIL: check the issue isn't already handled elsewhere or intentional.\n"
            "Format every check as — Check: <what> / Command: <exact> / Output: <actual observed, not "
            "paraphrased> / PASS or FAIL. A check with no command output is a SKIP, not a PASS. END your "
            "summary with EXACTLY one line: 'VERDICT: PASS' or 'VERDICT: FAIL' or 'VERDICT: PARTIAL' (PARTIAL "
            "only for environment limits — missing tool/deps/can't run — never for 'unsure'). Do NOT ask the user."
        ),
    ),
}


def _parse_agent_md(path: str) -> AgentSpec | None:
    """Parse an agent file: optional `---` frontmatter (name/description/tools/reasoning) + body = system
    prompt. Never raises — a malformed/unreadable file is skipped (returns None)."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end == -1:
            # opening fence but no closing one (authoring typo). FAIL CLOSED: don't fall through to the
            # no-frontmatter path, which would leave tools=None (= full writable surface) for a file that
            # was trying to declare a restrictive tool list. Skip it, per the "malformed → skipped" contract.
            return None
        if end != -1:
            for line in text[3:end].splitlines():
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
            body = text[end + 4:].lstrip("\n")
    name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
    if not name:
        return None
    tools_raw = meta.get("tools")
    # #58: accept both the scalar list `tools: a, b` AND inline YAML `tools: [a, b]` — strip brackets/quotes
    # before splitting so a bracketed value doesn't become tool names like "[a".
    # A PRESENT-but-blank `tools:` means restrict to ZERO tools (read-only, matching `tools: []`); only an
    # ABSENT key grants the full writable surface (None).
    if "tools" in meta and not str(tools_raw or "").strip():
        tools = ()
    elif tools_raw:
        tools = tuple(t for t in tools_raw.replace(",", " ").replace("[", " ").replace("]", " ")
                      .replace("'", " ").replace('"', " ").split() if t)
    else:
        tools = None
    reasoning = (meta.get("reasoning") or "").lower() or None
    return AgentSpec(name=name, description=meta.get("description", ""),
                     tools=tools, reasoning=reasoning, system_prompt=body.strip())


def load_agents(roots) -> dict[str, AgentSpec]:
    """Built-in agents overlaid with user-defined `<root>/agents/*.md` (later roots / user files win by
    name). `roots` are dirs that MAY contain an `agents/` subdir."""
    out = dict(BUILTIN_AGENTS)
    for root in roots or []:
        adir = os.path.join(root, "agents")
        if not os.path.isdir(adir):
            continue
        for path in sorted(glob.glob(os.path.join(adir, "*.md"))):
            spec = _parse_agent_md(path)
            if spec:
                out[spec.name] = spec
    return out
