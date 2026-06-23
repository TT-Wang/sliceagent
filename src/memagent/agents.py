"""Named-agent registry — file-defined subagent KINDS (borrowed structure from Kimi Code / Claude Code).

memagent's subagents were two HARDCODED kinds (read-only explorer + writable). Kimi/Claude make the
kinds a pluggable REGISTRY: each agent is a {name, description, tools-allowlist, reasoning, system-prompt}
definition, discovered from `<root>/agents/*.md` (markdown + frontmatter — memagent's own SKILL.md idiom),
and the model spawns one BY NAME via the generic `spawn_agent` tool. Built-ins (explorer, general) ship
in-tree; user files add or override by name.

Periphery, NOT the moat: a spawned agent still runs the bounded slice loop and returns only a summary.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

# An EXPLORER's read-only surface — the single source of truth (subagent.py imports this).
READ_ONLY_TOOLS = ("read_file", "list_files", "grep", "glob", "skill", "recall_history")
_READ_ONLY_SET = frozenset(READ_ONLY_TOOLS)   # mutability is decided against this KNOWN-safe set (pessimistic)

# Tools NO subagent may use, regardless of its allowlist (mirrors Kimi's SUBAGENT_EXCLUDED_TOOLS). A
# subagent must not stop to ask the END-USER — ambiguity is the parent's job; a child that blocks on input
# is a stall (and racy/meaningless when several run in parallel). It returns its summary instead.
SUBAGENT_EXCLUDED_TOOLS = frozenset({"ask_user"})

# Mutating tools — an agent whose allowlist includes ANY of these is "writable" (globally serialized vs
# other writers); an allowlist with none of them is read-only (parallelizes as a swarm).
WRITE_TOOLS = frozenset({
    "edit_file", "append_to_file", "str_replace", "run_command", "execute_code",
    "world_set", "world_clear", "require", "drop_requirement", "requirement_done", "update_plan",
    "set_mission", "mission_done",
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
    # An independent ADVERSARIAL verifier (borrowed from Claude Code's verification agent). Runs in a FRESH
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
        system_prompt=(
            "You are an independent VERIFICATION subagent. Your job is NOT to confirm the work is done — it is "
            "to TRY TO BREAK IT. You are given a task/claim and the change that was made; verify it "
            "INDEPENDENTLY and decide.\n"
            "Avoid two failure modes: (1) verification AVOIDANCE — reading code and narrating what you WOULD "
            "test, then writing PASS. Reading is NOT verification; RUN it. (2) being seduced by the first 80% "
            "— a passing test suite or the happy path is not proof; your value is the last 20%.\n"
            "DO NOT MODIFY THE PROJECT: no editing/creating/deleting project files, no installing deps, no git "
            "writes. You MAY write EPHEMERAL probe scripts under /tmp (via run_command/execute_code) and clean "
            "up after yourself.\n"
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
    tools = (tuple(t for t in tools_raw.replace(",", " ").replace("[", " ").replace("]", " ")
                   .replace("'", " ").replace('"', " ").split() if t)
             if tools_raw else None)
    reasoning = (meta.get("reasoning") or "").lower() or None
    return AgentSpec(name=name, description=meta.get("description", ""),
                     tools=tools, reasoning=reasoning, system_prompt=body.strip())


def load_agents(roots) -> dict[str, AgentSpec]:
    """Built-in agents overlaid with user-defined `<root>/agents/*.md` (later roots / user files win by
    name). `roots` are dirs that MAY contain an `agents/` subdir (mirrors Kimi's SUBAGENTS_DIRECTORY)."""
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
