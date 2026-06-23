"""Permission policy — authorization for tool calls (borrowed pattern: Kimi permission/policies).

An ordered chain of small policies. Each inspects (name, args) and returns a
ToolDecision to DENY, or None to abstain (defer to the next policy). First denial
wins; if every policy abstains, the call is allowed. The chain is a plain callable,
so it drops straight into hooks.PermissionHook(policy) — the loop and the moat are
untouched.

This is AUTHORIZATION (what's allowed at all). It's distinct from SAFE EXECUTION
(workspace path confinement + the sandbox), which lives in tools.py / sandbox.py.
Defense in depth: the policy denies catastrophic commands early with a clear reason;
the ToolHost still confines paths even if a policy abstains.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .agents import READ_ONLY_TOOLS   # single source of truth for the known read-only surface
from .hooks import ToolDecision

ALLOW = ToolDecision(True)

WRITE_TOOLS = frozenset(("edit_file", "append_to_file", "str_replace"))
EXEC_TOOLS = frozenset(("run_command", "execute_code"))

# Patterns that are almost never legitimate inside a coding-agent workspace.
# Kept deliberately narrow so normal dev commands (pytest, pip, npm, git add/commit,
# rm of a workspace file, mkdir, mv) pass untouched.
_DANGEROUS: list[tuple[re.Pattern, str]] = [
    (re.compile(r":\s*\(\s*\)\s*\{.*\|.*&", re.S),        "fork bomb"),
    (re.compile(r"\bsudo\b"),                              "privilege escalation (sudo)"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),  "system power control"),
    (re.compile(r"\b(mkfs|wipefs)\b"),                     "filesystem format"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/"),               "raw write to a device"),
    (re.compile(r">\s*/dev/(sd|nvme|disk|hd)"),           "raw write to a device"),
    # rm -rf targeting / ~ $HOME /* .. (workspace-relative rm is fine)
    (re.compile(r"\brm\b[^|;&\n]*\s-[a-z]*(?:rf|fr|r[a-z]*f|f[a-z]*r)\b[^|;&\n]*"
                r"\s(?:/|~|\$HOME|/\*|\.\.)(?=[\s/*'\"]|$)", re.IGNORECASE), "recursive delete of / ~ or parent"),
    (re.compile(r"\bchmod\b\s+-[a-z]*\s*[0-7]{3,4}\s+/(?:\s|$)", re.IGNORECASE), "chmod on /"),
    (re.compile(r"\bchown\b\s+-[a-z]*r[a-z]*\s+[^\n]*\s/(?:\s|$)", re.IGNORECASE), "recursive chown on /"),
    # remote code piped straight into a shell
    (re.compile(r"\b(curl|wget|fetch)\b[^|\n]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|python\d?)\b", re.IGNORECASE),
     "remote script piped to a shell"),
    # writes to / reads of sensitive locations
    (re.compile(r">>?\s*/etc/"),                          "write to /etc"),
    (re.compile(r"/etc/(passwd|shadow|sudoers)"),         "access to system credential files"),
    (re.compile(r"(\.ssh/|id_rsa|id_ed25519|\.aws/credentials|\.netrc)"), "access to private keys/credentials"),
    (re.compile(r"\bgit\b[^\n]*\bpush\b[^\n]*(--force\b|--force-with-lease\b|\s-f\b)", re.IGNORECASE),
     "force push"),
]


# DENY-BY-DEFAULT reader allowlist for readonly/ask modes. Keying on a mutator NAME set let unknown
# plugin/MCP tools (and even known mutating builtins absent from the small WRITE/EXEC sets — terminal_*,
# proc_*, world_set, update_plan, …) slip past these safety modes. Allow ONLY known-safe readers; treat
# everything else (including any unknown tool) as a mutation. ask_user is interactive, not a mutation.
_READERS = frozenset(READ_ONLY_TOOLS) | {"ask_user"}


def read_only(name: str, args: dict) -> Optional[ToolDecision]:
    """Deny anything that is not a KNOWN read-only tool (deny-by-default — an unknown tool may mutate)."""
    if name not in _READERS:
        return ToolDecision(False, "read-only mode: only read/list/search tools are allowed")
    return None


def ask_mutations(name: str, args: dict) -> Optional[ToolDecision]:
    """Confirm any tool that is not a KNOWN read-only tool (an unknown tool may modify state / run code)."""
    if name not in _READERS:
        return ToolDecision(False, f"{name} is not a known read-only tool (may modify state or run code)",
                            ask=True)
    return None


def no_dangerous_commands(name: str, args: dict) -> Optional[ToolDecision]:
    """Deny shell commands (or execute_code bodies) matching a narrow catastrophic list.
    Scanning arbitrary Python is best-effort — it catches obvious run('rm -rf /')-style
    bodies; the sandbox (cwd confine, secret scrub, timeout) is the real guardrail."""
    if name not in EXEC_TOOLS:
        return None
    cmd = str(args.get("command") or args.get("code") or "")
    for pat, reason in _DANGEROUS:
        if pat.search(cmd):
            return ToolDecision(False, f"blocked dangerous command: {reason}")
    return None


class PolicyChain:
    """An ordered list of `policy(name, args) -> ToolDecision | None`. Callable so it
    plugs directly into hooks.PermissionHook. First denial wins; all-abstain → ALLOW."""

    def __init__(self, *policies: Callable[[str, dict], Optional[ToolDecision]]):
        self.policies = policies

    def __call__(self, name: str, args: dict) -> ToolDecision:
        for p in self.policies:
            d = p(name, args)
            if d is not None and (not d.allow or d.ask):  # deny OR ask short-circuits
                return d
        return ALLOW


def make_policy(mode: str = "guard") -> PolicyChain:
    """Factory: 'guard' (block catastrophic commands), 'readonly' (no writes/exec),
    'ask' (block catastrophic + confirm every write/exec), or 'allow' (permissive)."""
    mode = (mode or "guard").lower()
    if mode == "allow":
        return PolicyChain()
    if mode == "readonly":
        return PolicyChain(read_only)
    if mode == "ask":
        return PolicyChain(no_dangerous_commands, ask_mutations)  # dangerous→deny, rest→ask
    if mode == "guard":
        return PolicyChain(no_dangerous_commands)
    # #28: a typo'd mode (e.g. "redonly") must NOT silently fall back to a weaker policy than intended.
    raise ValueError(f"unknown policy mode {mode!r} (expected 'guard', 'readonly', 'ask', or 'allow')")
