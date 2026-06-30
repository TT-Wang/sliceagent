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

import os
import re
import shlex
from typing import Callable, Optional

from .agents import READ_ONLY_TOOLS   # single source of truth for the known read-only surface
from .hooks import ToolDecision

ALLOW = ToolDecision(True)

WRITE_TOOLS = frozenset(("edit_file", "append_to_file", "str_replace"))
# every tool that runs a model-authored shell command/body — the catastrophic denylist must gate ALL of
# them, not just run_command/execute_code (proc_start/terminal_open/terminal_send execute shell verbatim too).
EXEC_TOOLS = frozenset(("run_command", "execute_code", "proc_start", "terminal_open", "terminal_send"))

# Patterns that are almost never legitimate inside a coding-agent workspace.
# Kept deliberately narrow so normal dev commands (pytest, pip, npm, git add/commit,
# rm of a workspace file, mkdir, mv) pass untouched.
# INVARIANT: every pattern is compiled case-INSENSITIVE (the comprehension below). The floor must not be
# defeated by casing — on a case-insensitive filesystem (macOS default) `SHUTDOWN` / `/ETC/PASSWD` resolve
# to the real command/path, so a case-sensitive denylist is a genuine bypass. A pattern needing DOTALL
# carries an inline `(?s)`.
# SCOPE: this denylist is a BEST-EFFORT defense-in-depth SPEED BUMP, not a complete sandbox. A regex cannot
# catch every shell encoding (globs, $(...), subshells, var-expansion, exotic wrappers); chasing each is a
# losing arms race. The BINDING guards are the sandbox (network=none fail-closed, cwd-confine, secret-scrub)
# and the permission modes (baby-sitter/teenager CONFIRM every command by default). Patterns here catch the
# obvious/common forms so an unattended (let-it-go) run still has a floor.
_DANGEROUS_SRC: list[tuple[str, str]] = [
    (r"(?s):\s*\(\s*\)\s*\{.*\|.*&",                       "fork bomb"),
    (r"\bsudo\b",                                          "privilege escalation (sudo)"),
    (r"\b(shutdown|reboot|halt|poweroff)\b",              "system power control"),
    (r"\b(mkfs|wipefs)\b",                                 "filesystem format"),
    (r"\bdd\b[^\n]*\bof=/dev/",                            "raw write to a device"),
    (r">\s*/dev/(sd|nvme|disk|hd)",                        "raw write to a device"),
    # any RECURSIVE rm targeting / ~ $HOME /* .. (workspace-relative rm is fine). Catches -rf, split "-r -f",
    # and long-form --recursive — the recursive flag (short -...r... or --recursive) anywhere before the target.
    (r"\brm\b(?=[^|;&\n]*(?:\s-[a-z]*r|\s--recursive))"
     r"[^|;&\n]*\s(?:/|~|\$HOME|/\*|\.\.)(?=[\s/*'\"]|$)", "recursive delete of / ~ or parent"),
    # chmod/chown on / — flags are OPTIONAL (plain `chmod 755 /` / `chown nobody /` are just as catastrophic
    # as the -R forms), and long-form flags (--recursive) count too. The target must be the bare root.
    (r"\bchmod\b\s+(?:-{1,2}[a-z]+\s+)*[0-7]{3,4}\s+/(?:[\s/*'\"]|$)", "chmod on /"),
    (r"\bchown\b\s+[^\n]*\s/(?:[\s/*'\"]|$)",              "chown on /"),
    # remote code piped straight into a shell
    (r"\b(curl|wget|fetch)\b[^|\n]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|python\d?)\b", "remote script piped to a shell"),
    # writes to / reads of sensitive locations
    (r">>?\s*/etc/",                                       "write to /etc"),
    # credential files + their common glob/prefix forms (cat /etc/pass*, /etc/shadow, /etc/sudoers.d, …)
    (r"/etc/(?:passwd|shadow|gshadow|sudoers|pass[\w*]*|shad[\w*]*|sudoer[\w*]*)", "access to system credential files"),
    (r"(\.ssh/|id_rsa|id_ed25519|\.aws/credentials|\.netrc)", "access to private keys/credentials"),
    (r"\bgit\b[^\n]*\bpush\b[^\n]*(--force\b|--force-with-lease\b|\s-f\b)", "force push"),
]
# Compile ALL case-insensitively in one place — uniform, so a future pattern can't silently reintroduce a
# casing bypass (the round-3 bug: shutdown/mkfs/dd/etc had no IGNORECASE while rm/chmod/curl did).
_DANGEROUS: list[tuple[re.Pattern, str]] = [(re.compile(p, re.IGNORECASE), why) for p, why in _DANGEROUS_SRC]


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


# Provably read-only shell verbs — a `run_command` whose verb is one of these (and which contains no
# shell metacharacter that could chain/redirect/substitute) reports state without changing it, so teenager
# mode runs it without a confirm prompt. Deny-by-default: an unknown verb still asks. The catastrophic floor
# (no_dangerous_commands) runs BEFORE this, so e.g. `cat /etc/passwd` is still denied.
_RO_VERBS = frozenset((
    "ls", "pwd", "cat", "head", "tail", "wc", "echo", "which", "type", "env", "printenv", "date",
    "whoami", "hostname", "uname", "id", "groups", "tree", "stat", "file", "du", "df", "realpath",
    "dirname", "basename", "grep", "rg", "egrep", "fgrep", "sort", "uniq", "nl", "cut", "column",
    "cksum", "md5", "md5sum", "sha1sum", "sha256sum", "true",
))
# git subcommands with NO mutating form (excludes branch/tag/config/remote/stash/reflog, which can write).
_RO_GIT_SUB = frozenset((
    "status", "log", "diff", "show", "rev-parse", "ls-files", "ls-tree", "describe", "blame",
    "shortlog", "rev-list", "cat-file", "for-each-ref", "name-rev", "whatchanged", "count-objects",
    "var", "ls-remote", "grep",
))
# find ACTIONS that run/delete (anything else is a read-only traversal)
_FIND_MUTATORS = frozenset(("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprintf", "-fprint", "-fls", "-fprint0"))
_SHELL_META = re.compile(r"[;&|<>`\n]|\$\(|\$\{|\breturn\b")


def _is_readonly_command(cmd: str) -> bool:
    """True only when `cmd` PROVABLY just reads/reports state — a tiny verb allowlist with no shell
    metacharacters. Conservative by design: anything unrecognized returns False (→ still confirmed)."""
    cmd = (cmd or "").strip()
    if not cmd or _SHELL_META.search(cmd):
        return False
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return False
    if not toks:
        return False
    verb = os.path.basename(toks[0])
    if verb == "git":
        sub = next((t for t in toks[1:] if not t.startswith("-")), "")
        return sub in _RO_GIT_SUB
    if verb == "find":
        return not any(t in _FIND_MUTATORS for t in toks)
    return verb in _RO_VERBS


def ask_commands(name: str, args: dict) -> Optional[ToolDecision]:
    """'teenager' middle ground: auto-allow known file EDITS + reads, but confirm anything that RUNS a
    command (EXEC tools) or is an unknown tool that might. Edits flow; running code pauses for a yes —
    EXCEPT a provably read-only `run_command` (git status, ls, cat, …), which reports state without
    changing it and so runs without a prompt (kills the confirm-hang on 'which repo am I in')."""
    if name in _READERS or name in WRITE_TOOLS:
        return None
    if name == "run_command" and _is_readonly_command(str(args.get("command") or "")):
        return None
    return ToolDecision(False, f"{name} runs a command — confirm before it executes", ask=True)


def no_dangerous_commands(name: str, args: dict) -> Optional[ToolDecision]:
    """Deny shell commands (or execute_code bodies) matching a narrow catastrophic list.
    Scanning arbitrary Python is best-effort — it catches obvious run('rm -rf /')-style
    bodies; the sandbox (cwd confine, secret scrub, timeout) is the real guardrail."""
    if name not in EXEC_TOOLS:
        return None
    cmd = str(args.get("command") or args.get("code") or args.get("input") or "")  # input = terminal_send line
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


# Three USER-FACING modes, all sharing the catastrophic-command floor (no fully-unrestricted UI mode).
# `allow`/`readonly` remain as LEGACY/eval escapes, not advertised. Friendly + legacy names both resolve.
USER_MODES = ("baby-sitter", "teenager", "let-it-go")
_MODE_ALIASES = {
    "baby-sitter": "babysitter", "babysitter": "babysitter", "baby": "babysitter", "ask": "babysitter",
    "teenager": "teenager", "teen": "teenager",
    "let-it-go": "letitgo", "letitgo": "letitgo", "letgo": "letitgo", "yolo": "letitgo", "guard": "letitgo",
    "allow": "allow", "readonly": "readonly",                       # legacy escapes
}
_MODE_LABELS = {"babysitter": "baby-sitter", "teenager": "teenager", "letitgo": "let-it-go",
                "allow": "allow", "readonly": "readonly"}
# canonical → True if the mode confirms (needs an interactive resolver; downgrade to let-it-go when headless)
CONFIRMS = {"babysitter": True, "teenager": True, "letitgo": False, "allow": False, "readonly": False}

# Legacy names still RESOLVE (back-compat for old configs/eval) but their connotation differs from the new
# modes — warn LOUDLY so a name like `guard` can't SILENTLY downgrade safety (it now means let-it-go = auto).
_LEGACY_WARN = {
    "guard": "AGENT_POLICY=guard is legacy and now maps to 'let-it-go' (auto-runs everything except "
             "catastrophic commands). For confirmations use 'baby-sitter' or 'teenager'.",
    "ask": "AGENT_POLICY=ask is legacy → use 'baby-sitter' (confirm every edit + command).",
    "allow": "AGENT_POLICY=allow is a legacy permissive eval mode (NO catastrophic floor) — for normal use "
             "pick baby-sitter / teenager / let-it-go.",
    "readonly": "AGENT_POLICY=readonly is a legacy mode (no writes/exec).",
}


def legacy_warning(name: str) -> Optional[str]:
    """A loud deprecation note when a LEGACY mode name is used, so a safety-connoting name like `guard` can't
    silently resolve to a more permissive mode. None for the current friendly names."""
    return _LEGACY_WARN.get((name or "").strip().lower().replace("_", "-").replace(" ", "-"))


def resolve_policy_mode(name: str) -> Optional[str]:
    """Friendly/legacy mode name → canonical key, or None if unrecognized (the caller warns + defaults)."""
    return _MODE_ALIASES.get((name or "").strip().lower().replace("_", "-").replace(" ", "-"))


def policy_label(canonical: str) -> str:
    """Canonical key → friendly display name for the toolbar/help."""
    return _MODE_LABELS.get(canonical, canonical)


def make_policy(mode: str = "teenager") -> PolicyChain:
    """Three modes, ALL with the catastrophic-command floor:
      baby-sitter — confirm every edit + command;  teenager — auto edits, confirm commands;
      let-it-go   — auto everything except catastrophic.  (legacy: allow=permissive, readonly=no writes.)"""
    canonical = resolve_policy_mode(mode)
    if canonical == "babysitter":
        return PolicyChain(no_dangerous_commands, ask_mutations)   # catastrophic→deny, every write/exec→ask
    if canonical == "teenager":
        return PolicyChain(no_dangerous_commands, ask_commands)    # catastrophic→deny, commands→ask, edits auto
    if canonical == "letitgo":
        return PolicyChain(no_dangerous_commands)                  # catastrophic→deny, everything else auto
    if canonical == "allow":
        return PolicyChain()                                       # LEGACY: fully permissive (eval)
    if canonical == "readonly":
        return PolicyChain(read_only)                              # LEGACY: no writes/exec
    # #28: a typo'd mode must NOT silently fall back to a weaker policy than intended.
    raise ValueError(f"unknown policy mode {mode!r} (expected one of {USER_MODES})")
