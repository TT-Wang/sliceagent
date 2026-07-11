"""Permission policy — authorization for tool calls.

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
NAVIGATION_TOOLS = frozenset(("change_workspace",))
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
# TRIMMED to the machine-destruction + remote-code-execution core (block review, 2026-07-11). A trusted
# LOCAL operator legitimately runs sudo, `git push --force` (own branch), chmod/chown, and edits /etc on their
# OWN machine — mature peers (Claude Code's most-permissive mode, Codex, Cursor, Aider) do NOT hard-block those.
# So this floor keeps ONLY the irreversible ones: wiping a disk, deleting the tree, or running remote code.
# Credential-file READS were moved OFF this hard-deny to a confirm (see _SENSITIVE_READ below): not a
# catastrophe, but — with web_search/MCP outbound — not silently auto-readable either.
_DANGEROUS_SRC: list[tuple[str, str]] = [
    (r"(?s):\s*\(\s*\)\s*\{.*\|.*&",                       "fork bomb"),
    (r"\b(shutdown|reboot|halt|poweroff)\b",              "system power control"),
    (r"\b(mkfs|wipefs)\b",                                 "filesystem format"),
    (r"\bdd\b[^\n]*\bof=/dev/",                            "raw write to a device"),
    (r">\s*/dev/(sd|nvme|disk|hd)",                        "raw write to a device"),
    # any RECURSIVE rm targeting / ~ $HOME /* .. (workspace-relative rm is fine). Catches -rf, split "-r -f",
    # and long-form --recursive — the recursive flag (short -...r... or --recursive) anywhere before the target.
    (r"\brm\b(?=[^|;&\n]*(?:\s-[a-z]*r|\s--recursive))"
     r"[^|;&\n]*\s(?:/|~|\$HOME|/\*|\.\.)(?=[\s/*'\"]|$)", "recursive delete of / ~ or parent"),
    # remote code piped straight into a shell (RCE — kept even on a trusted machine: it runs UNKNOWN code)
    (r"\b(curl|wget|fetch)\b[^|\n]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|python\d?)\b", "remote script piped to a shell"),
]
# Credential/secret READS: downgraded from a hard catastrophic deny to "not provably read-only", so a bare
# `cat ~/.aws/credentials` no longer auto-runs silently — it takes the ordinary confirm path (and a let-it-go
# run still surfaces it). Prevents silent secret exfiltration via the model's web/MCP tools without blocking a
# deliberate, confirmed read on your own machine.
_SENSITIVE_READ = re.compile(
    r"/etc/(?:passwd|shadow|gshadow|sudoers|pass[\w*]*|shad[\w*]*|sudoer[\w*]*)"
    r"|(?:\.ssh/|id_rsa\b|id_ed25519\b|\.aws/credentials\b|\.netrc\b)", re.IGNORECASE)
# Compile ALL case-insensitively in one place — uniform, so a future pattern can't silently reintroduce a
# casing bypass (the round-3 bug: shutdown/mkfs/dd/etc had no IGNORECASE while rm/chmod/curl did).
_DANGEROUS: list[tuple[re.Pattern, str]] = [(re.compile(p, re.IGNORECASE), why) for p, why in _DANGEROUS_SRC]


# DENY-BY-DEFAULT reader allowlist for readonly/ask modes. Keying on a mutator NAME set let unknown
# plugin/MCP tools (and even known mutating builtins absent from the small WRITE/EXEC sets — terminal_*,
# proc_*, world_set, update_plan, …) slip past these safety modes. Allow ONLY known-safe readers; treat
# everything else (including any unknown tool) as a mutation. ask_user is interactive, not a mutation.
_READERS = frozenset(READ_ONLY_TOOLS) | {"ask_user", "reconcile_execution"}


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
    # `cd` only re-points the shell's own cwd for the command that follows it (`cd repo && git status`); it
    # mutates no user state and takes no write option, so it is read-only as a chain prefix.
    "cd", "ls", "pwd", "cat", "head", "tail", "wc", "echo", "which", "type", "printenv", "date",
    "whoami", "hostname", "uname", "id", "groups", "tree", "stat", "file", "du", "df", "realpath",
    "dirname", "basename", "grep", "rg", "egrep", "fgrep", "sort", "nl", "cut", "column", "jq",
    "cksum", "md5", "md5sum", "sha1sum", "sha256sum", "true",
))
# `env` and `uniq` were REMOVED from the allowlist: `env <program>` executes an arbitrary program, and
# `uniq [IN] OUT` OVERWRITES its optional second positional — neither is provably read-only, so they now
# take the normal confirm path. A few remaining allowlisted verbs stay read-only ONLY without a specific
# WRITE/EXEC option; deny-by-default when that option is present (verb-scoped so read-only siblings like
# `du -s` / `grep -o` keep auto-running). Value-taking flags match `-o`, `-o=…`, `-oFILE`, `--output=…`.
_UNSAFE_OPTS = {
    "sort": ("-o", "--output"),   # -o FILE / --output=FILE overwrite a file
    "date": ("-s", "--set"),      # set the system clock (a mutation)
    "tree": ("-o",),              # -o FILE writes the listing to a file
}


def _has_unsafe_opt(verb: str, toks: list) -> bool:
    bad = _UNSAFE_OPTS.get(verb, ())
    for t in toks[1:]:
        for b in bad:
            if t == b or t.startswith(b + "=") or (len(b) == 2 and len(t) > 2 and t.startswith(b)):
                return True
    return False
# git subcommands with NO mutating form (excludes branch/tag/config/remote/stash/reflog, which can write).
_RO_GIT_SUB = frozenset((
    "status", "log", "diff", "show", "rev-parse", "ls-files", "ls-tree", "describe", "blame",
    "shortlog", "rev-list", "cat-file", "for-each-ref", "name-rev", "whatchanged", "count-objects",
    "var", "ls-remote", "grep",
))
# find ACTIONS that run/delete (anything else is a read-only traversal)
_FIND_MUTATORS = frozenset(("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprintf", "-fprint", "-fls", "-fprint0"))
_SHELL_META = re.compile(r"[;&|<>`\n]|\$\(|\$\{|\breturn\b")
_NULL_REDIRECT = re.compile(r"(?<!\S)(?:[012]?>|[012]?>>|&>)\s*/dev/null(?=\s|$)")


def _is_readonly_command(cmd: str) -> bool:
    """True only when `cmd` PROVABLY just reads/reports state — a tiny verb allowlist with no shell
    metacharacters. A narrow exception supports fallback discovery probes: ``reader 2>/dev/null || reader``
    is observational only when every branch independently passes this same classifier. No other redirect or
    compound shell syntax is promoted. Conservative by design: anything unrecognized returns False."""
    cmd = (cmd or "").strip()
    if _SENSITIVE_READ.search(cmd):     # a credential/secret read is never "provably safe" auto-run → confirm
        return False
    if not cmd:
        return False
    if "&&" in cmd or "||" in cmd:
        # && / || chain COMPLETE commands (looser than |). The whole chain only reads state when EVERY
        # branch independently proves read-only — e.g. `ls dir | head && echo --- && cat file` is purely
        # observational. A single `&` (backgrounding) is NOT this and still fails the metachar guard below.
        branches = re.split(r"&&|\|\|", cmd)
        return bool(branches) and all(
            branch.strip() and _is_readonly_command(branch) for branch in branches
        )
    if "|" in cmd:
        stages = cmd.split("|")
        return bool(stages) and all(
            stage.strip() and _is_readonly_command(stage) for stage in stages
        )
    # Discard-only redirection does not mutate user state. Strip it before the ordinary no-metachar proof;
    # a redirect anywhere else (including a near-miss path) remains visible and therefore fails closed.
    cmd = _NULL_REDIRECT.sub("", cmd).strip()
    if not cmd or _SHELL_META.search(cmd):
        return False
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return False
    if not toks:
        return False
    # A lookalike executable from /tmp or ~/bin is not the allowlisted reader merely because its basename
    # is ``rg``/``git``. PATH resolution remains the host's ordinary trust boundary; explicit paths do not.
    if os.path.isabs(toks[0]) or toks[0].startswith("~") or "/" in toks[0] or "\\" in toks[0]:
        return False
    verb = toks[0]
    if verb == "git":
        # split MAIN options (before the subcommand) from SUBCOMMAND options (after it).
        isub = next((i for i, t in enumerate(toks[1:], 1) if not t.startswith("-")), len(toks))
        sub = toks[isub] if isub < len(toks) else ""
        if sub not in _RO_GIT_SUB:
            return False
        main_opts, sub_opts = toks[1:isub], toks[isub + 1:]
        # MAIN -c/--config/--exec-path inject git config (diff.external, *.textconv, pager, alias, exec-path)
        # that points at an ARBITRARY command → exec. Refuse auto-approval (external review H-06).
        if any(
            t == "-c" or t.startswith("-c")
            or t in ("--config", "--config-env", "--exec-path", "--paginate")
            or t.startswith(("--config=", "--config-env=", "--exec-path="))
            for t in main_opts
        ):
            return False
        if any(t not in {"--no-pager", "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs"}
               for t in main_opts):
            return False
        pager_or_diff = {"log", "diff", "show", "blame", "shortlog", "whatchanged", "grep"}
        if sub in pager_or_diff and "--no-pager" not in main_opts:
            return False
        if sub in {"diff", "show"} and not {"--no-ext-diff", "--no-textconv"}.issubset(set(sub_opts)):
            return False
        # SUBCOMMAND options that WRITE a file (-o/--output) or RUN a configured helper (--ext-diff/--textconv,
        # `grep --open-files-in-pager`/`-O<cmd>` runs a pager). Any of these makes the "read-only" verb unsafe.
        for t in sub_opts:
            if (t in ("-o", "--output", "--ext-diff", "--textconv")
                    or t.startswith(("--output=", "-O", "--open-files-in-pager"))):
                return False
        return True
    if verb == "find":
        return not any(t in _FIND_MUTATORS for t in toks)
    if verb == "sed":
        return len(toks) >= 4 and toks[1] == "-n" \
            and re.fullmatch(r"(?:\d+|\$)(?:,(?:\d+|\$))?p", toks[2]) is not None \
            and all(not token.startswith("-") for token in toks[3:])
    if verb == "rg" and any(t == "--pre" or t.startswith("--pre=") for t in toks[1:]):
        return False
    return verb in _RO_VERBS and not _has_unsafe_opt(verb, toks)


def ask_commands(name: str, args: dict) -> Optional[ToolDecision]:
    """'teenager' middle ground: auto-allow known file EDITS + reads, but confirm anything that RUNS a
    command (EXEC tools) or is an unknown tool that might. Edits flow; running code pauses for a yes —
    EXCEPT a provably read-only `run_command` (git status, ls, cat, …), which reports state without
    changing it and so runs without a prompt (kills the confirm-hang on 'which repo am I in')."""
    if name in _READERS or name in WRITE_TOOLS or name in NAVIGATION_TOOLS:
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
