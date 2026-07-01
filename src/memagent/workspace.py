"""Workspace snapshot for the system prompt — a one-shot, cache-stable probe.

Ported (trimmed) from Hermes ``agent/coding_context.py`` (``_git`` :590,
``_parse_status`` :603, ``_project_facts`` :638, ``build_coding_workspace_block``
:687). The point is to hand the model its *verify loop* and current git posture up
front — which branch, how dirty the tree is, the exact test/lint/build commands —
instead of making it rediscover them every session.

MOAT / cache safety
-------------------
``build_workspace_snapshot`` is called **once per session** and its output is baked
into the *stable* (cacheable) system-prompt tier — never re-probed per turn (that
would shatter the prompt cache). Branch and dirty state drift mid-session, so the
caller's brief tells the model to re-check with ``git`` before acting on it. The
function is therefore deterministic per ``cwd`` within a session, never raises, and
returns ``""`` outside a workspace (no repo / no marker / git missing / empty cwd).

TRIMMED relative to Hermes: git **branch** + short **status counts**
(staged/modified/untracked) + detected **verify command(s)**. Deliberately dropped:
ahead/behind tracking, worktree detection, and the recent-commit log.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

# Project-root signals that mark a directory as a code workspace even when it
# isn't (yet) a git repo. Cheap filename checks — no parsing.
_PROJECT_MARKERS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "tsconfig.json", "deno.json",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "composer.json", "mix.exs", "pubspec.yaml",
    "CMakeLists.txt", "Makefile", "Dockerfile",
    "AGENTS.md", "CLAUDE.md", ".cursorrules",
)

# Agent-instruction files surfaced separately from manifests in the snapshot.
_CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md", ".cursorrules")

# Lockfile → package manager, checked in priority order.
_PY_LOCKFILES = (("uv.lock", "uv"), ("poetry.lock", "poetry"), ("Pipfile.lock", "pipenv"))
_JS_LOCKFILES = (
    ("pnpm-lock.yaml", "pnpm"), ("bun.lockb", "bun"), ("bun.lock", "bun"),
    ("yarn.lock", "yarn"), ("package-lock.json", "npm"),
)

# package.json scripts / Makefile targets worth surfacing as verify commands.
_VERIFY_TARGETS = ("test", "tests", "lint", "typecheck", "check", "build", "fmt", "format")
_MAX_VERIFY_COMMANDS = 8
_MAX_FACT_FILE_BYTES = 256 * 1024

_GIT_TIMEOUT = 2.5


# ── cwd / root resolution ────────────────────────────────────────────────────


def _resolve_cwd(cwd: Optional[str]) -> Optional[Path]:
    """Resolve ``cwd`` to a Path, or ``None`` if it cannot be used. Never raises."""
    try:
        if cwd:
            return Path(cwd).expanduser()
        return Path(os.getcwd())
    except (OSError, RuntimeError, ValueError):
        return None


def _git_root(cwd: Path) -> Optional[Path]:
    try:
        current = cwd.resolve()
        for parent in [current, *current.parents]:
            if (parent / ".git").exists():
                return parent
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def _home() -> Optional[Path]:
    try:
        return Path.home().resolve()
    except (OSError, RuntimeError):
        return None


def _marker_root(cwd: Path) -> Optional[Path]:
    """Nearest ancestor (≤6 levels) that looks like a project root, or ``None``.

    ``$HOME`` itself is skipped — a Makefile or AGENTS.md sitting in the home
    directory is global user config, not a project-root signal. Never raises.
    """
    try:
        current = cwd.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    home = _home()
    for depth, parent in enumerate([current, *current.parents]):
        if depth > 6:
            break
        if parent == home:
            continue
        try:
            for marker in _PROJECT_MARKERS:
                if (parent / marker).exists():
                    return parent
        except (OSError, ValueError):
            continue
    return None


# ── git/workspace probe ──────────────────────────────────────────────────────


def _git(cwd: Path, *args: str) -> str:
    """Run ``git -C cwd <args>`` and return stripped stdout, or ``""``. Never raises."""
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            errors="replace",   # a non-UTF-8 commit subject (%s) must not raise UnicodeDecodeError out of "never raises"
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _parse_status(porcelain: str) -> tuple[str, dict[str, int]]:
    """Parse ``git status --porcelain=2 --branch`` into (branch_head, counts).

    TRIMMED from Hermes: no upstream / ahead-behind tracking. Returns the branch
    head name (``""`` if absent) and counts of staged/modified/untracked/conflicts.
    """
    head = ""
    counts = {"staged": 0, "modified": 0, "untracked": 0, "conflicts": 0}
    for line in porcelain.splitlines():
        if line.startswith("# branch.head"):
            head = line.split(maxsplit=2)[-1]
        elif line.startswith(("1 ", "2 ")):
            parts = line.split(maxsplit=2)
            if len(parts) < 2:
                continue
            xy = parts[1]
            if len(xy) >= 2:
                if xy[0] != ".":
                    counts["staged"] += 1
                if xy[1] != ".":
                    counts["modified"] += 1
        elif line.startswith("u "):
            counts["conflicts"] += 1
        elif line.startswith("? "):
            counts["untracked"] += 1
    return head, counts


def _dirty_phrases(counts: dict[str, int]) -> list[str]:
    """The non-zero ``"<n> <label>"`` phrases of a parsed git status, in display order.

    Empty list == clean tree; callers join with ", " and fall back to "clean". Single source for
    both the one-line branch summary and the multi-line snapshot so their wording can't drift.
    """
    return [
        f"{n} {label}" for label, n in (
            ("staged", counts["staged"]),
            ("modified", counts["modified"]),
            ("untracked", counts["untracked"]),
            ("conflicts", counts["conflicts"]),
        ) if n
    ]


def _read_small(path: Path) -> str:
    """Read a small text file, or ``""`` — never raises, never reads huge files."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FACT_FILE_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _project_facts(root: Path) -> list[str]:
    """Detected project facts: manifest(s) + package manager, verify commands,
    agent-instruction context files. Cheap stat calls + a couple of small reads.
    Deterministic for a given tree; never raises.
    """
    facts: list[str] = []

    try:
        manifests = [
            m for m in _PROJECT_MARKERS
            if m not in _CONTEXT_FILES and (root / m).is_file()
        ]
        package_managers = [
            pm for lock, pm in (*_PY_LOCKFILES, *_JS_LOCKFILES) if (root / lock).is_file()
        ]
    except OSError:
        manifests, package_managers = [], []
    if manifests:
        line = f"- Project: {', '.join(manifests[:6])}"
        if package_managers:
            line += f" ({'/'.join(dict.fromkeys(package_managers))})"
        facts.append(line)

    verify: list[str] = []
    try:
        if (root / "scripts" / "run_tests.sh").is_file():
            verify.append("scripts/run_tests.sh")
        if (root / "package.json").is_file():
            try:
                scripts = json.loads(_read_small(root / "package.json") or "{}").get("scripts") or {}
            except (json.JSONDecodeError, AttributeError):
                scripts = {}
            js_pm = next((pm for lock, pm in _JS_LOCKFILES if (root / lock).is_file()), "npm")
            verify.extend(f"{js_pm} run {name}" for name in _VERIFY_TARGETS if name in scripts)
        if (root / "pytest.ini").is_file() or "[tool.pytest" in _read_small(root / "pyproject.toml"):
            verify.append("pytest")
        makefile = _read_small(root / "Makefile")
        if makefile:
            verify.extend(
                f"make {name}" for name in _VERIFY_TARGETS
                if re.search(rf"^{re.escape(name)}\s*:", makefile, re.MULTILINE)
            )
    except OSError:
        pass
    if verify:
        deduped = list(dict.fromkeys(verify))[:_MAX_VERIFY_COMMANDS]
        facts.append(f"- Verify: {'; '.join(deduped)}")

    try:
        context_files = [c for c in _CONTEXT_FILES if (root / c).is_file()]
    except OSError:
        context_files = []
    if context_files:
        facts.append(f"- Context files: {', '.join(context_files)}")

    return facts


def git_branch_status(cwd: str) -> str:
    """A compact one-line 'branch (status)' summary for the RE-OBSERVED ENVIRONMENT tier (I2).

    Reuses the same git probe as the snapshot, but collapsed to ONE line: e.g.
    "main (3 modified, 1 untracked)" or "main (clean)". Returns "" outside a repo / on any
    error. Deterministic per cwd within a session (intended to be computed ONCE per session and
    baked into the cache-stable system tier), never raises.
    """
    resolved = _resolve_cwd(cwd)
    if resolved is None:
        return ""
    git_root = _git_root(resolved)
    if git_root is None:
        return ""
    head, counts = _parse_status(_git(git_root, "status", "--porcelain=2", "--branch"))
    if not head:
        return ""
    branch = "(detached HEAD)" if head == "(detached)" else head
    dirty = _dirty_phrases(counts)
    base = f"{branch} ({', '.join(dirty) if dirty else 'clean'})"
    last = " ".join(_git(git_root, "log", "-1", "--format=%h %s").split())[:72]   # HEAD commit (orientation)
    return f"{base} · HEAD: {last}" if last else base


def build_workspace_snapshot(cwd: str) -> str:
    """Workspace snapshot body for the system prompt (``""`` outside a workspace).

    Git state (branch + short status counts) when ``cwd`` is in a repo, plus
    detected project facts (manifest, package manager, verify commands, context
    files) — so marker-only (non-git) projects still get a snapshot.

    Contract: ``''``-safe, NEVER raises, deterministic per ``cwd`` within a session.
    Intended to be called ONCE per session; the caller bakes the result into the
    stable (cacheable) system-prompt tier and supplies its own header.

    TRIMMED from Hermes: no ahead/behind, no worktree, no commit log. The leading
    "Root:" line is omitted — the caller's WORKSPACE header already frames it and
    a second absolute path tends to make the model run commands in the wrong dir.
    """
    resolved = _resolve_cwd(cwd)
    if resolved is None:
        return ""
    git_root = _git_root(resolved)
    root = git_root or _marker_root(resolved)
    if root is None:
        return ""

    lines: list[str] = []

    if git_root is not None:
        head, counts = _parse_status(_git(root, "status", "--porcelain=2", "--branch"))
        if head and head != "(detached)":
            lines.append(f"- Branch: {head}")
        elif head == "(detached)":
            lines.append("- Branch: (detached HEAD)")

        dirty = _dirty_phrases(counts)
        lines.append(f"- Status: {', '.join(dirty) if dirty else 'clean'}")

    lines.extend(_project_facts(root))
    return "\n".join(lines)


# ── LIVE world-state (SENSORY CORTEX — the derived-view, recomputed-each-build region) ────────────


def project_root(cwd: str) -> Optional[str]:
    """The project root for `cwd` — its git root, else the nearest ancestor holding a project marker
    (pyproject/package.json/…); None outside any project (e.g. a bare HOME dir). This is the session-
    static 'are we in a project at all?' decision that gates repo-derived slice content (the REPO MAP,
    facts, conventions, subdir hints) — so launching in HOME doesn't os.walk the whole home directory."""
    resolved = _resolve_cwd(cwd)
    if resolved is None:
        return None
    root = _git_root(resolved) or _marker_root(resolved)
    return str(root) if root else None


def workspace_facts(cwd: str) -> str:
    """STATIC project facts (manifest, package manager, verify commands, context files) for the
    cache-stable SYSTEM tier — the git-INDEPENDENT subset of build_workspace_snapshot. Live git
    state is deliberately NOT here; it lives in the volatile slice via git_worktree_state(), so the
    system message stays byte-stable (prompt-cache warm). '' outside a project; never raises."""
    resolved = _resolve_cwd(cwd)
    if resolved is None:
        return ""
    root = _git_root(resolved) or _marker_root(resolved)
    if root is None:
        return ""
    return "\n".join(_project_facts(root))


def project_conventions(cwd: str, *, max_chars: int = 4000) -> str:
    """The project's agent-convention file CONTENT (first present of AGENTS.md / CLAUDE.md / .cursorrules)
    — an ALWAYS-IN-FORCE contract that must outlive the bounded slice's eviction. Injection-neutralized
    (reuses subdir_hints._neutralize_injection) and capped. '' when none / outside a project.

    Deterministic per cwd, so it rides in the cacheable SYSTEM tier (100% prompt-cache after turn 1) and
    CANNOT be evicted/compacted — conventions persist across a long session at ~0 marginal cost, replacing
    the uncached, evictable manual re-read of AGENTS.md. Bounded to ONE file ≤ max_chars (smaller than a
    transcript agent's unbounded merged context). Treat as DATA: the live conversation overrides on conflict."""
    from .subdir_hints import _neutralize_injection
    resolved = _resolve_cwd(cwd)
    if resolved is None:
        return ""
    root = _git_root(resolved) or _marker_root(resolved)
    if root is None:
        return ""
    for name in _CONTEXT_FILES:
        text = _read_small(root / name)
        if text.strip():
            body = _neutralize_injection(text).strip()
            if len(body) > max_chars:
                body = body[:max_chars] + "\n[...truncated]"
            return f"{name}:\n{body}"
    return ""


def git_worktree_state(cwd: str, *, max_files: int = 20) -> str:
    """LIVE working-tree state for the VOLATILE slice tier (SENSORY CORTEX — the derived-view,
    recomputed-each-build region, never persisted): current branch + the CHANGED-FILE SET (staged/
    modified/untracked/conflicts), re-probed every build — unlike the one-shot session-start snapshot.
    This is the cure for the stale-snapshot 're-run git' smell: the model always sees the current git
    state. Bounded to max_files. '' outside a repo / on error; never raises (POMDP per-turn belief
    update analog)."""
    resolved = _resolve_cwd(cwd)
    if resolved is None:
        return ""
    git_root = _git_root(resolved)
    if git_root is None:
        return ""
    porcelain = _git(git_root, "status", "--porcelain=2", "--branch")
    head, _counts = _parse_status(porcelain)
    if not head:
        return ""
    branch = "(detached HEAD)" if head == "(detached)" else head
    changed: list[tuple[str, str]] = []
    for line in porcelain.splitlines():
        if line.startswith(("1 ", "2 ")):
            # porcelain v2 type-2 (rename/copy) has an extra <X><score> field AND joins the path as
            # "<path>\t<origpath>" — so split at maxsplit=9 and drop the tab-joined origpath, else the
            # reported path is mangled (origpath leaks in / the new path is truncated).
            n = 9 if line.startswith("2 ") else 8
            parts = line.split(maxsplit=n)
            if len(parts) >= 2 and len(parts[1]) >= 2:
                xy = parts[1]
                tag = "staged" if xy[0] != "." else "modified"
                changed.append((tag, parts[-1].split("\t", 1)[0]))
        elif line.startswith("u "):
            changed.append(("conflict", line.split(maxsplit=10)[-1]))
        elif line.startswith("? "):
            changed.append(("untracked", line[2:]))   # exact path (splitlines already dropped the newline); .strip() ate significant leading/trailing spaces
    if not changed:
        return f"branch {branch} · working tree clean"
    lines = [f"branch {branch} · {len(changed)} changed file(s)"]
    lines += [f"  {tag}: {path}" for tag, path in changed[:max_files]]
    if len(changed) > max_files:
        lines.append(f"  …and {len(changed) - max_files} more")
    return "\n".join(lines)
