"""platform_compat — the ONE Windows/POSIX seam (borrowed from Hermes' _subprocess_compat pattern).

Every win32 branch in sliceagent lives here so call sites stay one-liners and the POSIX path stays
EXACTLY what it was before this module existed: `sh()` returns the same `shell=True` string-exec,
`popen_group_kwargs()` returns the same `start_new_session=True`, `kill_tree()` runs the same
killpg ladder. On Windows: sh-syntax commands run under Git Bash (same strategy as Claude Code and
Hermes — the model's bash-flavored tool calls stay platform-invariant), process groups use
creationflags, and tree-kill uses `taskkill /T`.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"
_warned_no_bash = False

# SIGKILL doesn't exist on Windows. POSIX keeps the real signal; win32 uses a SENTINEL that can
# never equal SIGTERM — otherwise kill_tree's `sig == SIG_KILL` force-check would be True for the
# GRACEFUL phase too and procman's TERM->wait->KILL ladder would collapse to an immediate /F kill.
SIG_KILL = getattr(signal, "SIGKILL", None) or -9


def norm_rel(path: str) -> str:
    """win32 only: normalize separators to '/' in model-facing relative paths (repo-map
    keys/heads, hint labels). ripgrep and os.path emit backslashes on Windows, but the
    agent's shell commands run under Git Bash, which speaks '/'.
    POSIX: IDENTITY — backslash is a legal filename character there, never rewrite it.
    """
    if not IS_WINDOWS:
        return path
    return path.replace("\\", "/")


def find_bash() -> str | None:
    """win32 only: the bash.exe that runs the agent's sh-syntax commands (Git Bash).

    Resolution order (Hermes' _find_bash): SLICEAGENT_BASH env override → the app-owned
    PortableGit that install.ps1 drops under %LOCALAPPDATA%/sliceagent/git → well-known
    Git-for-Windows paths → PATH lookup, SKIPPING WSL's System32 bash.exe (it resolves
    Linux-side paths, not the Windows workspace).
    """
    env_bash = os.environ.get("SLICEAGENT_BASH")
    if env_bash and os.path.isfile(env_bash):
        return env_bash
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(local, "sliceagent", "git", "bin", "bash.exe") if local else "",
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin", "bash.exe"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    found = shutil.which("bash")
    if found and "system32" not in found.lower():  # dodge WSL's launcher bash
        return found
    return None


def sh(command: str) -> dict:
    """kwargs for subprocess.run/Popen of one sh-syntax command string.

    POSIX: {'args': command, 'shell': True} — byte-identical to the previous inline
    `subprocess.run(command, shell=True, ...)` at every call site.
    win32: run under Git Bash (['bash', '-c', command]); if no bash is found, fall back to
    shell=True (cmd.exe) so trivial commands still work rather than hard-failing.
    """
    if not IS_WINDOWS:
        return {"args": command, "shell": True}
    bash = find_bash()
    if bash:
        return {"args": [bash, "-c", command], "shell": False}
    global _warned_no_bash
    if not _warned_no_bash:  # once per process: silent cmd.exe downgrade breaks bash-syntax commands confusingly
        _warned_no_bash = True
        import sys as _s
        print("sliceagent: no Git Bash found — shell commands run under cmd.exe and bash syntax will fail. "
              "Install Git for Windows or set SLICEAGENT_BASH.", file=_s.stderr)
    return {"args": command, "shell": True}


def popen_group_kwargs() -> dict:
    """Own-process-group kwargs so a later kill can take down the whole tree.
    POSIX: {'start_new_session': True} — exactly what call sites passed before.
    win32: start_new_session is a silent no-op; use the creationflags equivalent."""
    if not IS_WINDOWS:
        return {"start_new_session": True}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flags}


def kill_tree(popen: subprocess.Popen, sig: int) -> None:
    """Terminate a process AND its descendants.
    POSIX: the pre-existing ladder — killpg(getpgid(pid)) falling back to send_signal.
    win32: `taskkill /T` (+/F when the caller asked for SIGKILL-strength), stdlib-only."""
    if not IS_WINDOWS:
        try:
            os.killpg(os.getpgid(popen.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                popen.send_signal(sig)
            except OSError:
                pass
        return
    force = ["/F"] if sig == SIG_KILL else []
    try:
        subprocess.run(["taskkill", *force, "/T", "/PID", str(popen.pid)],
                       capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        try:
            popen.kill() if force else popen.terminate()
        except OSError:
            pass


def capture_pgid(popen: subprocess.Popen):
    """POSIX: the process-group id of a just-spawned group leader (== its pid), captured while it is still
    alive — os.getpgid raises once the leader exits and is reaped. Returns None on Windows / on failure."""
    if IS_WINDOWS:
        return None
    try:
        return os.getpgid(popen.pid)
    except (AttributeError, OSError):
        return None


def signal_pgid(pgid, sig: int, popen: subprocess.Popen = None) -> None:
    """POSIX: signal a whole process GROUP by its stored pgid, so a background child is still reached after
    the leader was reaped (killpg on the pgid works while any member lives). Falls back to signalling the
    leader directly. Windows: defer to kill_tree (taskkill /T). No-op if there's nothing to signal."""
    if IS_WINDOWS:
        if popen is not None:
            kill_tree(popen, sig)
        return
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass   # group already fully gone
    if popen is not None:
        try:
            popen.send_signal(sig)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# win32 shell-path extraction (used by tools._grant_shell_paths, gated on IS_WINDOWS
# at the call site — POSIX never calls these).
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402  (kept local to this win32-only section)

# Absolute drive-letter path: C:\x or C:/x (either separator).
_WIN_ABS_RE = _re.compile(r"^[A-Za-z]:[\\/]")
# Drive-letter path tokens the POSIX extractor can't see: quoted (may contain spaces)
# OR bare, up to a shell metachar/space — the exact mirror of the POSIX token regex.
_WIN_TOKEN_RE = _re.compile(
    r"""['"]([A-Za-z]:[\\/][^'"]*)['"]|(?<![\w'"])([A-Za-z]:[\\/][^\s'"|&;<>()]+)""")
# Git-Bash (MSYS) drive mount: /c/Users/x  ->  C:/Users/x
_MSYS_DRIVE_RE = _re.compile(r"^/([A-Za-z])(?:/|$)")


def win_path_candidates(text: str) -> list[str]:
    """win32 only: drive-letter path tokens ('C:\\x', "C:/x", bare C:\\x) in one
    sh-syntax command string. Returns [] when none are present."""
    return [(q or uq).strip() for q, uq in _WIN_TOKEN_RE.findall(text)]


def msys_to_win(path: str) -> str:
    """win32 only: translate a Git-Bash mount path '/c/Users/x' -> 'C:/Users/x'.
    Any other string (including a plain POSIX-looking '/etc/hosts') is returned unchanged."""
    m = _MSYS_DRIVE_RE.match(path)
    if m and len(path) > 2:
        return m.group(1).upper() + ":" + path[2:]
    return path


def is_win_abs(path: str) -> bool:
    """True iff *path* is an absolute drive-letter path (C:\\... or C:/...)."""
    return bool(_WIN_ABS_RE.match(path))


class FileLock:
    """Best-effort EXCLUSIVE advisory lock on an already-open file, held for a `with` block. Real on POSIX
    (`fcntl.flock`); a graceful no-op where flock is unavailable (Windows / odd filesystem) — cache reads
    already tolerate a torn line, so the no-lock case degrades to today's behavior, never worse. Serializes
    concurrent APPENDERS to the SAME file (e.g. a resumed session that reuses its session_id, or a future
    off-thread writer) so their lines can't interleave into a corrupt record. Advisory: every writer of the
    file must go through here to get the guarantee. Never raises — a locking failure downgrades to unlocked."""

    def __init__(self, fileobj):
        self._f = fileobj
        self._locked = False

    def __enter__(self):
        try:
            import fcntl
            fcntl.flock(self._f.fileno(), fcntl.LOCK_EX)   # blocks until acquired; auto-released on fd close / process death
            self._locked = True
        except Exception:
            self._locked = False                           # no flock here → proceed unlocked (reads tolerate torn lines)
        return self

    def __exit__(self, *exc):
        try:
            self._f.flush()   # flush BEFORE releasing so the NEXT locker sees a complete file — otherwise a
        except Exception:     # count-then-append (read the file under the lock, then write) races the buffer.
            pass
        if self._locked:
            try:
                import fcntl
                fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        return False
