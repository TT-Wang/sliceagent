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
import time

IS_WINDOWS = sys.platform == "win32"
_warned_no_bash = False

# SIGKILL doesn't exist on Windows. POSIX keeps the real signal; win32 uses a SENTINEL that can
# never equal SIGTERM — otherwise kill_tree's `sig == SIG_KILL` force-check would be True for the
# GRACEFUL phase too and procman's TERM->wait->KILL ladder would collapse to an immediate /F kill.
SIG_KILL = getattr(signal, "SIGKILL", None) or -9


class ProcessGroupTerminationError(RuntimeError):
    """Raised when a process-tree teardown cannot prove that the owned group is extinct."""


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


def kill_tree(popen: subprocess.Popen, sig: int) -> bool:
    """Terminate a process AND its descendants.
    POSIX: the pre-existing ladder — killpg(getpgid(pid)) falling back to send_signal.
    win32: `taskkill /T` (+/F when the caller asked for SIGKILL-strength), stdlib-only.

    The return value says whether a tree-wide signal was delivered. A direct-leader fallback remains useful
    for best-effort cleanup but returns False because it cannot prove anything about descendants.
    """
    if not IS_WINDOWS:
        try:
            os.killpg(os.getpgid(popen.pid), sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            try:
                popen.send_signal(sig)
            except OSError:
                pass
            return False
    force = ["/F"] if sig == SIG_KILL else []
    try:
        result = subprocess.run(["taskkill", *force, "/T", "/PID", str(popen.pid)],
                                capture_output=True, timeout=10)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        try:
            popen.kill() if force else popen.terminate()
        except OSError:
            pass
        return False


def capture_pgid(popen: subprocess.Popen):
    """Return the owned group id for a process spawned with ``popen_group_kwargs``.

    POSIX ``start_new_session=True`` makes the child a group leader, so pgid is exactly its stable pid; using
    the known pid avoids a race where a very short-lived leader exits before ``os.getpgid`` runs while a
    background descendant remains. Windows tree termination is keyed by the live leader PID instead.
    """
    if IS_WINDOWS:
        return None
    try:
        pid = int(popen.pid)
        return pid if pid > 0 else None
    except (AttributeError, TypeError, ValueError):
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


def process_group_alive(pgid, popen: subprocess.Popen = None) -> bool | None:
    """Return the observed process-group state; ``None`` means extinction cannot be proven.

    POSIX uses the pgid captured while the group leader was alive, so descendants remain observable after
    that leader exits. If capture failed and the leader is gone, its former descendants cannot be identified
    safely and the result is deliberately unknown. Windows' tree operation is synchronous at the compatibility
    seam; the owned leader's settled state is the available proof there.
    """
    if popen is not None:
        try:
            leader_alive = popen.poll() is None
        except Exception:  # noqa: BLE001 - a broken process handle cannot prove extinction
            leader_alive = None
    else:
        leader_alive = None
    if IS_WINDOWS:
        # A dead leader alone says nothing about descendants; only a successful synchronous taskkill /T
        # in terminate_process_group can provide tree-wide proof on this platform.
        return True if leader_alive else None
    if pgid is None:
        return True if leader_alive else None
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        # Permission/race/OS errors are not evidence that the group disappeared.
        return True


def wait_process_group_extinct(pgid, popen: subprocess.Popen = None, timeout: float = 0.0) -> bool:
    """Wait up to *timeout* for a captured process group to disappear, returning proof only."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        if process_group_alive(pgid, popen) is False:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def terminate_process_group(pgid, popen: subprocess.Popen, *, term_timeout: float = 3.0,
                            kill_timeout: float = 2.0) -> bool:
    """TERM→KILL an owned process group and return only after proving group extinction.

    Crucially, signaling is keyed by the spawn-captured pgid and never gated on the leader still running.
    This reaches background descendants after their shell leader has already exited.
    """
    if IS_WINDOWS:
        # If the leader already vanished, taskkill can no longer identify its tree. Be honest instead of
        # treating leader exit as descendant extinction. While it is alive, taskkill /T is synchronous; a
        # zero return followed by leader settlement is the available tree-wide proof.
        try:
            if popen.poll() is not None:
                return False
        except Exception:  # noqa: BLE001
            return False
        delivered = kill_tree(popen, signal.SIGTERM)
        try:
            popen.wait(timeout=max(0.0, float(term_timeout)))
        except (OSError, subprocess.TimeoutExpired):
            pass
        if delivered and popen.poll() is not None:
            return True
        delivered = kill_tree(popen, SIG_KILL)
        try:
            popen.wait(timeout=max(0.0, float(kill_timeout)))
        except (OSError, subprocess.TimeoutExpired):
            pass
        return bool(delivered and popen.poll() is not None)
    if wait_process_group_extinct(pgid, popen, 0.0):
        return True
    signal_pgid(pgid, signal.SIGTERM, popen)
    if wait_process_group_extinct(pgid, popen, term_timeout):
        return True
    signal_pgid(pgid, SIG_KILL, popen)
    return wait_process_group_extinct(pgid, popen, kill_timeout)


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
    """Best-effort EXCLUSIVE advisory lock on an already-open file, held for a ``with`` block.

    POSIX uses ``fcntl.flock``. Windows uses a path-keyed kernel mutex because ``msvcrt.locking`` locks a
    byte range relative to each handle's file position and has a bounded retry loop; neither property is a
    sound fit for independently opened append handles. The named mutex covers threads *and* processes in
    the current Windows session without moving the caller's file position. Serializes concurrent APPENDERS
    to the SAME file (e.g. a resumed session that reuses its session_id, or a future off-thread writer) so
    their lines can't interleave or overwrite one another. Advisory: every writer of the file must go
    through here to get the guarantee. Never raises — a locking failure downgrades to unlocked."""

    def __init__(self, fileobj):
        self._f = fileobj
        self._locked = False
        self._win_handle = None
        self._win_kernel32 = None

    def _enter_windows(self) -> None:
        """Acquire one process-shared mutex for the file's canonical path.

        ``Local\\`` is intentionally scoped to the interactive Windows session: SliceAgent is a local
        single-user application, and using ``Global\\`` would require privileges that ordinary installs do
        not have. ``WAIT_ABANDONED`` still transfers ownership after a crashed writer, matching flock's
        process-death recovery.
        """
        import ctypes
        import hashlib
        from ctypes import wintypes

        raw_path = os.fspath(self._f.name)
        if not isinstance(raw_path, (str, bytes)):
            return
        canonical = os.fsdecode(os.path.normcase(os.path.realpath(os.path.abspath(raw_path))))
        digest = hashlib.sha256(canonical.encode("utf-8", "surrogatepass")).hexdigest()
        mutex_name = f"Local\\SliceAgentFileLock-{digest}"

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateMutexW(None, False, mutex_name)
        if not handle:
            return
        wait_result = kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)  # INFINITE
        if wait_result not in (0x00000000, 0x00000080):  # WAIT_OBJECT_0, WAIT_ABANDONED
            kernel32.CloseHandle(handle)
            return
        self._win_handle = handle
        self._win_kernel32 = kernel32
        self._locked = True

    def __enter__(self):
        try:
            if IS_WINDOWS:
                self._enter_windows()
            else:
                import fcntl
                # Blocks until acquired; automatically released on fd close or process death.
                fcntl.flock(self._f.fileno(), fcntl.LOCK_EX)
                self._locked = True
        except Exception:
            self._locked = False  # unavailable lock primitive → proceed unlocked (reads tolerate torn lines)
        return self

    def __exit__(self, *exc):
        try:
            self._f.flush()   # flush BEFORE releasing so the NEXT locker sees a complete file — otherwise a
        except Exception:     # count-then-append (read the file under the lock, then write) races the buffer.
            pass
        if self._locked and IS_WINDOWS:
            try:
                self._win_kernel32.ReleaseMutex(self._win_handle)
            except Exception:
                pass
            finally:
                try:
                    self._win_kernel32.CloseHandle(self._win_handle)
                except Exception:
                    pass
                self._win_handle = None
                self._win_kernel32 = None
                self._locked = False
        elif self._locked:
            try:
                import fcntl
                fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            self._locked = False
        return False
