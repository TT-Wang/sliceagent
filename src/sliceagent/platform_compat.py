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

# SIGKILL doesn't exist on Windows; POSIX value is unchanged.
SIG_KILL = getattr(signal, "SIGKILL", signal.SIGTERM)


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
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
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
