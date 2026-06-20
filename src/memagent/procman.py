"""procman — background / long-running processes for the agent (the gap the one-shot
``Sandbox.run`` can't fill).

``Sandbox.run`` blocks and returns only on exit, so two whole classes of work are
inexpressible: (1) "start a server, then probe it" (the server never exits), and (2)
multi-minute builds that overrun the run timeout and come back as exit 124. ``ProcManager``
keeps live children in a registry keyed by a short handle (``p1``, ``p2``, …) so the agent
can start a process, keep it alive across turns, ``poll`` / ``tail`` / ``wait``, then ``kill``.

Local subprocess backend (the eval path); cwd-confined and secret-env-scrubbed exactly like
``LocalSandbox``. Output streams to a temp LOGFILE (not a pipe) so ``tail``/``wait`` can read it
AFTER the call returns — a ``Popen`` pipe would deadlock once its OS buffer fills. Children run
in their own process group (``start_new_session=True``) so ``kill`` takes down the whole tree
(a server that forks workers, a build that spawns sub-makes). ``PYTHONUNBUFFERED`` is forced so
Python children flush to the logfile promptly instead of after exit.
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile

from .sandbox import _scrub_env

_TAIL_CHARS = 4000  # cap a tail read so a chatty process can't flood the slice


class _Proc:
    __slots__ = ("handle", "cmd", "popen", "log_path", "log_fh")

    def __init__(self, handle: str, cmd: str, popen, log_path: str, log_fh):
        self.handle = handle
        self.cmd = cmd
        self.popen = popen
        self.log_path = log_path
        self.log_fh = log_fh


class ProcManager:
    """Registry of live background processes. Not threadsafe (the agent loop is single-threaded)."""

    def __init__(self, *, scrub_secrets: bool = True):
        self.scrub_secrets = scrub_secrets
        self._procs: dict[str, _Proc] = {}
        self._n = 0

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self, command: str, *, cwd: str) -> str:
        """Launch `command` in the background; return a handle. Non-blocking."""
        self._n += 1
        handle = f"p{self._n}"
        fd, log_path = tempfile.mkstemp(prefix=f".memagent-{handle}-", suffix=".log")
        log_fh = os.fdopen(fd, "wb")
        env = _scrub_env() if self.scrub_secrets else dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        popen = subprocess.Popen(
            command, shell=True, cwd=cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._procs[handle] = _Proc(handle, command, popen, log_path, log_fh)
        return handle

    def poll(self, handle: str) -> str:
        rc = self._get(handle).popen.poll()
        return "running" if rc is None else f"exited {rc}"

    def tail(self, handle: str, lines: int = 40) -> str:
        p = self._get(handle)
        body = self._read_log(p, lines)
        return f"[{handle} {self.poll(handle)}]\n{body}"

    def wait(self, handle: str, timeout: float) -> str:
        p = self._get(handle)
        try:
            rc = p.popen.wait(timeout=timeout)
            status = f"exited {rc}"
        except subprocess.TimeoutExpired:
            status = f"running (still alive after {timeout:g}s)"
        return f"[{handle} {status}]\n{self._read_log(p, 40)}"

    def kill(self, handle: str) -> str:
        p = self._get(handle)
        if p.popen.poll() is None:
            self._signal_group(p, signal.SIGTERM)
            try:
                p.popen.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._signal_group(p, signal.SIGKILL)
                try:
                    p.popen.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        return f"killed {handle} ({self.poll(handle)})"

    def list(self) -> str:
        if not self._procs:
            return "(no background processes)"
        return "\n".join(f"{h}: {self.poll(h)} — {p.cmd}" for h, p in self._procs.items())

    def cleanup(self) -> None:
        """Kill every live child and remove its logfile. Call at session end; never raises."""
        for h in list(self._procs):
            try:
                self.kill(h)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            p = self._procs.pop(h, None)
            if not p:
                continue
            try:
                p.log_fh.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                os.unlink(p.log_path)
            except OSError:
                pass

    # ── internals ──────────────────────────────────────────────────────────
    def _get(self, handle: str) -> _Proc:
        p = self._procs.get(handle)
        if p is None:
            raise ValueError(
                f"unknown process handle {handle!r}. Live: {', '.join(self._procs) or '(none)'}")
        return p

    @staticmethod
    def _read_log(p: _Proc, lines: int) -> str:
        try:
            p.log_fh.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            with open(p.log_path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
        except OSError:
            data = ""
        tail = "\n".join(data.splitlines()[-max(1, lines):])
        if len(tail) > _TAIL_CHARS:
            tail = "…[earlier output elided]…\n" + tail[-_TAIL_CHARS:]
        return tail or "(no output yet)"

    @staticmethod
    def _signal_group(p: _Proc, sig: int) -> None:
        try:
            os.killpg(os.getpgid(p.popen.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                p.popen.send_signal(sig)
            except OSError:
                pass
