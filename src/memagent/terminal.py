"""terminal — persistent interactive PTY sessions (the other half of the live-process gap).

``Sandbox.run`` is one-shot and has no stdin, so a whole class of work is impossible: driving a
REPL or text game through successive prompts, navigating a TUI, sending keys into vim, or just
holding shell + env state (``cd``, ``export``, venv activation) across many tool calls. A
``PtySession`` allocates a real pseudo-terminal (stdlib ``pty``), launches a shell (or any program)
attached to it, and keeps it alive in a registry keyed by name so the agent can ``send`` keys,
``read`` the live output, ``wait`` for an expected pattern (the "expect" primitive that makes
interaction reliable), then ``close``.

stdlib only — ``pty`` + ``select`` + ``fcntl`` (``pexpect`` isn't a dependency). The master fd is
non-blocking; reads drain whatever the program has emitted (the output STREAM — not a rendered
screen-buffer, so full-curses TUIs show raw escape codes, but REPLs / games / line tools read
cleanly). Children run in their own process group so ``close`` takes down the whole tree.
"""
from __future__ import annotations

import codecs
import fcntl
import os
import pty
import re
import select
import signal
import subprocess
import time

from .sandbox import _scrub_env

_READ_CHUNK = 65536
_BUF_CAP = 200_000  # keep the tail of a chatty session bounded


class _Session:
    __slots__ = ("name", "cmd", "master", "popen", "buf", "_decoder")

    def __init__(self, name, cmd, master, popen):
        self.name = name
        self.cmd = cmd
        self.master = master
        self.popen = popen
        self.buf = ""  # output accumulated since the last read/wait returned it
        # STATEFUL utf-8 decoder: a multibyte char split across two os.read() boundaries must not decode to
        # U+FFFD on both halves — the decoder holds the partial sequence until its continuation bytes arrive.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")


class SessionManager:
    """Registry of live interactive PTY sessions. Single-threaded (the agent loop is)."""

    def __init__(self, *, scrub_secrets: bool = True):
        self.scrub_secrets = scrub_secrets
        self._s: dict[str, _Session] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────
    def open(self, name: str, *, cwd: str, command: str | None = None) -> str:
        if name in self._s:
            raise ValueError(f"session {name!r} is already open (close it first, or use another name)")
        master, slave = pty.openpty()
        env = _scrub_env() if self.scrub_secrets else dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TERM", "xterm")
        try:
            try:
                popen = self._spawn_pty(command, cwd, env, slave)
            finally:
                os.close(slave)                   # parent keeps only the master end
        except BaseException:
            os.close(master)                      # #19: Popen failed — don't leak the master fd too
            raise
        try:   # a fcntl failure AFTER spawn must tear down both the master fd AND the orphaned child (#19)
            flags = fcntl.fcntl(master, fcntl.F_GETFL)
            fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except BaseException:
            try:
                os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
            os.close(master)
            raise
        self._s[name] = _Session(name, command or "(shell)", master, popen)
        return name

    def _spawn_pty(self, command, cwd, env, slave):
        """Launch the PTY-attached process. OVERRIDABLE SEAM: a container variant relaunches the same
        command through `docker exec -it` so the session lives INSIDE the task container (host path here)."""
        if command:                               # run the given program directly on the PTY
            return subprocess.Popen(command, shell=True, cwd=cwd, env=env,
                                    stdin=slave, stdout=slave, stderr=slave,
                                    start_new_session=True, close_fds=True)
        shell = os.environ.get("SHELL") or "/bin/bash"   # interactive shell (holds cd/env across turns)
        return subprocess.Popen([shell], cwd=cwd, env=env,
                                stdin=slave, stdout=slave, stderr=slave,
                                start_new_session=True, close_fds=True)

    def send(self, name: str, keys: str, *, enter: bool = True, timeout: float = 30.0) -> str:
        sess = self._get(name)
        data = (keys + ("\n" if enter else "")).encode("utf-8", errors="replace")
        # The master fd is O_NONBLOCK: one os.write may write only PART of a large payload (the rest would be
        # silently dropped), and a FULL buffer raises BlockingIOError (EAGAIN) — back-pressure, not a dead
        # session. Loop until every byte is written, waiting for writability on EAGAIN — but under an OVERALL
        # deadline, so a child that has stopped reading stdin can't wedge the (possibly inline) loop thread
        # forever. Reserve the "may have exited" error for a genuine OSError.
        view = memoryview(data)
        deadline = time.time() + timeout
        try:
            while view:
                try:
                    view = view[os.write(sess.master, view):]
                except BlockingIOError:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise ValueError(f"session {name!r} not writable within {timeout:g}s; the program "
                                         "isn't reading its input (stdin) or its output buffer is full") from None
                    select.select([], [sess.master], [], min(remaining, 5))   # wait for the PTY to drain, then retry
        except OSError as e:
            raise ValueError(f"session {name!r} is not writable ({e}); it may have exited") from None
        # peek (NON-consuming) so the response is still there for a following read/wait — otherwise
        # send would eat the very output the model is about to wait for.
        return self.peek(name, timeout=0.4)

    def read(self, name: str, *, timeout: float = 1.0) -> str:
        sess = self._get(name)
        self._drain(sess, hard=timeout)
        out, sess.buf = sess.buf, ""
        if not out:
            return f"(no output; {self._status(sess)})"
        return out

    def peek(self, name: str, *, timeout: float = 0.6) -> str:
        """Drain output and return it WITHOUT consuming — so a later read/wait still sees it.
        Used right after open() so the program's first prompt isn't eaten by the banner read."""
        sess = self._get(name)
        self._drain(sess, hard=timeout)
        return sess.buf or f"(no output yet; {self._status(sess)})"

    def wait(self, name: str, pattern: str, *, timeout: float = 10.0) -> str:
        """Drain until `pattern` (regex) appears or timeout — the reliable interaction primitive."""
        sess = self._get(name)
        # #20: bound the (model-supplied) pattern — cap its length to limit catastrophic-backtracking
        # surface, and fail clearly on a bad regex instead of crashing the tool. (Python's re has no
        # match timeout; the haystack is this subprocess's own output, so length-capping is the mitigation.)
        if len(pattern) > 500:
            raise ValueError("wait pattern too long (max 500 chars)")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"invalid wait pattern: {e}")
        end = time.time() + timeout
        # inspect already-buffered output at least once BEFORE the timeout loop — so timeout<=0 still polls
        # (matches a buffered hit instead of false-negativing it and then clearing the buffer below).
        self._drain(sess, hard=min(0.4, timeout) if timeout > 0 else 0.0)
        m = rx.search(sess.buf)
        while m is None and time.time() < end:
            self._drain(sess, hard=0.4)
            m = rx.search(sess.buf)
            if m:
                break
            if sess.popen.poll() is not None:   # process died — one last drain, then stop
                self._drain(sess, hard=0.2)
                m = rx.search(sess.buf)
                break
        if m:                                   # expect semantics: consume up to the match, KEEP the rest
            cut = m.end()
            out, sess.buf = sess.buf[:cut], sess.buf[cut:]
            return f"[{name}: matched; {self._status(sess)}]\n{out}"
        out, sess.buf = sess.buf, ""            # timeout: return + clear everything seen
        return f"[{name}: NO match for {pattern!r} before timeout; {self._status(sess)}]\n{out or '(no output)'}"

    def close(self, name: str) -> str:
        sess = self._get(name)
        if sess.popen.poll() is None:
            try:
                os.killpg(os.getpgid(sess.popen.pid), signal.SIGTERM)
            except OSError:
                try:
                    sess.popen.terminate()
                except OSError:
                    pass
            try:
                sess.popen.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(sess.popen.pid), signal.SIGKILL)
                except OSError:
                    pass
                try:
                    sess.popen.wait(timeout=2)   # reap the SIGKILLed child so it isn't left a zombie at fd close
                except (subprocess.TimeoutExpired, OSError):
                    pass
        try:
            os.close(sess.master)
        except OSError:
            pass
        self._s.pop(name, None)
        return f"closed {name}"

    def list(self) -> str:
        if not self._s:
            return "(no terminal sessions)"
        return "\n".join(f"{n}: {self._status(s)} — {s.cmd}" for n, s in self._s.items())

    def cleanup(self) -> None:
        for n in list(self._s):
            try:
                self.close(n)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass

    # ── internals ──────────────────────────────────────────────────────────
    def _get(self, name: str) -> _Session:
        s = self._s.get(name)
        if s is None:
            raise ValueError(
                f"unknown session {name!r}. Open: {', '.join(self._s) or '(none)'}")
        return s

    def _drain(self, sess: _Session, *, hard: float) -> None:
        """Read everything currently available, stopping on a short idle gap or `hard` cap."""
        end = time.time() + hard
        while time.time() < end:
            r, _, _ = select.select([sess.master], [], [], min(0.15, hard))
            if not r:
                break  # no more output right now
            try:
                chunk = os.read(sess.master, _READ_CHUNK)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                break  # master closed / child gone
            if not chunk:
                break  # EOF
            sess.buf += sess._decoder.decode(chunk)   # incremental: holds a partial multibyte tail across reads
            if len(sess.buf) > _BUF_CAP:
                sess.buf = "…[earlier output elided]…\n" + sess.buf[-_BUF_CAP:]

    @staticmethod
    def _status(sess: _Session) -> str:
        rc = sess.popen.poll()
        return "alive" if rc is None else f"exited {rc}"
