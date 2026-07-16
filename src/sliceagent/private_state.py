"""Small, race-safe primitives for SliceAgent's user-private durable state.

These helpers are intentionally scoped to state that can contain prompts, model/tool
output, provider choices, or usage records.  They create/repair directories as 0700
and files as 0600, independent of the process umask.  Permission repair is best-effort
on platforms whose ACL model does not implement POSIX modes.
"""
from __future__ import annotations

import os
import tempfile
from typing import IO


def is_private_state_path(path: str) -> bool:
    """Whether ``path`` is under SliceAgent's personal state roots.

    The default ``~/.sliceagent`` remains a personal root even when the main cache is
    relocated, because user skills/preferences still default there.
    """
    candidate = os.path.realpath(os.path.expanduser(path))
    roots = {os.path.realpath(os.path.expanduser("~/.sliceagent"))}
    configured = os.environ.get("SLICEAGENT_CACHE_DIR")
    if configured:
        roots.add(os.path.realpath(os.path.expanduser(configured)))
    for root in roots:
        try:
            if os.path.commonpath((candidate, root)) == root:
                return True
        except ValueError:
            continue
    return False


def private_dir(path: str) -> str:
    # A relative ``.`` (or equivalent ``sub/..`` spelling) is the caller's workspace, not a SliceAgent-
    # owned state directory. Files created there can still be 0600, but silently chmodding the workspace
    # itself to 0700 is an unacceptable side effect. Absolute/custom state roots remain explicit and private.
    caller_cwd_alias = (not os.path.isabs(path) and
                        os.path.realpath(path) == os.path.realpath(os.curdir))
    os.makedirs(path, mode=0o700, exist_ok=True)
    if caller_cwd_alias:
        return path
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def private_file(path: str) -> str:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def open_private_append(path: str, *, encoding: str = "utf-8") -> IO[str]:
    """Open one append-only text record, creating and repairing it as 0600."""
    parent = os.path.dirname(path)
    if parent:
        private_dir(parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
        stream = os.fdopen(fd, "a", encoding=encoding)
        fd = -1
        return stream
    finally:
        if fd >= 0:
            os.close(fd)


def atomic_write_private(path: str, data: str | bytes, *, encoding: str = "utf-8",
                         prefix: str = ".sliceagent-state-") -> None:
    """Atomically replace ``path`` with private data using a unique same-dir temp.

    A unique ``mkstemp`` avoids fixed-``.tmp`` collisions between processes.  The
    replacement inherits 0600 from the temp and the explicit post-repair also fixes
    older permissive files on platforms/filesystems with unusual rename semantics.
    """
    parent = os.path.dirname(path)
    directory = private_dir(parent) if parent else "."
    fd, temp = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    published = False
    try:
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
        payload = data if isinstance(data, bytes) else data.encode(encoding)
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("private state write made no progress")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temp, path)
        published = True
        private_file(path)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if not published:
            try:
                os.unlink(temp)
            except OSError:
                pass
