"""Grounded filesystem reach for SliceAgent.

The workspace is the default frame, not the complete capability boundary.  ``ReachSet`` keeps the
primary workspace separate from additional focus roots discovered while carrying out the user's
request.  It contains no task semantics and grants no shell capability; it only gives every path-aware
tool one truthful view of the roots the host has already made reachable.

The model-facing internal context namespace is deliberately *not* represented here.  ``@sliceagent``
is a virtual, read-only filesystem routed before physical path resolution.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .platform_compat import IS_WINDOWS


SENSITIVE_DIR_NAMES = frozenset({
    ".ssh", ".aws", ".gnupg", ".gpg", ".kube", ".docker", ".config",
    "keyrings", ".password-store",
})


class ReachSteer(PermissionError):
    """A conclusive path-boundary refusal with a named recovery route.

    It remains a ``PermissionError`` so existing reach recovery can auto-ground an
    exact ordinary HOME target before the refusal reaches a tool outcome.
    """


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def is_sensitive_path(path: str) -> bool:
    """Whether ``path`` traverses a conventional credential directory.

    This is intentionally narrow.  It is the catastrophic privacy floor, not a broad policy engine.
    Exact user-directed secret work can still be handled through a separately explicit host capability.
    """
    parts = [part.casefold() for part in _real(path).split(os.sep) if part]
    return any(part in SENSITIVE_DIR_NAMES for part in parts)


@dataclass(frozen=True)
class FocusRoot:
    path: str
    source: str = "explicit"


class ReachSet:
    """Primary workspace plus grounded, task-local focus roots.

    ``primary`` may be a callable so the test/eval host that follows ``os.chdir`` retains its historical
    behavior.  Roots are ordered by admission and de-duplicated by their canonical real path.
    """

    def __init__(self, primary: str | Callable[[], str]):
        self._primary = primary
        self._focus_roots: list[FocusRoot] = []
        self._active_focus: str | None = None

    @property
    def primary(self) -> str:
        value = self._primary() if callable(self._primary) else self._primary
        return _real(value)

    @property
    def active_focus(self) -> str | None:
        return self._active_focus

    @active_focus.setter
    def active_focus(self, path: str | None) -> None:
        if path is None:
            self._active_focus = None
            return
        real = _real(path)
        if real == self.primary or real in self.roots:
            self._active_focus = real

    @property
    def roots(self) -> tuple[str, ...]:
        out = [self.primary]
        out.extend(root.path for root in self._focus_roots if root.path not in out)
        return tuple(out)

    @property
    def focus_roots(self) -> tuple[str, ...]:
        return tuple(root.path for root in self._focus_roots)

    def add(self, path: str, *, source: str = "explicit", allow_sensitive: bool = False) -> str | None:
        """Add one concrete directory without ever admitting a blanket filesystem/home root."""
        if not path:
            return None
        full = _real(path)
        home = _real("~")
        if full in (os.path.realpath(os.sep), home):
            return None
        if IS_WINDOWS and os.path.splitdrive(full)[1] in ("", os.sep, "/"):
            return None
        if not os.path.isdir(full):
            return None
        if is_sensitive_path(full) and not allow_sensitive:
            return None
        if full == self.primary:
            return full
        if full not in self.focus_roots:
            self._focus_roots.append(FocusRoot(full, str(source or "explicit")))
        return full

    def contains(self, path: str) -> bool:
        full = _real(path)
        return any(full == root or full.startswith(root + os.sep) for root in self.roots)

    def observation_root(self, path: str) -> str | None:
        """Derive the narrow directory needed to inspect one explicit physical path.

        Only existing targets below the user's home are auto-admitted.  This removes the ordinary
        cross-project read dead-end without widening to HOME itself or credential directories.
        """
        if not path:
            return None
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            return None
        full = _real(expanded)
        if self.contains(full):
            return next((root for root in self.roots
                         if full == root or full.startswith(root + os.sep)), self.primary)
        if not os.path.exists(full):
            return None
        home = _real("~")
        if not full.startswith(home + os.sep) or is_sensitive_path(full):
            return None
        directory = full if os.path.isdir(full) else os.path.dirname(full)
        return self.add(directory, source="explicit-observation")

    def target_root(self, path: str) -> str | None:
        """Admit the narrow parent of one explicit absolute target below HOME.

        Unlike :meth:`observation_root`, the leaf may not exist yet.  This supports a direct task-local write
        such as ``edit_file('/Users/me/other-project/new.py', ...)`` without making the whole home directory
        a capability.  Sensitive directories and non-home system locations remain outside automatic reach.
        """
        if not path:
            return None
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            return None
        full = _real(expanded)
        if self.contains(full):
            return next((root for root in self.roots
                         if full == root or full.startswith(root + os.sep)), self.primary)
        home = _real("~")
        if not full.startswith(home + os.sep) or is_sensitive_path(full):
            return None
        directory = full if os.path.isdir(full) else os.path.dirname(full)
        if not os.path.isdir(directory):
            return None
        return self.add(directory, source="explicit-target")
