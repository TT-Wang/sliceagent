"""Dependency-scoped workspace revisions.

A Git commit is not a workspace revision: it misses dirty bytes, untracked/generated files and
non-Git projects.  Observations therefore carry fingerprints for the exact paths they depended on.
Unrelated edits do not stale them; a changed dependency does.
"""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from typing import Iterable


def _digest_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _within(root: str, path: str) -> str:
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path if os.path.isabs(path) else os.path.join(root_real, path))
    try:
        if os.path.commonpath((root_real, path_real)) != root_real:
            raise ValueError(f"workspace dependency escapes root: {path}")
    except ValueError as exc:
        raise ValueError(f"workspace dependency escapes root: {path}") from exc
    return path_real


@dataclass(frozen=True)
class PathRevision:
    path: str
    kind: str
    fingerprint: str
    size: int = 0


def fingerprint_path(root: str, path: str) -> PathRevision:
    """Fingerprint one dependency without trusting mtime as content identity."""
    real = _within(root, path)
    rel = os.path.relpath(real, os.path.realpath(root)).replace(os.sep, "/")
    try:
        st = os.lstat(real)
    except FileNotFoundError:
        return PathRevision(rel, "missing", "missing", 0)
    if stat.S_ISLNK(st.st_mode):
        target = os.readlink(real)
        return PathRevision(rel, "symlink", hashlib.sha256(target.encode()).hexdigest(), len(target))
    if stat.S_ISREG(st.st_mode):
        return PathRevision(rel, "file", _digest_file(real), st.st_size)
    if stat.S_ISDIR(st.st_mode):
        # Directory observations depend on its immediate name/type set. File contents are separate
        # dependencies when read, which avoids globally staling a directory observation unnecessarily.
        rows = []
        for name in sorted(os.listdir(real)):
            p = os.path.join(real, name)
            try:
                mode = os.lstat(p).st_mode
                kind = "d" if stat.S_ISDIR(mode) else ("l" if stat.S_ISLNK(mode) else "f")
            except OSError:
                kind = "?"
            rows.append(f"{kind}:{name}")
        body = "\n".join(rows).encode("utf-8", "surrogateescape")
        return PathRevision(rel, "directory", hashlib.sha256(body).hexdigest(), len(rows))
    body = f"{st.st_mode}:{st.st_size}".encode()
    return PathRevision(rel, "other", hashlib.sha256(body).hexdigest(), st.st_size)


@dataclass(frozen=True)
class WorkspaceRevision:
    root: str
    dependencies: tuple[PathRevision, ...]

    @classmethod
    def capture(cls, root: str, paths: Iterable[str]) -> "WorkspaceRevision":
        root_real = os.path.realpath(root)
        unique = sorted({str(path) for path in paths})
        deps = tuple(fingerprint_path(root_real, path) for path in unique)
        return cls(root_real, deps)

    def drifted(self) -> tuple[PathRevision, ...]:
        """Current dependency revisions that differ from this observation."""
        changed = []
        for old in self.dependencies:
            cur = fingerprint_path(self.root, old.path)
            if cur != old:
                changed.append(cur)
        return tuple(changed)

    def is_current(self) -> bool:
        return not self.drifted()

    def as_dict(self) -> dict:
        return {
            "root": self.root,
            "dependencies": [
                {"path": d.path, "kind": d.kind, "fingerprint": d.fingerprint, "size": d.size}
                for d in self.dependencies
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspaceRevision":
        deps = tuple(PathRevision(
            path=str(row.get("path", "")), kind=str(row.get("kind", "missing")),
            fingerprint=str(row.get("fingerprint", "missing")), size=int(row.get("size", 0) or 0),
        ) for row in (data.get("dependencies") or []) if isinstance(row, dict))
        return cls(os.path.realpath(str(data.get("root") or ".")), deps)

