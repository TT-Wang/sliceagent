"""Resource-access model for safe tool parallelism (ported from Kimi Code's tool-access).

Each tool declares what it touches; the scheduler runs non-conflicting tool calls
concurrently and serializes conflicting ones. Two accesses conflict iff one is
`AllAccess` (globally exclusive, e.g. a shell), OR one writes AND their paths overlap.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileAccess:
    operation: str  # "read" | "write" | "readwrite" | "search"
    path: str
    recursive: bool = False


@dataclass(frozen=True)
class AllAccess:
    """An un-representable side effect (shell, network). Globally exclusive."""


@dataclass(frozen=True)
class ReadAllAccess:
    """A workspace-WIDE READ with no side effect (a read-only explorer subagent: it may read/list/grep/
    recall anywhere, but cannot write/run). Conflicts with any WRITER (AllAccess or a write FileAccess)
    but NEVER with another ReadAllAccess or a plain read — so N explorers fan out concurrently while any
    writing child still serializes. This is what turns the dormant one-at-a-time swarm into a real swarm."""


Access = FileAccess | AllAccess | ReadAllAccess
Accesses = list[Access]


# convenience builders
def none() -> Accesses:
    return []


def all_() -> Accesses:
    return [AllAccess()]


def read_file(path: str) -> Accesses:
    return [FileAccess("read", path)]


def write_file(path: str) -> Accesses:
    return [FileAccess("readwrite", path)]


def search_tree(path: str) -> Accesses:
    return [FileAccess("search", path, recursive=True)]


def _writes(op: str) -> bool:
    return op in ("write", "readwrite")


def _norm(path: str) -> str:
    p = path.replace("\\", "/")
    while "//" in p:
        p = p.replace("//", "/")
    p = p.lower()
    return p[:-1] if len(p) > 1 and p.endswith("/") else p


def _overlap(left: FileAccess, right: FileAccess) -> bool:
    lp, rp = _norm(left.path), _norm(right.path)
    if lp == rp:
        return True
    lpre = lp if lp.endswith("/") else lp + "/"
    rpre = rp if rp.endswith("/") else rp + "/"
    return (left.recursive and rp.startswith(lpre)) or (right.recursive and lp.startswith(rpre))


def _pair_conflict(left: Access, right: Access) -> bool:
    if isinstance(left, AllAccess) or isinstance(right, AllAccess):
        return True
    ra_left, ra_right = isinstance(left, ReadAllAccess), isinstance(right, ReadAllAccess)
    if ra_left or ra_right:
        if ra_left and ra_right:
            return False                                   # two workspace-wide reads never conflict
        other = right if ra_left else left                 # the other side is a FileAccess (AllAccess handled above)
        return _writes(other.operation)                    # read-all conflicts ONLY with a write, never with a read
    if not (_writes(left.operation) or _writes(right.operation)):
        return False  # read/read, read/search never conflict
    return _overlap(left, right)


def conflict(left: Accesses, right: Accesses) -> bool:
    return any(_pair_conflict(a, b) for a in left for b in right)
