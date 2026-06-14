"""Memory implementations — the RELEVANT MEMORY tier (cross-session lessons).

memem is the plug: its in-process hybrid retrieval (`memem.retrieve.retrieve`) feeds
the tier each task, and `memem.operations.memory_save` stores lessons. memem indexes a
curated lesson VAULT (not source code), so this is distinct from the code Retriever.

memem stays behind this interface — the moat never imports it directly — and we degrade
gracefully to NullMemory when memem (or its vault) isn't available, so memagent runs
either way. memem reads MEMEM_VAULT / MEMEM_DIR from the env at import time, so set
those (e.g. in .env) before constructing MememMemory.
"""
from __future__ import annotations

from .interfaces import Snippet


class NullMemory:
    """No cross-session memory (the default until a vault is configured)."""

    def recall(self, query: str, k: int = 6) -> list[Snippet]:
        return []

    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "") -> None:
        return None


class MememMemory:
    """Adapter over memem. Construction fails fast if memem isn't importable."""

    def __init__(self) -> None:
        import memem.retrieve  # noqa: F401  — fail fast if memem is absent

    def recall(self, query: str, k: int = 6) -> list[Snippet]:
        from memem.retrieve import retrieve
        try:
            hits = retrieve(query, k=k, log_call_type=None, writeback=False)
        except Exception:
            return []
        out: list[Snippet] = []
        for h in hits:
            text = h.get("body") or h.get("title") or ""
            out.append(Snippet(path=h.get("path", ""), text=text, score=float(h.get("score", 0.0))))
        return out

    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "") -> None:
        from memem.operations import memory_save
        try:
            memory_save(content, title=title, scope_id=scope, tags=tags)
        except Exception:
            pass


def make_memory(prefer_memem: bool = True):
    """Return MememMemory if memem is importable, else NullMemory (graceful)."""
    if prefer_memem:
        try:
            return MememMemory()
        except Exception:
            pass
    return NullMemory()
